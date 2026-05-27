"""Profile a few training steps of the world model to identify hotspots.

Records per-section CPU+CUDA time over a 2-step warmup + 5 measured
steps using torch.profiler. Prints a breakdown of:
  - dataloader_wait    (CPU time spent fetching the next batch)
  - encode_obs         (encoder forward on B*T windows)
  - rssm_unroll        (the per-step Python loop over T=64)
  - heads_and_loss     (decoder/reward/continue/KL inside the unroll)
  - backward
  - optim_step

This is read-only — does not save weights, doesn't run wandb. Output
goes to stdout only.

Run:
    python -m scripts.profile_smoke
"""
from __future__ import annotations

import time
from pathlib import Path

import lightning as L
import torch
from torch.profiler import ProfilerActivity, profile, record_function, schedule

from models.world_model import WorldModel
from training.datamodule import SpotBTCDataModule

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    L.seed_everything(42, workers=True)
    device = torch.device("cuda")

    dm = SpotBTCDataModule(
        klines_db=str(PROJECT_ROOT / "data" / "market_ro.duckdb"),
        steps_db=str(PROJECT_ROOT / "data" / "market.duckdb"),
        T=64, batch_size=16, num_workers=0, max_episodes=10,
    )
    dm.setup()
    train_dl = dm.train_dataloader()

    model = WorldModel(
        mae_checkpoint=str(PROJECT_ROOT / "checkpoints" / "encoder_mae_full_raw.pt"),
    ).to(device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-6)

    # Patch _step to record sub-section times. Hot patch: wrap encode_obs and
    # the unroll loop body in record_function blocks. We keep the original
    # method intact and just monkeypatch encode_obs via a wrapper.
    orig_encode = model.encode_obs

    def encode_with_record(obs):
        with record_function("encode_obs"):
            return orig_encode(obs)
    model.encode_obs = encode_with_record  # type: ignore[method-assign]

    # Warmup the dataloader iterator
    it = iter(train_dl)

    sched = schedule(skip_first=0, wait=0, warmup=2, active=5, repeat=1)
    print("Profiling 2 warmup + 5 measured steps...")
    print(f"{'step':>5} {'wall_ms':>9}")
    wall_history = []
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=sched,
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    ) as prof:
        for step in range(7):
            t0 = time.time()

            with record_function("dataloader_wait"):
                batch = next(it)
                batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            with record_function("forward_full"):
                loss = model._step(batch, "train")

            with record_function("backward"):
                opt.zero_grad(set_to_none=True)
                loss.backward()

            with record_function("optim_step"):
                opt.step()

            torch.cuda.synchronize()
            ms = (time.time() - t0) * 1000
            wall_history.append(ms)
            print(f"{step:>5} {ms:>9.1f}")
            prof.step()

    print("\n=== Top by CUDA time (self) ===")
    print(prof.key_averages().table(
        sort_by="self_cuda_time_total", row_limit=15, max_name_column_width=70,
    ))

    print("\n=== Top by CPU time (self) ===")
    print(prof.key_averages().table(
        sort_by="self_cpu_time_total", row_limit=15, max_name_column_width=70,
    ))

    print("\n=== Custom record_function regions (CUDA self) ===")
    rows = []
    for evt in prof.key_averages():
        if evt.key in {"dataloader_wait", "encode_obs", "forward_full", "backward",
                       "optim_step"}:
            rows.append((
                evt.key, evt.count,
                evt.self_cpu_time_total / 1000.0,
                evt.self_cuda_time_total / 1000.0,
                evt.cpu_time_total / 1000.0,
                evt.cuda_time_total / 1000.0,
            ))
    print(f"{'name':<20} {'n':>4} {'self_cpu_ms':>14} {'self_cuda_ms':>14} {'tot_cpu_ms':>14} {'tot_cuda_ms':>14}")
    for r in rows:
        print(f"{r[0]:<20} {r[1]:>4} {r[2]:>14.2f} {r[3]:>14.2f} {r[4]:>14.2f} {r[5]:>14.2f}")

    # Wall timing summary
    measured = wall_history[2:]
    avg = sum(measured) / max(len(measured), 1)
    print(f"\n=== Wall-clock summary ===")
    print(f"warmup steps (skipped): {wall_history[:2]}")
    print(f"measured steps        : {measured}")
    print(f"avg ms/step (measured): {avg:.1f}")

    # GPU memory
    print(f"\n=== GPU memory ===")
    print(f"allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GiB")
    print(f"max allocated (this run): {torch.cuda.max_memory_allocated() / 1024**3:.2f} GiB")
    print(f"reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GiB")


if __name__ == "__main__":
    main()
