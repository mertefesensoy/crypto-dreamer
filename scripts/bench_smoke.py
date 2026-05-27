"""Single-knob benchmark for world-model training throughput.

Configurable via CLI flags so each measurement is independent.

Run:
    # baseline
    python -m scripts.bench_smoke --tag baseline
    # +workers
    python -m scripts.bench_smoke --tag workers --num-workers 4 --pin-memory
    # +batch
    python -m scripts.bench_smoke --tag batch32 --num-workers 4 --pin-memory --batch-size 32
    # +compile
    python -m scripts.bench_smoke --tag compile --num-workers 4 --pin-memory --compile
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import lightning as L
import torch

from models.world_model import WorldModel
from training.datamodule import SpotBTCDataModule

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--T", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--pin-memory", action="store_true")
    p.add_argument("--persistent-workers", action="store_true")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--compile-mode", default="reduce-overhead")
    p.add_argument("--warmup-steps", type=int, default=3)
    p.add_argument("--measured-steps", type=int, default=10)
    p.add_argument("--max-episodes", type=int, default=20)
    p.add_argument("--precision", default="bf16-mixed")  # informational; AMP applied below
    args = p.parse_args()

    L.seed_everything(42, workers=True)
    device = torch.device("cuda")

    dm = SpotBTCDataModule(
        klines_db=str(PROJECT_ROOT / "data" / "market_ro.duckdb"),
        steps_db=str(PROJECT_ROOT / "data" / "market.duckdb"),
        T=args.T, batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_episodes=args.max_episodes,
    )
    # Patch dataloader factory to add pin_memory + persistent_workers we want.
    if args.pin_memory or args.persistent_workers:
        from torch.utils.data import DataLoader, WeightedRandomSampler

        def _train_dl():
            sampler = None
            if dm._train_weights is not None and len(dm._train_ds) > 0:
                g = torch.Generator(); g.manual_seed(42)
                sampler = WeightedRandomSampler(
                    weights=torch.from_numpy(dm._train_weights).double(),
                    num_samples=len(dm._train_ds),
                    replacement=True, generator=g,
                )
            return DataLoader(
                dm._train_ds, batch_size=dm.batch_size, sampler=sampler,
                num_workers=args.num_workers, drop_last=True,
                pin_memory=args.pin_memory,
                persistent_workers=args.persistent_workers and args.num_workers > 0,
            )
        dm.train_dataloader = _train_dl  # type: ignore[method-assign]

    dm.setup()
    train_dl = dm.train_dataloader()

    model = WorldModel(
        mae_checkpoint=str(PROJECT_ROOT / "checkpoints" / "encoder_mae_full_raw.pt"),
    ).to(device)
    model.train()
    if args.compile:
        try:
            model = torch.compile(model, mode=args.compile_mode)
            print(f"[compile] applied mode={args.compile_mode}")
        except Exception as e:
            print(f"[compile] FAILED: {type(e).__name__}: {e}")
            raise
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-6)
    use_bf16 = args.precision == "bf16-mixed"

    it = iter(train_dl)
    print(f"\n=== bench tag={args.tag} bs={args.batch_size} T={args.T} workers={args.num_workers} "
          f"pin={args.pin_memory} compile={args.compile} ===")
    timings = []
    for step in range(args.warmup_steps + args.measured_steps):
        torch.cuda.synchronize()
        t0 = time.time()
        batch = next(it)
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
            loss = model._step(batch, "train") if not args.compile else model.module._step(batch, "train") if hasattr(model, "module") else model._step(batch, "train")

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        torch.cuda.synchronize()
        ms = (time.time() - t0) * 1000
        if step >= args.warmup_steps:
            timings.append(ms)
            tag = "MEAS"
        else:
            tag = "warm"
        print(f"  [{tag}] step={step:>2} {ms:>9.1f} ms")

    timings.sort()
    n = len(timings)
    avg = sum(timings) / n
    median = timings[n // 2]
    print(f"\n=== RESULT tag={args.tag} ===")
    print(f"  measured steps : {n}")
    print(f"  avg  ms/step   : {avg:.1f}")
    print(f"  med  ms/step   : {median:.1f}")
    print(f"  min  ms/step   : {timings[0]:.1f}")
    print(f"  max  ms/step   : {timings[-1]:.1f}")
    print(f"  GPU mem peak   : {torch.cuda.max_memory_allocated() / 1024**3:.2f} GiB")
    print(f"  GPU mem reserve: {torch.cuda.max_memory_reserved() / 1024**3:.2f} GiB")


if __name__ == "__main__":
    main()
