"""ADR-007 Track A - plain PPO-clip training loop for the model-free baseline.

No Lightning: the on-policy rollout/update cycle needs explicit control over
env auto-resets, GAE truncation handling, episode-start recording, and
checkpoint cadence - all gate-relevant mechanics under ADR-007
(docs/design/ARCHITECTURE.md Section 12).

Contract:
- Envs come from ``training.ppo_env.make_training_envs``: partition-aware
  start sampling over the frozen snapshot ``data/market_ro.duckdb``. EVERY
  realized episode start row (``env._t0`` after each reset, including the
  first) is appended to ``artifacts/adr007/train_starts_seed{seed}.json``
  as ``{"seed": N, "starts": [...]}`` - the (G)(v) gate-precondition
  artifact - flushed at every checkpoint and at the end.
- GAE bootstraps ``V(terminal_obs)`` on truncation (time limit) and 0 on
  termination (equity guardrail): the truncation bootstrap is folded into
  the reward at the truncating step (r += gamma * V(terminal_obs)) and the
  done mask then cuts the GAE chain - equivalent to explicit per-boundary
  bootstrapping and exact for both boundary kinds.
- ``wandb.mode`` must be the literal string 'offline' (explicit ADR-007
  requirement, not inherited); anything else raises before any work starts.
- Checkpoints carry {'state_dict', 'arch' (ActorCritic ctor kwargs),
  'config' (resolved cfg as plain dict), 'seed', 'env_steps'} at every
  ``checkpoint_every_env_steps`` boundary and ALWAYS at exactly
  ``total_env_steps``. Intermediate checkpoints are forensic only and
  GATE-INELIGIBLE per ADR-007 amendment A2.
- NaN/Inf guard: a non-finite loss saves an emergency checkpoint, writes a
  'NONFINITE' heartbeat line, and exits non-zero.

Determinism: ``torch.manual_seed`` + numpy seeding + per-env spawned RNGs
make CPU runs reproducible end to end. GPU training is NOT bit-deterministic
(non-deterministic CUDA kernels) - ADR-007 acknowledges this, which is why
relaunches are never agent-discretionary.

Usage (from the project root):
    uv run python -m training.train_ppo --config configs/ppo_baseline.yaml \
        --seed 42 [dotted.overrides ...]
e.g. smoke:
    uv run python -m training.train_ppo --config configs/ppo_baseline.yaml \
        --seed 42 total_env_steps=4000 n_envs=4 rollout_steps=125 \
        device=cpu wandb.enabled=false checkpoint_every_env_steps=2000
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.distributions import Categorical

from models.ppo import ActorCritic
from training.ppo_env import load_market_data, make_training_envs

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CSV_HEADER = (
    "env_steps,update,loss_pi,loss_v,entropy,approx_kl,clip_frac,"
    "mean_ep_return_recent,mean_ep_len_recent,sps"
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00")


def write_heartbeat(
    path: Path, update: int, env_steps: int, loss_pi: str, status: str = "FINITE"
) -> None:
    line = (
        f"{utc_now_iso()} update={update} env_steps={env_steps} "
        f"loss_pi={loss_pi} {status}\n"
    )
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        f.write(line)


def append_csv_row(path: Path, row: str) -> None:
    new_file = not path.exists()
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        if new_file:
            f.write(CSV_HEADER + "\n")
        f.write(row + "\n")


def flush_train_starts(path: Path, seed: int, starts: list[int]) -> None:
    """(G)(v) artifact: every realized episode start row, in reset order."""
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump({"seed": seed, "starts": starts}, f)
        f.write("\n")


def save_checkpoint(
    path: Path, model: ActorCritic, cfg_dict: dict, seed: int, env_steps: int
) -> None:
    torch.save(
        {
            "state_dict": model.state_dict(),
            "arch": dict(model.ctor_kwargs),
            "config": cfg_dict,
            "seed": seed,
            "env_steps": env_steps,
        },
        path,
    )


def obs_to_tensors(
    obs_list: list[dict], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stack N env obs dicts into (N,256,12) and (N,3) float32 tensors."""
    win = torch.as_tensor(
        np.stack([o["window"] for o in obs_list]), dtype=torch.float32, device=device
    )
    port = torch.as_tensor(
        np.stack([o["portfolio"] for o in obs_list]),
        dtype=torch.float32,
        device=device,
    )
    return win, port


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ADR-007 PPO baseline trainer")
    parser.add_argument("--config", default="configs/ppo_baseline.yaml")
    parser.add_argument("--seed", type=int, default=None)
    args, extra = parser.parse_known_args(argv)

    cfg = OmegaConf.load(args.config)
    if extra:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(extra))
    if args.seed is not None:
        cfg.seed = args.seed

    # ADR-007: wandb offline is an explicit, load-bearing requirement.
    if str(cfg.wandb.mode) != "offline":
        raise RuntimeError(
            f"ADR-007 violation: wandb.mode must be the literal string "
            f"'offline', got {cfg.wandb.mode!r}"
        )

    seed = int(cfg.seed)
    total_env_steps = int(cfg.total_env_steps)
    n_envs = int(cfg.n_envs)
    rollout_steps = int(cfg.rollout_steps)
    epochs = int(cfg.epochs)
    minibatches = int(cfg.minibatches)
    gamma = float(cfg.gamma)
    gae_lambda = float(cfg.gae_lambda)
    clip = float(cfg.clip)
    ent_coef = float(cfg.ent_coef)
    vf_coef = float(cfg.vf_coef)
    max_grad_norm = float(cfg.max_grad_norm)
    episode_steps = int(cfg.episode_steps)
    ckpt_every = int(cfg.checkpoint_every_env_steps)
    hb_every = int(cfg.heartbeat_every_updates)

    batch_env_steps = n_envs * rollout_steps
    assert total_env_steps % batch_env_steps == 0, (
        f"total_env_steps={total_env_steps} must be divisible by "
        f"n_envs*rollout_steps={batch_env_steps}"
    )
    n_updates = total_env_steps // batch_env_steps
    flat_batch = batch_env_steps
    assert flat_batch % minibatches == 0, (
        f"rollout batch {flat_batch} must be divisible by minibatches={minibatches}"
    )
    mb_size = flat_batch // minibatches

    # Determinism: full bit-determinism holds on CPU; GPU training is NOT
    # bit-deterministic (ADR-007 acknowledges; relaunches need amendments).
    torch.manual_seed(seed)
    np.random.seed(seed)

    if str(cfg.device) == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device=cuda requested but CUDA is unavailable")
    device = torch.device(str(cfg.device))

    ckpt_dir = PROJECT_ROOT / str(cfg.out.checkpoints_dir)
    logs_dir = PROJECT_ROOT / str(cfg.out.logs_dir)
    artifacts_dir = PROJECT_ROOT / str(cfg.out.artifacts_dir)
    for d in (ckpt_dir, logs_dir, artifacts_dir):
        d.mkdir(parents=True, exist_ok=True)
    csv_path = logs_dir / f"ppo_baseline_seed{seed}_metrics.csv"
    hb_path = logs_dir / f"heartbeat_ppo_seed{seed}.log"
    starts_path = artifacts_dir / f"train_starts_seed{seed}.json"

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    run = None
    if bool(cfg.wandb.enabled):
        import wandb

        run = wandb.init(
            project=str(cfg.wandb.project),
            name=str(cfg.wandb.run_name).format(seed=seed),
            mode=str(cfg.wandb.mode),
            config=cfg_dict,
        )

    market = load_market_data()
    envs = make_training_envs(market, n_envs, seed, episode_steps=episode_steps)

    model = ActorCritic(
        n_actions=int(cfg.model.n_actions), hidden=int(cfg.model.hidden)
    ).to(device)
    model.train()
    n_params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.lr), eps=1e-5)

    # Initial resets; record every episode start row ((G)(v) precondition).
    starts: list[int] = []
    obs_list = []
    for env in envs:
        o, _ = env.reset()
        obs_list.append(o)
        starts.append(int(env._t0))
    win, port = obs_to_tensors(obs_list, device)

    # Rollout buffers, on device; obs as two tensors (window, portfolio).
    b_win = torch.zeros((rollout_steps, n_envs, 256, 12), device=device)
    b_port = torch.zeros((rollout_steps, n_envs, 3), device=device)
    b_actions = torch.zeros((rollout_steps, n_envs), dtype=torch.long, device=device)
    b_logprobs = torch.zeros((rollout_steps, n_envs), device=device)
    b_rewards = torch.zeros((rollout_steps, n_envs), device=device)
    b_values = torch.zeros((rollout_steps, n_envs), device=device)
    b_dones = torch.zeros((rollout_steps, n_envs), device=device)

    ep_ret = np.zeros(n_envs, dtype=np.float64)
    ep_len = np.zeros(n_envs, dtype=np.int64)
    recent_returns: deque = deque(maxlen=100)
    recent_lens: deque = deque(maxlen=100)

    print(
        f"ppo_baseline start: seed={seed} device={device.type} "
        f"total_env_steps={total_env_steps} updates={n_updates} "
        f"n_envs={n_envs} rollout_steps={rollout_steps} params={n_params}"
    )
    write_heartbeat(hb_path, 0, 0, "na")

    env_steps = 0
    t_train_start = time.perf_counter()
    final_ckpt_path: Path | None = None
    last_loss_pi = float("nan")

    for update in range(1, n_updates + 1):
        t_update_start = time.perf_counter()

        # ---- rollout ------------------------------------------------------
        for t in range(rollout_steps):
            b_win[t] = win
            b_port[t] = port
            with torch.no_grad():
                logits, value = model(win, port)
                dist = Categorical(logits=logits)
                actions = dist.sample()
                logprobs = dist.log_prob(actions)
            b_actions[t] = actions
            b_logprobs[t] = logprobs
            b_values[t] = value

            actions_np = actions.cpu().numpy()
            rewards_np = np.zeros(n_envs, dtype=np.float64)
            dones_np = np.zeros(n_envs, dtype=np.float32)
            next_obs = []
            trunc_env_idx: list[int] = []
            trunc_obs: list[dict] = []
            for i, env in enumerate(envs):
                o, r, terminated, truncated, _info = env.step(int(actions_np[i]))
                rewards_np[i] = r
                ep_ret[i] += r
                ep_len[i] += 1
                if terminated or truncated:
                    dones_np[i] = 1.0
                    if truncated and not terminated:
                        # Time-limit boundary: bootstrap V(terminal_obs).
                        trunc_env_idx.append(i)
                        trunc_obs.append(o)
                    recent_returns.append(float(ep_ret[i]))
                    recent_lens.append(int(ep_len[i]))
                    ep_ret[i] = 0.0
                    ep_len[i] = 0
                    o, _ = env.reset()  # auto-reset
                    starts.append(int(env._t0))
                next_obs.append(o)
            if trunc_env_idx:
                tw, tp = obs_to_tensors(trunc_obs, device)
                with torch.no_grad():
                    _, tv = model(tw, tp)
                for j, i in enumerate(trunc_env_idx):
                    rewards_np[i] += gamma * float(tv[j])
            b_rewards[t] = torch.as_tensor(
                rewards_np, dtype=torch.float32, device=device
            )
            b_dones[t] = torch.as_tensor(dones_np, device=device)
            win, port = obs_to_tensors(next_obs, device)
        env_steps += batch_env_steps

        # ---- GAE (truncation bootstrap already folded into rewards) -------
        with torch.no_grad():
            _, next_value = model(win, port)
        advantages = torch.zeros_like(b_rewards)
        lastgae = torch.zeros(n_envs, device=device)
        for t in reversed(range(rollout_steps)):
            nextvalue = next_value if t == rollout_steps - 1 else b_values[t + 1]
            nonterminal = 1.0 - b_dones[t]
            delta = b_rewards[t] + gamma * nextvalue * nonterminal - b_values[t]
            lastgae = delta + gamma * gae_lambda * nonterminal * lastgae
            advantages[t] = lastgae
        returns = advantages + b_values

        # ---- PPO-clip update ----------------------------------------------
        f_win = b_win.reshape(flat_batch, 256, 12)
        f_port = b_port.reshape(flat_batch, 3)
        f_actions = b_actions.reshape(flat_batch)
        f_logprobs = b_logprobs.reshape(flat_batch)
        f_adv = advantages.reshape(flat_batch)
        f_ret = returns.reshape(flat_batch)

        m_loss_pi, m_loss_v, m_entropy, m_kl, m_clipfrac = [], [], [], [], []
        for _epoch in range(epochs):
            perm = torch.randperm(flat_batch, device=device)
            for start in range(0, flat_batch, mb_size):
                mb = perm[start : start + mb_size]
                logits, newvalue = model(f_win[mb], f_port[mb])
                dist = Categorical(logits=logits)
                newlogprob = dist.log_prob(f_actions[mb])
                entropy = dist.entropy().mean()
                logratio = newlogprob - f_logprobs[mb]
                ratio = logratio.exp()
                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - logratio).mean()
                    clip_frac = ((ratio - 1.0).abs() > clip).float().mean()

                adv = f_adv[mb]
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                pg1 = -adv * ratio
                pg2 = -adv * torch.clamp(ratio, 1.0 - clip, 1.0 + clip)
                loss_pi = torch.max(pg1, pg2).mean()
                loss_v = 0.5 * ((newvalue - f_ret[mb]) ** 2).mean()
                loss = loss_pi - ent_coef * entropy + vf_coef * loss_v

                if not (
                    torch.isfinite(loss_pi)
                    and torch.isfinite(loss_v)
                    and torch.isfinite(entropy)
                ):
                    # NaN/Inf guard: emergency checkpoint, heartbeat, exit 1.
                    emergency = (
                        ckpt_dir
                        / f"ppo_baseline_seed{seed}_step{env_steps}_EMERGENCY.ckpt"
                    )
                    save_checkpoint(emergency, model, cfg_dict, seed, env_steps)
                    flush_train_starts(starts_path, seed, starts)
                    write_heartbeat(
                        hb_path, update, env_steps, "nan", status="NONFINITE"
                    )
                    print(
                        f"NONFINITE loss at update {update} env_steps {env_steps}; "
                        f"emergency checkpoint {emergency.name}"
                    )
                    if run is not None:
                        run.finish(exit_code=1)
                    return 1

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

                m_loss_pi.append(float(loss_pi))
                m_loss_v.append(float(loss_v))
                m_entropy.append(float(entropy))
                m_kl.append(float(approx_kl))
                m_clipfrac.append(float(clip_frac))

        # ---- logging --------------------------------------------------
        sps = batch_env_steps / max(time.perf_counter() - t_update_start, 1e-9)
        loss_pi_mean = float(np.mean(m_loss_pi))
        loss_v_mean = float(np.mean(m_loss_v))
        entropy_mean = float(np.mean(m_entropy))
        kl_mean = float(np.mean(m_kl))
        clipfrac_mean = float(np.mean(m_clipfrac))
        ep_ret_recent = (
            float(np.mean(recent_returns)) if recent_returns else float("nan")
        )
        ep_len_recent = float(np.mean(recent_lens)) if recent_lens else float("nan")
        last_loss_pi = loss_pi_mean

        append_csv_row(
            csv_path,
            f"{env_steps},{update},{loss_pi_mean:.6f},{loss_v_mean:.6f},"
            f"{entropy_mean:.6f},{kl_mean:.6f},{clipfrac_mean:.6f},"
            f"{ep_ret_recent:.6f},{ep_len_recent:.2f},{sps:.1f}",
        )
        if run is not None:
            run.log(
                {
                    "env_steps": env_steps,
                    "update": update,
                    "loss_pi": loss_pi_mean,
                    "loss_v": loss_v_mean,
                    "entropy": entropy_mean,
                    "approx_kl": kl_mean,
                    "clip_frac": clipfrac_mean,
                    "mean_ep_return_recent": ep_ret_recent,
                    "mean_ep_len_recent": ep_len_recent,
                    "sps": sps,
                },
                step=env_steps,
            )
        if update % hb_every == 0:
            write_heartbeat(hb_path, update, env_steps, f"{loss_pi_mean:.6f}")
            print(
                f"update={update}/{n_updates} env_steps={env_steps} "
                f"loss_pi={loss_pi_mean:.6f} loss_v={loss_v_mean:.6f} "
                f"entropy={entropy_mean:.4f} sps={sps:.1f}"
            )

        # ---- checkpoints (cadence + guaranteed final) -------------------
        crossed = (env_steps // ckpt_every) > ((env_steps - batch_env_steps) // ckpt_every)
        if crossed or env_steps == total_env_steps:
            ckpt_path = ckpt_dir / f"ppo_baseline_seed{seed}_step{env_steps}.ckpt"
            save_checkpoint(ckpt_path, model, cfg_dict, seed, env_steps)
            flush_train_starts(starts_path, seed, starts)
            if env_steps == total_env_steps:
                final_ckpt_path = ckpt_path
            print(f"checkpoint saved: {ckpt_path.name} (env_steps={env_steps})")

    # ---- end of training -------------------------------------------------
    flush_train_starts(starts_path, seed, starts)
    write_heartbeat(hb_path, n_updates, env_steps, f"{last_loss_pi:.6f}")
    elapsed = time.perf_counter() - t_train_start
    if run is not None:
        run.finish()
    assert final_ckpt_path is not None and final_ckpt_path.exists(), (
        "final checkpoint missing - ADR-007 requires it at exactly total_env_steps"
    )
    print(
        f"ppo_baseline done: seed={seed} env_steps={env_steps} "
        f"updates={n_updates} final_ckpt={final_ckpt_path.name} "
        f"loss_pi={last_loss_pi:.6f} episodes_started={len(starts)} "
        f"elapsed_s={elapsed:.1f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
