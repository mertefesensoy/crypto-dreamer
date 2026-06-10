"""Comparator and agent policies for the ADR-007 baseline evaluation (Track D).

Implements ADR-007 Section (C) (docs/design/ARCHITECTURE.md): the four
policies the eval harness rolls out over the frozen 72-episode set, plus the
closed-form buy-and-hold integrity value the harness checks per span.

API contract (the eval harness imports exactly this - do not change):

- ``Policy``           - base: ``reset(episode_index) -> None``, ``act(obs) -> int``
- ``FlatPolicy``       - constant action 0 (0% allocation; never trades)
- ``BuyHoldPolicy``    - constant action 4 (100% allocation)
- ``RandomPolicy``     - uniform over the 5 actions, dedicated RNG seed 7
- ``AgentPolicy``      - final PPO checkpoint, deterministic argmax actions
- ``get_policy(name)`` - factory for names 'agent' / 'bh' / 'flat' / 'random'
- ``bh_closed_form_logret(...)`` - ADR-007 (C) closed-form kline B&H value

``obs`` is the env-native dict from ``training.ppo_env.BaselineSpotEnv``:
``window`` (256, 12) float32 and ``portfolio`` (3,) float32. Policies are
feedforward / stateless per ADR-007 (C): the per-step information set is
exactly the env observation, never cross-episode memory, so ``reset`` carries
no persistent state for any policy here.
"""

from __future__ import annotations

import math

import numpy as np

N_ACTIONS = 5  # Discrete(5) target allocation {0, 25, 50, 75, 100}% of equity


class Policy:
    """Base/protocol for all evaluated policies.

    Contract:
    - ``reset(episode_index)`` is called by the harness once before each
      enumerated episode. It must NOT reseed or otherwise reset any RNG
      (ADR-007 (C): RandomPolicy's stream is consumed across episodes in
      enumerated order) and carries no persistent cross-episode state.
    - ``act(obs)`` maps a single env observation dict to an int action in
      ``range(5)``. No side effects beyond internal RNG consumption.
    """

    def reset(self, episode_index: int) -> None:
        """Per-episode hook; default no-op (all policies here are stateless)."""

    def act(self, obs: dict) -> int:
        raise NotImplementedError


class FlatPolicy(Policy):
    """Constant action 0 (0% target allocation). Never trades (ADR-007 (C))."""

    def act(self, obs: dict) -> int:
        return 0


class BuyHoldPolicy(Policy):
    """Constant action 4 (100% target allocation) (ADR-007 (C))."""

    def act(self, obs: dict) -> int:
        return 4


class RandomPolicy(Policy):
    """Uniform random over the 5 actions - sanity reference only.

    ADR-007 (C): dedicated RNG seed 7, created ONCE at construction and
    consumed in enumerated episode order. ``reset`` deliberately does NOT
    reseed - the stream continues across episodes, so two fresh instances
    fed the same step sequence produce identical actions.
    """

    def __init__(self) -> None:
        self._rng = np.random.default_rng(7)

    def act(self, obs: dict) -> int:
        return int(self._rng.integers(0, N_ACTIONS))


class AgentPolicy(Policy):
    """Final PPO checkpoint evaluated deterministically (ADR-007 (C)).

    Action = argmax over policy logits via ``np.argmax``, whose
    lowest-index-wins behavior implements the pre-registered exact-tie
    rule. The network is feedforward; ``reset`` re-initializes nothing
    persistent - there is no cross-episode state per ADR.

    Checkpoint format (Track A contract): a torch-saved dict with at least
    ``state_dict`` (model weights) and ``arch`` (ActorCritic constructor
    kwargs). ``models.ppo`` is imported lazily inside ``__init__`` because
    Track A lands concurrently with this file.
    """

    def __init__(self, ckpt_path: str, device: str = "cpu") -> None:
        try:
            from models.ppo import ActorCritic
        except ImportError as exc:
            raise ImportError(
                "AgentPolicy requires models/ppo.py (Track A deliverable) to "
                "provide ActorCritic; it is not importable yet. Original "
                f"error: {exc}"
            ) from exc
        import torch

        self._torch = torch
        self.device = torch.device(device)
        # torch 2.6 defaults weights_only=True; the checkpoint is a plain
        # dict of tensors + primitive arch kwargs, which that mode allows.
        # Fall back to weights_only=False only for our own trusted artifact
        # if extra non-tensor objects were saved alongside.
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        except Exception:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if "state_dict" not in ckpt or "arch" not in ckpt:
            raise KeyError(
                "checkpoint must contain 'state_dict' and 'arch' keys "
                f"(Track A contract); got keys: {sorted(ckpt.keys())}"
            )
        self.model = ActorCritic(**ckpt["arch"])
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.to(self.device)
        self.model.eval()

    def act(self, obs: dict) -> int:
        torch = self._torch
        with torch.no_grad():
            window = torch.as_tensor(
                np.asarray(obs["window"], dtype=np.float32), device=self.device
            ).unsqueeze(0)
            portfolio = torch.as_tensor(
                np.asarray(obs["portfolio"], dtype=np.float32), device=self.device
            ).unsqueeze(0)
            try:
                out = self.model({"window": window, "portfolio": portfolio})
            except TypeError:
                # ActorCritic.forward(window, portfolio) signature variant.
                out = self.model(window, portfolio)
        if isinstance(out, (tuple, list)):
            out = out[0]  # (logits, value) convention - logits first
        if hasattr(out, "logits"):
            out = out.logits
        logits = out.detach().to("cpu").numpy().reshape(-1)
        if logits.shape[0] != N_ACTIONS:
            raise ValueError(
                f"expected {N_ACTIONS} policy logits, got shape {logits.shape}"
            )
        # np.argmax returns the LOWEST index on exact ties - the pinned
        # ADR-007 (C) tie-break.
        return int(np.argmax(logits))


def get_policy(
    name: str, *, ckpt_path: str | None = None, device: str = "cpu"
) -> Policy:
    """Factory for the four ADR-007 (C) policies.

    Names: 'agent' (requires ckpt_path), 'bh', 'flat', 'random'.
    """
    if name == "agent":
        if ckpt_path is None:
            raise ValueError("get_policy('agent') requires ckpt_path")
        return AgentPolicy(ckpt_path, device=device)
    if name == "bh":
        return BuyHoldPolicy()
    if name == "flat":
        return FlatPolicy()
    if name == "random":
        return RandomPolicy()
    raise ValueError(
        f"unknown policy name {name!r}; expected 'agent', 'bh', 'flat' or 'random'"
    )


def bh_closed_form_logret(
    close_start: float,
    close_end: float,
    fee: float = 0.001,
    slippage: float = 0.0002,
    initial_cash: float = 10000.0,
) -> float:
    """Closed-form kline B&H cumulative net log-return (ADR-007 (C)).

    Enter at the span-start close paying the taker fee (fee x notional,
    10.0 on the 10,000 default) and linear slippage (price marked up by
    ``1 + slippage``), mark to market at the span-end close:

        ln((initial_cash * close_end / (close_start * (1 + slippage))
            - initial_cash * fee) / initial_cash)

    With defaults this is exactly the ADR text
    ``ln((10000 * close_end / (close_start * 1.0002) - 10) / 10000)``.
    The harness asserts its env-rolled B&H span return matches this value
    within |diff| <= 1e-4 (integrity precondition, not a gate). Pure
    float64 - the Phase-3 classifier compares at full precision.
    """
    return math.log(
        (initial_cash * close_end / (close_start * (1.0 + slippage)) - initial_cash * fee)
        / initial_cash
    )


def _self_test() -> None:
    """Inline self-test. Touches NO env and NO eval episode (ADR-007 rule:
    only the pre-named smoke episode may be rolled out pre-gate; this test
    rolls out nothing - fabricated observations only)."""
    import importlib.util

    obs = {
        "window": np.zeros((256, 12), dtype=np.float32),
        "portfolio": np.zeros((3,), dtype=np.float32),
    }

    # Flat / B&H constants.
    flat = get_policy("flat")
    bh = get_policy("bh")
    flat.reset(0)
    bh.reset(0)
    assert flat.act(obs) == 0, "FlatPolicy must always return action 0"
    assert bh.act(obs) == 4, "BuyHoldPolicy must always return action 4"
    for policy in (flat, bh):
        assert isinstance(policy.act(obs), int)
    print("OK flat=0, bh=4")

    # Random: range, determinism across instances, reset() not reseeding.
    r1 = get_policy("random")
    r2 = get_policy("random")
    seq1, seq2 = [], []
    for episode in range(4):
        r1.reset(episode)  # must NOT reseed - interleaved to prove it
        r2.reset(episode)
        seq1.extend(r1.act(obs) for _ in range(50))
        seq2.extend(r2.act(obs) for _ in range(50))
    assert all(a in range(N_ACTIONS) for a in seq1), "random action out of range"
    assert seq1 == seq2, "two RandomPolicy instances diverged (seed-7 stream broken)"
    r3 = RandomPolicy()
    head = [r3.act(obs) for _ in range(200)]
    assert head == seq1, "reset() must not reseed: stream must equal uninterrupted draw"
    print(f"OK random: 200 actions identical across instances, head={seq1[:8]}")

    # Closed-form B&H vs hand-computed reference:
    # 10000*101/(100*1.0002) = 10097.980403919217; -10 fee -> 10087.980403919217;
    # /10000 -> 1.0087980403919217; ln -> 0.008759563152730545.
    hand = 0.008759563152730545
    got = bh_closed_form_logret(100.0, 101.0)
    assert abs(got - hand) <= 1e-12, f"bh closed form {got!r} != hand value {hand!r}"
    # Degenerate sanity: no price move -> pure entry cost, strictly negative.
    assert bh_closed_form_logret(100.0, 100.0) < 0.0
    print(f"OK bh_closed_form_logret(100,101) = {got!r}")

    # Factory error paths.
    for bad in ("nope", "AGENT"):
        try:
            get_policy(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"get_policy({bad!r}) should raise ValueError")
    try:
        get_policy("agent")
    except ValueError:
        pass
    else:
        raise AssertionError("get_policy('agent') without ckpt_path should raise")
    print("OK factory error paths")

    # AgentPolicy: smoke only if Track A's models/ppo.py exists yet.
    if importlib.util.find_spec("models.ppo") is None:
        try:
            AgentPolicy("does_not_matter.ckpt")
        except ImportError as exc:
            assert "models/ppo.py" in str(exc)
            print("OK AgentPolicy raises clear ImportError without models/ppo.py")
        else:
            raise AssertionError("AgentPolicy should raise ImportError pre-Track-A")
        print("SKIP AgentPolicy (models/ppo.py not present yet)")
    else:
        import inspect
        import os
        import tempfile

        import torch

        from models.ppo import ActorCritic

        sig = inspect.signature(ActorCritic.__init__)
        required = [
            p.name
            for p in sig.parameters.values()
            if p.name != "self"
            and p.default is inspect.Parameter.empty
            and p.kind
            in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            )
        ]
        if required:
            print(
                "SKIP AgentPolicy smoke (ActorCritic has required ctor args "
                f"{required}; arch kwargs unknown to Track D)"
            )
        else:
            net = ActorCritic()
            fd, tmp = tempfile.mkstemp(suffix=".ckpt")
            os.close(fd)
            try:
                torch.save({"state_dict": net.state_dict(), "arch": {}}, tmp)
                agent = get_policy("agent", ckpt_path=tmp)
                a0 = agent.act(obs)
                agent.reset(0)
                a1 = agent.act(obs)
                assert a0 in range(N_ACTIONS) and a0 == a1, (
                    "AgentPolicy must be deterministic on identical obs"
                )
                print(f"OK AgentPolicy smoke: deterministic action {a0}")
            finally:
                os.remove(tmp)

    print("SELF-TEST PASS training/baseline_policies.py")


if __name__ == "__main__":
    _self_test()
