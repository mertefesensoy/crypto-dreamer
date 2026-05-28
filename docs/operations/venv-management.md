# crypto-dreamer · venv management

## Why this exists

CUDA PyTorch wheels are not on PyPI. A plain `torch>=X` declaration resolves to the CPU-only wheel and silently uninstalls the working CUDA build on the next `uv sync`. The fix is in `pyproject.toml`: a `[[tool.uv.index]]` entry for `https://download.pytorch.org/whl/cu124` plus a `[tool.uv.sources]` rule pinning torch to that index when `sys_platform == 'win32'`. The torch version is pinned to `==2.6.0` for reproducibility. Non-Windows platforms resolve torch from PyPI, so the project remains portable to Linux. With this config, `uv sync` installs `torch==2.6.0+cu124` on Windows and does not drift.

## Fresh setup on a new Windows machine

```
uv venv
uv sync --extra dev
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
```

Expected: `2.6.0+cu124 True 12.4`.

## Common operations

- **Adding a dependency** · always `uv add <pkg>`. Never hand-edit `pyproject.toml` then `uv sync` separately.
- **Installing dev tools after a clean sync** · always `uv sync --extra dev`. Plain `uv sync` drops pytest, ruff, mypy, httpx, websockets, ipython.
- **CUDA torch broke after some change** · re-run `uv sync --extra dev` and the import check above. If the version is wrong, check `uv.lock` for the `torch` entry · it must reference `https://download.pytorch.org/whl/cu124` on win32.

## What NOT to do

- Do NOT run plain `uv sync` without `--extra dev` unless you want a runtime-only env.
- Do NOT `pip install torch ...` to "fix" CUDA outside the uv flow · the next `uv sync` will undo it (this is the 2026-05-27 drift incident).
- Do NOT change `torch==2.6.0` in `pyproject.toml` without re-running the full verification battery.
- Do NOT drop the `explicit = true` flag on the pytorch-cu124 index · without it, uv may pull unrelated packages from PyTorch's index.

## OMI freeze checklist

Before going cold, and again on thaw before training, run all five checks. All must pass:

1. `.\.venv\Scripts\python.exe -c "import torch; print('torch:', torch.__version__); print('cuda available:', torch.cuda.is_available()); print('cuda version:', torch.version.cuda)"` · expect `2.6.0+cu124`, `True`, `12.4`.
2. `.\.venv\Scripts\python.exe -m pytest tests/ -x -q --ignore=tests/test_dream_endpoint.py` · expect `42 passed`.
3. `.\.venv\Scripts\python.exe -m pytest --version` · expect `9.0.x` or newer.
4. `uv sync --extra dev` run a SECOND time · expect `Resolved N packages` with no installs, uninstalls, or torch drift.
5. Re-run check 1 · expect identical output.

If any check fails on thaw, do not start training. Inspect the lock file and the pytorch-cu124 source declaration first.
