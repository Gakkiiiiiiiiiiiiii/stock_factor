# CI profiles and main-branch protection

This repository separates deterministic core checks from the optional Technical
Transformer runtime. The profiles are defined in `pyproject.toml`:

| Profile | Installs | Intended use |
| --- | --- | --- |
| `test-core` | pytest, coverage, Ruff | Core unit, contract, research-integrity, and lint jobs; no Torch |
| `ml-cpu` | CPU-capable Torch runtime | Local/CI ML runtime base |
| `test-ml` | `test-core` dependencies plus pandas, pyarrow, Torch, safetensors, and pyqlib | Complete `tests/technical_transformer` CPU suite |
| `ml-gpu` | Torch runtime | GPU environments; select the CUDA-compatible Torch index at install time |
| `test` / `technical` | Legacy-compatible aliases | Existing callers; retained during migration |

The blocking workflow job contexts are:

- `lint`
- `core-unit`
- `ml-unit-cpu`
- `research-integrity`
- `contract`
- `integration`
- `docker-build`

`ml-unit-cpu` installs `.[test-ml]` and runs the complete
`tests/technical_transformer` directory. It must not blanket-skip tests when
Torch is unavailable; a missing or broken ML dependency is an environment/job
failure. Individual tests may use `importorskip` only when the test itself is
an explicitly optional compatibility probe.

## Branch-protection checklist

Configure the `main` branch in GitHub repository settings with the seven job
contexts above as required status checks, require branches to be up to date,
and require CODEOWNERS review for protected paths. Keep force-push and branch
deletion disabled. The stable names are an interface used by branch
protection, so rename a job only with a coordinated settings update.

This document records the required baseline; it does **not** claim that remote
GitHub branch protection has been enabled. Verify the repository settings and
capture the settings export or review evidence separately after enabling it.

## Local commands

```powershell
python -m pip install -e ".[test-core]"
python -m ruff check src tests
python -m ruff format --check src tests
python -m pytest -q --ignore=tests/technical_transformer --ignore=tests/test_integration.py --basetemp=.test-tmp-core

python -m pip install -e ".[test-ml]"
python -m pytest -q tests/technical_transformer --basetemp=.test-tmp-ml
```

On Windows, use repository-local `--basetemp` paths to avoid default temporary
directory permission failures. Do not commit those directories.
