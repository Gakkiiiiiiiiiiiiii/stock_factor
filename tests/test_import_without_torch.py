"""Ensure lightweight technical imports do not require the ML runtime."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path


def test_lightweight_imports_do_not_load_torch() -> None:
    """Importing core evaluation/data helpers must work with Torch blocked."""
    source_root = Path(__file__).resolve().parents[1] / "src"
    script = textwrap.dedent(
        """
        import builtins

        original_import = builtins.__import__

        def block_torch(name, *args, **kwargs):
            if name == "torch" or name.startswith("torch."):
                raise AssertionError("lightweight import unexpectedly loaded torch")
            return original_import(name, *args, **kwargs)

        builtins.__import__ = block_torch

        import stock_factor.technical_transformer.evaluation as evaluation
        from stock_factor.technical_transformer.data.features import build_features

        assert callable(build_features)
        assert callable(evaluation.run_reliability_evaluation)
        """
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join([str(source_root), environment.get("PYTHONPATH", "")])
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=environment,
        check=False,
    )
    assert result.returncode == 0, result.stderr
