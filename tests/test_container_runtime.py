import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_container_config_root_and_worker_entrypoint_are_explicit_and_real():
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")

    assert "ENV STOCK_FACTOR_CONFIG_ROOT=/app/config" in dockerfile
    assert 'CMD ["python", "-m", "stock_factor.workers.factor_worker"]' in dockerfile
    assert "STOCK_FACTOR_CONFIG_ROOT: /app/config" in compose
    assert 'command: ["python", "-m", "stock_factor.workers.factor_worker"]' in compose


def test_worker_module_is_importable_without_starting_poll_loop():
    from stock_factor.workers import factor_worker

    assert callable(factor_worker.main)
    assert callable(factor_worker.run_forever)


def test_installed_style_config_root_honors_explicit_runtime_environment(tmp_path):
    environment = os.environ.copy()
    environment["STOCK_FACTOR_CONFIG_ROOT"] = str(tmp_path)
    result = subprocess.run(
        [sys.executable, "-c", "from stock_factor.config import schema; print(schema.CONFIG_ROOT)"],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert Path(result.stdout.strip()) == tmp_path.resolve()
