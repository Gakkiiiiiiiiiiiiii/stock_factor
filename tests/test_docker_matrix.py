from pathlib import Path


def test_docker_profiles_are_separate_and_commands_are_executable():
    dockerfile = (Path(__file__).parents[1] / "Dockerfile").read_text(encoding="utf-8")
    for target in ("core", "worker", "ml-cpu", "ml-gpu"):
        assert f"AS {target}" in dockerfile
    core_section = dockerfile.split("FROM base AS core", 1)[1].split("FROM core AS worker", 1)[0]
    assert "torch" not in core_section.lower()
    assert "locks/core.lock" in core_section
    for target in ("ml-cpu", "ml-gpu"):
        marker = f"FROM base AS {target}" if target == "ml-cpu" else "FROM ml-cpu AS ml-gpu"
        section = dockerfile.split(marker, 1)[1]
        assert "uvicorn" in section.split("CMD", 1)[-1]
        assert f"locks/{target}.lock" in section
    gpu = dockerfile.split("FROM ml-cpu AS ml-gpu", 1)[1]
    assert "download.pytorch.org/whl/cu128" in gpu
    cpu = dockerfile.split("FROM base AS ml-cpu", 1)[1].split("# GPU builds", 1)[0]
    assert "download.pytorch.org/whl/cpu" in cpu
    assert "torch==2.11.0+cpu" in cpu
    assert "cu128" not in cpu and "nvidia-" not in cpu
    assert "--no-deps -r" in cpu
    assert "pip check" in cpu
    assert "pip check" in gpu
