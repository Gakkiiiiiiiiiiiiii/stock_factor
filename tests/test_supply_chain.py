from scripts.verify_supply_chain import verify


def test_versioned_locks_and_sbom_are_consistent_offline():
    result = verify()
    assert result["valid"] is True
    assert result["locks"]["core.lock"] >= 20
    assert result["locks"]["ml-cpu.lock"] > result["locks"]["core.lock"]


def test_linux_profile_locks_separate_cpu_and_cuda_runtime_closures():
    from scripts.verify_supply_chain import ROOT, _read_lock

    cpu = _read_lock(ROOT / "locks" / "ml-cpu.lock")
    gpu = _read_lock(ROOT / "locks" / "ml-gpu.lock")
    assert cpu["torch"] == "2.11.0+cpu"
    assert not any(name.startswith(("cuda_", "nvidia_")) or name == "triton" for name in cpu)
    assert gpu["torch"] == "2.11.0+cu128"
    assert {"cuda_toolkit", "nvidia_cudnn_cu12", "nvidia_nccl_cu12", "triton"} <= set(gpu)
