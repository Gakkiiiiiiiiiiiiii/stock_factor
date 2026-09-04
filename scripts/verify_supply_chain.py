"""Offline consistency checks for complete dependency locks and the versioned SBOM."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).parents[1]
LOCKS = (ROOT / "locks" / "core.lock", ROOT / "locks" / "ml-cpu.lock", ROOT / "locks" / "ml-gpu.lock")
SBOM = ROOT / "sbom" / "stock-factor-sbom.v1.json"
PIN = re.compile(r"^([A-Za-z0-9_.-]+)==([^=\s]+)$")
PROFILES = {"core": "core.lock", "ml-cpu": "ml-cpu.lock", "ml-gpu": "ml-gpu.lock"}


def _key(name: str) -> str:
    return name.lower().replace("-", "_")


def _read_lock(path: Path) -> dict[str, str]:
    packages: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        match = PIN.fullmatch(line)
        if match is None:
            raise ValueError(f"un-pinned, editable, or malformed lock line: {path}:{line}")
        key = _key(match.group(1))
        if key in packages:
            raise ValueError(f"duplicate package in lock: {path}:{key}")
        packages[key] = match.group(2)
    if len(packages) < 20:
        raise ValueError(f"lock is unexpectedly small: {path}")
    return packages


def _sbom_packages(sbom: dict) -> dict[tuple[str, str], set[str]]:
    result: dict[tuple[str, str], set[str]] = {}
    for component in sbom.get("components", []):
        if component.get("type") != "library" or not component.get("name") or not component.get("version"):
            raise ValueError("SBOM contains an invalid library component")
        profiles = component.get("profiles")
        if not isinstance(profiles, list) or not profiles or any(profile not in PROFILES for profile in profiles):
            raise ValueError(f"SBOM component has invalid profiles: {component}")
        key = (_key(str(component["name"])), str(component["version"]))
        if key in result:
            raise ValueError(f"duplicate SBOM component: {key}")
        result[key] = set(profiles)
    return result


def verify() -> dict[str, object]:
    parsed = {path.name: _read_lock(path) for path in LOCKS}
    core = parsed["core.lock"]
    cpu = parsed["ml-cpu.lock"]
    gpu = parsed["ml-gpu.lock"]
    for name, profile in (("ml-cpu.lock", cpu), ("ml-gpu.lock", gpu)):
        drift = {key: (core[key], value) for key, value in profile.items() if key in core and core[key] != value}
        if drift:
            raise ValueError(f"{name} drifts from core lock: {sorted(drift)}")
    if "torch" in core:
        raise ValueError("core lock must not contain torch/GPU dependencies")
    if "pyqlib" not in cpu or "pyarrow" not in cpu or "safetensors" not in cpu:
        raise ValueError("ML lock is missing required top-level/transitive closure packages")
    if "torch" not in cpu or not cpu["torch"].endswith("+cpu"):
        raise ValueError("CPU lock must pin an official CPU torch wheel")
    if "torch" not in gpu or not gpu["torch"].endswith("+cu128"):
        raise ValueError("GPU lock must pin a CUDA 12.8 torch wheel")
    if gpu["torch"].split("+")[0] != cpu["torch"].split("+")[0]:
        raise ValueError("CPU/GPU lock torch wheels must share the same base version")
    cpu_header = (ROOT / "locks" / "ml-cpu.lock").read_text(encoding="utf-8")
    if "https://download.pytorch.org/whl/cpu" not in cpu_header:
        raise ValueError("CPU lock is missing its controlled CPU wheel index")
    gpu_header = (ROOT / "locks" / "ml-gpu.lock").read_text(encoding="utf-8")
    if "https://download.pytorch.org/whl/cu128" not in gpu_header:
        raise ValueError("GPU lock is missing its controlled CUDA wheel index")
    cpu_accidental_cuda = sorted(
        name for name in cpu if name.startswith("nvidia_") or name.startswith("cuda_") or name in {"triton"}
    )
    if cpu_accidental_cuda:
        raise ValueError(f"CPU lock must not contain CUDA packages: {cpu_accidental_cuda}")
    required_gpu = {
        "cuda_bindings",
        "cuda_pathfinder",
        "cuda_toolkit",
        "nvidia_cublas_cu12",
        "nvidia_cuda_runtime_cu12",
        "nvidia_cudnn_cu12",
        "nvidia_nccl_cu12",
        "triton",
    }
    missing_gpu = sorted(required_gpu - set(gpu))
    if missing_gpu:
        raise ValueError(f"GPU lock is missing Linux CUDA closure packages: {missing_gpu}")

    sbom = json.loads(SBOM.read_text(encoding="utf-8"))
    if sbom.get("version") != 1 or sbom.get("bomFormat") != "CycloneDX":
        raise ValueError("unsupported SBOM version")
    components = _sbom_packages(sbom)
    expected: dict[tuple[str, str], set[str]] = {}
    for profile, filename in PROFILES.items():
        for name, version in parsed[filename].items():
            expected.setdefault((name, version), set()).add(profile)
    if components != expected:
        missing = sorted(set(expected) - set(components))
        extra = sorted(set(components) - set(expected))
        raise ValueError(f"SBOM does not exactly map lock closure (missing={missing[:5]}, extra={extra[:5]})")
    for path in LOCKS:
        relative = path.relative_to(ROOT).as_posix()
        expected_hash = sbom["metadata"].get("lock_sha256", {}).get(relative)
        actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if expected_hash != actual_hash:
            raise ValueError(f"SBOM lock checksum mismatch: {path}")
    return {
        "valid": True,
        "locks": {name: len(items) for name, items in parsed.items()},
        "sbom_components": len(components),
        "sbom": str(SBOM),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generate-sbom", action="store_true", help="verify only; generation is intentionally offline")
    parser.parse_args()
    print(json.dumps(verify(), sort_keys=True))


if __name__ == "__main__":
    main()
