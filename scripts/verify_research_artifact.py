"""Offline verifier for a sealed ResearchArtifactV2 JSON document."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from stock_factor.application.research_artifact_service import ResearchArtifactError
from stock_factor.domain.research_artifact import ResearchArtifactV2


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify a ResearchArtifactV2 without network access")
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--dependency-lock-hash")
    parser.add_argument("--dependency-lock-file", type=Path)
    parser.add_argument("--contract-file", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument(
        "--contract-checksum",
        action="append",
        default=[],
        metavar="NAME=CHECKSUM",
        help="Expected contract checksum; may be supplied more than once",
    )
    args = parser.parse_args()
    try:
        payload = json.loads(args.artifact.read_text(encoding="utf-8"))
        artifact = ResearchArtifactV2.from_payload(payload)
        schema_path = Path(__file__).resolve().parents[1] / "contracts" / "research-artifact.v2.json"
        try:
            from jsonschema import Draft202012Validator

            schema = json.loads(schema_path.read_text(encoding="utf-8"))
            Draft202012Validator(schema).validate(artifact.to_payload())
        except ImportError:
            # Domain validation above remains available in minimal runtime images.
            pass
        if args.dependency_lock_hash and artifact.dependency_lock_hash != args.dependency_lock_hash:
            raise ResearchArtifactError("dependency lock hash mismatch")
        if args.dependency_lock_file:
            lock_hash = hashlib.sha256(args.dependency_lock_file.read_bytes()).hexdigest()
            if artifact.dependency_lock_hash not in {lock_hash, "sha256:" + lock_hash}:
                raise ResearchArtifactError("dependency lock file mismatch")
        expected_checksums: dict[str, str] = {}
        for item in args.contract_checksum:
            name, separator, checksum = item.partition("=")
            if not separator or not name or not checksum:
                raise ResearchArtifactError("contract checksum must use NAME=CHECKSUM")
            expected_checksums[name] = checksum
        if expected_checksums and any(
            artifact.contract_checksums.get(name) != value for name, value in expected_checksums.items()
        ):
            raise ResearchArtifactError("contract checksum mismatch")
        for item in args.contract_file:
            name, separator, path = item.partition("=")
            if not separator or not name or not path:
                raise ResearchArtifactError("contract file must use NAME=PATH")
            checksum = "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()
            if artifact.contract_checksums.get(name) != checksum:
                raise ResearchArtifactError("contract file checksum mismatch")
        if not artifact.verify():
            raise ResearchArtifactError("artifact hash mismatch")
        print(json.dumps({"artifact_id": artifact.artifact_id, "verified": True}, sort_keys=True))
        return 0
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        print(json.dumps({"verified": False, "error": str(exc)}, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
