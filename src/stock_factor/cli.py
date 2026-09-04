"""Offline administrative CLI for configuration and research artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stock_factor.config.schema import ConfigError, load_config, load_config_inventory
from stock_factor.domain.research_artifact import ResearchArtifactV2


def _config_inspect(args: argparse.Namespace) -> int:
    print(json.dumps(load_config(args.path).to_dict(), ensure_ascii=False, indent=2, default=str))
    return 0


def _config_verify(args: argparse.Namespace) -> int:
    loaded = load_config(args.path, environment=args.environment)
    print(json.dumps({"verified": True, **loaded.to_dict()}, ensure_ascii=False, indent=2, default=str))
    return 0


def _artifact_verify(args: argparse.Namespace) -> int:
    payload = json.loads(Path(args.path).read_text(encoding="utf-8"))
    artifact = ResearchArtifactV2.from_payload(payload)
    print(json.dumps({"verified": artifact.verify(), "artifact_id": artifact.artifact_id}, indent=2))
    return 0


def _oos_status(args: argparse.Namespace) -> int:
    payload = json.loads(Path(args.path).read_text(encoding="utf-8"))
    print(json.dumps({"status": payload.get("status", "UNKNOWN"), "source": str(Path(args.path))}, indent=2))
    return 0


def _experimental_status(_args: argparse.Namespace) -> int:
    print(json.dumps({"formal_eligible": False, "authority": "experimental"}))
    return 0


def _config_inventory(_args: argparse.Namespace) -> int:
    inventory = load_config_inventory()
    print(json.dumps({"verified": True, "items": [item.to_dict() for item in inventory.values()]}, indent=2))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="stock-factor", description="Offline stock_factor research administration")
    groups = parser.add_subparsers(dest="group", required=True)
    research = groups.add_parser("research", help="research configuration inspection")
    research_commands = research.add_subparsers(dest="command", required=True)
    inspect_cmd = research_commands.add_parser("config-inspect")
    inspect_cmd.add_argument("--path", required=True)
    inspect_cmd.set_defaults(handler=_config_inspect)
    verify_cmd = research_commands.add_parser("config-verify")
    verify_cmd.add_argument("--path", required=True)
    verify_cmd.add_argument("--environment")
    verify_cmd.set_defaults(handler=_config_verify)
    artifact = groups.add_parser("artifact", help="immutable artifact administration")
    artifact_commands = artifact.add_subparsers(dest="command", required=True)
    artifact_verify = artifact_commands.add_parser("verify")
    artifact_verify.add_argument("--path", required=True)
    artifact_verify.set_defaults(handler=_artifact_verify)
    oos = groups.add_parser("oos", help="offline OOS status")
    oos_commands = oos.add_subparsers(dest="command", required=True)
    oos_status = oos_commands.add_parser("status")
    oos_status.add_argument("--path", required=True)
    oos_status.set_defaults(handler=_oos_status)
    experimental = groups.add_parser("experimental", help="non-formal experimental utilities")
    experimental.set_defaults(handler=_experimental_status)
    admin = groups.add_parser("admin", help="administrative utilities")
    admin_commands = admin.add_subparsers(dest="command", required=True)
    inventory = admin_commands.add_parser("config-inventory")
    inventory.set_defaults(handler=_config_inventory)
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
        return args.handler(args)
    except (ConfigError, ValueError, OSError, json.JSONDecodeError) as exc:
        print(json.dumps({"verified": False, "error": str(exc)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
