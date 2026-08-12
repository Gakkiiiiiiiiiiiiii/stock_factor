"""Worker entry point reserved for factor-owned mining/evaluation jobs."""


def main() -> None:
    raise SystemExit("No factor task backend is configured; migrate repositories before starting the worker.")


if __name__ == "__main__":
    main()
