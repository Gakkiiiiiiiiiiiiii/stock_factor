from __future__ import annotations

import logging
import os
import socket
import time

from stock_factor.api.dependencies import build_application

LOGGER = logging.getLogger("stock_factor.worker")


def run_forever() -> None:
    logging.basicConfig(level=os.getenv("FACTOR_LOG_LEVEL", "INFO"))
    application = build_application()
    worker_id = os.getenv("FACTOR_WORKER_ID", f"{socket.gethostname()}:{os.getpid()}")
    poll_seconds = float(os.getenv("FACTOR_WORKER_POLL_SECONDS", "2"))
    lease_seconds = int(os.getenv("FACTOR_JOB_LEASE_SECONDS", "900"))
    while True:
        result = application.process_next(worker_id, lease_seconds)
        if result is None:
            time.sleep(poll_seconds)
        else:
            LOGGER.info("factor job processed", extra=result)


def main() -> None:
    run_forever()


if __name__ == "__main__":
    main()
