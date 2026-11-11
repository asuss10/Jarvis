"""Logging setup. Call once at program start."""

import logging


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )
    # httpx logs every HTTP request at INFO, which is just noise for us.
    logging.getLogger("httpx").setLevel(logging.WARNING)