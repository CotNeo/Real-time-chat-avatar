"""
Structured logging (Section 19 groundwork).

One JSON object per line to stdout in production; a readable console renderer
in dev. No Prometheus/Grafana for this MVP (Section 19/31) — metrics are read
via the FastAPI /metrics endpoint (Milestone 14), logs are for events/errors.
"""
from __future__ import annotations

import logging
import sys

import structlog


def configure_logging(level: str = "info", fmt: str = "json") -> None:
    logging.basicConfig(
        format="%(message)s", stream=sys.stdout, level=level.upper()
    )

    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    renderer = (
        structlog.processors.JSONRenderer()
        if fmt == "json"
        else structlog.dev.ConsoleRenderer()
    )

    structlog.configure(
        processors=[*shared_processors, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelName(level.upper())
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)
