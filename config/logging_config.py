"""Centralized logging configuration for orchestrator and agents.

This module configures Python logging so that:
- Application logs are emitted with timestamps and module names.
- HTTP client libraries (httpx, httpcore, requests, urllib3) log
  request/response details at DEBUG level.
- LangGraph-related logs are visible at DEBUG.

Use LOG_LEVEL env var to control the base level (default: INFO).
"""

import logging
import os
from typing import Iterable


HTTP_LOGGERS: Iterable[str] = (
    "httpx",
    "httpcore",
    "urllib3",
    "requests",
    # Google client stack (Gmail, etc.)
    "googleapiclient.discovery",
    "google.auth.transport.requests",
)

LANGGRAPH_LOGGERS: Iterable[str] = (
    "langgraph",
    "langgraph_sdk",
)


def setup_logging() -> None:
    """Configure root logging and enable verbose HTTP + LangGraph logs.

    Idempotent: safe to call from multiple modules / processes.
    """
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    root = logging.getLogger()

    # If no handlers are configured yet, set a default format.
    if not root.handlers:
        logging.basicConfig(
            level=level,
            format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        )
    else:
        root.setLevel(level)

    # HTTP clients: always log at DEBUG so we can see URLs, methods, status codes.
    for name in HTTP_LOGGERS:
        logging.getLogger(name).setLevel(logging.DEBUG)

    # LangGraph libraries for routing / RemoteGraph behaviour.
    for name in LANGGRAPH_LOGGERS:
        logging.getLogger(name).setLevel(logging.DEBUG)

    # Optional: reduce noise from very chatty libraries by lowering their level
    # here if needed in the future.
