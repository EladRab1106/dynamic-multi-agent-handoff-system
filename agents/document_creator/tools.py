"""
Tools for Document Creator agent.

Local copy to ensure the agent is fully self-contained.
"""

from pathlib import Path
from datetime import datetime
from langchain_core.tools import tool
from typing import Dict, Any
import logging
import os
import base64


logger = logging.getLogger(__name__)


@tool
def write_markdown_file(markdown: str) -> Dict[str, Any]:
    """
    Write markdown content to a file on disk.

    Returns:
        {
          "file_path": "outputs/report_YYYYMMDD_HHMMSS.md",
          "abs_file_path": "/absolute/path/to/outputs/report_YYYYMMDD_HHMMSS.md",
          "file_base64": "<base64-encoded file content>",
          "filename": "report_YYYYMMDD_HHMMSS.md",
          "mime_type": "text/markdown"
        }
    """
    logger.info(
        "write_markdown_file: starting file creation",
        extra={
            "cwd": os.getcwd(),
            "markdown_length": len(markdown or ""),
        },
    )

    if not markdown or not markdown.strip():
        logger.error("write_markdown_file: markdown content is empty; aborting")
        raise ValueError("markdown content cannot be empty")

    try:
        # Log presence of potential mount point used in Cloud Run
        mounted_outputs = Path("/outputs")
        if mounted_outputs.exists():
            logger.info(
                "write_markdown_file: detected '/outputs' directory on filesystem",
                extra={"mounted_outputs_abs": str(mounted_outputs.resolve())},
            )
        else:
            logger.info("write_markdown_file: '/outputs' directory does not exist")

        # Ensure local outputs directory exists (this is where we actually write)
        outputs_dir = Path("outputs")
        os.makedirs(outputs_dir, exist_ok=True)
        logger.info(
            "write_markdown_file: ensured outputs directory exists",
            extra={"outputs_dir_abs": str(outputs_dir.resolve())},
        )

        # Generate filename with timestamp
        filename = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        path = outputs_dir / filename
        logger.info(
            "write_markdown_file: attempting to write markdown file",
            extra={"target_path": str(path), "target_abs_path": str(path.resolve())},
        )

        # Write markdown content to file
        path.write_text(markdown, encoding="utf-8")

        # HARD VERIFICATION: File must exist and not be empty
        if not path.exists():
            logger.error(
                "write_markdown_file: file does not exist after write",
                extra={"target_abs_path": str(path.resolve())},
            )
            raise RuntimeError(f"File creation failed: {path}")

        file_size = path.stat().st_size
        if file_size == 0:
            logger.error(
                "write_markdown_file: file is empty after write",
                extra={"target_abs_path": str(path.resolve())},
            )
            raise RuntimeError(f"File creation failed: {path} is empty")

        # Compute absolute path for cross-service compatibility
        abs_path = path.resolve()
        abs_file_path = abs_path.as_posix()

        # Read file bytes and encode as base64 so other agents can reconstruct the file
        try:
            file_bytes = path.read_bytes()
            file_base64 = base64.b64encode(file_bytes).decode("utf-8")
        except Exception:
            logger.exception(
                "write_markdown_file: failed to read file bytes for base64 encoding",
                extra={"target_abs_path": str(path.resolve())},
            )
            raise

        result = {
            "file_path": str(path),
            "abs_file_path": abs_file_path,
            "file_base64": file_base64,
            "filename": filename,
            "mime_type": "text/markdown",
        }
        logger.info(
            "write_markdown_file: file successfully created",
            extra={
                "file_path": result["file_path"],
                "abs_file_path": result["abs_file_path"],
                "file_size_bytes": file_size,
            },
        )
        return result

    except Exception:
        logger.exception("write_markdown_file: unexpected error while creating markdown file")
        raise
