"""
Tools for Document Creator agent.

Local copy to ensure the agent is fully self-contained.
"""

from pathlib import Path
from datetime import datetime
from langchain_core.tools import tool
from typing import Dict, Any
import os


@tool
def write_markdown_file(markdown: str) -> Dict[str, Any]:
    """
    Write markdown content to a file on disk.

    Returns:
        {
          "file_path": "outputs/report_YYYYMMDD_HHMMSS.md",
          "abs_file_path": "/absolute/path/to/outputs/report_YYYYMMDD_HHMMSS.md"
        }
    """
    if not markdown or not markdown.strip():
        raise ValueError("markdown content cannot be empty")
    
    # Ensure outputs directory exists
    os.makedirs("outputs", exist_ok=True)

    # Generate filename with timestamp
    filename = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    path = Path("outputs") / filename

    # Write markdown content to file
    path.write_text(markdown, encoding="utf-8")

    # HARD VERIFICATION: File must exist and not be empty
    if not path.exists():
        raise RuntimeError(f"File creation failed: {path}")
    
    if path.stat().st_size == 0:
        raise RuntimeError(f"File creation failed: {path} is empty")

    # Compute absolute path for cross-service compatibility
    abs_path = path.resolve()
    abs_file_path = abs_path.as_posix()

    return {
        "file_path": str(path),
        "abs_file_path": abs_file_path
    }
