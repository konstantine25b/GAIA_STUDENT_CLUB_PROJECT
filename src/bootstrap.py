"""Add project root to sys.path and set working directory."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def setup_project() -> Path:
    root = project_root()
    os.chdir(root)
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root


def setup_project_from_cwd() -> Path:
    """Find project root when cwd may be notebooks/ or elsewhere."""
    for candidate in [Path.cwd().resolve(), *Path.cwd().resolve().parents]:
        if (candidate / "src" / "llm_client.py").is_file():
            os.chdir(candidate)
            root_str = str(candidate)
            if root_str not in sys.path:
                sys.path.insert(0, root_str)
            return candidate
    raise RuntimeError(
        "Could not find project root. Open the repo folder in Jupyter or run from the project root."
    )
