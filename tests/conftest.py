"""Local pytest fixtures for stable filesystem behavior on Windows."""

from __future__ import annotations

import shutil
from pathlib import Path
from uuid import uuid4

import pytest

ROOT = Path(__file__).parent.parent


@pytest.fixture
def tmp_path() -> Path:
    """Provide a writable temporary directory inside the repository workspace."""
    base_dir = ROOT / "tests" / "_tmp"
    base_dir.mkdir(parents=True, exist_ok=True)
    path = base_dir / f"pytest-work-{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)
