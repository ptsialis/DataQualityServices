from __future__ import annotations

import os
import re
import secrets
import shutil
from pathlib import Path
from typing import Optional

from flask import session


WORKSPACE_ROOT = Path(
    os.environ.get("WORKSPACE_ROOT", "/tmp/ba_workspaces")
).resolve()

WORKSPACE_ID_PATTERN = re.compile(r"^[a-f0-9]{32}$")


def get_workspace_id(*, create: bool = True) -> Optional[str]:
    workspace_id = session.get("workspace_id")

    if workspace_id is None and create:
        workspace_id = secrets.token_hex(16)
        session["workspace_id"] = workspace_id
        session.modified = True

    if workspace_id is None:
        return None

    if (
        not isinstance(workspace_id, str)
        or not WORKSPACE_ID_PATTERN.fullmatch(workspace_id)
    ):
        raise ValueError("Invalid workspace ID")

    return workspace_id


def _safe_workspace_path(*parts: str, create_workspace: bool = True) -> Path:
    workspace_id = get_workspace_id(create=create_workspace)

    if workspace_id is None:
        raise ValueError("No active workspace")

    WORKSPACE_ROOT.mkdir(mode=0o700, parents=True, exist_ok=True)

    base = (WORKSPACE_ROOT / workspace_id).resolve()

    if base.parent != WORKSPACE_ROOT:
        raise ValueError("Invalid workspace path")

    if create_workspace:
        base.mkdir(mode=0o700, parents=True, exist_ok=True)

    path = base.joinpath(*parts).resolve()

    if path != base and base not in path.parents:
        raise ValueError("Path escapes workspace")

    return path


def workspace_dir(*parts: str) -> Path:
    path = _safe_workspace_path(*parts)
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    return path


def workspace_file(*parts: str) -> Path:
    path = _safe_workspace_path(*parts)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    return path


def remove_current_workspace() -> None:
    workspace_id = get_workspace_id(create=False)

    if workspace_id is None:
        return

    base = _safe_workspace_path(create_workspace=False)

    if base.exists():
        shutil.rmtree(base)