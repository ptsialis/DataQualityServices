from __future__ import annotations

import os
import re
import secrets
import shutil
import tempfile
from pathlib import Path

import pandas as pd
from flask import session


# Root folder can be overridden in Docker or production.
SESSION_DATA_ROOT = Path(
    os.environ.get("SESSION_DATA_PATH", "session_data")
).resolve()

# Only known dataframe keys may become filenames.
ALLOWED_DATAFRAME_KEYS = {
    "original",
    "anomaly",
    "impute",
    "inference",
    "personal_0",
    "personal_1",
    "metadata_store",
}

WORKSPACE_ID_PATTERN = re.compile(r"^[a-f0-9]{32}$")


def get_workspace_id() -> str:
    """
    Returns a random workspace ID tied to the current Flask session.

    The ID is stored in the signed Flask session cookie.
    Each browser session receives its own server-side data directory.
    """
    workspace_id = session.get("workspace_id")

    if workspace_id is None:
        workspace_id = secrets.token_hex(16)
        session["workspace_id"] = workspace_id
        session.modified = True

    if (
        not isinstance(workspace_id, str)
        or not WORKSPACE_ID_PATTERN.fullmatch(workspace_id)
    ):
        raise ValueError("Invalid workspace ID")

    return workspace_id


def get_workspace_path() -> Path:
    """
    Returns the isolated directory for the current browser session.
    """
    SESSION_DATA_ROOT.mkdir(mode=0o700, parents=True, exist_ok=True)

    workspace_path = (SESSION_DATA_ROOT / get_workspace_id()).resolve()

    # Prevent path traversal.
    if workspace_path.parent != SESSION_DATA_ROOT:
        raise ValueError("Invalid workspace path")

    workspace_path.mkdir(mode=0o700, parents=True, exist_ok=True)
    return workspace_path

def get_session_file_path(*parts: str) -> Path:
    """
    Safely resolves a file path inside the current session workspace.

    Use this for exports, uploads, and model artifacts.
    """
    base = get_workspace_path()
    path = base.joinpath(*parts).resolve()

    if path != base and base not in path.parents:
        raise ValueError("Path escapes workspace")

    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    return path


def _get_dataframe_path(key: str) -> Path:
    """
    Resolves a dataframe path safely inside the current workspace.
    """
    if key not in ALLOWED_DATAFRAME_KEYS:
        raise ValueError(f"Unsupported dataframe key: {key}")

    return get_workspace_path() / f"{key}.csv"


def init_session_state() -> None:
    """
    Initializes primitive Flask-session values and creates the isolated
    workspace directory for the current browser session.

    Empty CSV placeholder files are intentionally not created.
    Missing dataframe files are treated as empty DataFrames when loaded.
    """
    defaults = {
        "language": "de",
        "starting_process": True,
        "intermediate_process": False,
        "ending_process": False,
        "intial_app_run": True,
        "show_upload": False,
        "dataset_name": "",
        "button_clicked": False,
    }

    for key, value in defaults.items():
        if key not in session:
            session[key] = value

    get_workspace_path()
    session.modified = True


def save_dataframe_to_session(key: str, df: pd.DataFrame) -> None:
    """
    Saves a dataframe atomically inside the current session workspace.

    Atomic replacement prevents partially written CSV files if two requests
    from the same browser session overlap.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pandas DataFrame for '{key}'")

    target_path = _get_dataframe_path(key)

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".tmp",
        dir=target_path.parent,
        delete=False,
        encoding="utf-8",
        newline="",
    ) as tmp:
        temp_path = Path(tmp.name)

    try:
        df.to_csv(temp_path, index=True)
        os.replace(temp_path, target_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def load_dataframe_from_session(key: str) -> pd.DataFrame:
    """
    Loads a dataframe from the current session workspace.

    Returns an empty DataFrame when the file does not exist or cannot be read.
    """
    path = _get_dataframe_path(key)

    if not path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(path, index_col=0)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    except Exception as exc:
        print(f"Fehler beim Laden des DataFrames für '{key}': {exc}")
        return pd.DataFrame()


def remove_current_workspace() -> None:
    """
    Deletes only the files belonging to the current browser session.
    """
    workspace_id = session.get("workspace_id")

    if workspace_id is None:
        return

    if (
        not isinstance(workspace_id, str)
        or not WORKSPACE_ID_PATTERN.fullmatch(workspace_id)
    ):
        raise ValueError("Invalid workspace ID")

    workspace_path = (SESSION_DATA_ROOT / workspace_id).resolve()

    if workspace_path.parent != SESSION_DATA_ROOT:
        raise ValueError("Invalid workspace path")

    shutil.rmtree(workspace_path, ignore_errors=True)