from __future__ import annotations

from pathlib import Path

from . import config
from .db import source_ref_registered_for_kb


def _resolve_source_path(raw_source: str) -> Path:
    source_path = Path(raw_source.replace("\\", "/"))
    if source_path.is_absolute():
        return source_path.resolve()
    return (Path.cwd() / source_path).resolve()


def resolve_allowed_source_path(raw_source: str, kb_id: int) -> Path:
    source_path = _resolve_source_path(raw_source)
    upload_root = (Path(config.settings.data_dir) / "uploads" / f"kb_{kb_id}").resolve()

    inside_upload_root = False
    try:
        source_path.relative_to(upload_root)
        inside_upload_root = True
    except ValueError:
        inside_upload_root = False

    if not inside_upload_root and not source_ref_registered_for_kb(source_path, kb_id):
        raise PermissionError("source_outside_kb_root")

    if not source_path.exists() or not source_path.is_file():
        raise FileNotFoundError("source_file_not_found")

    return source_path
