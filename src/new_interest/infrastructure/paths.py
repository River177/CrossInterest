from __future__ import annotations

from pathlib import Path


def portable_path(path: Path | str, *, root: Path | None = None) -> str:
    """Serialize repository paths without embedding a checkout location."""
    resolved = Path(path).resolve()
    base = (root or Path.cwd()).resolve()
    try:
        return str(resolved.relative_to(base))
    except ValueError:
        return str(resolved)
