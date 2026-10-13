"""Minimal ``.env`` loader + typed environment helpers (stdlib only).

RAGRAM_Pro keeps its core dependency-free, so instead of python-dotenv this
module parses ``KEY=VALUE`` files itself. Entry points (``python app.py``,
``examples/fastapi_app.py``) call :func:`load_env` once at startup; library
modules then read configuration from ``os.environ`` as usual. Real
environment variables (shell/CI) always win over file values.

File discovery order (first hit wins):

1. explicit ``path`` argument (missing file raises ``FileNotFoundError``)
2. ``$RAGRAM_ENV_FILE``
3. ``./.env`` (current working directory)
4. ``<directory of this module>/.env`` (repo root when running from source)

Syntax accepted per line: ``KEY=value``, ``export KEY=value``, optional
quotes (``'`` / ``"``), full-line and unquoted-inline ``#`` comments.
Blank values are treated as *unset* so placeholders never mask defaults.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

__all__ = ["env_float", "env_int", "env_str", "load_env"]

_ASSIGNMENT = re.compile(r"^(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=(.*)$")


def _resolve(path: str | os.PathLike[str] | None) -> Path | None:
    if path is not None:
        candidate = Path(path)
        if not candidate.is_file():
            raise FileNotFoundError(f".env file not found: {candidate}")
        return candidate
    override = os.environ.get("RAGRAM_ENV_FILE")
    if override:
        candidate = Path(override)
        if not candidate.is_file():
            raise FileNotFoundError(f"$RAGRAM_ENV_FILE points to a missing file: {candidate}")
        return candidate
    for candidate in (Path.cwd() / ".env", Path(__file__).resolve().parent / ".env"):
        if candidate.is_file():
            return candidate
    return None


def _unquote(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        inner = value[1:-1]
        return inner.replace("\\n", "\n") if value[0] == '"' else inner
    # unquoted: strip an inline "# comment"
    return re.split(r"\s+#", value, maxsplit=1)[0].strip()


def load_env(
    path: str | os.PathLike[str] | None = None, *, override: bool = False
) -> dict[str, str]:
    """Load a ``.env`` file into ``os.environ`` (idempotent, non-destructive).

    Returns the key/value pairs found in the file. Values already present in
    ``os.environ`` are left untouched unless ``override=True``. Empty values
    are skipped entirely (treated as unset).
    """
    resolved = _resolve(path)
    if resolved is None:
        return {}
    loaded: dict[str, str] = {}
    for raw_line in resolved.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = _ASSIGNMENT.match(line)
        if match is None:
            continue  # tolerate blank/junk lines instead of failing startup
        key, raw_value = match.group(1), match.group(2).strip()
        value = _unquote(raw_value)
        if not value:
            continue  # blank placeholder -> unset
        loaded[key] = value
        if override or key not in os.environ:
            os.environ[key] = value
    return loaded


def env_str(name: str, default: str | None = None) -> str | None:
    """Read a string env var; blank/whitespace-only counts as unset."""
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    return value


def env_int(name: str, default: int) -> int:
    """Read an int env var; raises ``ValueError`` on a non-numeric value."""
    value = env_str(name)
    if value is None:
        return default
    try:
        return int(value.strip())
    except ValueError as exc:
        raise ValueError(f"environment variable {name}={value!r} is not an integer") from exc


def env_float(name: str, default: float) -> float:
    """Read a float env var; raises ``ValueError`` on a non-numeric value."""
    value = env_str(name)
    if value is None:
        return default
    try:
        return float(value.strip())
    except ValueError as exc:
        raise ValueError(f"environment variable {name}={value!r} is not a number") from exc
