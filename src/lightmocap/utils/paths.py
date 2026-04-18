from pathlib import Path


def resolve_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def ensure_path(path: str | Path) -> Path:
    resolved = resolve_path(path)
    if not resolved.exists():
        raise FileNotFoundError(resolved)
    return resolved
