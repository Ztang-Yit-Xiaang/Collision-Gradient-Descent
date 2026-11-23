"""Resolve recorded source paths across the script directory move."""

from pathlib import Path


def source_path(name, root=Path(".")):
    relative = Path(name)
    original = root / relative
    if relative.parent != Path(".") or original.is_file():
        return original
    script = root / "scripts" / relative
    if script.is_file():
        return script
    return root / "src" / "collision_optim" / relative


def manifest_path(path, manifest):
    """Find the recorded key for a current repository-relative source path."""
    name = path.as_posix()
    if name not in manifest and path.parent == Path("scripts"):
        return path.name
    return name
