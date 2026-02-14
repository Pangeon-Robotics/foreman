from pathlib import Path


def get_version() -> str:
    """Return the current version string."""
    return (Path(__file__).parent / "VERSION").read_text().strip()
