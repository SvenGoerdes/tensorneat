"""Capture git provenance (SHA + dirty flag) for run reproducibility."""

import subprocess
from pathlib import Path


def capture_git_state(repo_path: str | Path | None = None) -> dict[str, str]:
    """Return git SHA and dirty status for the repo containing repo_path.

    If git is unavailable or the path is not a repo, returns sentinel values
    so logging callers don't have to special-case the failure.
    """
    cwd = str(Path(repo_path).resolve()) if repo_path else None
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        porcelain = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=cwd,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=cwd,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return {
            "git_sha": sha,
            "git_branch": branch,
            "git_dirty": "true" if porcelain else "false",
        }
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {
            "git_sha": "unknown",
            "git_branch": "unknown",
            "git_dirty": "unknown",
        }
