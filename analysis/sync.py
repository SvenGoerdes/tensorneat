"""Sync mlflow.db and .npz result folders from the remote GPU server."""

import os
import subprocess
import sys

SSH_HOST = "20240503@10.10.80.3"
REMOTE_BASE = "~/tensorneat"
REMOTE_EXPERIMENTS = [
    "multi_task_neat_vs_haneat_final",
    "multi_task_haneat_finalv2",
]


def _ssh_pass() -> str:
    pw = os.environ.get("TENSORNEAT_SSH_PASS", "")
    if not pw:
        print("ERROR: Set TENSORNEAT_SSH_PASS env var before syncing.", file=sys.stderr)
        sys.exit(1)
    return pw


def _sshpass_prefix(pw: str) -> list[str]:
    return ["sshpass", "-p", pw]


def _check_sshpass() -> None:
    result = subprocess.run(["which", "sshpass"], capture_output=True)
    if result.returncode != 0:
        print("sshpass not found. Installing via brew...")
        subprocess.run(["brew", "install", "sshpass"], check=True)


def sync_db(local_db_path: str) -> None:
    """Flush WAL on remote then rsync mlflow.db locally."""
    pw = _ssh_pass()
    _check_sshpass()

    print("Flushing WAL on remote mlflow.db...")
    subprocess.run(
        _sshpass_prefix(pw) + [
            "ssh", "-o", "StrictHostKeyChecking=no",
            SSH_HOST,
            f'sqlite3 {REMOTE_BASE}/mlflow.db "PRAGMA wal_checkpoint(TRUNCATE);"',
        ],
        check=True,
    )

    print(f"Syncing mlflow.db → {local_db_path} ...")
    subprocess.run(
        _sshpass_prefix(pw) + [
            "rsync", "-az", "--partial",
            "-e", "ssh -o StrictHostKeyChecking=no",
            f"{SSH_HOST}:{REMOTE_BASE}/mlflow.db",
            local_db_path,
        ],
        check=True,
    )
    print("  mlflow.db synced.")


def sync_results(local_results_root: str) -> None:
    """rsync both experiment result folders into local results/."""
    pw = _ssh_pass()
    _check_sshpass()

    os.makedirs(local_results_root, exist_ok=True)
    for exp in REMOTE_EXPERIMENTS:
        remote_path = f"{SSH_HOST}:{REMOTE_BASE}/results/{exp}/"
        local_path = os.path.join(local_results_root, exp) + "/"
        os.makedirs(local_path, exist_ok=True)
        print(f"Syncing results/{exp}/ ...")
        subprocess.run(
            _sshpass_prefix(pw) + [
                "rsync", "-avz", "--partial",
                "-e", "ssh -o StrictHostKeyChecking=no",
                remote_path,
                local_path,
            ],
            check=True,
        )
        print(f"  results/{exp}/ synced.")
