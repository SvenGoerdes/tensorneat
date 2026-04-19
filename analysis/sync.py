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

SSHPASS = "/opt/homebrew/bin/sshpass"
SSH = "/usr/bin/ssh"
SCP = "/usr/bin/scp"
TAR = "/usr/bin/tar"

SSH_OPTS = ["-o", "StrictHostKeyChecking=no", "-o", "BatchMode=no"]


def _pw() -> str:
    pw = os.environ.get("TENSORNEAT_SSH_PASS", "")
    if not pw:
        print("ERROR: Set TENSORNEAT_SSH_PASS env var before syncing.", file=sys.stderr)
        sys.exit(1)
    return pw


def _sshpass(pw: str, cmd: list[str]) -> list[str]:
    return [SSHPASS, "-p", pw] + cmd


def _run(cmd: list[str], **kwargs) -> None:
    subprocess.run(cmd, check=True, **kwargs)


def sync_db(local_db_path: str) -> None:
    """Flush WAL on remote then scp mlflow.db locally."""
    pw = _pw()

    print("Flushing WAL on remote mlflow.db...")
    _run(_sshpass(pw, [SSH] + SSH_OPTS + [
        SSH_HOST,
        f'sqlite3 {REMOTE_BASE}/mlflow.db "PRAGMA wal_checkpoint(TRUNCATE);" 2>/dev/null || true',
    ]))

    print(f"Copying mlflow.db → {local_db_path} ...")
    _run(_sshpass(pw, [SCP, "-o", "StrictHostKeyChecking=no",
                       f"{SSH_HOST}:{REMOTE_BASE}/mlflow.db",
                       local_db_path]))
    print("  mlflow.db copied.")


def sync_results(local_results_root: str) -> None:
    """Copy experiment result folders via tar-pipe over SSH (no rsync needed on remote)."""
    pw = _pw()
    os.makedirs(local_results_root, exist_ok=True)

    for exp in REMOTE_EXPERIMENTS:
        local_exp_dir = os.path.join(local_results_root, exp)
        os.makedirs(local_exp_dir, exist_ok=True)
        print(f"Copying results/{exp}/ via tar-pipe ...")

        # Stream: ssh "tar -C remote_dir -czf - ." | tar -C local_dir -xzf -
        ssh_cmd = _sshpass(pw, [SSH] + SSH_OPTS + [
            SSH_HOST,
            f"tar -C {REMOTE_BASE}/results/{exp} -czf - . 2>/dev/null",
        ])
        ssh_proc = subprocess.Popen(ssh_cmd, stdout=subprocess.PIPE)
        tar_proc = subprocess.Popen(
            [TAR, "-C", local_exp_dir, "-xzf", "-"],
            stdin=ssh_proc.stdout,
        )
        if ssh_proc.stdout:
            ssh_proc.stdout.close()
        tar_proc.wait()
        ssh_proc.wait()
        if tar_proc.returncode != 0 or ssh_proc.returncode != 0:
            print(f"  WARNING: tar-pipe for {exp} exited with ssh={ssh_proc.returncode} tar={tar_proc.returncode}")
        else:
            print(f"  results/{exp}/ synced.")
