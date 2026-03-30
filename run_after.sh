#!/bin/bash
# Usage: bash run_after.sh <hours>
# Example: bash run_after.sh 3.5

HOURS=${1:-15}
SECONDS_WAIT=$(awk "BEGIN {printf \"%.0f\", $HOURS * 3600}")

echo "Waiting ${HOURS}h (${SECONDS_WAIT}s) before starting..."
sleep "$SECONDS_WAIT"

echo "Starting experiment at $(date)"
cd "$(dirname "$0")"
source /usr/local/anaconda/etc/profile.d/conda.sh
conda activate tensorneat
nohup python -u main.py > outputv3.log 2>&1 &
echo "Started with PID $! — tail outputv3.log to follow"