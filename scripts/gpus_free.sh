#!/bin/bash
# test_gpu_free.sh
# Usage examples:
#   bash test_gpu_free.sh                 # Check all GPUs once
#   bash test_gpu_free.sh -g 0,2,6        # Check only GPUs 0/2/6
#   bash test_gpu_free.sh -w 2            # Refresh every 2 seconds
#   MIN_FREE_MEM_MB=2000 MAX_UTIL=10 bash test_gpu_free.sh  # Custom thresholds
#   bash test_gpu_free.sh -v              # Verbose output

set -euo pipefail

MIN_FREE_MEM_MB="${MIN_FREE_MEM_MB:-2000}"
MAX_UTIL="${MAX_UTIL:-10}"
INTERVAL=""
GPU_FILTER=""
VERBOSE=0

while getopts ":g:w:v" opt; do
  case "$opt" in
    g) GPU_FILTER="$OPTARG" ;;   # Comma-separated
    w) INTERVAL="$OPTARG" ;;
    v) VERBOSE=1 ;;
    *) echo "Usage: $0 [-g gpu_ids_comma] [-w seconds] [-v]"; exit 1 ;;
  esac
done

has_nvidia_smi() { command -v nvidia-smi &>/dev/null; }

get_all_gpus() {
  if ! has_nvidia_smi; then
    echo 0
    return
  fi
  nvidia-smi --query-gpu=index --format=csv,noheader,nounits 2>/dev/null \
    | sed -E 's/^[[:space:]]+|[[:space:]]+$//g' \
    | awk 'NF>0'
}

list_pids_filtered() {
  local gpu_id="$1"
  nvidia-smi -i "$gpu_id" --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
  | awk 'BEGIN{IGNORECASE=1}
         /^[[:space:]]*$/ {next}
         /No running processes found|N\/A|Not Supported/ {next}
         $1 ~ /^[0-9]+$/ {print $1}'
}

is_gpu_free() {
  local gpu_id="$1"

  # Treat an existing lock directory as busy.
  if [ -d "/tmp/gpu-lock-$gpu_id" ]; then
    return 1
  fi

  if ! has_nvidia_smi; then
    # If nvidia-smi is unavailable, this helper script treats the GPU as free.
    return 0
  fi

  # Running compute processes (strictly numeric PIDs only).
  local procs_count
  procs_count="$(list_pids_filtered "$gpu_id" | wc -l | tr -d ' ')"
  if [ "${procs_count:-0}" -gt 0 ]; then
    return 1
  fi

  # Free memory and utilization.
  local mem_free util
  read -r mem_free util < <(
    nvidia-smi -i "$gpu_id" --query-gpu=memory.free,utilization.gpu --format=csv,noheader,nounits \
      | head -n1 | tr -d ' ' | tr ',' ' '
  )
  [[ "$mem_free" =~ ^[0-9]+$ ]] || return 1
  [[ "$util"     =~ ^[0-9]+$ ]] || return 1
  (( mem_free >= MIN_FREE_MEM_MB )) || return 1
  (( util     <= MAX_UTIL        )) || return 1
  return 0
}

print_one_gpu() {
  local id="$1"
  local lock_dir="/tmp/gpu-lock-$id"
  local lock="no"
  local lock_owner="-"
  if [ -d "$lock_dir" ]; then
    lock="yes"
    if [ -f "$lock_dir/owner" ]; then
      lock_owner="$(cat "$lock_dir/owner" 2>/dev/null || echo "-")"
    fi
  fi

  local mem_free="-" util="-" name="-"
  if has_nvidia_smi; then
    name="$(nvidia-smi -i "$id" --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1 | sed -E 's/^[[:space:]]+|[[:space:]]+$//g')"
    read -r mem_free util < <(
      nvidia-smi -i "$id" --query-gpu=memory.free,utilization.gpu --format=csv,noheader,nounits 2>/dev/null \
        | head -n1 | tr -d ' ' | tr ',' ' '
    )
  fi
  local pids="$(has_nvidia_smi && list_pids_filtered "$id" | xargs || echo "")"
  if is_gpu_free "$id"; then
    local status="FREE"
  else
    local status="BUSY"
  fi

  printf "GPU %-2s | %-18s | status=%-4s | lock=%-3s (owner=%s) | mem.free=%-6s MiB | util=%-3s %%\n" \
         "$id" "$name" "$status" "$lock" "$lock_owner" "${mem_free:-"-"}" "${util:-"-"}"
  if [ $VERBOSE -eq 1 ]; then
    echo "         procs: ${pids:-<none>}"
  fi
}

one_pass() {
  local ids=()
  if [ -n "$GPU_FILTER" ]; then
    IFS=',' read -r -a ids <<< "$GPU_FILTER"
  else
    mapfile -t ids < <(get_all_gpus)
  fi
  echo "---- $(date '+%F %T')  MIN_FREE_MEM_MB=${MIN_FREE_MEM_MB}  MAX_UTIL=${MAX_UTIL} ----"
  for id in "${ids[@]}"; do
    print_one_gpu "$id"
  done
}

# Main flow: one-shot or watch mode.
if [ -n "$INTERVAL" ]; then
  while true; do
    one_pass
    sleep "$INTERVAL"
  done
else
  one_pass
fi
