#!/bin/bash
# Usage:
#   ./auto_gpu_from_csv.sh tasks.csv            # Serial execution
#   ./auto_gpu_from_csv.sh tasks.csv parallel   # Parallel execution (limited by MAX_PARALLEL)
# CSV format (two columns, comma-separated, header/comments allowed):
#   train_cfg_path,eval_cfg_path
#   cfgs/SAVA-X/seed/42.yml,cfgs/SAVA-X/eval/seed/42.yml

set -euo pipefail

LOG_FILE="auto_gpu_execution.log"
LOG_DIR="${LOG_DIR:-logs}"
TQDM_DISABLE="${TQDM_DISABLE:-1}"
TQDM_MININTERVAL="${TQDM_MININTERVAL:-2}"
TQDM_MINITER="${TQDM_MINITER:-50}"
PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-0}"
mkdir -p "$LOG_DIR"

MAX_PARALLEL=${MAX_PARALLEL:-8}
MIN_FREE_MEM_MB="${MIN_FREE_MEM_MB:-2000}"
MAX_UTIL="${MAX_UTIL:-10}"
CHECK_INTERVAL="${CHECK_INTERVAL:-5}"
ROOT_PID="$$"

if ! command -v nvidia-smi &>/dev/null; then
  DEFAULT_GPU=0
else
  DEFAULT_GPU=0
fi

cleanup() {
  echo "$(date): Script interrupted by user" | tee -a "$LOG_FILE"
  jobs -p | xargs -r kill 2>/dev/null || true
  for d in /tmp/gpu-lock-*; do
    [ -d "$d" ] || continue
    if [[ -f "$d/owner" ]] && grep -qx "$ROOT_PID" "$d/owner"; then
      rm -f "$d/owner" 2>/dev/null || true
      rmdir "$d" 2>/dev/null || true
    fi
  done
  exit 1
}
trap cleanup SIGINT SIGTERM

is_gpu_free() {
  local gpu_id="$1"

  # If the GPU is locked by this script, treat it as busy.
  if [ -d "/tmp/gpu-lock-$gpu_id" ]; then
    return 1
  fi

  # If nvidia-smi is unavailable, allow the task to proceed.
  if ! command -v nvidia-smi &>/dev/null; then
    return 0
  fi

  # 1) Check whether compute processes are running (robust filtering).
  # Filter blank/status lines and keep only numeric PIDs.
  local procs_count
  procs_count="$(
    nvidia-smi -i "$gpu_id" --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
    | awk 'BEGIN{IGNORECASE=1}
           /^[[:space:]]*$/ {next}
           /No running processes found|N\/A|Not Supported/ {next}
           $1 ~ /^[0-9]+$/ {c++}
           END{print c+0}'
  )"
  if [ "${procs_count:-0}" -gt 0 ]; then
    return 1
  fi

  # 2) Check the free-memory and utilization thresholds.
  local mem_free util
  read -r mem_free util < <(
    nvidia-smi -i "$gpu_id" \
      --query-gpu=memory.free,utilization.gpu \
      --format=csv,noheader,nounits \
    | head -n1 | tr -d ' ' | tr ',' ' '
  )

  # If unavailable or non-numeric, treat the GPU as busy to avoid false positives.
  [[ "$mem_free" =~ ^[0-9]+$ ]] || return 1
  [[ "$util"     =~ ^[0-9]+$ ]] || return 1

  (( mem_free >= MIN_FREE_MEM_MB )) || return 1
  (( util     <= MAX_UTIL        )) || return 1

  return 0
}

is_none() {
  local v="${1:-}"
  [[ "${v,,}" == "none" ]]
}

lock_gpu() {
  local gpu_id="$1"
  local lock_dir="/tmp/gpu-lock-$gpu_id"
  if mkdir "$lock_dir" 2>/dev/null; then
    echo "$ROOT_PID" > "$lock_dir/owner"
    return 0
  else
    return 1
  fi
}

unlock_gpu() {
  local gpu_id="$1"
  local lock_dir="/tmp/gpu-lock-$gpu_id"
  rm -f "$lock_dir/owner" 2>/dev/null || true
  rmdir "$lock_dir" 2>/dev/null || true
}

get_free_gpu() {
  if ! command -v nvidia-smi &>/dev/null; then
    local gpu_id="$DEFAULT_GPU"
    if lock_gpu "$gpu_id"; then echo "$gpu_id"; return 0; fi
    while true; do
      if lock_gpu "$gpu_id"; then echo "$gpu_id"; return 0; fi
      echo "$(date): Default GPU $gpu_id locked, waiting..." >>"$LOG_FILE"
      sleep "$CHECK_INTERVAL"
    done
  fi
  local all_gpus=()
  mapfile -t all_gpus < <(nvidia-smi --query-gpu=index --format=csv,noheader,nounits || true)
  if [ "${#all_gpus[@]}" -eq 0 ]; then all_gpus=(0); fi
  while true; do
    echo "$(date): Scanning GPUs... (min_free=${MIN_FREE_MEM_MB}MB, max_util=${MAX_UTIL}%)" >>"$LOG_FILE"
    for gpu_id in "${all_gpus[@]}"; do
      if is_gpu_free "$gpu_id"; then
        if lock_gpu "$gpu_id"; then echo "$gpu_id"; return 0; fi
      fi
    done
    echo "$(date): No free GPU fits thresholds, waiting ${CHECK_INTERVAL}s..." >>"$LOG_FILE"
    sleep "$CHECK_INTERVAL"
  done
}

# ===== Update: allow training-only or evaluation-only runs =====
run_training_evaluation() {
  local train_cfg="$1"
  local eval_cfg="$2"

  local do_train=1
  local do_eval=1
  if is_none "$train_cfg"; then do_train=0; fi
  if is_none "$eval_cfg"; then do_eval=0; fi

  if (( do_train==0 && do_eval==0 )); then
    echo "$(date): [SKIP] Both train_cfg and eval_cfg are 'none'" | tee -a "$LOG_FILE"
    return 0
  fi

  local gpu_id
  gpu_id="$(get_free_gpu)"
  
  # --- Key improvement: use a subshell-style block / manual error handling ---
  # Wrap the Python execution so unlock always runs whether the job succeeds or fails.
  {
    local stamp="$(date +%F_%H-%M-%S)"
    local train_name="$(basename "${train_cfg:-none}" .yml)"
    local eval_name="$(basename "${eval_cfg:-none}" .yml)"

    echo "$(date): Using GPU $gpu_id for train_cfg=$train_cfg | eval_cfg=$eval_cfg" | tee -a "$LOG_FILE"

    # -------- TRAIN stage --------
    if (( do_train==1 )); then
      if [ ! -f "$train_cfg" ]; then
        echo "$(date): [ERROR] Train cfg not found: $train_cfg" | tee -a "$LOG_FILE"
        # Do not return here directly; unlock must still run first.
      else
        local train_args="--gpu_id ${gpu_id} --cfg_path ${train_cfg}"
        local train_log="${LOG_DIR}/train_${train_name}_${stamp}.log"
        # Run the Python job without letting the shell exit immediately on failure.
        TQDM_DISABLE="${TQDM_DISABLE}" CUDA_VISIBLE_DEVICES=${gpu_id} \
        python3 train_SAVAX.py ${train_args} >>"$train_log" 2>&1 || echo "$(date): [FAIL] Python training exited with error" | tee -a "$LOG_FILE"
      fi
    fi

    # -------- EVAL stage --------
    if (( do_eval==1 )); then
      if [ ! -f "$eval_cfg" ]; then
        echo "$(date): [ERROR] Eval cfg not found: $eval_cfg" | tee -a "$LOG_FILE"
      else
        local eval_args="--gpu_id ${gpu_id} --cfg_path ${eval_cfg} --re_eval"
        local eval_log="${LOG_DIR}/eval_${eval_name}_${stamp}.log"
        TQDM_DISABLE="${TQDM_DISABLE}" CUDA_VISIBLE_DEVICES=${gpu_id} \
        python3 eval_SAVAX.py ${eval_args} >>"$eval_log" 2>&1 || echo "$(date): [FAIL] Python evaluation exited with error" | tee -a "$LOG_FILE"
      fi
    fi
  } 

  # --- Make sure unlock always runs ---
  echo "$(date): Releasing GPU $gpu_id" | tee -a "$LOG_FILE"
  unlock_gpu "$gpu_id"
}

wait_for_slot() {
  while true; do
    local running
    running=$(jobs -rp | wc -l | tr -d ' ')
    if (( running < MAX_PARALLEL )); then break; fi
    sleep 1
  done
}

run_from_csv_serial() {
  local csv="$1"
  echo "$(date): Starting SERIAL execution from CSV: $csv" | tee -a "$LOG_FILE"
  while IFS=, read -r train_cfg eval_cfg; do
    line="${train_cfg:-},${eval_cfg:-}"
    [[ -z "${line//,/}" ]] && continue
    [[ "$train_cfg" =~ ^# ]] && continue
    [[ "$train_cfg" == "train_cfg"* ]] && continue
    train_cfg="${train_cfg//[$'\r\t ']}"
    eval_cfg="${eval_cfg//[$'\r\t ']}"
    [ -z "$train_cfg" ] && continue
    [ -z "$eval_cfg" ] && continue
    run_training_evaluation "$train_cfg" "$eval_cfg"
  done < "$csv"
  echo "$(date): All SERIAL tasks completed!" | tee -a "$LOG_FILE"
}

run_from_csv_parallel() {
  local csv="$1"
  echo "$(date): Starting PARALLEL execution from CSV (max $MAX_PARALLEL): $csv" | tee -a "$LOG_FILE"
  while IFS=, read -r train_cfg eval_cfg; do
    line="${train_cfg:-},${eval_cfg:-}"
    [[ -z "${line//,/}" ]] && continue
    [[ "$train_cfg" =~ ^# ]] && continue
    [[ "$train_cfg" == "train_cfg"* ]] && continue
    train_cfg="${train_cfg//[$'\r\t ']}"
    eval_cfg="${eval_cfg//[$'\r\t ']}"
    [ -z "$train_cfg" ] && continue
    [ -z "$eval_cfg" ] && continue
    wait_for_slot
    run_training_evaluation "$train_cfg" "$eval_cfg" &
    sleep 2
  done < "$csv"
  wait
  echo "$(date): All PARALLEL tasks completed!" | tee -a "$LOG_FILE"
}

main() {
  if [ $# -lt 1 ]; then
    echo "Usage: $0 <tasks.csv> [serial|parallel]"
    exit 1
  fi
  local csv="$1"
  local mode="${2:-serial}"
  if [ ! -f "$csv" ]; then echo "[ERROR] CSV not found: $csv"; exit 1; fi
  echo "$(date): Auto GPU allocation script started" | tee "$LOG_FILE"
  echo "$(date): Current GPU status:" | tee -a "$LOG_FILE"
  if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu --format=csv | tee -a "$LOG_FILE"
  else
    echo "nvidia-smi not available" | tee -a "$LOG_FILE"
  fi
  if [ "$mode" = "parallel" ]; then
    run_from_csv_parallel "$csv"
  else
    run_from_csv_serial "$csv"
  fi
  echo "$(date): Script execution finished" | tee -a "$LOG_FILE"
}
main "$@"
