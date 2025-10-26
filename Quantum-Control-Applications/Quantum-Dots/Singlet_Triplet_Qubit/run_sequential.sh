#!/usr/bin/env bash
set -euo pipefail

# ===== Configurable parameters =====
LOGDIR="logs"           # Log directory
PY="python3"            # Python interpreter (can be changed to .venv/bin/python)
SLEEP_BETWEEN=0         # Seconds to sleep between scripts (avoid instrument contention); 0 means no sleep
TIMEOUT_SEC=0           # Per-script timeout in seconds; 0 means no timeout (requires coreutils for gtimeout)
EXCLUDE=("run_sequential.sh")   # Filenames to exclude
# If you want a custom execution order, put run_order.txt in the same directory (one .py filename per line)

mkdir -p "$LOGDIR"
: > "$LOGDIR/failed.txt"   # Clear previous failure list

have_gtimeout() { command -v gtimeout >/dev/null 2>&1; }

should_exclude() {
  local base="$1"
  for x in "${EXCLUDE[@]}"; do
    [[ "$base" == "$x" ]] && return 0
  done
  return 1
}

run_one() {
  local f="$1"
  local base="$(basename "$f")"
  local stem="${base%.py}"
  local out="$LOGDIR/$stem.out"
  local err="$LOGDIR/$stem.err"

  echo "========== [RUN] $f =========="
  echo "[$(date '+%F %T')] START $f" > "$out"
  echo "[$(date '+%F %T')] START $f" > "$err"

  if (( TIMEOUT_SEC > 0 )) && have_gtimeout; then
    if ! gtimeout "$TIMEOUT_SEC" "$PY" -u "$f" >>"$out" 2>>"$err"; then
      echo "$f" >> "$LOGDIR/failed.txt"
    fi
  else
    if ! "$PY" -u "$f" >>"$out" 2>>"$err"; then
      echo "$f" >> "$LOGDIR/failed.txt"
    fi
  fi

  if (( SLEEP_BETWEEN > 0 )); then
    echo "[INFO] Sleeping ${SLEEP_BETWEEN}s to let the instrument settle..."
    sleep "$SLEEP_BETWEEN"
  fi
}

# Build file list (prefer run_order.txt; otherwise automatically collect *.py in the current directory)
declare -a FILES=()
if [[ -f run_order.txt ]]; then
  # Execute according to run_order.txt (ignore empty lines and lines starting with #)
  while IFS= read -r line; do
    [[ -z "$line" || "$line" =~ ^# ]] && continue
    [[ "$line" != *.py ]] && line="${line}.py"
    FILES+=("$line")
  done < run_order.txt
else
  # Only run files in the current directory (do not descend into subdirectories)
  shopt -s nullglob
  for f in ./*.py; do
    FILES+=("$f")
  done
  shopt -u nullglob
fi

if (( ${#FILES[@]} == 0 )); then
  echo "No .py files found."
  exit 0
fi

echo "Executing ${#FILES[@]} file(s) sequentially (non-parallel). Logs: $LOGDIR/"

FAILED=0
for f in "${FILES[@]}"; do
  # Skip if the file doesn't exist or doesn't end with .py
  [[ -f "$f" && "$f" == *.py ]] || { echo "[SKIP] $f"; continue; }
  base="$(basename "$f")"
  if should_exclude "$base"; then
    echo "[EXCLUDE] $base"
    continue
  fi
  run_one "$f"
done

if [[ -s "$LOGDIR/failed.txt" ]]; then
  FAILED=$(wc -l < "$LOGDIR/failed.txt" | tr -d ' ')
  echo
  echo "Completed with $FAILED failure(s). See $LOGDIR/failed.txt and each script's .err log."
else
  echo
  echo "All succeeded 🎉  Logs: $LOGDIR/"
fi

# chmod +x run_sequential.sh                             
# ./run_sequential.sh