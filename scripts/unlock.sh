for d in /tmp/gpu-lock-*; do
  [ -d "$d" ] || continue
    rm -rf "$d"
    echo "removed $d (owned by $ROOT_PID)"
  # fi
done