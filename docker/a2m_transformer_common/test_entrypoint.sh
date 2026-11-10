#!/usr/bin/env bash
# Run inside the image with the same non-root UID/GID used by Slurm.
set -euo pipefail
launcher=/usr/local/bin/train-a2m
test "$(id -u)" -ne 0
test -r "$launcher"
test -x "$launcher"
test "$(stat -c '%a' "$launcher")" = 755

# Reach the script's argument guard without needing a GPU or starting training.
if result="$(bash "$launcher" permission-check 2>&1)"; then
    echo "El lanzador debía rechazar argumentos adicionales" >&2
    exit 1
fi
[[ "$result" == *"Este lanzador no acepta argumentos"* ]]
echo "PASS: el usuario $(id -u):$(id -g) puede leer y arrancar train-a2m"
