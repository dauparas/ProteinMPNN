# !/bin/bash

DIR="${1}"

for script in "$DIR"/*.sh; do
  if [ -f "$script" ]; then
    echo "doing $script..."
    bash "$script"
  fi
done