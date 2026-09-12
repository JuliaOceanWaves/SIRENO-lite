#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root using the active Python environment.
cd "$(dirname "$0")/.."
python3 -m pip install ".[build]"

MPL_SEED_DIR="build/mplconfig_seed"
python3 scripts/prewarm_matplotlib_cache.py --output-dir "$MPL_SEED_DIR"

python3 -m PyInstaller \
  --clean \
  --noconfirm \
  --onedir \
  --windowed \
  --collect-all numpy \
  --collect-all matplotlib \
  --collect-all gekko \
  --hidden-import matplotlib.backends.backend_pdf \
  --add-data "sirenolite/data/load_resource_data.csv:sirenolite/data" \
  --add-data "sirenolite/data/sirenolite_config.json:sirenolite/data" \
  --add-data "${MPL_SEED_DIR}:mplconfig_seed" \
  --name sirenolite_gui \
  --paths . \
  sirenolite/gui.py
