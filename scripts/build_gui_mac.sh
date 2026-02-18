#!/usr/bin/env bash
set -euo pipefail

rm -rf build dist

python3 -m pip install --upgrade pip
python3 -m pip install gekko pandas "numpy<2" matplotlib "pyinstaller>=6.6"

MPL_SEED_DIR="build/mplconfig_seed"
python3 scripts/prewarm_matplotlib_cache.py --output-dir "$MPL_SEED_DIR"

pyinstaller \
  --clean \
  --noconfirm \
  --onefile \
  --windowed \
  --collect-all numpy \
  --collect-all matplotlib \
  --hidden-import matplotlib.backends.backend_pdf \
  --add-data "sirenolite/data/load_resource_data.csv:sirenolite/data" \
  --add-data "${MPL_SEED_DIR}:mplconfig_seed" \
  --name sirenolite_gui \
  --paths . \
  sirenolite/gui.py
