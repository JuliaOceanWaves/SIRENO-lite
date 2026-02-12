#!/usr/bin/env bash
set -euo pipefail

rm -rf build dist

python3 -m pip install --upgrade pip
python3 -m pip install gekko pandas "numpy<2" matplotlib "pyinstaller>=6.6"

pyinstaller \
  --clean \
  --noconfirm \
  --onefile \
  --windowed \
  --collect-all numpy \
  --collect-all matplotlib \
  --hidden-import matplotlib.backends.backend_pdf \
  --add-data "sirenolite/data/load_resource_data.csv:sirenolite/data" \
  --name sirenolite_gui \
  --paths . \
  sirenolite/gui.py
