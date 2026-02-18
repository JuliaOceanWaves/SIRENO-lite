$ErrorActionPreference = "Stop"

if (Test-Path "build") { Remove-Item -Recurse -Force "build" }
if (Test-Path "dist") { Remove-Item -Recurse -Force "dist" }

python -m pip install --upgrade pip
python -m pip install gekko pandas "numpy<2" matplotlib "pyinstaller>=6.6"

$MplSeedDir = "build/mplconfig_seed"
python scripts/prewarm_matplotlib_cache.py --output-dir $MplSeedDir

python -m PyInstaller `
  --clean `
  --noconfirm `
  --onefile `
  --windowed `
  --collect-all numpy `
  --collect-all matplotlib `
  --hidden-import matplotlib.backends.backend_pdf `
  --add-data "sirenolite/data/load_resource_data.csv;sirenolite/data" `
  --add-data "$MplSeedDir;mplconfig_seed" `
  --name sirenolite_gui `
  --paths . `
  sirenolite/gui.py
