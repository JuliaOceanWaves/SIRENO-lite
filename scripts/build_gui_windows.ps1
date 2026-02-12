$ErrorActionPreference = "Stop"

if (Test-Path "build") { Remove-Item -Recurse -Force "build" }
if (Test-Path "dist") { Remove-Item -Recurse -Force "dist" }

python -m pip install --upgrade pip
python -m pip install gekko pandas "numpy<2" matplotlib "pyinstaller>=6.6"

python -m PyInstaller `
  --clean `
  --noconfirm `
  --onefile `
  --windowed `
  --collect-all numpy `
  --collect-all matplotlib `
  --hidden-import matplotlib.backends.backend_pdf `
  --add-data "sirenolite/data/load_resource_data.csv;sirenolite/data" `
  --name sirenolite_gui `
  --paths . `
  sirenolite/gui.py
