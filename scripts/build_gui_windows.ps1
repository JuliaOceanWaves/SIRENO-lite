$ErrorActionPreference = "Stop"

Set-Location (Join-Path $PSScriptRoot "..")
python -m pip install ".[build]"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$MplSeedDir = "build/mplconfig_seed"
python scripts/prewarm_matplotlib_cache.py --output-dir $MplSeedDir
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

python -m PyInstaller `
  --clean `
  --noconfirm `
  --onefile `
  --windowed `
  --collect-all numpy `
  --collect-all matplotlib `
  --collect-all gekko `
  --hidden-import matplotlib.backends.backend_pdf `
  --add-data "sirenolite/data/load_resource_data.csv;sirenolite/data" `
  --add-data "sirenolite/data/sirenolite_config.json;sirenolite/data" `
  --add-data "$MplSeedDir;mplconfig_seed" `
  --name sirenolite_gui `
  --paths . `
  sirenolite/gui.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
