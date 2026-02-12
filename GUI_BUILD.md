GUI Packaging Guide

Overview
This GUI can be packaged into standalone executables with PyInstaller. Builds are OS-specific, so you must build on each target OS (macOS, Windows, Linux). Cross-compiling from one OS to another is not supported by PyInstaller.

Prerequisites
- Python 3.9+ installed on the target OS
- Dependencies installed in a virtual environment:
  - gekko
  - pandas
  - numpy (use `<2` when building the GUI)
  - matplotlib
  - pyinstaller

Quick Start (per OS)
1) Create and activate a virtual environment.
2) Install dependencies.
3) Run the build script for your OS in `scripts/`.

Example dependency install:
pip install gekko pandas "numpy<2" matplotlib "pyinstaller>=6.6"

Build Scripts
- macOS: `scripts/build_gui_mac.sh`
- Linux: `scripts/build_gui_linux.sh`
- Windows (PowerShell): `scripts/build_gui_windows.ps1`

Windows notes
- Prefer Python from https://www.python.org/downloads/ over the Microsoft Store build.
- If your network blocks `files.pythonhosted.org`, configure a proxy or index mirror for pip.

GitHub Actions Builds
- Workflow: `.github/workflows/build_gui.yml`
- Trigger: run manually (workflow_dispatch) or push a version tag like `v1.2.3`.
- Outputs: zipped binaries are uploaded as workflow artifacts and attached to the GitHub release for tags.

Output
The executable will be in `dist/` (e.g., `dist/sirenolite_gui` or `dist/sirenolite_gui.exe`).

Notes
- The GUI defaults to a packaged sample CSV at `sirenolite/data/load_resource_data.csv`.
- If you bundle the GUI with PyInstaller, include the sample data file with `--add-data` (the build scripts already do). The syntax differs by OS:
  - macOS/Linux: --add-data "sirenolite/data/load_resource_data.csv:sirenolite/data"
  - Windows: --add-data "sirenolite/data/load_resource_data.csv;sirenolite/data"
- If you see NumPy import errors from a built executable, rebuild in a clean environment with `numpy<2`.
- The build scripts include `--clean` to avoid stale artifacts; if you still see issues, manually delete `build/` and `dist/` before rebuilding.
- PDF export uses the matplotlib PDF backend. The build scripts include it via `--collect-all matplotlib` and `--hidden-import matplotlib.backends.backend_pdf`.

Troubleshooting
- If matplotlib is missing, ensure the GUI backend is installed via `pip install matplotlib`.
- If PyInstaller fails on macOS with permissions, try running in a clean virtual environment.
