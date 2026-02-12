# Troubleshooting

## PyInstaller fails on macOS due to `pathlib` backport
**Symptom**
- PyInstaller reports: "The 'pathlib' package is an obsolete backport of a standard library package and is incompatible with PyInstaller."

**Cause**
- A `pathlib` backport is installed in the Python environment, shadowing the standard library module.

**Fix**
1. Uninstall the backport from the same environment you use to build:
```bash
python -m pip uninstall pathlib
```
2. Confirm Python resolves the standard library module:
```bash
python -c "import pathlib; print(pathlib.__file__)"
```
The path should point to the Python stdlib (e.g., `.../lib/python3.11/pathlib.py`) and not `site-packages`.

**Notes**
- If you are using conda, `conda remove pathlib` will not find pip-installed packages.
- Ensure the `python` used here is the same one used in the build script (`scripts/build_gui_mac.sh`).

## pip install fails with HTTP 403
**Symptom**
- `ERROR: 403 Client Error: Forbidden for url: https://files.pythonhosted.org/...`

**Cause**
- Network or proxy restrictions blocking PyPI downloads.

**Fix**
- Try a different network or configure a proxy/index mirror:
  - https://pip.pypa.io/en/stable/topics/configuration/

## Python or pip not found
**Symptom**
- `python` or `pip` command not found.

**Fix**
- Install Python from https://www.python.org/downloads/
- Confirm versions:
```bash
python3 --version
python3 -m pip --version
```

## Virtual environment confusion
**Symptom**
- Packages install, but the app still fails to import them.

**Fix**
- Use an isolated environment and activate it:
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```
- Learn more: https://docs.python.org/3/library/venv.html

## Permission errors while installing packages
**Symptom**
- `Permission denied` or `not writable` errors during `pip install`.

**Fix**
- Use a virtual environment (recommended) or:
```bash
python3 -m pip install --user <package>
```
- pip docs: https://pip.pypa.io/en/stable/user_guide/

## Tkinter or matplotlib GUI issues
**Symptom**
- GUI fails to start or matplotlib backend errors.

**Fix**
- Tkinter is included with most Python installs. If missing on macOS Homebrew Python:
  - https://docs.python.org/3/library/tkinter.html
- For matplotlib backend guidance:
  - https://matplotlib.org/stable/users/explain/backends.html

## GEKKO remote solve errors
**Symptom**
- Remote solve fails or times out.

**Fix**
- Set `solver.remote = False` to run locally.
- GEKKO docs: https://gekko.readthedocs.io/

## CSV format errors
**Symptom**
- Errors about missing columns or incorrect time steps.

**Fix**
- Ensure CSV has columns: `Time`, `Load`, `Solar`, `Wind`, `Wave`, `Hydrogen`, `PotableWater`.
- `Time` must be 1-hour increments (0, 1, 2, ...).
- Example file: `sirenolite/data/load_resource_data.csv`.
