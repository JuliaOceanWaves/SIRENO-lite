# GUI packaging and release verification

Build on each target operating system using a Python installation with Tk. PyInstaller packages the interpreter, plotting libraries, sample inputs, and GEKKO's local solver executables. A build is accepted only after the packaged app itself completes a local solve and exports its result.

## Build

From an activated, isolated Python 3.9+ environment:

```bash
python -m pip install ".[build]"
```

Run the matching script:

```bash
bash scripts/build_gui_mac.sh
# or: bash scripts/build_gui_linux.sh
# Windows PowerShell: scripts/build_gui_windows.ps1
```

The scripts run from the repository root, install build dependencies into the active environment, create a Matplotlib font-cache seed, and run PyInstaller. They preserve unrelated build outputs. Generated specifications are build artifacts; the scripts are the maintained build definition.

Outputs are `dist/sirenolite_gui.app` on macOS, `dist/sirenolite_gui` on Linux, and `dist/sirenolite_gui.exe` on Windows. The macOS build uses directory mode; package the complete `.app` bundle. Builds are specific to OS and architecture; cross-compilation is not a supported PyInstaller workflow. See [PyInstaller's operating modes](https://pyinstaller.org/en/stable/operating-mode.html) and [bundle data documentation](https://pyinstaller.org/en/stable/spec-files.html).

## Test the native executable

```bash
# macOS
dist/sirenolite_gui.app/Contents/MacOS/sirenolite_gui --smoke-test build/frozen-smoke
# Linux with a virtual display
xvfb-run -a dist/sirenolite_gui --smoke-test build/frozen-smoke
```

On Windows, wait for the windowed process and check its exit status:

```powershell
$p = Start-Process dist/sirenolite_gui.exe -ArgumentList '--smoke-test', 'build/frozen-smoke' -Wait -PassThru
if ($p.ExitCode -ne 0) { throw 'Packaged GUI check failed' }
```

The check constructs actual Tk widgets and runs two cases through the GUI worker
and event loop. A constant 10 W wind-only GEKKO case verifies the analytical
10 W minimum capacity and electrical balance. A 168-hour Hawaii combined-service
paper case verifies the SciPy/HiGHS backend, bundled resources, power balance,
and cyclic battery/water inventories. Both export CSV, PDF, and JSON; the paper
case uses the `paper/` output subdirectory. Inspect the top-level
`smoke_report.json` and its `passed` field. A successful process launch alone
does not establish solver functionality.

Also launch the app normally and inspect Inputs and the summary, log, and plots on Run / Plot at a usable window size. Exercise a custom CSV beside its JSON, fixed sizing, an invalid input, failed-solve recovery, and both export dialogs. Automated tests cannot establish every native dialog or accessibility behavior.

## Distribution checks

```bash
python -m build
python -m pip install --force-reinstall --no-deps dist/*.whl
python scripts/smoke_cli.py --output-dir build/installed-smoke
```

The installed CLI check runs outside the source tree, rejects an editable source
import, and verifies agreement between package and runtime versions. It solves
an analytical constant 100 W generator-only case and a 48-hour paper case,
checks capacity, mass, electrical and cyclic inventory balances, then replays
the exported paper configuration and retained compressed input. Use a separate
test environment when verifying a wheel to avoid editable-install confusion.

The source archive includes the test fixtures, browser assets, documentation, examples, and build scripts. Extract it into a new directory and run the full test command from there with Tk and Chrome required. The test workflow also performs this check, so files available only in a developer's checkout cannot hide an incomplete source distribution.

`.github/workflows/build_gui.yml` builds and smoke-tests Linux, macOS, and Windows
before uploading binaries. A version tag must match the installed package
version. Release publication also requires the reusable verification workflow:
Python 3.9/3.14 tests with real Tk and Chrome, documentation, installed wheel,
extracted source archive, and representative annual paper replays. Only the
release job receives repository write permission. It publishes the three GUI
archives, wheel, source archive, and `SHA256SUMS` after those checks pass.
These permissions follow [GitHub's job permission model](https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-syntax#jobsjob_idpermissions).
Signing and macOS notarization remain separate; the generated macOS app has an
ad hoc signature. Successful local macOS checks do not establish Linux or
Windows execution.

## Troubleshooting

- A missing local solver usually indicates that GEKKO's `bin` directory was omitted. The maintained scripts include `--collect-all gekko`; verify a local solve in the built app.
- A missing PDF backend requires `matplotlib.backends.backend_pdf`; the scripts include it and collect Matplotlib data.
- On restricted machines, configure the package index/proxy for pip. Do not substitute a successful source run for an unbuilt executable.
- Use a clean virtual environment when old PyInstaller hooks conflict with NumPy or setuptools. Record actual versions in the release evidence rather than assuming an old version pin fixes every import error.
- If the build cannot write or clean PyInstaller's user cache, point `PYINSTALLER_CONFIG_DIR` at a writable build directory before running the script. On macOS/Linux, for example: `export PYINSTALLER_CONFIG_DIR="$PWD/build/pyinstaller-cache"`.
- Tk requires a display. On Linux CI use Xvfb; a headless server without a display cannot run the desktop app. See [Python's Tk documentation](https://docs.python.org/3/library/tkinter.html).
- Browser CI uses the hosted runner's installed Chrome, whose sandbox is configured for Ubuntu. Downloaded Chrome for Testing binaries can fail under Ubuntu's AppArmor user-namespace policy; see [Puppeteer's sandbox guidance](https://pptr.dev/troubleshooting#issues-with-apparmor-on-ubuntu). Keep browser isolation enabled when reproducing this check.
- On macOS, switching notebook tabs can leave a blank page until the native window redraws; the same behavior is reported in [CPython issue 136095](https://github.com/python/cpython/issues/136095). A small ordinary window resize restored both Inputs and Results in our desktop checks. Moving the pointer into the page may also refresh it. This is a display issue: a completed solve and its exports remain available. Inspect both tabs after switching when reviewing a macOS build.

## Paper-case GUI resources

The build scripts collect SciPy/HiGHS and all `sirenolite` package data, including
the 19 compressed annual resource profiles, their source manifest, and the
retained annual comparison evidence. The default desktop case is annual Hawaii
combined service; it runs without Julia or the manuscript checkout. Preserve
`sirenolite/data/paper_sites.json` and all referenced CSV.gz files in platform
bundles. The native smoke test exercises both supported backends; the annual
comparison is a separate numerical gate described in the
[paper reproduction guide](docs/paper-reproducibility.md).
