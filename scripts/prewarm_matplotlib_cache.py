#!/usr/bin/env python3
"""Create a matplotlib cache seed for packaged GUI builds."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, help="Directory to receive cache files.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLBACKEND"] = "Agg"
    os.environ["MPLCONFIGDIR"] = str(output_dir)

    import matplotlib.font_manager as font_manager  # pylint: disable=import-outside-toplevel
    import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel

    font_manager.findSystemFonts()
    fig = plt.figure()
    fig.text(0.5, 0.5, "SIRENO-lite")
    fig.canvas.draw()
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
