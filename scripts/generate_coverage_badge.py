#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def _color_for(percent: float) -> str:
    if percent < 50:
        return "#e05d44"
    if percent < 70:
        return "#fe7d37"
    if percent < 80:
        return "#dfb317"
    if percent < 90:
        return "#a4a61d"
    return "#4c1"


def _format_percent(percent: float) -> str:
    text = f"{percent:.1f}".rstrip("0").rstrip(".")
    return f"{text}%"


def _svg_badge(label: str, value: str, color: str) -> str:
    label_width = max(50, 6 * len(label) + 10)
    value_width = max(50, 6 * len(value) + 10)
    total_width = label_width + value_width
    label_center = label_width / 2
    value_center = label_width + (value_width / 2)

    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{total_width}" height="20" role="img" aria-label="{label}: {value}">
<linearGradient id="s" x2="0" y2="100%">
  <stop offset="0" stop-color="#bbb" stop-opacity=".1"/>
  <stop offset="1" stop-opacity=".1"/>
</linearGradient>
<clipPath id="r">
  <rect width="{total_width}" height="20" rx="3" fill="#fff"/>
</clipPath>
<g clip-path="url(#r)">
  <rect width="{label_width}" height="20" fill="#555"/>
  <rect x="{label_width}" width="{value_width}" height="20" fill="{color}"/>
  <rect width="{total_width}" height="20" fill="url(#s)"/>
</g>
<g fill="#fff" text-anchor="middle" font-family="Verdana,Geneva,DejaVu Sans,sans-serif" font-size="11">
  <text x="{label_center}" y="14">{label}</text>
  <text x="{value_center}" y="14">{value}</text>
</g>
</svg>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate an SVG coverage badge.")
    parser.add_argument("--input", default="coverage.xml", help="Path to coverage.xml")
    parser.add_argument("--output", default="badges/coverage.svg", help="Output SVG path")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = Path.cwd() / input_path
    if not input_path.exists():
        print(f"coverage file not found: {input_path}", file=sys.stderr)
        return 1

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = Path.cwd() / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    root = ET.parse(input_path).getroot()
    rate = float(root.attrib.get("line-rate", "0"))
    percent = round(rate * 100, 1)
    value = _format_percent(percent)
    color = _color_for(percent)
    svg = _svg_badge("coverage", value, color)
    output_path.write_text(svg, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
