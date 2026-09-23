#!/usr/bin/env python3
"""Render a table as an SVG (+ PNG) image, for embedding in Medium articles.

This script lives in the public repo (code/scripts/) — it's reusable tooling. What it
generates is article content, which is not: write output under ../output/stages/articles/
(a sibling of code/, never committed/pushed), matching the same code-vs-output split used
in the ragproject sibling series.

Medium strips/mangles raw HTML <table> markup — confirmed from the rag-tsb-safety article
series (../../../ragproject/outputs/rag-tsb-safety/articles/assets/*.png). The fix there, reused
here: never put a markdown/HTML table in an article file. Render it as a small styled SVG,
export a PNG, embed the PNG.

Usage (either import and call, or edit TABLES below and run directly):

    from table_gen import render_table
    render_table(
        out_stem=OUTPUT_ARTICLES / "assets" / "01-question-and-data-table-cost-breakdown",
        headers=["Item", "Est. cost"],
        rows=[["Socrata pull", "< $0.50"], ["Glue ETL + crawler", "~$0.50"]],
        col_widths=[560, 260],
    )

Produces `<out_stem>.svg` always, and `<out_stem>.png` if `cairosvg` is importable (it is an
optional dev dependency — `uv sync --extra dev`; if it's missing, the SVG is still written
and a note is printed so the PNG can be exported by hand in draw.io/a browser instead).

Visual style matches the reference series: dark navy header (#232F3E) with white bold text,
alternating white / #F5F5F5 row shading, #CCCCCC grid lines, #232F3E outer border.

    python code/scripts/table_gen.py

stdlib only for SVG generation; cairosvg is optional, only needed for the PNG step.
"""

from __future__ import annotations

import html
import pathlib
import textwrap

OUTPUT_ARTICLES = pathlib.Path(__file__).resolve().parents[2] / "output" / "stages" / "articles"

FONT_SIZE = 13.5
LINE_HEIGHT = 18
CELL_PAD_X = 14
CELL_PAD_Y = 23.5
HEADER_HEIGHT = 38
HEADER_FILL = "#232F3E"
HEADER_TEXT = "#FFFFFF"
ROW_FILL_A = "#FFFFFF"
ROW_FILL_B = "#F5F5F5"
GRID = "#CCCCCC"
TEXT_COLOR = "#222222"
CHAR_WIDTH = 7.2  # rough average glyph width at FONT_SIZE, Helvetica/Arial


def _wrap(text: str, width_px: int) -> list[str]:
    chars_per_line = max(8, int((width_px - 2 * CELL_PAD_X) / CHAR_WIDTH))
    return textwrap.wrap(text, chars_per_line) or [""]


def render_table(
    out_stem: pathlib.Path,
    headers: list[str],
    rows: list[list[str]],
    col_widths: list[int],
    *,
    font_family: str = "Helvetica, Arial, sans-serif",
) -> pathlib.Path:
    """Write `<out_stem>.svg` (and `.png` if cairosvg is available). Returns the .svg path."""
    assert len(headers) == len(col_widths)
    width = sum(col_widths)
    col_x = [sum(col_widths[:i]) for i in range(len(col_widths))]

    wrapped_rows = [[_wrap(str(cell), col_widths[i]) for i, cell in enumerate(row)] for row in rows]
    row_line_counts = [max(len(c) for c in wr) for wr in wrapped_rows]
    row_heights = [
        max(2 * CELL_PAD_Y - LINE_HEIGHT, n * LINE_HEIGHT + 2 * (CELL_PAD_Y - LINE_HEIGHT))
        for n in row_line_counts
    ]
    height = HEADER_HEIGHT + sum(row_heights)

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="{font_family}">',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'<rect x="0" y="0" width="{width}" height="{HEADER_HEIGHT}" fill="{HEADER_FILL}"/>',
    ]
    header_y = HEADER_HEIGHT / 2 + FONT_SIZE / 2.6
    for x, h in zip(col_x, headers, strict=True):
        svg.append(
            f'<text x="{x + CELL_PAD_X}" y="{header_y:.1f}" font-size="{FONT_SIZE}" '
            f'font-weight="700" fill="{HEADER_TEXT}">{html.escape(h)}</text>'
        )

    y = HEADER_HEIGHT
    for i, (wr, rh) in enumerate(zip(wrapped_rows, row_heights, strict=True)):
        fill = ROW_FILL_A if i % 2 == 0 else ROW_FILL_B
        svg.append(f'<rect x="0" y="{y}" width="{width}" height="{rh}" fill="{fill}"/>')
        for x, lines in zip(col_x, wr, strict=True):
            for li, line in enumerate(lines):
                ty = (
                    y
                    + (CELL_PAD_Y - LINE_HEIGHT)
                    + LINE_HEIGHT * (li + 1)
                    - (LINE_HEIGHT - FONT_SIZE)
                )
                svg.append(
                    f'<text x="{x + CELL_PAD_X}" y="{ty:.1f}" font-size="{FONT_SIZE}" '
                    f'fill="{TEXT_COLOR}">{html.escape(line)}</text>'
                )
        y += rh
        svg.append(
            f'<line x1="0" y1="{y}" x2="{width}" y2="{y}" stroke="{GRID}" stroke-width="1"/>'
        )

    for x in col_x[1:]:
        svg.append(
            f'<line x1="{x}" y1="0" x2="{x}" y2="{height}" stroke="{GRID}" stroke-width="1"/>'
        )
    svg.append(
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="none" '
        f'stroke="{HEADER_FILL}" stroke-width="1.5"/>'
    )
    svg.append("</svg>")

    out_stem.parent.mkdir(parents=True, exist_ok=True)
    svg_path = out_stem.with_suffix(".svg")
    svg_path.write_text("\n".join(svg))

    try:
        import cairosvg  # optional

        cairosvg.svg2png(url=str(svg_path), write_to=str(out_stem.with_suffix(".png")), scale=2)
    except ImportError:
        print(
            f"wrote {svg_path} (cairosvg not installed — export the PNG by hand, "
            f"e.g. open the SVG in a browser and screenshot, or `uv sync --extra dev`)"
        )
    return svg_path


# Example / smoke-test table — delete once real tables are wired in per-article.
TABLES = [
    dict(
        out_stem=OUTPUT_ARTICLES / "assets" / "_example-table",
        headers=["Item", "Est. cost"],
        rows=[
            ["Socrata pull", "< $0.50"],
            ["Glue ETL + crawler", "~$0.50"],
            ["SageMaker Training (XGBoost, spot)", "~$0.80"],
        ],
        col_widths=[420, 200],
    ),
]

if __name__ == "__main__":
    for t in TABLES:
        render_table(**t)
