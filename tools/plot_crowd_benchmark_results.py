#!/usr/bin/env python3
"""Plot crowd-benchmark outcome rates as stacked bar charts."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Ellipse


FIGURE_WIDTH = 707
FIGURE_HEIGHT = 352

SUCCESS_COLOR = "#4A80FF"
INFEASIBLE_COLOR = "#FF4D85"
COLLISION_COLOR = "#E6D9FF"
OPEN_SANS_FONT = (
    Path(__file__).resolve().parent
    / "fonts"
    / "OpenSans-VariableFont_wdth,wght.ttf"
)

# Values are (success, collision, infeasible) percentages.
METHODS = [
    {
        "title": "Control-Tree",
        "double_integrator": [(48, 0, 52), (10, 0, 90), (3, 0, 97)],
        "unicycle": [(17, 0, 83), (4, 0, 96), (1, 0, 99)],
    },
    {
        "title": "Single-Risk",
        "double_integrator": [(25, 0, 75), (1, 0, 99), (0, 0, 100)],
        "unicycle": [(20, 0, 80), (3, 0, 97), (1, 0, 99)],
    },
    {
        "title": "OA-MPC",
        "double_integrator": [(0, 0, 100), (0, 0, 100), (0, 1, 99)],
        "unicycle": [(0, 15, 85), (0, 8, 92), (0, 7, 93)],
    },
    {
        "title": "OACP",
        "double_integrator": [(50, 1, 49), (13, 7, 80), (2, 3, 95)],
        "unicycle": [(52, 48, 0), (29, 70, 1), (12, 87, 1)],
    },
    {
        "title": "CBF-QP",
        "double_integrator": [(86, 13, 1), (49, 35, 16), (32, 38, 30)],
        "unicycle": [(63, 3, 34), (33, 5, 62), (13, 8, 79)],
    },
    {
        "title": "Occlusion-CBF",
        "double_integrator": [(99, 0, 1), (89, 0, 11), (70, 0, 30)],
        "unicycle": [(98, 0, 2), (90, 0, 10), (86, 0, 14)],
    },
    {
        "title": "Relaxed terminal",
        "double_integrator": [(100, 0, 0), (93, 0, 7), (76, 0, 24)],
        "unicycle": None,
    },
]


def _configure_style() -> None:
    if not OPEN_SANS_FONT.is_file():
        raise FileNotFoundError(f"Bundled Open Sans font not found: {OPEN_SANS_FONT}")
    font_manager.fontManager.addfont(str(OPEN_SANS_FONT))
    mpl.rcParams.update(
        {
            "font.family": "Open Sans",
            "font.sans-serif": ["Open Sans"],
            "svg.fonttype": "path",
            "axes.linewidth": 1.2,
            "xtick.major.width": 1.2,
            "ytick.major.width": 1.2,
            "xtick.major.size": 4.0,
            "ytick.major.size": 4.0,
        }
    )


def _plot_panel(
    ax: plt.Axes,
    values: list[tuple[int, int, int]],
    x_tick_labels: tuple[str, str, str],
    *,
    show_y_labels: bool,
) -> None:
    x_positions = range(3)
    success = [value[0] for value in values]
    collision = [value[1] for value in values]
    infeasible = [value[2] for value in values]

    ax.bar(
        x_positions,
        success,
        width=0.87,
        color=SUCCESS_COLOR,
        linewidth=0,
        clip_on=False,
        zorder=2,
    )
    ax.bar(
        x_positions,
        infeasible,
        width=0.87,
        bottom=success,
        color=INFEASIBLE_COLOR,
        linewidth=0,
        clip_on=False,
        zorder=2,
    )
    infeasible_bottom = [s + i for s, i in zip(success, infeasible)]
    ax.bar(
        x_positions,
        collision,
        width=0.87,
        bottom=infeasible_bottom,
        color=COLLISION_COLOR,
        linewidth=0,
        clip_on=False,
        zorder=2,
    )

    # Label only segments large enough to hold the value.
    for bar_index, (success_value, collision_value, infeasible_value) in enumerate(values):
        segments = (
            (success_value, success_value / 2, "white"),
            (
                infeasible_value,
                success_value + infeasible_value / 2,
                "white",
            ),
            (
                collision_value,
                success_value + infeasible_value + collision_value / 2,
                "black",
            ),
        )
        for segment_value, label_y, text_color in segments:
            if segment_value >= 6:
                ax.text(
                    bar_index,
                    label_y,
                    str(segment_value),
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=15,
                    clip_on=False,
                    zorder=3,
                )

    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(0, 100)
    ax.set_xticks([0, 1, 2], x_tick_labels)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.tick_params(
        axis="both",
        which="major",
        direction="out",
        labelsize=15,
        pad=1.5,
        top=False,
        right=False,
    )
    ax.tick_params(axis="y", labelleft=show_y_labels)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.2)


def _add_legend(fig: plt.Figure) -> None:
    legend_items = (
        (53.0, SUCCESS_COLOR, "Success Rate", 64.5),
        (182.0, INFEASIBLE_COLOR, "Infeasible Rate", 193.5),
        (333.0, COLLISION_COLOR, "Collision Rate", 344.5),
    )
    center_y = 8.5
    marker_size = 16.0
    for center_x, color, label, text_x in legend_items:
        marker = Ellipse(
            (center_x / FIGURE_WIDTH, center_y / FIGURE_HEIGHT),
            width=marker_size / FIGURE_WIDTH,
            height=marker_size / FIGURE_HEIGHT,
            transform=fig.transFigure,
            facecolor=color,
            edgecolor="none",
            clip_on=False,
        )
        fig.add_artist(marker)
        fig.text(
            text_x / FIGURE_WIDTH,
            center_y / FIGURE_HEIGHT,
            label,
            ha="left",
            va="center",
            fontsize=15,
            color="black",
        )


def create_figure(output_path: Path, preview_path: Path | None = None) -> None:
    _configure_style()
    fig = plt.figure(
        figsize=(FIGURE_WIDTH / 72, FIGURE_HEIGHT / 72),
        facecolor="white",
    )

    # Fill the canvas tightly and widen each panel for single-column legibility.
    panel_left = 48.0
    panel_width = 84.0
    panel_gap = 10.0
    top_bottom = 205.0
    top_height = 108.0
    bottom_bottom = 68.0
    bottom_height = 108.0

    for column_index, method in enumerate(METHODS):
        left = panel_left + column_index * (panel_width + panel_gap)
        top_ax = fig.add_axes(
            [
                left / FIGURE_WIDTH,
                top_bottom / FIGURE_HEIGHT,
                panel_width / FIGURE_WIDTH,
                top_height / FIGURE_HEIGHT,
            ]
        )
        _plot_panel(
            top_ax,
            method["double_integrator"],
            ("10", "30", "50"),
            show_y_labels=column_index == 0,
        )
        display_title = {
            "Control-Tree": "Control-\nTree",
            "Single-Risk": "Single-\nRisk",
            "OA-MPC": "\nOA-MPC",
            "OACP": "\nOACP",
            "CBF-QP": "\nCBF-QP",
            "Occlusion-CBF": "Occlusion-\nCBF",
            "Relaxed terminal": "Relaxed\nterminal",
        }.get(method["title"], method["title"])
        top_ax.set_title(
            display_title,
            fontsize=16,
            fontweight="normal",
            linespacing=0.9,
            pad=2,
        )

        if method["unicycle"] is None:
            continue
        bottom_ax = fig.add_axes(
            [
                left / FIGURE_WIDTH,
                bottom_bottom / FIGURE_HEIGHT,
                panel_width / FIGURE_WIDTH,
                bottom_height / FIGURE_HEIGHT,
            ]
        )
        _plot_panel(
            bottom_ax,
            method["unicycle"],
            ("10", "20", "30"),
            show_y_labels=column_index == 0,
        )

    fig.text(
        8.0 / FIGURE_WIDTH,
        259.0 / FIGURE_HEIGHT,
        "Double Integrator",
        ha="center",
        va="center",
        rotation=90,
        fontsize=17,
        fontweight="normal",
    )
    fig.text(
        8.0 / FIGURE_WIDTH,
        122.0 / FIGURE_HEIGHT,
        "Unicycle",
        ha="center",
        va="center",
        rotation=90,
        fontsize=17,
        fontweight="normal",
    )
    fig.text(
        0.5,
        31.5 / FIGURE_HEIGHT,
        "Obstacle number",
        ha="center",
        va="center",
        fontsize=18,
    )
    _add_legend(fig)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if preview_path is not None:
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            preview_path,
            format="png",
            dpi=72,
            facecolor="white",
            edgecolor="none",
        )
    fig.savefig(
        output_path,
        format="svg",
        facecolor="white",
        edgecolor="none",
        metadata={
            "Title": "Crowd benchmark results",
            "Description": (
                "Stacked success, infeasible, and collision rates for "
                "double-integrator and unicycle crowd benchmarks."
            ),
            "Creator": "Matplotlib",
            "Date": None,
        },
    )
    plt.close(fig)

    # Use unitless SVG dimensions for consistent embedding.
    svg = output_path.read_text(encoding="utf-8")
    svg = svg.replace(
        f'width="{FIGURE_WIDTH}pt" height="{FIGURE_HEIGHT}pt"',
        f'width="{FIGURE_WIDTH}" height="{FIGURE_HEIGHT}"',
        1,
    )
    output_path.write_text(svg, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    default_output = (
        Path(__file__).resolve().parents[1]
        / "output"
        / "crowd_benchmark_results.svg"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output,
        help=f"SVG destination (default: {default_output})",
    )
    parser.add_argument(
        "--preview-png",
        type=Path,
        help="Optional 707 x 352 PNG preview used for visual verification",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    preview = arguments.preview_png.resolve() if arguments.preview_png else None
    create_figure(arguments.output.resolve(), preview)
