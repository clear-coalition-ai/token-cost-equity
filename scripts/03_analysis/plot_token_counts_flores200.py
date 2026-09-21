"""Plot selected FLORES-200 token-count distributions from best to worst grade."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
import pandas as pd


load_dotenv()
PROJECT_ROOT = os.getenv("PROJECT_ROOT")
if not PROJECT_ROOT:
    raise EnvironmentError("PROJECT_ROOT is not set. Check your .env file.")
PROJECT_ROOT = Path(PROJECT_ROOT)

DEFAULT_INPUT = (
    PROJECT_ROOT / "02_output_model_experiments" / "flores200_token_counts_langinfo.csv"
)
DEFAULT_RATIO_INPUT = PROJECT_ROOT / "03_output_analysis" / "ratio_flores200.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "03_output_analysis" / "token_counts_flores200.png"

GRADE_EXAMPLE_MODELS = [
    {
        "model": "facebook/nllb-200-3.3B",
        "label": "Meta NLLB",
    },
    {
        "model": "gemini-3.8-flash",
        "label": "Gemini Flash",
    },
    {
        "model": "claude-fable-5",
        "label": "Claude Fable",
    },
    {
        "model": "moonshotai/Kimi-K3",
        "label": "Kimi K3",
    },
    {
        "model": "gpt-5",
        "label": "GPT-5",
    },
    {
        "model": "tiiuae/Falcon3-7B-Instruct",
        "label": "Falcon 3",
    },
]

SCRIPT_CATEGORY_COLORS = {
    "Latin": "#10A37F",
    "Common": "#4D6BFE",
    "Rare Script": "#F2C94C",
}

SCRIPT_CATEGORY_ANNOTATION_COLORS = {
    "Latin": "#00785B",
    "Common": "#2436D9",
    "Rare Script": "#9A6400",
}

RANK_DELTA_LABEL_PLACEMENTS = {
    ("facebook/nllb-200-3.3B", "Banjar"): {"x": 20, "y": 300, "ha": "left"},
    ("facebook/nllb-200-3.3B", "Pangasinan"): {"x": 32, "y": 230, "ha": "left"},
    ("facebook/nllb-200-3.3B", "Asturian"): {"x": 46, "y": 160, "ha": "left"},
    ("facebook/nllb-200-3.3B", "Thai"): {"x": 157, "y": 300, "ha": "right"},
    ("facebook/nllb-200-3.3B", "Georgian"): {"x": 171, "y": 230, "ha": "right"},
    ("facebook/nllb-200-3.3B", "Greek"): {"x": 162, "y": 160, "ha": "right"},
    ("gemini-3.8-flash", "Asturian"): {"x": 37, "y": 300, "ha": "left"},
    ("gemini-3.8-flash", "Pangasinan"): {"x": 49, "y": 230, "ha": "left"},
    ("gemini-3.8-flash", "Najdi Arabic"): {"x": 54, "y": 160, "ha": "left"},
    ("gemini-3.8-flash", "Greek"): {"x": 150, "y": 300, "ha": "right"},
    ("gemini-3.8-flash", "Georgian"): {"x": 158, "y": 155, "ha": "right"},
    ("gemini-3.8-flash", "Armenian"): {"x": 183, "y": 230, "ha": "right"},
    ("claude-fable-5", "Asturian"): {"x": 27, "y": 300, "ha": "left"},
    ("claude-fable-5", "Venetian"): {"x": 38, "y": 230, "ha": "left"},
    ("claude-fable-5", "Friulian"): {"x": 50, "y": 160, "ha": "left"},
    ("claude-fable-5", "Greek"): {"x": 158, "y": 300, "ha": "right"},
    ("claude-fable-5", "Thai"): {"x": 163, "y": 230, "ha": "right"},
    ("claude-fable-5", "Kannada"): {"x": 190, "y": 260, "ha": "right"},
    ("moonshotai/Kimi-K3", "Pangasinan"): {"x": 18, "y": 300, "ha": "left"},
    ("moonshotai/Kimi-K3", "Asturian"): {"x": 29, "y": 230, "ha": "left"},
    ("moonshotai/Kimi-K3", "Kabuverdianu"): {"x": 43, "y": 160, "ha": "left"},
    ("moonshotai/Kimi-K3", "Greek"): {"x": 164, "y": 300, "ha": "right"},
    ("moonshotai/Kimi-K3", "Thai"): {"x": 167, "y": 230, "ha": "right"},
    ("moonshotai/Kimi-K3", "Georgian"): {"x": 174, "y": 160, "ha": "right"},
    ("gpt-5", "Najdi Arabic"): {"x": 28, "y": 300, "ha": "left"},
    ("gpt-5", "Asturian"): {"x": 39, "y": 230, "ha": "left"},
    ("gpt-5", "Pangasinan"): {"x": 52, "y": 160, "ha": "left"},
    ("gpt-5", "Thai"): {"x": 143, "y": 300, "ha": "right"},
    ("gpt-5", "Greek"): {"x": 158, "y": 230, "ha": "right"},
    ("gpt-5", "Myanmar (Burmese)"): {"x": 151, "y": 160, "ha": "right"},
    ("tiiuae/Falcon3-7B-Instruct", "Asturian"): {"x": 17, "y": 300, "ha": "left"},
    ("tiiuae/Falcon3-7B-Instruct", "Kabuverdianu"): {"x": 29, "y": 230, "ha": "left"},
    ("tiiuae/Falcon3-7B-Instruct", "Papiamento"): {"x": 40, "y": 160, "ha": "left"},
    ("tiiuae/Falcon3-7B-Instruct", "Russian"): {"x": 134, "y": 300, "ha": "right"},
    ("tiiuae/Falcon3-7B-Instruct", "Greek"): {"x": 184, "y": 340, "ha": "left"},
    ("tiiuae/Falcon3-7B-Instruct", "Thai"): {"x": 165, "y": 210, "ha": "right"},
}

GRADE_BADGE_COLORS = {
    "A": {"face": "#b8d9bb", "text": "#2f6b38"},
    "B": {"face": "#a7a2ee", "text": "#3f3a9c"},
    "D": {"face": "#f9e2a4", "text": "#8a5a00"},
    "F": {"face": "#f5a7aa", "text": "#9b2f35"},
}

COMMON_SCRIPTS = {
    "Arab",
    "Cyrl",
    "Deva",
}


def configure_fonts() -> None:
    """Use a consistent sans-serif font with broadly available fallbacks."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create stacked token-count charts for grade example models."
    )
    parser.add_argument(
        "--ratio-input",
        type=Path,
        default=DEFAULT_RATIO_INPUT,
        help=f"Ratio metric CSV used to calculate CLEAR grades. Default: {DEFAULT_RATIO_INPUT}",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input CSV path. Default: {DEFAULT_INPUT}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output image path. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--width-px",
        type=int,
        default=1200,
        help="Output image width in pixels. Default: 1200",
    )
    parser.add_argument(
        "--height-px",
        type=int,
        default=1500,
        help="Output image height in pixels. Default: 1500",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=160,
        help="Output image DPI. Figure inches are derived from pixel size / DPI.",
    )
    parser.add_argument(
        "--label-rank-deltas",
        dest="label_rank_deltas",
        action="store_true",
        default=True,
        help="Label the three largest positive and negative token-rank vs cc_spk_rank gaps in each panel.",
    )
    parser.add_argument(
        "--no-label-rank-deltas",
        dest="label_rank_deltas",
        action="store_false",
        help="Hide token-rank vs cc_spk_rank annotations.",
    )
    return parser.parse_args()


def script_category(row: pd.Series, script_counts: pd.Series) -> str:
    if row["script"] == "Latn":
        return "Latin"
    if row["script"] in COMMON_SCRIPTS:
        return "Common"
    if script_counts[row["script"]] < 5:
        return "Rare Script"
    raise ValueError(f"Unexpected uncategorized script: {row['script']}")


def model_grade(ratio_df: pd.DataFrame, model: str) -> str:
    model_ratios = ratio_df.loc[ratio_df["model"].eq(model)].sort_values("N")
    if model_ratios.empty:
        raise ValueError(f"No ratio metrics found for {model}.")

    ratio = model_ratios["max_min_ratio"].iloc[-1]
    if ratio <= 3:
        return "A"
    if ratio <= 6:
        return "B"
    if ratio <= 9:
        return "C"
    if ratio <= 12:
        return "D"
    return "F"


def model_data(df: pd.DataFrame, model: str) -> pd.DataFrame:
    chart_data = df.loc[df["model"].eq(model)].copy()
    if chart_data.empty:
        raise ValueError(f"No rows found for {model}.")

    script_counts = chart_data["script"].value_counts()
    chart_data["script_category"] = chart_data.apply(
        script_category,
        axis=1,
        script_counts=script_counts,
    )
    chart_data = chart_data.sort_values("token_count", ignore_index=True)
    chart_data["token_count_rank"] = chart_data.index + 1
    chart_data["rank_delta"] = chart_data["token_count_rank"] - chart_data["cc_spk_rank"]
    chart_data["abs_rank_delta"] = chart_data["rank_delta"].abs()
    chart_data["token_count_thousands"] = chart_data["token_count"] / 1000
    return chart_data


def draw_rank_delta_labels(ax: plt.Axes, chart_data: pd.DataFrame) -> None:
    label_candidates = chart_data.loc[~chart_data["file"].eq("arb_Latn.dev")]
    positive = label_candidates.nlargest(3, "rank_delta").sort_values("token_count_rank")
    negative = label_candidates.nsmallest(3, "rank_delta").sort_values("token_count_rank")
    annotations = [
        *(
            {"row": row, "label_y": label_y, "side": "left", "offset": offset}
            for offset, (label_y, (_, row)) in enumerate(
                zip([300, 230, 160], negative.iterrows())
            )
        ),
        *(
            {"row": row, "label_y": label_y, "side": "right", "offset": offset}
            for offset, (label_y, (_, row)) in enumerate(
                zip([305, 225, 145], positive.iterrows())
            )
        ),
    ]

    for annotation in annotations:
        row = annotation["row"]
        color = SCRIPT_CATEGORY_ANNOTATION_COLORS[row["script_category"]]
        x = row["token_count_rank"] - 1
        y = row["token_count_thousands"]
        label_y = annotation["label_y"]
        placement = RANK_DELTA_LABEL_PLACEMENTS.get((row["model"], row["name"]))
        if placement:
            x_label = placement["x"]
            label_y = placement["y"]
            ha = placement["ha"]
        elif annotation["side"] == "left":
            x_label = max(x + 12 + annotation["offset"] * 3, 9)
            ha = "left"
        else:
            x_label = min(x - 18 - annotation["offset"] * 9, len(chart_data) - 10)
            ha = "right"
        if ha == "left":
            x_line_end = x_label - 2
        else:
            x_line_end = x_label + 2
        y_start = y + 8 if label_y >= y else y - 8
        joint_y = label_y
        ax.plot(
            [x, x, x_line_end],
            [y_start, joint_y, joint_y],
            color=color,
            linewidth=0.9,
            alpha=0.9,
            solid_capstyle="round",
        )
        ax.scatter(
            [x],
            [y],
            s=11,
            facecolor="#ffffff",
            edgecolor=color,
            linewidth=0.7,
            zorder=4,
        )
        ax.text(
            x_label,
            label_y,
            row["name"],
            fontsize=7.2,
            fontweight="bold",
            color=color,
            ha=ha,
            va="center",
            zorder=5,
            clip_on=False,
            bbox={
                "boxstyle": "square,pad=0.04",
                "facecolor": "#ffffff",
                "edgecolor": "none",
                "alpha": 0.9,
            },
        )


def draw_plot(
    df: pd.DataFrame,
    ratio_df: pd.DataFrame,
    output_path: Path,
    width_px: int,
    height_px: int,
    dpi: int,
    label_rank_deltas: bool,
) -> None:
    model_frames = [
        {
            **model_info,
            "grade_note": model_grade(ratio_df, model_info["model"]),
            "data": model_data(df, model_info["model"]),
        }
        for model_info in GRADE_EXAMPLE_MODELS
    ]
    ymax = max(
        500,
        max(frame["data"]["token_count_thousands"].max() for frame in model_frames)
        * 1.08,
    )

    fig_size = (width_px / dpi, height_px / dpi)
    fig, axs = plt.subplots(
        len(model_frames),
        1,
        figsize=fig_size,
        dpi=dpi,
        constrained_layout=False,
        sharex=True,
        sharey=True,
    )
    fig.subplots_adjust(left=0.13, right=0.985, top=0.86, bottom=0.1, hspace=0.22)

    for ax, frame in zip(axs, model_frames):
        chart_data = frame["data"]
        x_positions = range(len(chart_data))
        bar_colors = chart_data["script_category"].map(SCRIPT_CATEGORY_COLORS)
        ax.bar(
            x_positions,
            chart_data["token_count_thousands"],
            color=bar_colors,
            width=1.0,
            edgecolor="none",
            linewidth=0,
        )
        ax.set_xlim(-0.5, len(chart_data) - 0.5)
        if label_rank_deltas:
            draw_rank_delta_labels(ax, chart_data)

        model_label = ax.text(
            0.01,
            0.84,
            frame["label"],
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            ha="left",
            va="center",
        )
        fig.canvas.draw()
        label_bbox = model_label.get_window_extent(renderer=fig.canvas.get_renderer())
        label_right_axes_x = ax.transAxes.inverted().transform(
            (label_bbox.x1, label_bbox.y0)
        )[0]
        grade_style = GRADE_BADGE_COLORS[frame["grade_note"]]
        ax.text(
            label_right_axes_x + 0.04,
            0.86,
            frame["grade_note"],
            transform=ax.transAxes,
            fontsize=10,
            fontweight="bold",
            color=grade_style["text"],
            ha="center",
            va="center",
            bbox={
                "boxstyle": "circle,pad=0.36",
                "facecolor": grade_style["face"],
                "edgecolor": grade_style["text"],
                "linewidth": 1.1,
                "alpha": 0.95,
            },
        )
        ax.set_ylim(0, ymax)
        ax.set_xticks([])
        ax.set_yticks([100, 200, 300, 400, 500])
        ax.yaxis.set_major_formatter(
            FuncFormatter(lambda tick, _: f"{int(tick):,}k")
        )
        ax.tick_params(axis="both", length=0, labelsize=8)
        ax.grid(axis="y", color="#dedede", linewidth=0.8, alpha=0.75)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(
        "Token Count Distribution by CLEAR Grade",
        fontsize=14,
        fontweight="bold",
        x=0.065,
        y=0.965,
        ha="left",
    )
    fig.text(
        0.065,
        0.937,
        "FLORES-200 languages sorted lowest to highest token count within each model",
        fontsize=9,
        color="#666666",
        ha="left",
    )
    fig.supxlabel("Languages Sorted by Token Count", fontsize=12, y=0.065)
    fig.supylabel(
        "Tokens Needed to Represent Flores 200 Sentences\nin Language",
        fontsize=11,
        x=0.035,
        y=0.5,
        ha="center",
        va="center",
        multialignment="center",
    )

    legend_handles = [
        Patch(facecolor=color, edgecolor="none", label=label)
        for label, color in SCRIPT_CATEGORY_COLORS.items()
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(0.065, 0.915),
        ncol=4,
        frameon=False,
        fontsize=8,
        handlelength=1.0,
        handletextpad=0.4,
        columnspacing=1.2,
    )

    fig.text(
        0.04,
        0.035,
        "Common scripts: Arab, Cyrl, Deva.",
        fontsize=8,
        color="#555555",
    )
    fig.text(
        0.04,
        0.02,
        "Rare scripts appear in fewer than five languages in this FLORES-200 data file.",
        fontsize=8,
        color="#555555",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    configure_fonts()
    args = parse_args()
    df = pd.read_csv(args.input)
    ratio_df = pd.read_csv(args.ratio_input)
    draw_plot(
        df,
        ratio_df,
        args.output,
        args.width_px,
        args.height_px,
        args.dpi,
        args.label_rank_deltas,
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
