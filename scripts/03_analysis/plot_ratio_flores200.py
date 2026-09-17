"""Plot FLORES-200 max/min token-count ratios by number of languages.

The input CSV is expected to contain:
    model, N, max_min_ratio

By default this script plots the curated model subset used in the final figure.
Use ``--models all`` to include every model in the metric output.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from textwrap import wrap

import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, HPacker, TextArea
from matplotlib.patches import FancyArrowPatch
import pandas as pd
from dotenv import load_dotenv


load_dotenv()
PROJECT_ROOT = os.getenv("PROJECT_ROOT")
if not PROJECT_ROOT:
    raise EnvironmentError("PROJECT_ROOT is not set. Check your .env file.")
PROJECT_ROOT = Path(PROJECT_ROOT)

DEFAULT_INPUT = PROJECT_ROOT / "03_output_analysis" / "ratio_flores200.csv"
DEFAULT_LANGUAGE_LIST = PROJECT_ROOT / "01_data_processed" / "flores200_langinfo.tsv"
DEFAULT_OUTPUT = PROJECT_ROOT / "03_output_analysis" / "ratio_flores200_plot.png"
DEFAULT_JUMP_OUTPUT = (
    PROJECT_ROOT / "03_output_analysis" / "ratio_flores200_plot_language_jumps.png"
)
RANK_VAR = "cc_spk_rank"


def configure_fonts() -> None:
    """Use a consistent sans-serif font with broadly available fallbacks."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
        }
    )

DEFAULT_MODELS = [
    "tiiuae/Falcon3-7B-Instruct",
    "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4",
    "CohereLabs/command-a-plus-05-2026-bf16",
    "openai/gpt-oss-20b",
    "gpt-5",
    "zai-org/GLM-5.3",
    "Qwen/Qwen3.6-35B-A3B",
    "moonshotai/Kimi-K3",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    "claude-fable-5",
    "google/gemma-4-E4B-it",
    "gemini-3.8-flash",
    "facebook/nllb-200-3.3B",
]

CLOSED_SOURCE_MODELS = {
    "gpt-5",
    "claude-fable-5",
    "gemini-3.8-flash",
}

DEFAULT_LINE_STYLE = (0, (3, 2))
SOLID_LINE_STYLE = "solid"

MODEL_LINE_STYLE_OVERRIDES = {
    "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4": (0, (3, 2)),
    "openai/gpt-oss-20b": (0, (5, 2)),
    "gpt-5": (0, (5, 2)),
    "google/gemma-4-E4B-it": (0, (1, 2)),
    "gemini-3.8-flash": (0, (1, 2)),
    "CohereLabs/command-a-plus-05-2026-bf16": (0, (3, 1, 1, 1)),
    "Qwen/Qwen3.6-35B-A3B": (0, (5, 2)),
    "claude-fable-5": "dashdot",
    "tiiuae/Falcon3-7B-Instruct": (0, (3, 2)),
    "moonshotai/Kimi-K3": (0, (5, 2)),
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": (0, (1, 2)),
    "meta-llama/Llama-4-Scout-17B-16E-Instruct": "dashdot",
}

# Brand-inspired colors. Prefer official/public brand colors where available;
# keep NLLB black and shared-tokenizer pairs on a single color for readability.
MODEL_COLOR_OVERRIDES = {
    "tiiuae/Falcon3-7B-Instruct": "#6302fd",  # TII/Falcon blue, approximated from public branding.
    "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4": "#76B900",
    "CohereLabs/command-a-plus-05-2026-bf16": "#FF7759",
    "openai/gpt-oss-20b": "#10A37F",
    "gpt-5": "#10A37F",
    "zai-org/GLM-5.3": "#1a94ff",  # Z.ai/Zhipu public branding is less standardized.
    "Qwen/Qwen3.6-35B-A3B": "#FF6701",  # Alibaba/Qwen orange.
    "moonshotai/Kimi-K3": "#0000ee",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": "#4D6BFE",
    "meta-llama/Llama-4-Scout-17B-16E-Instruct": "#0082FB",
    "claude-fable-5": "#d97757",
    "google/gemma-4-E4B-it": "#0D652D",
    "gemini-3.8-flash": "#0D652D",
    "facebook/nllb-200-3.3B": "black",
}

SHARED_TOKENIZER_GROUPS = [
    ("openai/gpt-oss-20b", "gpt-5"),
    ("google/gemma-4-E4B-it", "gemini-3.8-flash"),
]

LABEL_JOINT_X_OFFSETS = {
    "Qwen/Qwen3.6-35B-A3B": -5.0,
}

LABEL_Y_OVERRIDES = {
    "zai-org/GLM-5.3": "endpoint",
    "moonshotai/Kimi-K3": "endpoint",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": "endpoint",
}

SIMPLIFIED_MODEL_NAMES = {
    "tiiuae/Falcon3-7B-Instruct": "Falcon 3",
    "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4": "NVIDIA Nemotron Lightning",
    "CohereLabs/command-a-plus-05-2026-bf16": "CohereLabs Command A+",
    "openai/gpt-oss-20b": "OpenAI GPT-OSS",
    "gpt-5": "GPT-5",
    "zai-org/GLM-5.3": "Z.ai GLM 5.3",
    "Qwen/Qwen3.6-35B-A3B": "Qwen 3.6",
    "moonshotai/Kimi-K3": "Kimi K3",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": "DeepSeek R1 Distill Qwen",
    "meta-llama/Llama-4-Scout-17B-16E-Instruct": "Meta Llama 4 Scout",
    "claude-fable-5": "Claude Fable",
    "google/gemma-4-E4B-it": "Google Gemma",
    "gemini-3.8-flash": "Gemini Flash",
    "facebook/nllb-200-3.3B": "Meta NLLB",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a readable FLORES-200 max/min ratio chart."
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
        "--language-list",
        type=Path,
        default=DEFAULT_LANGUAGE_LIST,
        help=f"TSV containing the language ordering. Default: {DEFAULT_LANGUAGE_LIST}",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help='Models to plot, or pass "all" to use all models in the metric output.',
    )
    parser.add_argument(
        "--ylim",
        type=float,
        default=18,
        help="Upper y-axis limit for the readable chart. Default: 18",
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
        default=1000,
        help="Output image height in pixels. Default: 1000",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=160,
        help="Output image DPI. Figure inches are derived from pixel size / DPI.",
    )
    parser.add_argument(
        "--annotate-jumps",
        action="store_true",
        help="Annotate language codes at the top for the largest line jumps.",
    )
    parser.add_argument(
        "--jump-label-count",
        type=int,
        default=8,
        help="Number of distinct language jump positions to annotate. Default: 8",
    )
    parser.add_argument(
        "--solid-lines",
        action="store_true",
        help="Force all model lines and label arrows to use solid strokes.",
    )
    return parser.parse_args()


def load_language_order(path: Path, rank_var: str) -> pd.DataFrame:
    lang_df = pd.read_csv(path, sep="\t")
    required_columns = {"iso_script", "name", rank_var}
    missing_columns = required_columns - set(lang_df.columns)
    if missing_columns:
        missing_list = ", ".join(sorted(missing_columns))
        raise ValueError(f"Language list is missing required column(s): {missing_list}")

    lang_df = lang_df.dropna(subset=[rank_var]).copy()
    lang_df[rank_var] = lang_df[rank_var].astype(int)
    if lang_df[rank_var].duplicated().any():
        duplicates = lang_df.loc[lang_df[rank_var].duplicated(), rank_var].tolist()
        raise ValueError(f"Language list contains duplicate {rank_var} values: {duplicates}")

    return lang_df.sort_values(rank_var, ignore_index=True)


def select_models(df: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    if len(models) == 1 and models[0].lower() == "all":
        return df.copy()

    available = set(df["model"].unique())
    missing = [model for model in models if model not in available]
    if missing:
        missing_list = "\n  - ".join(missing)
        raise ValueError(f"Requested model(s) not found in CSV:\n  - {missing_list}")

    return df.loc[df["model"].isin(models)].copy()


def model_source_category(model: str) -> str:
    if "nllb" in model.lower():
        return "nllb"
    if model in CLOSED_SOURCE_MODELS:
        return "closed-source"
    return "open-source"


def model_color(model: str, fallback_color: str) -> str:
    return MODEL_COLOR_OVERRIDES.get(model, fallback_color)


def model_line_style(model: str):
    return MODEL_LINE_STYLE_OVERRIDES.get(model, DEFAULT_LINE_STYLE)


def display_model_name(model: str) -> str:
    return SIMPLIFIED_MODEL_NAMES.get(model, model)


def largest_jump_annotations(
    df: pd.DataFrame,
    language_order: pd.DataFrame,
    rank_var: str,
    count: int,
) -> list[dict[str, object]]:
    rank_to_code = dict(zip(language_order[rank_var], language_order["iso_script"]))
    jumps_by_n: dict[int, dict[str, object]] = {}

    for model, model_df in df.groupby("model", sort=False):
        model_df = model_df.sort_values("N")
        deltas = model_df["max_min_ratio"].diff()
        for row, delta in zip(model_df.itertuples(index=False), deltas):
            if pd.isna(delta) or delta <= 0:
                continue

            n = int(row.N)
            existing = jumps_by_n.get(n)
            if existing is None or float(delta) > float(existing["jump"]):
                jumps_by_n[n] = {
                    "N": n,
                    "code": rank_to_code.get(n, str(n)),
                    "jump": float(delta),
                    "model": model,
                }

    return sorted(
        jumps_by_n.values(),
        key=lambda annotation: float(annotation["jump"]),
        reverse=True,
    )[:count]


def build_label_groups(
    df: pd.DataFrame,
    model_order: list[str],
    precision: int = 2,
) -> list[dict[str, object]]:
    tokenizer_group_keys = {
        model: ("shared-tokenizer", group_index)
        for group_index, models in enumerate(SHARED_TOKENIZER_GROUPS)
        for model in models
    }
    grouped_curves: dict[tuple[object, ...], list[str]] = {}

    for model in model_order:
        if model in tokenizer_group_keys:
            group_key = tokenizer_group_keys[model]
        else:
            model_df = df.loc[df["model"] == model].sort_values("N")
            group_key = tuple(
                round(value, precision) for value in model_df["max_min_ratio"]
            )
        grouped_curves.setdefault(group_key, []).append(model)

    label_groups = []
    for i, models in enumerate(grouped_curves.values()):
        endpoints = [
            float(
                df.loc[df["model"] == model]
                .sort_values("N")["max_min_ratio"]
                .iloc[-1]
            )
            for model in models
        ]
        label_groups.append(
            {
                "id": f"group_{i}",
                "models": models,
                "endpoint": sum(endpoints) / len(endpoints),
            }
        )

    return sorted(label_groups, key=lambda group: group["endpoint"], reverse=True)


def label_lines_for_group(
    models: list[str],
    model_colors: dict[str, str],
    wrap_width: int = 38,
) -> list[list[tuple[str, str]]]:
    if len(models) > 1:
        segments = []
        for model_index, model in enumerate(models):
            if model_index > 0:
                segments.append((" + ", "#555555"))
            segments.append((display_model_name(model), model_colors[model]))
        return [segments]

    label_lines = []
    for model_index, model in enumerate(models):
        wrapped_model = wrap(display_model_name(model), width=wrap_width)
        for line_index, line in enumerate(wrapped_model):
            label_lines.append([(line, model_colors[model])])

    return label_lines


def packed_label_positions(
    label_groups: list[dict[str, object]],
    gap: float,
    lower: float,
    upper: float,
) -> dict[str, float]:
    """Place label blocks near endpoints, centering labels within dense clusters."""
    sorted_groups = sorted(label_groups, key=lambda group: group["endpoint"])
    clusters: list[list[dict[str, object]]] = []

    for group in sorted_groups:
        if not clusters:
            clusters.append([group])
            continue

        previous_endpoint = float(clusters[-1][-1]["endpoint"])
        if float(group["endpoint"]) - previous_endpoint <= 1.0:
            clusters[-1].append(group)
        else:
            clusters.append([group])

    cluster_layouts = []
    previous_top = lower
    for cluster in clusters:
        total_height = sum(float(group["label_height"]) for group in cluster)
        total_height += gap * (len(cluster) - 1)
        target_center = sum(float(group["endpoint"]) for group in cluster) / len(cluster)
        bottom = target_center - total_height / 2
        bottom = max(bottom, lower, previous_top + gap)
        top = bottom + total_height
        cluster_layouts.append({"cluster": cluster, "bottom": bottom, "top": top})
        previous_top = top

    overflow = cluster_layouts[-1]["top"] - upper if cluster_layouts else 0
    if overflow > 0:
        for layout in cluster_layouts:
            layout["bottom"] -= overflow
            layout["top"] -= overflow

    previous_bottom = upper
    for layout in reversed(cluster_layouts):
        if layout["top"] > previous_bottom - gap:
            shift = layout["top"] - (previous_bottom - gap)
            layout["bottom"] -= shift
            layout["top"] -= shift
        if layout["bottom"] < lower:
            shift = lower - layout["bottom"]
            layout["bottom"] += shift
            layout["top"] += shift
        previous_bottom = layout["bottom"]

    positions: dict[str, float] = {}
    for layout in cluster_layouts:
        y_cursor = layout["bottom"]
        for group in layout["cluster"]:
            height = float(group["label_height"])
            positions[group["id"]] = y_cursor + height / 2
            y_cursor += height + gap

    return positions


def draw_ratio_plot(
    df: pd.DataFrame,
    language_order: pd.DataFrame,
    rank_var: str,
    output_path: Path,
    ylim: float,
    width_px: int,
    height_px: int,
    dpi: int,
    annotate_jumps: bool = False,
    jump_label_count: int = 8,
    solid_lines: bool = False,
) -> None:
    model_order = (
        df.groupby("model", sort=False)["max_min_ratio"]
        .last()
        .sort_values(ascending=False)
        .index.tolist()
    )

    fig_size = (width_px / dpi, height_px / dpi)
    fig, ax = plt.subplots(figsize=fig_size, dpi=dpi, constrained_layout=False)
    fig.subplots_adjust(left=0.1, right=0.56, top=0.82, bottom=0.19)

    bands = [
        (0, 3, "#b8d9bb", "#2f6b38", "A"),
        (3, 6, "#a7a2ee", "#3f3a9c", "B"),
        (6, 9, "#ffffb3", "#797a00", "C"),
        (9, 12, "#f9e2a4", "#8a5a00", "D"),
        (12, ylim, "#f5a7aa", "#9b2f35", "F"),
    ]

    band_ax = ax.inset_axes(
        [1.015, 0, 0.045, 1],
        transform=ax.transAxes,
        sharey=ax,
    )
    for y0, y1, color, text_color, grade in bands:
        band_ax.axhspan(y0, y1, color=color, alpha=0.9, linewidth=0)
        band_ax.text(
            0.5,
            (y0 + y1) / 2,
            grade,
            color=text_color,
            fontsize=11,
            fontweight="bold",
            ha="center",
            va="center",
        )
    band_ax.set_xlim(0, 1)
    band_ax.set_ylim(0, ylim)
    band_ax.set_xticks([])
    band_ax.tick_params(axis="y", left=False, labelleft=False)
    for spine in band_ax.spines.values():
        spine.set_visible(False)

    if annotate_jumps:
        script_counts = language_order["script"].value_counts()
        single_language_script_ranks = language_order.loc[
            language_order["script"].map(script_counts).eq(1),
            rank_var,
        ]
        for x in single_language_script_ranks:
            ax.axvline(
                int(x),
                color="#f2c94c",
                linewidth=1.6,
                alpha=0.28,
                zorder=0,
            )

    colors = plt.get_cmap("tab20").colors
    endpoint_values = {}
    model_colors = {}

    for i, model in enumerate(model_order):
        model_df = df.loc[df["model"] == model].sort_values("N")
        color = model_color(model, colors[i % len(colors)])
        model_colors[model] = color
        ax.step(
            model_df["N"],
            model_df["max_min_ratio"],
            where="post",
            linewidth=1.6,
            color=color,
            linestyle=SOLID_LINE_STYLE if solid_lines else model_line_style(model),
            alpha=0.95,
        )
        endpoint_values[model] = float(model_df["max_min_ratio"].iloc[-1])

    label_groups = build_label_groups(df, model_order)
    for group in label_groups:
        label_lines = label_lines_for_group(group["models"], model_colors)
        line_gap = 0.34 if len(group["models"]) > 1 else 0.42
        group["label_lines"] = label_lines
        group["line_gap"] = line_gap
        group["label_height"] = max(line_gap, line_gap * len(label_lines))

    label_positions = packed_label_positions(
        label_groups=label_groups,
        gap=0.22,
        lower=0.6,
        upper=ylim - 0.35,
    )

    x_max = float(min(df["N"].max(), language_order[rank_var].max()))
    grade_bar_tip_x = x_max + 10.0
    label_x_base = x_max + 20.0
    label_x_offsets = [0.0, 4.0, 8.0, 2.0, 6.0, 10.0, 14.0]
    for group_index, group in enumerate(label_groups):
        models = group["models"]
        representative_model = models[0]
        color = model_colors[representative_model]
        line_style = (
            SOLID_LINE_STYLE if solid_lines else model_line_style(representative_model)
        )
        y_end = group["endpoint"]
        y_label = label_positions[group["id"]]
        if LABEL_Y_OVERRIDES.get(representative_model) == "endpoint":
            y_label = y_end
        label_x = label_x_base + label_x_offsets[group_index % len(label_x_offsets)]
        arrow_start_x = label_x - 1.4
        arrow_start_x += LABEL_JOINT_X_OFFSETS.get(representative_model, 0)
        arrow = FancyArrowPatch(
            (arrow_start_x, y_end),
            (grade_bar_tip_x, y_end),
            arrowstyle="-|>",
            color=color,
            linewidth=1.6,
            linestyle=line_style,
            mutation_scale=8,
            transform=ax.transData,
            clip_on=False,
            zorder=20,
        )
        ax.add_patch(arrow)

        label_lines = group["label_lines"]
        line_gap = group["line_gap"]
        start_y = y_label + line_gap * (len(label_lines) - 1) / 2
        end_y = start_y - line_gap * (len(label_lines) - 1)

        if abs(y_label - y_end) > 0.03:
            ax.plot(
                [arrow_start_x, arrow_start_x],
                [y_label, y_end],
                color=color,
                linewidth=1.6,
                linestyle=line_style,
                clip_on=False,
                zorder=19,
            )
            ax.plot(
                [arrow_start_x, label_x - 0.9],
                [y_label, y_label],
                color=color,
                linewidth=1.6,
                linestyle=line_style,
                clip_on=False,
                zorder=19,
            )

        for line_index, line_segments in enumerate(label_lines):
            line_y = start_y - line_index * line_gap
            if len(line_segments) == 1:
                line, line_color = line_segments[0]
                ax.text(
                    label_x,
                    line_y,
                    line,
                    color=line_color,
                    fontsize=9,
                    fontweight="heavy",
                    va="center",
                    ha="left",
                    bbox={
                        "facecolor": "white",
                        "edgecolor": "none",
                        "alpha": 0.82,
                        "pad": 0.25,
                    },
                )
                continue

            packed_label = HPacker(
                children=[
                    TextArea(
                        text,
                        textprops={
                            "color": text_color,
                            "fontsize": 9,
                            "fontweight": "heavy",
                            "bbox": {
                                "facecolor": "white",
                                "edgecolor": "none",
                                "alpha": 0.82,
                                "pad": 0.25,
                            },
                        },
                    )
                    for text, text_color in line_segments
                ],
                align="center",
                pad=0,
                sep=0,
            )
            ax.add_artist(
                AnnotationBbox(
                    packed_label,
                    (label_x, line_y),
                    xycoords=ax.transData,
                    box_alignment=(0, 0.5),
                    frameon=False,
                    pad=0,
                    annotation_clip=False,
                )
            )

    ax.set_title(
        "Tokenizer Tax by Language Set",
        fontsize=12.5,
        fontweight="bold",
        loc="left",
        pad=24,
    )
    ax.text(
        0,
        1.02,
        "Languages Ranked Highest to Lowest Resource",
        transform=ax.transAxes,
        fontsize=9,
        color="#666666",
        ha="left",
        va="bottom",
    )
    ax.set_xlabel("Number of Languages in Set", fontsize=10)
    ax.set_ylabel(
        "Max Tokenizer Tax",
        fontsize=10,
        labelpad=14,
    )
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, ylim)
    ax.set_xticks(range(0, int(x_max) + 1, 25))
    y_ticks = list(range(0, int(ylim) + 1, 3))
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(["" if tick == 0 else f"{tick}x" for tick in y_ticks])
    ax.tick_params(axis="both", length=0)
    ax.grid(axis="y", color="#dedede", linewidth=1.0, alpha=0.8)
    ax.grid(axis="x", color="#ececec", linewidth=0.6, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if annotate_jumps:
        fig.text(
            0.04,
            0.06,
            "Faint yellow lines mark languages written in scripts represented by only one "
            "language in the FLORES-200 set.",
            fontsize=8,
            color="#555555",
        )

    fig.text(
        0.04,
        0.035,
        "CLEAR grade rubric across 200 languages: A <= 3x, B <= 6x, "
        "C <= 9x, D <= 12x, F > 12x.",
        fontsize=8,
        color="#333333",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    configure_fonts()

    args = parse_args()
    language_order = load_language_order(args.language_list, RANK_VAR)

    df = pd.read_csv(args.input)
    df = select_models(df, args.models)
    df = df.loc[df["N"] <= language_order[RANK_VAR].max()].copy()

    if df.empty:
        raise ValueError("No rows left to plot after filtering.")

    draw_ratio_plot(
        df,
        language_order,
        RANK_VAR,
        args.output,
        args.ylim,
        args.width_px,
        args.height_px,
        args.dpi,
        args.annotate_jumps,
        args.jump_label_count,
        args.solid_lines,
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
