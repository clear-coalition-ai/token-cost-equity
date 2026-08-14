
# %% Setup
# Python standard library
import os

# Third-party libraries
from dotenv import load_dotenv
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde


# %% Project folder
load_dotenv()
PROJECT_ROOT = os.getenv("PROJECT_ROOT")
if not PROJECT_ROOT:
    raise EnvironmentError("PROJECT_ROOT is not set. Check your .env file.")
os.chdir(PROJECT_ROOT)


# %% Read token count data
results_df = pd.read_csv("02_output_model_experiments/flores200_token_counts_langinfo.csv", keep_default_na = False, na_values = [""])

# Get list of models
model_list = results_df["model"].unique().tolist()

# Show token counts in thousands for ease of reading chart axis labels
results_df["token_count"] = results_df["token_count"] / 1000

# Create script color map
script_df = results_df["script"].value_counts()
script_df = script_df.reset_index(drop = False)
script_df = script_df.drop(columns = "count")

script_color_dict = {"Latn": "blue",
                     "Arab": "red",
                     "Cyrl": "red",
                     "Deva": "red",
                     "Hans": "red",
                     "Hant": "red",
                     }

script_df["color"] = script_df["script"].map(script_color_dict)
script_df.loc[script_df["color"].isna(), "color"] = "lightblue"
    # make all other scripts light blue

# Merge to add script colors mapping to results_df
results_df = pd.merge(left = results_df, right = script_df, on = "script", how = "outer", indicator = True)
assert all(results_df["_merge"] == "both")
results_df = results_df.drop(columns="_merge")



# %% Define functions to plot Lorenz and concentration curves
def lorenz_curve(token_var):
    """
    Compute Lorenz curve coordinates from a vector of raw token count values.
 
    Parameters
    ----------
    vec : array-like
        Raw (non-negative) token values, one per language.
 
    Returns
    -------
    lang_share : np.ndarray
        Cumulative share of languages (0 to 1), including the origin.
    income_share : np.ndarray
        Cumulative share of total token count (0 to 1), including the origin.
    gini : float
        Gini coefficient computed from the same sorted data.
    """
    vec = np.asarray(token_var, dtype=float)
    if np.any(vec < 0):
        raise ValueError("Token count values must be non-negative.")
 
    n = len(vec)
    sorted_vec = np.sort(vec)
        # sort values in vec in ascending order
 
    cum_tokens = np.cumsum(sorted_vec)
    token_share = cum_tokens / cum_tokens[-1]
    lang_share = np.arange(1, n + 1) / n
 
    # Prepend the origin (0, 0) so the curve starts there
    token_share = np.insert(token_share, 0, 0.0)
    lang_share = np.insert(lang_share, 0, 0.0)
 
    # Gini coefficient via the trapezoidal-area relationship to the Lorenz curve
    # G = 1 - 2 * area under Lorenz curve
    area_under_curve = np.trapezoid(token_share, lang_share)
    gini = 1 - 2 * area_under_curve
 
    return lang_share, token_share, gini
 
 
def plot_lorenz(token_var, ax=None):
    """Plot the Lorenz curve for a raw income vector against the line of equality."""
    lang_share, token_share, gini = lorenz_curve(token_var)
 
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
 
    ax.plot(lang_share, token_share, color="steelblue", lw=2, label="Lorenz curve")
    ax.plot([0, 1], [0, 1], color="gray", ls="--", lw=1, label="Line of equality")
    ax.fill_between(lang_share, token_share, lang_share, color="steelblue", alpha=0.15)
 
    ax.set_xlabel("Cumulative share of languages\nRanking: Ascending Order by Token Count")
    ax.set_ylabel("Cumulative share of token count")
    ax.set_title(f"Lorenz Curve (Gini Index = {gini:.3f})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.set_aspect("equal")
 
    return ax #, gini


def concentration_curve(rank_var, cost_var, lower_rank_is_bigger=True):
    """
    Compute concentration curve coordinates.
 
    Parameters
    ----------
    rank_var : array-like
        Variable used to rank units (e.g. speaker count, training-data
        share). Units are sorted ascending on this variable.
    cost_var : array-like
        The outcome being distributed (e.g. tokens-per-word fertility,
        or total tokens on a fixed reference corpus). Each unit
        contributes equally on the x-axis (one language = one unit).
    lower_rank_is_bigger : bool
        Whether smaller rank values correspond to bigger languages 
        (bigger as in more training data, high speaker count, etc.).
        For example, if ranking by amount of training data and rank 1
        corresponds to the language with most training data, then set this
        argument to True.
 
    Returns
    -------
    pop_share, cost_share : np.ndarray
        Cumulative share of units (x) and cumulative share of cost (y),
        including the (0, 0) origin.
    ci : float
        Concentration index. Positive = progressive (cost concentrated
        among high-rank units). Negative = regressive (cost concentrated
        among low-rank units).
    """
    rank_var = np.asarray(rank_var, dtype=float)
    cost_var = np.asarray(cost_var, dtype=float)
    n = len(rank_var)

    # Always sort so the smallest/least-resourced unit is first.
    sort_key = -rank_var if lower_rank_is_bigger else rank_var
    order = np.argsort(sort_key)
    sorted_cost = cost_var[order]
 
    cost_share = np.cumsum(sorted_cost) / sorted_cost.sum()
    pop_share = np.arange(1, n + 1) / n
 
    cost_share = np.insert(cost_share, 0, 0.0)
    pop_share = np.insert(pop_share, 0, 0.0)
 
    ci = 1 - 2 * np.trapezoid(cost_share, pop_share)
    return pop_share, cost_share, ci
 
 
def plot_concentration(rank_var, cost_var, rank_label="languages\nRanking: Descending by Common Crawl Pages", ax=None):
    """Plot the concentration curve against the diagonal line of equality."""
    pop_share, cost_share, ci = concentration_curve(rank_var, cost_var)
 
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
 
    ax.plot(pop_share, cost_share, marker="o", ms=3, color="#993C1D", label="Concentration curve")
    ax.plot([0, 1], [0, 1], ls="--", color="gray", lw=1, label="Line of equality")
    ax.fill_between(pop_share, cost_share, pop_share, color="#993C1D", alpha=0.12)
 
    ax.set_xlabel(f"Cumulative share of {rank_label}")
    ax.set_ylabel("Cumulative share of token cost")
    ax.set_title(f"Concentration Curve (Concentration Index = {ci:.3f})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.legend()
 
    return ax #, ci


# %% Plot concentration curves for each model
with PdfPages("03_output_analysis/Token Count Concentration Charts - FLORES200.pdf") as pdf:
    for model in model_list:

        chart_data = results_df.loc[(results_df["model"] == model) & (results_df["cc_spk_rank"] <= 200),]
        
        tok_max = chart_data["token_count"].max() + 10
            # +10 for visual padding in chart

        axs0_max = 2
        axs1_max = 2

        fig, axs = plt.subplots(axs0_max, axs1_max, figsize = (14,14))
        fig.suptitle(f"Differences in Token Cost Across Languages\nModel: {model}\nData: FLORES-200 dev")

        # Bar Chart
        chart_data = chart_data.sort_values(by = "token_count", ascending = True, ignore_index = True)
        axs[0,0].bar(x = chart_data["iso_script"], height = chart_data["token_count"], color = chart_data["color"])
        axs[0,0].set_title("Token Count by Language")
        axs[0,0].set_xlabel("Languages\nRanking: Ascending Order by Token Count")
        axs[0,0].set_ylabel("Token Count (in Thousands)")
        axs[0,0].set_ylim(0, tok_max)
        axs[0,0].tick_params(axis = "x", labelbottom = False)

        axs[0,0].text(0.02, 0.98, f"Bar Colors Vary by Script:\nBlue = Latn\nRed = Arab, Cyrl, Deva, Hans, Hant\nLight Blue = All Others",
                      transform = axs[0,0].transAxes,
                      fontsize = 8,
                      verticalalignment = "top",
                      horizontalalignment = "left",
                      bbox=dict(boxstyle = "round", facecolor = "white", alpha = 0.8, edgecolor = "gray"))

        # Lorenz Curve
        plot_lorenz(chart_data["token_count"], ax=axs[1,0])

        # Bar Chart
        chart_data = chart_data.sort_values(by = "cc_spk_rank", ascending = False, ignore_index = True)
        axs[0,1].bar(x = chart_data["iso_script"], height = chart_data["token_count"], color = chart_data["color"])
        axs[0,1].set_title("Token Count by Language (With Kernel Density Plot)")
        axs[0,1].set_xlabel("Languages\nRanking: From Lowest Resource to Highest")
        axs[0,1].set_ylabel("Token Count (in Thousands)")
        axs[0,1].set_ylim(0, tok_max)
        axs[0,1].tick_params(axis = "x", labelbottom = False)

        positions = np.arange(len(chart_data))
        weights = chart_data["token_count"].values
        kde = gaussian_kde(positions, weights=weights, bw_method=0.1)
        y = kde(positions)
        ax2 = axs[0,1].twinx()
        ax2.plot(positions, y, color="gray", lw=1.2, linestyle="--", alpha=0.7)
        ax2.set_ylabel("Density")

        tok_argmax = chart_data.iloc[weights.argmax(),][["name", "script"]].tolist()
        tok_max_lang = tok_argmax[0] + " [" + tok_argmax[1] + "]"

        kde_argmax = chart_data.iloc[y.argmax(),][["name", "script"]].tolist()
        kde_max_lang = kde_argmax[0] + " [" + kde_argmax[1] + "]"

        axs[0,1].text(0.98, 0.98, f"argmax(density) = {kde_max_lang}\nargmax(token count) = {tok_max_lang}",
                      transform = axs[0,1].transAxes,
                      fontsize = 8,
                      verticalalignment = "top",
                      horizontalalignment = "right",
                      bbox=dict(boxstyle = "round", facecolor = "white", alpha = 0.8, edgecolor = "gray"))

        # Concentration Curve
        plot_concentration(rank_var = chart_data["cc_spk_rank"], cost_var = chart_data["token_count"],
                            rank_label = "languages\nRanking: From Lowest Resource to Highest",
                            ax = axs[1,1])
        
        pdf.savefig()
        plt.close()


# TODO: For consideration:
#   There is a discrepancy between the concentration curve and the kernel density chart
#   The concentration curve is a cumulative share - the calculation of each share only takes the values "to the left" into account
#   The current kernel being used in the k-density plot is a Gaussian kernel, which takes values both to the left and to the right
#   While the Gaussian kernel density is a useful summary of the data in its own right, if the goal is to use the kernel density chart to
#   interpret specific segments of the concentration curve, a weighted average or a kernel that only looks to the left may be more appropriate.
        
