
# %% Setup
# Python standard library
import os

# Third-party libraries
from dotenv import load_dotenv
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# %% Project folder
load_dotenv()
PROJECT_ROOT = os.getenv("PROJECT_ROOT")
if not PROJECT_ROOT:
    raise EnvironmentError("PROJECT_ROOT is not set. Check your .env file.")
os.chdir(PROJECT_ROOT)


# %% Clean results DataFrame for plotting and analysis
results_df = pd.read_csv("02_output_model_experiments/token_counts_flores200.csv")

# Get list of models
model_list = results_df["model"].unique().tolist()


# %% Add language info
flores200_langinfo = pd.read_table("01_data_processed/flores200_langinfo.tsv")

results_df = pd.merge(left = results_df, right = flores200_langinfo, how = "outer", on = "file", indicator = True)

assert all(results_df["_merge"] == "both")
    # confirming merge resulted in full match
results_df = results_df.drop(columns = "_merge")

assert results_df["speakers"].isna().sum() == 0
    # no more missing values in "speakers" column


# %% Concentration curve
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
    #order = np.argsort(rank_var)  # ascending: smallest/least-resourced first
    #sorted_cost = cost_var[order]
 
    cost_share = np.cumsum(sorted_cost) / sorted_cost.sum()
    pop_share = np.arange(1, n + 1) / n
 
    cost_share = np.insert(cost_share, 0, 0.0)
    pop_share = np.insert(pop_share, 0, 0.0)
 
    ci = 1 - 2 * np.trapezoid(cost_share, pop_share)
    return pop_share, cost_share, ci
 
 
def plot_concentration(rank_var, cost_var, model, labels=None, rank_label="languages, ranked by Common Crawl pages", ax=None):
    """Plot the concentration curve against the diagonal line of equality."""
    pop_share, cost_share, ci = concentration_curve(rank_var, cost_var)
 
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
 
    ax.plot(pop_share, cost_share, marker="o", ms=3, color="#993C1D", label="Concentration curve")
    ax.plot([0, 1], [0, 1], ls="--", color="gray", lw=1, label="Line of equality")
    ax.fill_between(pop_share, cost_share, pop_share, color="#993C1D", alpha=0.12)
 
    ax.set_xlabel(f"Cumulative share of {rank_label}")
    ax.set_ylabel("Cumulative share of token cost")
    ax.set_title(f"Tokenization cost concentration (CI = {ci:.3f})\n{model}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.legend()
 
    return ax, ci

with PdfPages("03_output_analysis/Token Count Concetration Charts - FLORES200.pdf") as pdf:
    for model in model_list:
        chart_data = results_df.loc[results_df["model"] == model,]
        chart_data = chart_data.sort_values(by = "cc_spk_rank", ignore_index = True)
        plot_concentration(rank_var = chart_data["cc_spk_rank"], cost_var = chart_data["token_count"], model = model,
                           rank_label = "languages\nRanked by Common Crawl pages (smallest to largest)" )
        pdf.savefig()
        plt.close()



        chart_data = results_df.loc[results_df["model"] == "google/gemma-4-E4B-it",]
        chart_data["flipped_rank"] = chart_data.shape[0] + 1 - chart_data["cc_spk_rank"]
        plot_concentration(rank_var = chart_data["flipped_rank"], cost_var = chart_data["token_count"])

        chart_data = results_df.loc[results_df["model"] == "gpt-5",]
        plot_concentration(rank_var = chart_data["cc_spk_rank"], cost_var = chart_data["token_count"])

        chart_data = results_df.loc[results_df["model"] == "meta-llama/Llama-3.3-70B-Instruct",]
        plot_concentration(rank_var = chart_data["cc_spk_rank"], cost_var = chart_data["token_count"])

        chart_data = results_df.loc[results_df["model"] == "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",]
        plot_concentration(rank_var = chart_data["cc_spk_rank"], cost_var = chart_data["token_count"])


# %% Prepare for plotting
# Show token counts in thousands for ease of reading chart axis labels
results_df["token_count"] = results_df["token_count"] / 1000

# Find languages with multiple scripts
N_model = len(model_list)
script_cnt = results_df.groupby("iso_639_3").size() / N_model
script_cnt = script_cnt.reset_index(drop = False)
script_cnt.columns = ["iso_639_3", "count"]
iso_multiscript_list = script_cnt.loc[script_cnt["count"] > 1, "iso_639_3"].tolist()

# Build dictionary for mapping colors to ISO 639-3 codes
color_list = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:gray", "tab:pink"]
assert len(iso_multiscript_list) == len(color_list) 
N_iso = len(iso_multiscript_list)
iso_color_map = {iso_multiscript_list[i]: color_list[i] for i in range(N_iso)}   


# %% Generate plots for each model
with PdfPages("03_output_analysis/Token Count Charts - FLORES200.pdf") as pdf:
    for model in model_list:
        print(f"Working on report: {model}")
        
        chart_data = results_df.loc[results_df["model"] == model,]
        chart_data = chart_data.sort_values(by = "token_count", ascending = True, ignore_index = True)

        tok_max = chart_data["token_count"].max() + 10
            # +10 for visual padding in chart
            
        axs0_max = 2
        axs1_max = 2
   
        fig, axs = plt.subplots(axs0_max, axs1_max, figsize = (14,14))
        fig.suptitle("Differences in Token Cost Across Languages\nModel: " + model + "\nData: FLORES-200 dev")
        
        # Bar Chart
        axs[0,0].bar(x = chart_data["iso_script"], height = chart_data["token_count"])
        axs[0,0].set_title("Token Count by Language")
        axs[0,0].set_xlabel("Language-Script Pair (in Ascending Order by Token Count)")
        axs[0,0].set_ylabel("Token Count (in Thousands)")
        axs[0,0].set_ylim(0, tok_max)
        axs[0,0].tick_params(axis = "x", labelbottom = False)

        # Histogram         
        axs[0,1].hist(x = chart_data["token_count"], bins = 40, range = (0, tok_max))
        axs[0,1].set_title("Histogram of Token Counts")
        axs[0,1].set_xlabel("Token Count (in Thousands)")
        axs[0,1].set_ylabel("Number of Languages")

        # Scatter Plot by Script
        axs[1,0].scatter(x = chart_data["script"], y = chart_data["token_count"], s = 20, alpha = 0.5)
        axs[1,0].set_title("Token Counts by Script")
        axs[1,0].set_xlabel("Script (ISO 15924)")
        axs[1,0].set_ylabel("Token Count (in Thousands)")
        axs[1,0].set_ylim(0, tok_max)
        axs[1,0].tick_params(axis = "x", labelrotation = 90, labelsize = 8)

        # Scatter Plot for Multiscript Languages
        chart_data = chart_data.loc[chart_data["iso_639_3"].isin(iso_multiscript_list),]
            # Filter chart data to only languages with multiple scripts
        chart_data["color"] = chart_data["iso_639_3"].map(iso_color_map)
            # Assign color to each ISO 639-3 code for plotting
        chart_data = chart_data.sort_values(by = "iso_script", ascending = True, ignore_index = True)
                
        axs[1,1].scatter(x = chart_data["iso_script"], y = chart_data["token_count"], s = 20, alpha = 1,
                         c = chart_data["color"])
        axs[1,1].set_title("Languages with Multiple Scripts")
        axs[1,1].set_xlabel("Language-Script Pair (ISO 639-3, ISO 15924)")
        axs[1,1].set_ylabel("Token Count (in Thousands)")
        axs[1,1].set_ylim(0, tok_max)
        axs[1,1].tick_params(axis = "x", labelrotation = 90, labelsize = 8)

        plt.tight_layout()
        pdf.savefig()
        plt.close()


# %% Define Gini coefficient function
def gini(x):
    """
    Compute the Gini coefficient of a dataset using the sorting shortcut.
 
    G = (2 * sum(i * x_i)) / (n * sum(x_i)) - (n + 1) / n
 
    where x_i are sorted ascending and i in {1, 2, ... , n}
 
    Parameters
    ----------
    x : array-like
        Non-negative values (e.g., incomes, wealth). Must contain at
        least one positive value.
 
    Returns
    -------
    float
        Gini coefficient, ranging from 0 (perfect equality) to
        just under 1 (maximal inequality).
    """
    x = np.asarray(x, dtype=np.float64).flatten()
 
    if np.any(x < 0):
        raise ValueError("Gini coefficient requires non-negative values.")
    if x.sum() == 0:
        raise ValueError("Gini coefficient is undefined when all values are zero.")
 
    n = len(x)
    x_sorted = np.sort(x)
    ranks = np.arange(1, n + 1)
 
    return (2 * np.sum(ranks * x_sorted)) / (n * np.sum(x_sorted)) - (n + 1) / n


# %% Calculate Gini coefficients
#gini_N_list = np.arange(5, 200 + 1, step = 5)
gini_N_list = np.arange(2, 200 + 1, step = 1)

    # +1 to include 200
gini_results = []
ratio_results = []

for model in model_list:
    print(f"Working on {model}")
        
    calc_data = results_df.loc[results_df["model"] == model,]
    #calc_data = calc_data.sort_values(by = "speakers", ascending = False, ignore_index = True)

    #if model == model_list[0]:
    #    calc_data[["iso_script", "name", "speakers"]].to_csv("03_output_analysis/gini_languages.csv", index = True)
    #        # exporting list of languages that Gini coefficients are based on

    for rank_var in ["cc_spk_rank", "spk_rank"]:
        for N in gini_N_list:
            #token_count_vector = calc_data.loc[calc_data.index < N, "token_count"]
            token_count_vector = calc_data.loc[calc_data[rank_var] <= N, "token_count"]
            
            g = gini(token_count_vector)
            gini_result_i = {"model": model, "rank_var": rank_var, "N": N, "gini": g}
            gini_results.append(gini_result_i)

            max_min_ratio = token_count_vector.max() / token_count_vector.min()
            ratio_result_i = {"model": model, "rank_var": rank_var, "N": N, "max_min_ratio": max_min_ratio}
            ratio_results.append(ratio_result_i)

gini_df = pd.DataFrame(data = gini_results)
ratio_df = pd.DataFrame(data = ratio_results)


# %% Generate Gini Charts
with PdfPages("03_output_analysis/Token Gini Charts - FLORES200.pdf") as pdf:
    axs0_max = 2
    axs1_max = 2

    for i in range(len(model_list)):
        print(f"Working on charts for {model}")
        model = model_list[i]

        chart_data_T = results_df.loc[results_df["model"] == model,]
        chart_data_TL = chart_data_T.sort_values(by = "spk_rank", ascending = True, ignore_index = True)
        chart_data_TR = chart_data_T.sort_values(by = "cc_spk_rank", ascending = True, ignore_index = True)

        tok_max = chart_data_T["token_count"].max() + 10

        chart_data_BL = gini_df.loc[(gini_df["model"] == model) & (gini_df["rank_var"] == "spk_rank"),]
        chart_data_BL = chart_data_BL.sort_values(by = "N", ascending = True, ignore_index = True)

        chart_data_BR = gini_df.loc[(gini_df["model"] == model) & (gini_df["rank_var"] == "cc_spk_rank"),]
        chart_data_BR = chart_data_BR.sort_values(by = "N", ascending = True, ignore_index = True)

        gini_max_BL = max(0.5, chart_data_BL["gini"].max() + 0.05)
        gini_max_BR = max(0.5, chart_data_BR["gini"].max() + 0.05)
        corr_BL = round(chart_data_BL[["N", "gini"]].corr().iloc[0,1], 2)
        corr_BR = round(chart_data_BR[["N", "gini"]].corr().iloc[0,1], 2)
            # Pearson correlation between N and gini
     
        fig, axs = plt.subplots(axs0_max, axs1_max, figsize = (14,14))
        fig.suptitle("Gini Coefficient as Function of Language Count\nModel:" + model + "\nData: FLORES-200 dev\n")

        # Bar Chart
        axs[0,0].bar(x = chart_data_TL["iso_script"], height = chart_data_TL["token_count"])
        axs[0,0].set_title("Token Count by Language")
        axs[0,0].set_xlabel("Language-Script Pair\nDescending Order by Speaker Population")
        axs[0,0].set_ylabel("Token Count (in Thousands)")
        axs[0,0].set_ylim(0, tok_max)
        axs[0,0].tick_params(axis = "x", labelbottom = False)

        axs[0,1].bar(x = chart_data_TR["iso_script"], height = chart_data_TR["token_count"])
        axs[0,1].set_title("Token Count by Language")
        axs[0,1].set_xlabel("Language-Script Pair\nDescending Order by Common Crawl Pages (Primarily) and Speaker Population (Secondarily)")
        axs[0,1].set_ylabel("Token Count (in Thousands)")
        axs[0,1].set_ylim(0, tok_max)
        axs[0,1].tick_params(axis = "x", labelbottom = False)

        # Line Chart
        axs[1,0].plot(chart_data_BL["N"], chart_data_BL["gini"])
        axs[1,0].set_title("Gini Coefficient by Top N Languages\nBy Speaker Population")
        axs[1,0].set_xlabel("Number of Languages (N)")
        axs[1,0].set_ylabel("Gini Coefficient")
        axs[1,0].set_ylim(0, gini_max_BL)
        axs[1,0].text(0.02, 0.98, f"Correlation(N, Gini) = {corr_BL}",
                    transform = axs[1,0].transAxes,
                    fontsize = 14,
                    verticalalignment = "top",
                    horizontalalignment = "left",
                    bbox=dict(boxstyle = "round", facecolor = "white", alpha = 0.8, edgecolor = "gray"))

        axs[1,1].plot(chart_data_BR["N"], chart_data_BR["gini"])
        axs[1,1].set_title("Gini Coefficient by Top N Languages\nBy Common Crawl Pages (Primarily) and Speaker Population (Secondarily)")
        axs[1,1].set_xlabel("Number of Languages (N)")
        axs[1,1].set_ylabel("Gini Coefficient")
        axs[1,1].set_ylim(0, gini_max_BR)
        axs[1,1].text(0.02, 0.98, f"Correlation(N, Gini) = {corr_BR}",
                    transform = axs[1,1].transAxes,
                    fontsize = 14,
                    verticalalignment = "top",
                    horizontalalignment = "left",
                    bbox=dict(boxstyle = "round", facecolor = "white", alpha = 0.8, edgecolor = "gray"))


        plt.tight_layout()
        pdf.savefig()
        plt.close()


# %% Prepare Final Gini Results
rank_var_final = "cc_spk_rank"
out_N_list = [5, 10, 20, 50, 100, 200]

gini_df_long = gini_df.loc[(gini_df["N"].isin(out_N_list)) & (gini_df["rank_var"] == rank_var_final),]
gini_df_long["gini"] = gini_df_long["gini"].round(2)

ratio_df_long = ratio_df.loc[(ratio_df["N"].isin(out_N_list)) & (ratio_df["rank_var"] == rank_var_final),]
ratio_df_long["max_min_ratio"] = ratio_df_long["max_min_ratio"].round(1)

gini_df_wide = gini_df_long.pivot(index="model", columns="N", values="gini")
gini_df_wide.columns = ["gini_" + str(N) for N in out_N_list]
gini_df_wide = gini_df_wide.reset_index(drop = False)
gini_df_wide = gini_df_wide.sort_values(by = "gini_50", ascending = True, ignore_index = True)

gini_50_sort = gini_df_wide[["model"]].reset_index(drop = False)
gini_50_sort.columns = ["sort_order", "model"]

ratio_df_wide = ratio_df_long.pivot(index="model", columns="N", values="max_min_ratio")
ratio_df_wide.columns = ["max_min_ratio_" + str(N) for N in out_N_list]
ratio_df_wide = ratio_df_wide.reset_index(drop = False)
ratio_df_wide = pd.merge(left = ratio_df_wide, right = gini_50_sort, how = "outer", on = "model", indicator = True)
assert all(ratio_df_wide["_merge"] == "both")
ratio_df_wide = ratio_df_wide.sort_values(by = "sort_order", ascending = True, ignore_index = True)
ratio_df_wide = ratio_df_wide.drop(columns = ["_merge", "sort_order"])


# %% Write Final Gini Results to CSV Files
gini_df_wide.to_csv("03_output_analysis/gini_flores200.csv", index = False)
ratio_df_wide.to_csv("03_output_analysis/ratio_flores200.csv", index = False)

# TODO: move gini() function to new src/util/equity_metrics file/folder


# %% Summarizing gini as function of N curves in a metric
gini_df_fin = gini_df.loc[gini_df["rank_var"] == rank_var_final,]

metric_results = []
for model in model_list:   
    calc_df = gini_df_fin.loc[gini_df_fin["model"] == model,]
    calc_df = calc_df.sort_values(by = "N", ascending = True, ignore_index = True)
    
    gini_vector = calc_df["gini"]
    
    gini_diff = gini_vector.diff()
    #metric_min = gini_diff.min()
    #assert metric_min < 0
    #langrank_min = gini_diff.idxmin() + 2
    
    #metric_max = gini_diff.max()
    #langrank_max = gini_diff.idxmax() + 2

    gini_diff_abs = gini_diff.abs()
    metric_abs_max = gini_diff_abs.max().round(4)
    langrank_abs_max = gini_diff_abs.idxmax() + 2
        # use to find language where bigger jump in gini occurs
        # Gini_AbsDiff_0 = Missing value
        # Gini_AbsDiff_1 = | Gini_3 - Gini_2 |
        # ...
        # Gini_AbsDiff_19 = | Gini_21 - Gini_20 |
        # ...
        # Gini_AbsDiff_198 = | Gini_200 - Gini_199 |

    
    metric_result_i = {"model": model, 
                       #"metric_min": metric_min, "langrank_min": langrank_min,
                       #"metric_max": metric_max, "langrank_max": langrank_max,
                       "metric_abs_max": metric_abs_max, "langrank_abs_max": langrank_abs_max}
    metric_results.append(metric_result_i)

metric_df = pd.DataFrame(data = metric_results)

langrank = flores200_langinfo[["cc_spk_rank", "name", "iso_script"]]

"""
langrank_max = langrank.rename(columns = {"cc_spk_rank": "langrank_max",
                                          "name": "name_max",
                                          "iso_script": "iso_script_max"})
assert langrank_max["langrank_max"].is_unique
metric_df = pd.merge(left = metric_df, right = langrank_max, how = "left", on = "langrank_max")

langrank_min = langrank.rename(columns = {"cc_spk_rank": "langrank_min",
                                          "name": "name_min",
                                          "iso_script": "iso_script_min"})
assert langrank_min["langrank_min"].is_unique
metric_df = pd.merge(left = metric_df, right = langrank_min, how = "left", on = "langrank_min")
"""

langrank_abs_max = langrank.rename(columns = {"cc_spk_rank": "langrank_abs_max",
                                              "name": "name_abs_max",
                                              "iso_script": "iso_script_abs_max"})
assert langrank_abs_max["langrank_abs_max"].is_unique
metric_df = pd.merge(left = metric_df, right = langrank_abs_max, how = "left", on = "langrank_abs_max")

# Finalize output
metric_df = metric_df[["model", "metric_abs_max", "langrank_abs_max", "name_abs_max", "iso_script_abs_max"]]
metric_df = metric_df.rename(columns = {"metric_abs_max": "metric", 
                                        "langrank_abs_max": "langrank", 
                                        "name_abs_max": "name", 
                                        "iso_script_abs_max": "iso_script"})

metric_df.to_csv("03_output_analysis/gini_curve_summary_metric.csv", index = False)



# %% Add models plotted together
gini_df_fin = gini_df_fin.loc[gini_df_fin["model"] != "gemini-2.5-pro",]
gini_df_fin = gini_df_fin.loc[gini_df_fin["model"] != "gemini-3.1-pro-preview",]
gini_df_fin = gini_df_fin.loc[gini_df_fin["model"] != "openai/gpt-oss-20b",]
gini_df_fin = gini_df_fin.loc[gini_df_fin["model"] != "claude-opus-4-8",]
    # drop some models to avoid overplotting where multiple models from same company/family are included

chart_model_list = gini_df_fin["model"].unique().tolist()

label_offset = {'Qwen/Qwen3.6-35B-A3B': 0,
                'microsoft/phi-4': 0,
                'google/gemma-4-31B-it': -0.005,
                'ibm-granite/granite-4.1-30b': -0.01,
                'deepseek-ai/DeepSeek-R1-Distill-Qwen-32B': 0,
                'deepseek-ai/DeepSeek-R1-Distill-Llama-70B': 0,
                'meta-llama/Llama-3.3-70B-Instruct': 0.02,
                'CohereLabs/tiny-aya-base': 0.005,
                'mistralai/Ministral-3-14B-Instruct-2512': 0,
                'facebook/nllb-200-3.3B': 0,
                'gpt-5': 0,
                'claude-fable-5': 0,
                'gemini-3.5-flash': 0.005}

with PdfPages("03_output_analysis/Token Gini Charts - FLORES200 - All Models.pdf") as pdf:
    fig, ax = plt.subplots(figsize=(10, 6))
    for i in range(len(chart_model_list)):
        model = chart_model_list[i]
        print(f"Working on {model}")

        chart_data = gini_df_fin.loc[gini_df_fin["model"] == model,]
        chart_data = chart_data.sort_values(by = "N", ascending = True, ignore_index = True)

        gini200 = chart_data.loc[gini_df_fin["N"] == 200, "gini"]

        line, = ax.plot(chart_data["N"], chart_data["gini"])

        y_last = chart_data["gini"].iloc[-1] + label_offset[model]
        #print(y_last)

        ax.text(200, y_last, 
                f"  {model}",
                va="center", ha="left",
                fontsize=8
        )


    ax.set_title("Gini Coefficient as a Function of Languages Included\nLanguages Ranked by Common Crawl Page Count")
    ax.set_xlabel("Number of Languages (N)")
    ax.set_ylabel("Gini Coefficient")

    ax.set_ylim(0, 0.6)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.tight_layout()
    pdf.savefig()
    plt.close()


