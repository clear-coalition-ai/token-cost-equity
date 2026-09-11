
# %% Setup
# Python standard library
import os

# Third-party libraries
from dotenv import load_dotenv
import numpy as np
import pandas as pd


# %% Project folder
load_dotenv()
PROJECT_ROOT = os.getenv("PROJECT_ROOT")
if not PROJECT_ROOT:
    raise EnvironmentError("PROJECT_ROOT is not set. Check your .env file.")
os.chdir(PROJECT_ROOT)


# %% Clean results DataFrame for plotting and analysis
count_df = pd.read_csv("02_output_model_experiments/flores200_token_counts_langinfo.csv")

N_langs = len(count_df["iso_script"].unique().tolist())

# Get list of models
model_list = count_df["model"].unique().tolist()

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


# %% Calculate equity metrics
gini_results = []
ratio_results = []

for model in model_list:    
    calc_data = count_df.loc[count_df["model"] == model,]

    for N in np.arange(2, N_langs + 1, step = 1):
        token_count_vector = calc_data.loc[calc_data["cc_spk_rank"] <= N, "token_count"]

        # Gini coefficient
        g = gini(token_count_vector)
        gini_result_i = {"model": model, "N": N, "gini": g}
        gini_results.append(gini_result_i)

        # Max/min ratio
        max_min_ratio = token_count_vector.max() / token_count_vector.min()
        ratio_result_i = {"model": model, "N": N, "max_min_ratio": max_min_ratio}
        ratio_results.append(ratio_result_i)

gini_df = pd.DataFrame(data = gini_results)
ratio_df = pd.DataFrame(data = ratio_results)


# %% Reshape to Wide Format for Results Table
out_N_list = [5, 10, 20, 50, 100, 150, 200, N_langs]

gini_df_long = gini_df.loc[gini_df["N"].isin(out_N_list),]
gini_df_long["gini"] = gini_df_long["gini"].round(2)

ratio_df_long = ratio_df.loc[ratio_df["N"].isin(out_N_list),]
ratio_df_long["max_min_ratio"] = ratio_df_long["max_min_ratio"].round(1)

gini_df_wide = gini_df_long.pivot(index="model", columns="N", values="gini")
gini_df_wide.columns = ["gini_" + str(N) for N in out_N_list]
gini_df_wide = gini_df_wide.reset_index(drop = False)
gini_df_wide = gini_df_wide.sort_values(by = "gini_200", ascending = True, ignore_index = True)

ratio_df_wide = ratio_df_long.pivot(index="model", columns="N", values="max_min_ratio")
ratio_df_wide.columns = ["max_min_ratio_" + str(N) for N in out_N_list]
ratio_df_wide = ratio_df_wide.reset_index(drop = False)
ratio_df_wide = ratio_df_wide.sort_values(by = "max_min_ratio_200", ascending = True, ignore_index = True)


# %% Write to CSV Files
gini_df.to_csv("03_output_analysis/gini_flores200.csv", index = False)
ratio_df.to_csv("03_output_analysis/ratio_flores200.csv", index = False)

gini_df_wide.to_csv("03_output_analysis/gini_for_table_flores200.csv", index = False)
ratio_df_wide.to_csv("03_output_analysis/ratio_for_table_flores200.csv", index = False)

# TODO: move gini() function to new src/util/equity_metrics file/folder



