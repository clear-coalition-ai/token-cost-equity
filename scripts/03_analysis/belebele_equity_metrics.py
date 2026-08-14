
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
results_df = pd.read_csv("02_output_model_experiments/belebele_results.csv")

# Get list of models
model_list = results_df["model"].unique().tolist()

# Get list of languages
lang_list = results_df["iso_639_3"].unique().tolist()
assert len(lang_list) == 115


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
gini_N_list = np.arange(2, 115 + 1, step = 1)
    # +1 to include all languages

gini_results = []

for model in model_list:
    print(f"Working on {model}")
        
    score_vector = results_df.loc[results_df["model"] == model, "belebele"]
    
    g = gini(score_vector).round(4)
    mean_score = score_vector.mean().round(4)
    min_score = score_vector.min().round(4)
    max_score = score_vector.max().round(4)
    gini_result_i = {"model": model, "gini_115": g, "mean_score": mean_score, "min_score": min_score, "max_score": max_score}
    gini_results.append(gini_result_i)

gini_df = pd.DataFrame(data = gini_results)
gini_df = gini_df.sort_values(by = "gini_115", ascending = True, ignore_index = True)


# %% Write Gini Results to CSV Files
gini_df.to_csv("03_output_analysis/gini_belebele.csv", index = False)


# TODO: add ranking