
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
gini_df = pd.read_csv("03_output_analysis/gini_flores200.csv")

# Get list of models
model_list = gini_df["model"].unique().tolist()

# Get language info
flores200_langinfo = pd.read_table("01_data_processed/flores200_langinfo.tsv")


# %% Summarizing gini as function of N curves in a metric
rank_var_final = "cc_spk_rank"
gini_df_fin = gini_df.loc[gini_df["rank_var"] == rank_var_final,]

metric_results = []
for model in model_list:   
    calc_df = gini_df_fin.loc[gini_df_fin["model"] == model,]
    calc_df = calc_df.sort_values(by = "N", ascending = True, ignore_index = True)
    
    gini_vector = calc_df["gini"]
    
    gini_diff = gini_vector.diff()
    gini_diff_abs = gini_diff.abs()
    
    metric_min = gini_diff.min()
    assert metric_min < 0
    langrank_min = gini_diff.idxmin() + 2
    
    metric_max = gini_diff.max()
    langrank_max = gini_diff.idxmax() + 2

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
                       "metric_min": metric_min, "langrank_min": langrank_min,
                       "metric_max": metric_max, "langrank_max": langrank_max,
                       "metric_abs_max": metric_abs_max, "langrank_abs_max": langrank_abs_max}
    metric_results.append(metric_result_i)

metric_df = pd.DataFrame(data = metric_results)

langrank = flores200_langinfo[["cc_spk_rank", "name", "iso_script"]]


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


langrank_abs_max = langrank.rename(columns = {"cc_spk_rank": "langrank_abs_max",
                                              "name": "name_abs_max",
                                              "iso_script": "iso_script_abs_max"})
assert langrank_abs_max["langrank_abs_max"].is_unique
metric_df = pd.merge(left = metric_df, right = langrank_abs_max, how = "left", on = "langrank_abs_max")

# Finalize output
metric_df = metric_df[["model",
                       "metric_min", "langrank_min", "name_min", "iso_script_min",
                       "metric_max", "langrank_max", "name_max", "iso_script_max",
                       "metric_abs_max", "langrank_abs_max", "name_abs_max", "iso_script_abs_max",
                       ]]

metric_df.to_csv("03_output_analysis/gini_curve_summary_metric.csv", index = False)



# %% Add models plotted together
models_to_include = ["facebook/nllb-200-3.3B",
                     "gemini-3.5-flash",
                     "google/gemma-4-E4B-it",
                     "moonshotai/Kimi-K3",
                     "claude-fable-5",
                     "meta-llama/Llama-4-Scout-17B-16E-Instruct",
                     "gpt-5",
                     "openai/gpt-oss-20b",
                     "Qwen/Qwen3.6-35B-A3B",
                     "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
                     "mistralai/Ministral-3-8B-Instruct-2512",
                     "CohereLabs/command-a-plus-05-2026-bf16",
                     "zai-org/GLM-5.2",
                     "microsoft/phi-4",
                     "allenai/Olmo-3.1-32B-Instruct",
                     "ibm-granite/granite-4.1-8b",
                     "tiiuae/Falcon3-7B-Instruct"
                     ]


gini_df_fin = gini_df_fin.loc[gini_df_fin["model"].isin(models_to_include),]
    # drop some models to avoid overplotting where multiple models from same company/family are included

chart_model_list = gini_df_fin["model"].unique().tolist()

label_offset = {"facebook/nllb-200-3.3B": 0,
                "gemini-3.5-flash": -0.002, 
                "google/gemma-4-E4B-it": 0.002,
                "moonshotai/Kimi-K3": 0,
                "claude-fable-5": 0.001,
                "meta-llama/Llama-4-Scout-17B-16E-Instruct": 0,
                "gpt-5": -0.002,
                "openai/gpt-oss-20b": 0.002,
                "Qwen/Qwen3.6-35B-A3B": 0,
                "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": -0.002,
                "mistralai/Ministral-3-8B-Instruct-2512": -0.006,
                "CohereLabs/command-a-plus-05-2026-bf16": 0.005,
                "zai-org/GLM-5.2": 0,
                "microsoft/phi-4": -0.007,
                "allenai/Olmo-3.1-32B-Instruct": -0.002,                
                "ibm-granite/granite-4.1-8b": 0.003,
                "tiiuae/Falcon3-7B-Instruct": 0,
                }
    # creating offsets to avoid overlapping labels

with PdfPages("03_output_analysis/Token Gini Charts - FLORES200 - All Models.pdf") as pdf:
    fig, ax = plt.subplots(figsize=(10, 10))
    for i in range(len(chart_model_list)):
        model = chart_model_list[i]
        print(f"Working on {model}")

        chart_data = gini_df_fin.loc[gini_df_fin["model"] == model,]
        chart_data = chart_data.loc[chart_data["N"] >= 5,]
        chart_data = chart_data.sort_values(by = "N", ascending = True, ignore_index = True)

        gini200 = chart_data.loc[chart_data["N"] == 200, "gini"]

        line, = ax.plot(chart_data["N"], chart_data["gini"])

        y_last = chart_data["gini"].iloc[-1] + label_offset[model]
        #print(y_last)

        ax.text(200, y_last, 
                f"  {model}",
                va="center", ha="left",
                fontsize=8
        )


    ax.set_title("Gini Coefficient as a Function of Languages Included\nLanguages Ranked from High to Low Resource")
    ax.set_xlabel("Number of Languages (N)")
    ax.set_ylabel("Gini Coefficient")

    ax.text(0.5, -0.08, "Basis for Resourceness Ranking: 1) Common Crawl Page Count, 2) Speaker Count, 3) Script Prevalence",
            transform=ax.transAxes, ha='center', fontsize=9, style='italic')

    ax.set_xlim(0,200)
    ax.set_ylim(0, 0.4)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.tight_layout()
    pdf.savefig()
    plt.close()


