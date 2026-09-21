
# %% Setup
# Python standard library
import os

# Third-party libraries
from dotenv import load_dotenv
import pandas as pd


# %% Project folder
load_dotenv()
PROJECT_ROOT = os.getenv("PROJECT_ROOT")
if not PROJECT_ROOT:
    raise EnvironmentError("PROJECT_ROOT is not set. Check your .env file.")
os.chdir(PROJECT_ROOT)


# %% Find top 3 models for each FLORES-200 language
count_df = pd.read_csv("02_output_model_experiments/flores200_token_counts_langinfo.csv")

# For each language-script pair, compute token ratio
model_min = count_df.groupby("model").agg(token_count_min = ("token_count", "min"))
model_min = model_min.reset_index(drop = False)
count_df = pd.merge(left = count_df, right = model_min, how = "outer", on = "model", indicator = True)
    # merge to add token_count_min
assert all( count_df["_merge"] == "both" )
count_df = count_df.drop(columns = "_merge")
count_df["token_ratio"] = count_df["token_count"] / count_df["token_count_min"]

# Exclude select models from analysis
models_to_exclude = ["facebook/nllb-200-3.3B", "facebook/nllb-moe-54b", # excluding translation models
                     "google/gemma-3-4b-it", "google/gemma-4-E4B-it", "gemini-3.1-pro-preview", # include gemini-3.8-flash
                     "claude-opus-4-8", # include claude-fable-5
                     "gpt-5", "openai/gpt-oss-20b", # include gpt-6-astra
                     "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4", # include nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4
                     "meta-llama/Llama-4-Scout-17B-16E-Instruct", # include meta-models/Muse-Glimmer-30B
                     "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", # include Qwen/Qwen3-8B
                     "CohereLabs/aya-expanse-8b", "CohereLabs/c4ai-command-r-08-2024", # include CohereLabs/c4ai-command-a-03-2025
                     "deepseek-ai/DeepSeek-V4-Flash", # include deepseek-ai/DeepSeek-V4.1-Flash
                     "meta-llama/Llama-3.2-3B-Instruct", # include meta-llama/Llama-3.3-70B-Instruct
                     "allenai/Olmo-3-1125-32B", # include allenai/Olmo-3.1-32B-Instruct
                     "ibm-granite/granite-4.1-3b", # include ibm-granite/granite-4.1-8b
                     "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B", "deepseek-ai/DeepSeek-R1-Distill-Llama-8B", # excluding models with major tokenizer issues
                    ]

count_df = count_df.loc[count_df["model"].isin(models_to_exclude) == False,]

count_df = count_df.sort_values(by = ["iso_script", "name", "token_ratio", "model"], ascending = True, ignore_index = True)
    # models are sorted by model name where token_ratio is tied (relevant for eng_Latn and zho_Hans)
all_models = count_df[["model", "iso_script", "name", "token_ratio"]]

# Aggregate data by language to get top 3 models
top3_models = count_df.groupby(["iso_script", "name"]).head(3)
top3_models = top3_models.groupby(["iso_script", "name"]).agg(top3 = ("model", ", ".join))
top3_models[["model_rank1", "model_rank2", "model_rank3"]] = top3_models["top3"].str.split(",", expand = True)
top3_models = top3_models.reset_index(drop = False)
top3_models = top3_models[["iso_script", "name", "model_rank1", "model_rank2", "model_rank3"]]

# %% Export data
all_models.to_csv("03_output_analysis/leaderboard_by_language_all_models.csv", index = False)
top3_models.to_csv("03_output_analysis/leaderboard_by_language_top3_models.csv", index = False)
