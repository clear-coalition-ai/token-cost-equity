
# %% Setup
# Python standard library
import os
import hashlib

# Third-party libraries
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
import pandas as pd


# %% Project folder
load_dotenv()
PROJECT_ROOT = os.getenv("PROJECT_ROOT")
if not PROJECT_ROOT:
    raise EnvironmentError("PROJECT_ROOT is not set. Check your .env file.")
os.chdir(PROJECT_ROOT)


# %% Read chosen models table
chosen_models = pd.read_table("00_data_raw/clear_benchmarking_team/LLM CLEAR Sheet - Chosen Models.tsv")

# Rename columns and drop unneeded columns
chosen_models = chosen_models.rename(columns = {"Link": "link", 
                                                "Release Date / Last Update": "release_date_str"})

added_models = [
    {"link": "https://huggingface.co/zai-org/GLM-5.2",                        "release_date_str": "June 2026"},
    {"link": "https://huggingface.co/moonshotai/Kimi-K3",                     "release_date_str": "July 2026"},
    {"link": "https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash",          "release_date_str": "April 2026"},
    {"link": "https://huggingface.co/CohereLabs/c4ai-command-a-03-2025",      "release_date_str": "March 2025"},
    {"link": "https://huggingface.co/CohereLabs/command-a-plus-05-2026-bf16", "release_date_str": "May 2026"},
    {"link": "https://huggingface.co/openai/gpt-oss-20b",                     "release_date_str": "August 2025"},
    {"link": "https://huggingface.co/openai/gpt-oss-120b",                    "release_date_str": "August 2025"},  
    {"link": "https://huggingface.co/facebook/nllb-200-3.3B",                 "release_date_str": "July 2022"},
    {"link": "https://huggingface.co/facebook/nllb-moe-54b",                  "release_date_str": "March 2023"},
    {"link": "https://huggingface.co/meta-models/Muse-Glimmer-30B",           "release_date_str": "August 2026"},
    {"link": "https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4",   "release_date_str": "August 2026"},
    {"link": "https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4",       "release_date_str": "June 2026"},
    {"link": "https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4",       "release_date_str": "March 2026"},
    {"link": "https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4",          "release_date_str": "December 2025"},
    ]

added_models = pd.DataFrame(added_models)

chosen_models = pd.concat([chosen_models, added_models], ignore_index = True)

# Extract model from Hugging Face link
chosen_models["model"] = chosen_models["link"].str.replace("https://huggingface.co/", "")
chosen_models[["organization", "model_name"]] = chosen_models["model"].str.split("/" , expand = True)

# Create sortable release_date column
chosen_models["release_date"] = pd.to_datetime(chosen_models["release_date_str"], format = "mixed")

# Drop unneeded columns
chosen_models = chosen_models[["model", "organization", "model_name", "release_date"]]

# Make tokenizer_repo_id column
llama2_list = ["apple/OpenELM-1_1B-Instruct", "apple/OpenELM-3B-Instruct"]

chosen_models["tokenizer_repo_id"] = chosen_models["model"]
chosen_models.loc[chosen_models["model"].isin(llama2_list), "tokenizer_repo_id"] = "meta-llama/Llama-2-7b-hf"
    # special case: tokenizer file not included in model repo, get tokenizer from other repo

# Make tokenizer_file column
tiktoken_list = ["moonshotai/Kimi-Linear-48B-A3B-Instruct", "moonshotai/Moonlight-16B-A3B-Instruct", "moonshotai/Kimi-K3"]

chosen_models["tokenizer_file"] = "tokenizer.json"
chosen_models.loc[chosen_models["model"].isin(tiktoken_list), "tokenizer_file"] = "tiktoken.model"
    # special case: tokenizer.json not include in model repo, get other file


# %% Hash each model's tokenizer file
results = []
for row_index, row_data in chosen_models.iterrows():
    model = row_data["model"]
    tokenizer_repo_id = row_data["tokenizer_repo_id"]
    tokenizer_file = row_data["tokenizer_file"]
    
    path = hf_hub_download(repo_id = tokenizer_repo_id, filename = tokenizer_file)
    tokenizer_hash = hashlib.sha256(open(path, "rb").read()).hexdigest()
    result_i = {"model": model, "tokenizer_hash": tokenizer_hash}

    results.append(result_i)

    tokenizer_hashes = pd.DataFrame(results)


# %% Merge to add tokenizer_hashes to chosen_models DataFrame
assert tokenizer_hashes["model"].is_unique
assert chosen_models["model"].is_unique

chosen_models = pd.merge(left = chosen_models, right = tokenizer_hashes,
                         how = "outer", on = "model", indicator = True)

assert all(chosen_models["_merge"] == "both")
chosen_models = chosen_models.drop(columns = "_merge")


# %% Summary checks
tokenizer_summary = chosen_models.groupby("tokenizer_hash").agg(n_orgs = ("organization", "nunique"),
                                                                n_models = ("model_name", "count"),
                                                                n_dates = ("release_date", "nunique"),
                                                                models = ("model_name", list))
tokenizer_summary = tokenizer_summary.reset_index(drop = False)
assert all(tokenizer_summary["n_orgs"] == 1)
    # tokenizers are never shared by models from different organizations
assert any(tokenizer_summary["n_dates"] > 1)
    # there exist groups of models that have the same tokenizer but different release dates


# %% Only keep one model when multiple models use the same tokenizer
chosen_models = chosen_models.sort_values(by = ["tokenizer_hash", "release_date", "model"], ascending = True, ignore_index = True)
chosen_models["dup"] = chosen_models["tokenizer_hash"].duplicated(keep = "last")
    # For each tokenizer, keep model with latest release date. Break ties using model (keep last in sort order)

chosen_tokenizers = chosen_models.loc[chosen_models["dup"] == False, ["tokenizer_repo_id", "tokenizer_hash"]]
    # Tokenizers to include in token cost experiments

models_by_tokenizer = chosen_models.groupby("tokenizer_hash").agg(model_list = ("model_name", list))
models_by_tokenizer = models_by_tokenizer.reset_index(drop = False)

tokenizer_groups = pd.merge(left = chosen_tokenizers, right = models_by_tokenizer, how = "outer", on = "tokenizer_hash")
    # each row corresponds to a group of models with the same tokenizer

tokenizer_groups = tokenizer_groups[["tokenizer_repo_id", "model_list"]]


# %% Write to TSV file
assert tokenizer_groups["tokenizer_repo_id"].str.contains("\t").sum() == 0
assert tokenizer_groups["model_list"].str.contains("\t").sum() == 0

tokenizer_groups.to_csv("01_data_processed/hf_models_by_tokenizer.tsv", sep = "\t", index = False)
