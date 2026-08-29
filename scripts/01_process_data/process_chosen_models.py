
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
chosen_models = pd.read_csv("00_data_raw/model_lists/hugging_face_models.csv")

# Extract model repo ID from Hugging Face link
chosen_models["model_repo_id"] = chosen_models["link"].str.replace("https://huggingface.co/", "")
chosen_models[["organization", "model_name"]] = chosen_models["model_repo_id"].str.split("/" , expand = True)

# Create sortable release_date column
chosen_models["release_date"] = pd.to_datetime(chosen_models["release_date_str"], format = "%B %Y")

# Fill in tokenizer_repo_id column
chosen_models.loc[chosen_models["tokenizer_repo_id"].isna(), "tokenizer_repo_id"] = chosen_models["model_repo_id"]
    # tokenizer_repo_id is blank in hugging_face_models.csv where the model's repo includes the tokenizer
    
# Keep only needed columns
chosen_models = chosen_models[["model_repo_id", "organization", "model_name", "release_date", "tokenizer_repo_id", "tokenizer_file"]]


# %% Hash each model's tokenizer file
results = []
for row_index, row_data in chosen_models.iterrows():
    model_repo_id = row_data["model_repo_id"]
    tokenizer_repo_id = row_data["tokenizer_repo_id"]
    tokenizer_file = row_data["tokenizer_file"]
    
    path = hf_hub_download(repo_id = tokenizer_repo_id, filename = tokenizer_file)
    tokenizer_hash = hashlib.sha256(open(path, "rb").read()).hexdigest()
    result_i = {"model_repo_id": model_repo_id, "tokenizer_hash": tokenizer_hash}

    results.append(result_i)

    tokenizer_hashes = pd.DataFrame(results)


# %% Merge to add tokenizer_hashes to chosen_models DataFrame
assert tokenizer_hashes["model_repo_id"].is_unique
assert chosen_models["model_repo_id"].is_unique

chosen_models = pd.merge(left = chosen_models, right = tokenizer_hashes,
                         how = "outer", on = "model_repo_id", indicator = True)

assert all(chosen_models["_merge"] == "both")
chosen_models = chosen_models.drop(columns = "_merge")


# %% Summary checks
tokenizer_summary = chosen_models.groupby("tokenizer_hash").agg(n_orgs = ("organization", "nunique"),
                                                                n_models = ("model_name", "count"),
                                                                n_dates = ("release_date", "nunique"),
                                                                models = ("model_name", list))
tokenizer_summary = tokenizer_summary.reset_index(drop = False)
assert all(tokenizer_summary["n_orgs"] == 1)
    # among chosen models, tokenizers are never shared by models from different organizations
assert any(tokenizer_summary["n_dates"] > 1)
    # among chosen models, there exist groups of models that have the same tokenizer but different release dates


# %% Only keep one model when multiple models use the same tokenizer
chosen_models = chosen_models.sort_values(by = ["tokenizer_hash", "release_date", "model_repo_id"], ascending = True, ignore_index = True)
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
