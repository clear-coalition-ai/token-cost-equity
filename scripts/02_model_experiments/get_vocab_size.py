
# %% Setup
# Python standard library
import os

# Third-party libraries
#import anthropic
from dotenv import load_dotenv
#import google.genai
import pandas as pd
#import tiktoken
from transformers import AutoTokenizer


# %% Project folder
load_dotenv()
PROJECT_ROOT = os.getenv("PROJECT_ROOT")
if not PROJECT_ROOT:
    raise EnvironmentError("PROJECT_ROOT is not set. Check your .env file.")
os.chdir(PROJECT_ROOT)


# %% Lists of tokenizers to use
hf_models = pd.read_table("01_data_processed/hf_models_by_tokenizer.tsv")
hf_model_list = hf_models["tokenizer_repo_id"].tolist()

hf_remote_code_list = ["moonshotai/Kimi-K3"]
    # requires `trust_remote_code = True` to run

results = []
for model in hf_model_list:
    print(f"Working on model: {model}")
    if model in hf_remote_code_list:
        tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code = True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model)
    #tokenizer = AutoTokenizer.from_pretrained(model)

    vocab_size1 = tokenizer.vocab_size
    vocab_size2 = len(tokenizer)
  
    # Append results for this iteration in results list
    result_i = {"model": model, "vocab_size1": vocab_size1, "vocab_size2": vocab_size2}
    results.append(result_i)


vocab_size_df = pd.DataFrame(data = results)

hf_models = hf_models.rename(columns = {"tokenizer_repo_id": "model"})

vocab_size_df = pd.merge(left = vocab_size_df, right = hf_models, how = "outer", on = "model", indicator = True)
assert all( vocab_size_df["_merge"] == "both" )
vocab_size_df = vocab_size_df.drop(columns = "_merge")

vocab_size_df.to_csv('02_output_model_experiments/vocab_size.tsv', sep = "\t", index = False)

