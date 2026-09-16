
# %% Setup
# Python standard library
import os

# Third-party libraries
import anthropic
from dotenv import load_dotenv
import google.genai
import openai
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

openai_model_list = ["gpt-5", "gpt-6-astra"]
anthropic_model_list = ["claude-opus-4-8", "claude-fable-5"]  
google_model_list = ["gemini-3.1-pro-preview", "gemini-3.8-flash"]
    

# %% Read FLORES-200 dev data
flores200_dev = pd.read_table("01_data_processed/flores200_dev.tsv")


# %% Check for existing token counts file - avoids full rerun when adding new models
if os.path.exists("02_output_model_experiments/flores200_token_counts.csv"):
    old_results_df = pd.read_csv("02_output_model_experiments/flores200_token_counts.csv")
    model_list = old_results_df["model"].unique().tolist()
        # get models for which token counts have already been collected

    models_to_drop = list(set(model_list) - set(hf_model_list + openai_model_list + anthropic_model_list + google_model_list))
        # find eliminated models
    if len(models_to_drop) > 0:
        old_results_df = old_results_df.loc[old_results_df["model"].isin(models_to_drop) == False,]
            # drop eliminated models from data

    hf_model_list = list(set(hf_model_list) - set(model_list))
    openai_model_list = list(set(openai_model_list) - set(model_list))
    anthropic_model_list = list(set(anthropic_model_list) - set(model_list))
    google_model_list = list(set(google_model_list) - set(model_list))
        # keep newly added models
else:
    old_results_df = pd.DataFrame()


# %% Loop through each model and get token counts
results = []

# Hugging Face open-source models
for model in hf_model_list:
    print(f"Working on model: {model}")
    if model in hf_remote_code_list:
        tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code = True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model)

    for row_index, row_data in flores200_dev.iterrows():
        file = row_data["file"]
        text = row_data["text"]
                
        # Tokenize data
        tokens = tokenizer.encode(text)
        token_count = len(tokens)
        
        # Append results for this iteration in results list
        result_i = {"model": model, "file": file, "token_count": token_count}
        results.append(result_i)

# OpenAI closed-source models
#for model in openai_model_list:
#    print(f"Working on model: {model}") 
#    tokenizer = tiktoken.encoding_for_model(model)
#
#    for row_index, row_data in flores200_dev.iterrows():
#        file = row_data["file"]
#        text = row_data["text"]
#                
#        # Tokenize data
#        tokens = tokenizer.encode(text)
#        token_count = len(tokens)
#        
#        # Append results for this iteration in results list
#        result_i = {"model": model, "file": file, "token_count": token_count}
#        results.append(result_i)


client = openai.OpenAI()
for model in openai_model_list:
    print(f"Working on model: {model}") 

    for row_index, row_data in flores200_dev.iterrows():
        file = row_data["file"]
        text = row_data["text"]
                
        # Tokenize data
        response = client.responses.input_tokens.count(model = model, input = text)
        token_count = response.input_tokens
        
        # Append results for this iteration in results list
        result_i = {"model": model, "file": file, "token_count": token_count}
        results.append(result_i)


# Anthropic closed-source models
client = anthropic.Anthropic()
for model in anthropic_model_list:
    print(f"Working on model: {model}") 

    for row_index, row_data in flores200_dev.iterrows():
        file = row_data["file"]
        text = row_data["text"]
                
        # Tokenize data
        response = client.messages.count_tokens(model = model, messages = [{"role": "user", "content": text}])
        token_count = response.input_tokens
        
        # Append results for this iteration in results list
        result_i = {"model": model, "file": file, "token_count": token_count}
        results.append(result_i)

# Google closed-source models
client = google.genai.Client()
for model in google_model_list:
    print(f"Working on model: {model}") 

    for row_index, row_data in flores200_dev.iterrows():
        file = row_data["file"]
        text = row_data["text"]
                
        # Tokenize data
        response = client.models.count_tokens(model = model, contents = text)
        token_count = response.total_tokens
        
        # Append results for this iteration in results list
        result_i = {"model": model, "file": file, "token_count": token_count}
        results.append(result_i)

# Create results DataFrame
new_results_df = pd.DataFrame(data = results)
flores200_token_counts = pd.concat(objs = [old_results_df, new_results_df], ignore_index = True)


# %% Write results to CSV file
flores200_token_counts.to_csv("02_output_model_experiments/flores200_token_counts.csv", index = False)


# %% Add language info
flores200_langinfo = pd.read_table("01_data_processed/flores200_langinfo.tsv", keep_default_na = False, na_values = [""])

flores200_token_counts_langinfo = pd.merge(left = flores200_token_counts, right = flores200_langinfo, how = "outer", on = "file", indicator = True)

assert all(flores200_token_counts_langinfo["_merge"] == "both")
    # confirming merge resulted in full match
flores200_token_counts_langinfo = flores200_token_counts_langinfo.drop(columns = "_merge")

assert flores200_token_counts_langinfo["speakers"].isna().sum() == 0
    # no more missing values in "speakers" column


# %% Write results to CSV file
assert all(flores200_token_counts_langinfo.dtypes.astype(str).isin(["str", "int64", "float64"]))
    # confirming all columns are either of type str, int64, or float64
for c in flores200_token_counts_langinfo.columns:
    if flores200_token_counts_langinfo[c].dtype == "str":
        assert flores200_token_counts_langinfo[c].str.contains(r",").sum() == 0
    # confirming there are no commas in string columns - safe to save as CSV

flores200_token_counts_langinfo.to_csv("02_output_model_experiments/flores200_token_counts_langinfo.csv", index = False)


# TODO: try roundtrip tokenization to make sure the text is tokenized correctly (for deepseek llama issue, UNK tokens)
# TODO: pin version for Hugging Face models with trust_remote_code = True
