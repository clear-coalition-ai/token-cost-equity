
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


# %% Combine all score files into a DataFrame
os.chdir(os.path.join(PROJECT_ROOT, "00_data_raw/African-Languages-Lab/proxy-mt-benchmark-scores/scores"))
file_list = os.listdir() 
    # each file contains data for a different language-script combination

scores_df = pd.DataFrame()
for file in file_list:
    data = pd.read_csv(file)
    data["model"] = file.replace(".csv", "")
    scores_df = pd.concat(objs = [scores_df, data])


# %% Cleaning and checks
scores_df = scores_df[["model", "lang_code", "belebele"]]

scores_df_wide = scores_df.pivot(index = "model", columns = "lang_code", values = "belebele")
N_models = scores_df_wide.shape[0]
assert all( scores_df_wide.isna().sum().isin([0, N_models]) )
    # checking that belebele is missing/non-missing for the same languages across all models

scores_df = scores_df.loc[scores_df["belebele"].isna() == False,]
scores_df = scores_df.rename(columns = {"lang_code": "iso_639_3"})


# %% Write to file
os.chdir(PROJECT_ROOT)
scores_df.to_csv("02_output_model_experiments/belebele_results.csv", index = False)




