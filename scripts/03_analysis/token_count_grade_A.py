
# %% Setup
# Python standard library
import os

# Third-party libraries
from dotenv import load_dotenv
#import numpy as np
import pandas as pd


# %% Project folder
load_dotenv()
PROJECT_ROOT = os.getenv("PROJECT_ROOT")
if not PROJECT_ROOT:
    raise EnvironmentError("PROJECT_ROOT is not set. Check your .env file.")
os.chdir(PROJECT_ROOT)


# %% Count language-script pairs where max/min ratio <= grade A threshold
count_df = pd.read_csv("02_output_model_experiments/flores200_token_counts_langinfo.csv")

N_langs = len(count_df["iso_script"].unique().tolist())
    # count all language-script pairs

# For each language-script pair, compute token ratio
model_min = count_df.groupby("model").agg(token_count_min = ("token_count", "min"))
model_min = model_min.reset_index(drop = False)
count_df = pd.merge(left = count_df, right = model_min, how = "outer", on = "model", indicator = True)
    # merge to add token_count_min
assert all( count_df["_merge"] == "both" )
count_df = count_df.drop(columns = "_merge")
count_df["token_ratio"] = count_df["token_count"] / count_df["token_count_min"]

# Divide grade A (max/min ratio in [1,3]) into subintervals of size 0.25
count_df["interval"] = ""
count_df.loc[(count_df["token_ratio"] >= 1.0) & (count_df["token_ratio"] <= 1.25), "interval"] = "[1.00, 1.25]"
count_df.loc[(count_df["token_ratio"] > 1.25) & (count_df["token_ratio"] <= 1.50), "interval"] = "(1.25, 1.50]"
count_df.loc[(count_df["token_ratio"] > 1.50) & (count_df["token_ratio"] <= 1.75), "interval"] = "(1.50, 1.75]"
count_df.loc[(count_df["token_ratio"] > 1.75) & (count_df["token_ratio"] <= 2.00), "interval"] = "(1.75, 2.00]"
count_df.loc[(count_df["token_ratio"] > 2.00) & (count_df["token_ratio"] <= 2.25), "interval"] = "(2.00, 2.25]"
count_df.loc[(count_df["token_ratio"] > 2.25) & (count_df["token_ratio"] <= 2.50), "interval"] = "(2.25, 2.50]"
count_df.loc[(count_df["token_ratio"] > 2.50) & (count_df["token_ratio"] <= 2.75), "interval"] = "(2.50, 2.75]"
count_df.loc[(count_df["token_ratio"] > 2.75) & (count_df["token_ratio"] <= 3.00), "interval"] = "(2.75, 3.00]"
count_df.loc[count_df["token_ratio"] > 3.00, "interval"] = "(3.00, inf)"

assert count_df["interval"].isna().sum() == 0

# Count languages in each interval by model
grade_A_subinterval = count_df.groupby(["model", "interval"]).agg(lang_count = ("iso_script", "count"))
grade_A_subinterval = grade_A_subinterval.reset_index(drop = False)

# Export data
grade_A_subinterval.to_csv("03_output_analysis/grade_A_subintervals.csv", index = False)


# %% Count languages below certain thresholds
sub2 = count_df.loc[count_df["token_ratio"] < 2]    
sub2 = sub2.groupby("model").agg(count_sub2 = ("iso_script", "count"))
sub2 = sub2.reset_index(drop = False)
sub2["pct_sub2"] = sub2["count_sub2"] / N_langs
sub2 = sub2.sort_values(by = "pct_sub2", ascending = False, ignore_index = True)
sub2["pct_sub2"] = sub2["pct_sub2"].round(4)

sub3 = count_df.loc[count_df["token_ratio"] < 3]    
sub3 = sub3.groupby("model").agg(count_sub3 = ("iso_script", "count"))
sub3 = sub3.reset_index(drop = False)
sub3["pct_sub3"] = sub3["count_sub3"] / N_langs
sub3 = sub3.sort_values(by = "pct_sub3", ascending = False, ignore_index = True)
sub3["pct_sub3"] = sub3["pct_sub3"].round(4)

sub23 = pd.merge(left = sub2, right = sub3, how = "outer", on = "model", indicator = True)
assert all( sub23["_merge"] == "both" )
sub23 = sub23.drop(columns = "_merge")


# %% Find N where max/min ratio curve exceeds equity margin (uses ranking)
ratio_df = pd.read_csv("03_output_analysis/ratio_flores200.csv")
flores200_langinfo = pd.read_table("01_data_processed/flores200_langinfo.tsv", keep_default_na = False, na_values = [""])
flores200_langinfo = flores200_langinfo[["iso_script", "name", "cc_spk_rank"]]

lastrank_A = ratio_df.loc[ratio_df["max_min_ratio"] <= 3,]
lastrank_A = lastrank_A.groupby("model").agg(lastrank_grade_A = ("N", "max"))
lastrank_A = lastrank_A.reset_index(drop = False)

lastrank_A["lastrank_grade_A_pct"] = lastrank_A["lastrank_grade_A"] / N_langs   
lastrank_A = lastrank_A.sort_values(by = "lastrank_grade_A_pct", ascending = False, ignore_index = True)
lastrank_A["lastrank_grade_A_pct"] = lastrank_A["lastrank_grade_A_pct"].round(4)

lastrank_A["firstrank_grade_B"] = lastrank_A["lastrank_grade_A"] + 1
lastrank_A.loc[lastrank_A["firstrank_grade_B"] > N_langs, "firstrank_grade_B"] = pd.NA

flores200_langinfo = flores200_langinfo.rename(columns = {"iso_script": "lastrank_grade_A_iso_script", 
                                                          "name": "lastrank_grade_A_name",
                                                          "cc_spk_rank": "lastrank_grade_A"})
lastrank_A = pd.merge(left = lastrank_A, right = flores200_langinfo, how = "left", on = "lastrank_grade_A", indicator = True)
assert all( lastrank_A["_merge"].isin(["both", "left_only"]) )
lastrank_A = lastrank_A.drop(columns = "_merge")
    # merge to add lastrank_grade_A_iso_script and lastrank_grade_A_name

flores200_langinfo = flores200_langinfo.rename(columns = {"lastrank_grade_A_iso_script": "firstrank_grade_B_iso_script", 
                                                          "lastrank_grade_A_name": "firstrank_grade_B_name",
                                                          "lastrank_grade_A": "firstrank_grade_B"})
lastrank_A = pd.merge(left = lastrank_A, right = flores200_langinfo, how = "left", on = "firstrank_grade_B", indicator = True)
assert all( lastrank_A["_merge"].isin(["both", "left_only"]) )
lastrank_A = lastrank_A.drop(columns = "_merge")
    # merge to add firstrank_grade_B_iso_script and firstrank_grade_B_name

# Combine DataFrames
grade_A = pd.merge(left = sub23, right = lastrank_A, how = "outer", on = "model", indicator = True)
assert all( grade_A["_merge"].isin(["both", "left_only"]) )
    # some models are not in lastrank_A because max_min_ratio is never below grade A threshold
grade_A = grade_A.drop(columns = "_merge")

# Export data
grade_A.to_csv("03_output_analysis/grade_A.csv", index = False)

