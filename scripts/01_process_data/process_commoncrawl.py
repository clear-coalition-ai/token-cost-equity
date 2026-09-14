
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


# %% Read data
cc = pd.read_csv("00_data_raw/commoncrawl/languages.csv", keep_default_na = False, na_values = [""])


# %% Clean data
crawl_to_use = "CC-MAIN-2026-25"
    # most recent crawl at time of analysis

cc_last = cc.loc[cc["crawl"] == crawl_to_use,]
    # filter data to one crawl

cc_last = cc_last.loc[cc_last["primary_language"] != "<unknown>",]
    # exclude data where language is <unknown>

cc_last = cc_last.sort_values(by = "pages", ascending = False, ignore_index = True)
cc_last["cc_rank"] = cc_last.index + 1
    # rank languages by number of pages crawled, ranking starts at 1

cc_last = cc_last.rename(columns = {"primary_language": "iso_639_3", "pages": "cc_pages"})
cc_last = cc_last[["iso_639_3", "cc_pages", "cc_rank"]]


# %% Write data
cc_last.to_csv("01_data_processed/commoncrawl_clean.csv", index = False)
