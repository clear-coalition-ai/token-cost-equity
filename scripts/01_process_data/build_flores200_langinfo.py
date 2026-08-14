

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


# %% Read FLORES-200 data
flores200_dev = pd.read_table("01_data_processed/flores200_dev.tsv")
flores200_dev[["iso_script", "filetype"]] = flores200_dev["file"].str.split("." , expand = True)
flores200_dev[["iso_639_3", "script"]] = flores200_dev["iso_script"].str.split("_", expand = True)

flores200_langs = flores200_dev[["file", "iso_script", "iso_639_3", "script"]]
assert flores200_langs["iso_script"].is_unique


# %% Read macrolanguage mappings
macro = pd.read_csv("01_data_processed/macrolanguage_mappings.csv")
assert macro["iso_639_3_indiv"].is_unique

macro_isos = macro["iso_639_3_macro"]
macro_isos = macro_isos.drop_duplicates()
macro_isos.name = "iso_639_3"
    # make Series of all macrolanguage ISO 639-3 codes



# %% Read Common Crawl data
cc = pd.read_csv("01_data_processed/commoncrawl_clean.csv")
assert cc["iso_639_3"].is_unique


# %% Read LinguaMeta data
lm = pd.read_table("01_data_processed/linguameta_clean.tsv")
assert lm["iso_639_3"].is_unique


# %% Distinguish between individual language and macrolanguages ISO 639-3 codes in FLORES-200
flores200_isos = flores200_langs["iso_639_3"].drop_duplicates()
    # make Series of all ISO 639_3 codes in FLORES-200

flores200_isos = pd.merge(left = flores200_isos, right = macro_isos, how = "left", on = "iso_639_3", indicator = True)
assert all( flores200_isos["_merge"].isin(["both", "left_only"]) )
flores200_isos["iso_type"] = flores200_isos["_merge"].map({"both": "macro", "left_only": "indiv"})
flores200_isos = flores200_isos.drop(columns = "_merge")
    # merge to detect macrolanguages in FLORES-200

flores200_macro = flores200_isos.loc[flores200_isos["iso_type"] == "macro", "iso_639_3"]
flores200_indiv = flores200_isos.loc[flores200_isos["iso_type"] == "indiv", "iso_639_3"]
    # split up FLORES-200 ISO 639-3 codes into two DataFrames based on type


# %% Distinguish individual language and macrolanguages ISO 639-3 codes in Common Crawl
cc = pd.merge(left = cc, right = macro_isos, how = "left", on = "iso_639_3", indicator = True)
assert all( cc["_merge"].isin(["both", "left_only"]) )
cc["iso_type"] = cc["_merge"].map({"both": "macro", "left_only": "indiv"})
cc = cc.drop(columns = "_merge")
    # merge to detect macrolanguages in Common Crawl

cc_macro = cc.loc[cc["iso_type"] == "macro", ["iso_639_3", "cc_pages"]]
cc_indiv = cc.loc[cc["iso_type"] == "indiv", ["iso_639_3", "cc_pages"]]
    # split up Common Crawl ISO 639-3 codes into two DataFrames based on type


# %% Search for matches for FLORES-200 macrolanguages in Common Crawl 
flores200_macro = pd.merge(left = flores200_macro, right = cc_macro, how = "left", on = "iso_639_3", indicator = True)
    # merge to add cc_pages

assert all( flores200_macro["_merge"].isin(["both", "left_only"]) )
flores200_macro["match_result"] = flores200_macro["_merge"].map({"both": "matched", "left_only": "unmatched"})
flores200_macro = flores200_macro.drop(columns = "_merge")

flores200_macro_matched = flores200_macro.loc[flores200_macro["match_result"] == "matched", "iso_639_3"]
flores200_macro_matched.name = "iso_639_3_macro"
    # make Series of macrolanguage ISO 639-3 codes found in both

flores200_macro_unmatched = flores200_macro.loc[flores200_macro["match_result"] == "unmatched", "iso_639_3"]
    # make Series of macrolanguage ISO 639-3 codes not found in Common Crawl

macro_m = macro.rename(columns = {"iso_639_3_macro": "iso_639_3"})
assert macro_m["iso_639_3"].is_unique == False
    # merge will increase number of rows in DataFrame returned by merge
flores200_macro_unmatched = pd.merge(left = flores200_macro_unmatched, right = macro_m, how = "left", on = "iso_639_3")
flores200_macro_unmatched = flores200_macro_unmatched.rename(columns = {"iso_639_3": "iso_639_3_macro", "iso_639_3_indiv": "iso_639_3"})
    # get individual languages ISO 639-3 codes corresponding to unmatched macrolanguage ISO 639-3 codes

flores200_macro_unmatched = pd.merge(left = flores200_macro_unmatched, right = cc_indiv, how = "left", on = "iso_639_3", indicator = True)
assert all( flores200_macro_unmatched["_merge"] == "left_only" )
    # confirming that no matches in Common Crawl were found

del flores200_macro_unmatched
flores200_macro = flores200_macro[["iso_639_3", "cc_pages"]]
    # use flores200_macro, nothing to incorporate from second attempt at matching


# %% Search for matches for FLORES-200 individual languages in Common Crawl
flores200_indiv = pd.merge(left = flores200_indiv, right = cc_indiv, how = "left", on = "iso_639_3", indicator = True)
    # merge to add cc_pages

assert all( flores200_indiv["_merge"].isin(["both", "left_only"]) )
flores200_indiv["match_result"] = flores200_indiv["_merge"].map({"both": "matched", "left_only": "unmatched"})
flores200_indiv = flores200_indiv.drop(columns = "_merge")
flores200_indiv_unmatched = flores200_indiv.loc[flores200_indiv["match_result"] == "unmatched", "iso_639_3"]
flores200_indiv = flores200_indiv.drop(columns = "match_result")
    # make Series of individual language ISO 639-3 codes not found in Common Crawl

macro_i = macro.rename(columns = {"iso_639_3_indiv": "iso_639_3"})
flores200_indiv_unmatched = pd.merge(left = flores200_indiv_unmatched, right = macro_i, how = "left", on = "iso_639_3")
    # merge to add iso_639_3_macro
flores200_indiv_unmatched = pd.merge(left = flores200_indiv_unmatched, right = flores200_macro_matched, how = "left", on = "iso_639_3_macro", indicator = True)
    # merge to find already matched macrolanguages
flores200_indiv_unmatched.loc[flores200_indiv_unmatched["_merge"] == "both", "iso_639_3_macro"] = None
    # don't match on macrolanguage ISO 639-3 for individual languages that were matched macro-to-macro (e.g. yue is part of macrolanguage zho)
flores200_indiv_unmatched = flores200_indiv_unmatched.drop(columns = "_merge")
    
flores200_indiv_unmatched = pd.merge(left = flores200_indiv_unmatched, right = lm, how = "left", on = "iso_639_3")
    # merge to add speakers
assert all( flores200_indiv_unmatched["speakers"].isna() == False )
flores200_indiv_unmatched = flores200_indiv_unmatched.sort_values(by = ["iso_639_3_macro", "speakers"], ascending = [True, False], ignore_index = True)
    # within each macrolanguage, sort languages by number of speakers (largest to smallest)
im_map = flores200_indiv_unmatched.loc[flores200_indiv_unmatched["iso_639_3_macro"].isna() == False,]
im_map = im_map.groupby(["iso_639_3_macro"]).agg(iso_639_3 = ("iso_639_3", "first"))
im_map = im_map.reset_index(drop = False)
    # for each unmatched individual languages in FLORES-200 that is part of a macrolanguage,
    #   match the largest individual ISO 639-3 code (by speaker count) with the Common Crawl macrolanguage 

flores200_indiv = pd.merge(left = flores200_indiv, right = im_map, how = "left", on = "iso_639_3")
    # merge to add iso_639_3_macro

cc_macro = cc_macro.rename(columns = {"iso_639_3": "iso_639_3_macro",
                                      "cc_pages": "cc_pages_macro"})
flores200_indiv = pd.merge(left = flores200_indiv, right = cc_macro, how = "left", on = "iso_639_3_macro")
    # merge to add cc_pages_macro

flores200_indiv.loc[flores200_indiv["cc_pages"].isna() == True, "cc_pages"] = flores200_indiv["cc_pages_macro"]
    # replace cc_pages with cc_pages_macro where cc_pages is missing
flores200_indiv = flores200_indiv[["iso_639_3", "cc_pages"]]


# %% Combine split sets of FLORES-200 languages
flores200_isos_cc = pd.concat(objs = [flores200_macro, flores200_indiv], ignore_index = True)
assert flores200_isos_cc["iso_639_3"].is_unique


# %% Add Common Crawl pages
flores200_langinfo = pd.merge(left = flores200_langs, right = flores200_isos_cc, how = "left", on = "iso_639_3", indicator = True)
    # merge to add cc_pages
assert all( flores200_langinfo["_merge"] == "both" )
flores200_langinfo = flores200_langinfo.drop(columns = "_merge")


# %% Add speaker counts from linguameta
flores200_langinfo = pd.merge(left = flores200_langinfo, right = lm, how = "left", on = "iso_639_3", indicator = True)
    # merge to add name and speaker
assert all( flores200_langinfo["_merge"] == "both" )
flores200_langinfo = flores200_langinfo.drop(columns = "_merge")


# %% Add script prevalence from own research
assert flores200_langinfo["iso_script"].is_unique
script_count = flores200_langinfo.groupby(["iso_639_3"]).size()
script_count.name = "script_count"
script_count = script_count.reset_index(drop = False)

flores200_langinfo = pd.merge(left = flores200_langinfo, right = script_count, how = "left", on = "iso_639_3")
    # merge to add script_count
multiscript = flores200_langinfo.loc[flores200_langinfo["script_count"] > 1,]
assert all( multiscript["script_count"] == 2 )

script_prevalence_dict = {"ace_Arab": 2, # Source: https://en.wikipedia.org/wiki/Acehnese_language
                          "ace_Latn": 1,
                          "arb_Arab": 1, # Source: https://en.wikipedia.org/wiki/Modern_Standard_Arabic
                          "arb_Latn": 2,
                          "bjn_Arab": 2, # Source: https://en.wikipedia.org/wiki/Banjarese_language
                          "bjn_Latn": 1,
                          "kas_Arab": 1, # Source: https://en.wikipedia.org/wiki/Kashmiri_language
                          "kas_Deva": 2,
                          "knc_Arab": 2, # Source: https://languagesgulper.com/eng/Kanuri.html
                          "knc_Latn": 1,
                          "min_Arab": 2, # Source: https://en.wikipedia.org/wiki/Minangkabau_language
                          "min_Latn": 1,
                          "taq_Latn": 1, # Source: https://www.omniglot.com/writing/tamasheq.htm
                          "taq_Tfng": 2,
                          "zho_Hans": 1, # Source: https://en.wikipedia.org/wiki/Chinese_language
                          "zho_Hant": 2,
                          }
    # 1 is higher prevalence, 2 is lower prevalence

multiscript["script_prevalence"] = multiscript["iso_script"].map(script_prevalence_dict)
multiscript = multiscript[["iso_script", "script_prevalence"]]

flores200_langinfo = pd.merge(left = flores200_langinfo, right = multiscript, how = "left", on = "iso_script")
    # merge to add script_prevalence

# %% Rank FLORES-200 language-script pairs by...
    # 1. Common Crawl pages
    # 2. Speaker count (breaks ties for languages not found in Common Crawl data)
    # 3. Script prevalence (breaks ties for languages with multiple scripts)

flores200_langinfo = flores200_langinfo.sort_values(by = ["cc_pages", "speakers", "script_prevalence"], ascending = [False, False, True], ignore_index = True)
flores200_langinfo["cc_spk_rank"] = flores200_langinfo.index + 1


# %% Write data to file
flores200_langinfo = flores200_langinfo[["file", "iso_script", "iso_639_3", "script", "name",
                                         "cc_pages", "speakers", "script_prevalence", "cc_spk_rank"]]

flores200_langinfo.to_csv("01_data_processed/flores200_langinfo.tsv", sep = "\t", index = False)





