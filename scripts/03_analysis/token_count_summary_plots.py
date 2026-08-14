
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
results_df = pd.read_csv("02_output_model_experiments/flores200_token_counts_langinfo.csv")

# Get list of models
model_list = results_df["model"].unique().tolist()


# %% Prepare for plotting
# Show token counts in thousands for ease of reading chart axis labels
results_df["token_count"] = results_df["token_count"] / 1000

# Find languages with multiple scripts
N_model = len(model_list)
script_cnt = results_df.groupby("iso_639_3").size() / N_model
script_cnt = script_cnt.reset_index(drop = False)
script_cnt.columns = ["iso_639_3", "count"]
iso_multiscript_list = script_cnt.loc[script_cnt["count"] > 1, "iso_639_3"].tolist()

# Build dictionary for mapping colors to ISO 639-3 codes
color_list = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:gray", "tab:pink"]
assert len(iso_multiscript_list) == len(color_list) 
N_iso = len(iso_multiscript_list)
iso_color_map = {iso_multiscript_list[i]: color_list[i] for i in range(N_iso)}   


# %% Generate plots for each model
with PdfPages("03_output_analysis/Token Count Charts - FLORES200.pdf") as pdf:
    for model in model_list:
        print(f"Working on report: {model}")
        
        chart_data = results_df.loc[results_df["model"] == model,]
        chart_data = chart_data.sort_values(by = "token_count", ascending = True, ignore_index = True)

        tok_max = chart_data["token_count"].max() + 10
            # +10 for visual padding in chart
            
        axs0_max = 2
        axs1_max = 2
   
        fig, axs = plt.subplots(axs0_max, axs1_max, figsize = (14,14))
        fig.suptitle("Differences in Token Cost Across Languages\nModel: " + model + "\nData: FLORES-200 dev")
        
        # Bar Chart
        axs[0,0].bar(x = chart_data["iso_script"], height = chart_data["token_count"])
        axs[0,0].set_title("Token Count by Language")
        axs[0,0].set_xlabel("Language-Script Pair (in Ascending Order by Token Count)")
        axs[0,0].set_ylabel("Token Count (in Thousands)")
        axs[0,0].set_ylim(0, tok_max)
        axs[0,0].tick_params(axis = "x", labelbottom = False)

        # Histogram         
        axs[0,1].hist(x = chart_data["token_count"], bins = 40, range = (0, tok_max))
        axs[0,1].set_title("Histogram of Token Counts")
        axs[0,1].set_xlabel("Token Count (in Thousands)")
        axs[0,1].set_ylabel("Number of Languages")

        # Scatter Plot by Script
        axs[1,0].scatter(x = chart_data["script"], y = chart_data["token_count"], s = 20, alpha = 0.5)
        axs[1,0].set_title("Token Counts by Script")
        axs[1,0].set_xlabel("Script (ISO 15924)")
        axs[1,0].set_ylabel("Token Count (in Thousands)")
        axs[1,0].set_ylim(0, tok_max)
        axs[1,0].tick_params(axis = "x", labelrotation = 90, labelsize = 8)

        # Scatter Plot for Multiscript Languages
        chart_data = chart_data.loc[chart_data["iso_639_3"].isin(iso_multiscript_list),]
            # Filter chart data to only languages with multiple scripts
        chart_data["color"] = chart_data["iso_639_3"].map(iso_color_map)
            # Assign color to each ISO 639-3 code for plotting
        chart_data = chart_data.sort_values(by = "iso_script", ascending = True, ignore_index = True)
                
        axs[1,1].scatter(x = chart_data["iso_script"], y = chart_data["token_count"], s = 20, alpha = 1,
                         c = chart_data["color"])
        axs[1,1].set_title("Languages with Multiple Scripts")
        axs[1,1].set_xlabel("Language-Script Pair (ISO 639-3, ISO 15924)")
        axs[1,1].set_ylabel("Token Count (in Thousands)")
        axs[1,1].set_ylim(0, tok_max)
        axs[1,1].tick_params(axis = "x", labelrotation = 90, labelsize = 8)

        plt.tight_layout()
        pdf.savefig()
        plt.close()

