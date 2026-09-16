"""Run the full pipeline to reproduce results"""

# %% Setup
# Python standard library
import os
import subprocess
import sys

# Third-party libraries
from dotenv import load_dotenv


# %% Project folder
load_dotenv()
PROJECT_ROOT = os.getenv("PROJECT_ROOT")
if not PROJECT_ROOT:
    raise EnvironmentError("PROJECT_ROOT is not set. Check your .env file.")
os.chdir(PROJECT_ROOT)

SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "scripts")


# Folders and scripts in the order they should run
PIPELINE = {
    "01_process_data": [
        "process_chosen_models.py",
        "process_flores200.py",
        "process_linguameta.py",
        "process_commoncrawl.py",
        "process_macrolanguage_mappings.py",
        "build_flores200_langinfo.py",
    ],
    "02_model_experiments": [
        "count_tokens_flores200.py",
        "vocab_size.py",
    ],
    "03_analysis": [
        "token_count_equity_metrics.py",
        "token_count_grade_A.py",
    ],
}

for stage, scripts in PIPELINE.items():
    folder = os.path.join(SCRIPTS_DIR, stage)
    print(f"=== {stage} ===")

    for name in scripts:
        print(f"Running {name}...")
        result = subprocess.run([sys.executable, name], cwd=folder)
        if result.returncode != 0:
            sys.exit(f"Failed: {stage}/{name}")

print("Pipeline complete.")