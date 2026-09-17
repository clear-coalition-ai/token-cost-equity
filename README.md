# Token Cost Equity

Measuring differences in tokenization efficiency and cost across languages

## Repository Structure

```
├───00_data_raw
│   ├───commoncrawl
│   ├───flores200_dataset
│   ├───iso_639_3
│   ├───linguameta
│   └───model_lists
├───01_data_processed
├───02_output_model_experiments
├───03_output_analysis
├───scripts
│   ├───01_process_data
│   ├───02_model_experiments
│   └───03_analysis
└───src
```

This repo is structured as a sequential pipeline and contains only code files. 

Empty `data_*` and `output_*` folders are included to provide the directory structure used in the scripts.

In the `scripts` folder, the scripts in each subfolder `0(N)_*` follow these conventions: 
- Read from data/output folder of same or lower number: `0(M)_data_*`/`0(M)_output_*` where `M <= N`
- Write to data/output folder of the same number: `0(N)_data_*`/`0(N)_output_*`


## Reproducibility

Analytical results can be reproduced as follows:
1. Create .env file with the following:
 ```
ANTHROPIC_API_KEY = [your key here] 
GEMINI_API_KEY = [your key here]
HF_TOKEN = [your key here]
OPENAI_API_KEY = [your key here]
PROJECT_ROOT = [your filepath here]
```
2. Download the data files listed in `00_data_raw/README.md`
3. Place the data files in the appropriate subfolders within `00_data_raw`
4. Run `scripts/run_all.py`


## Evaluating Additional Models

The `src/evaluate_equity.py` script is provided to evaluate token cost equity for additional models using FLORES-200.

The script retrieves FLORES-200 from https://huggingface.co/datasets/facebook/flores, which is a gated repo.

Example usage:
`python src/evaluate_equity.py "hf:Qwen/Qwen3.8-27B" --split "dev" --languages "eng_Latn" "zho_Hans"`

