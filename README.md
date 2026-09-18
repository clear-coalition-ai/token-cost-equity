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

### Arguments
| Argument | Supported Values | Usage | Description |
| :------  | :--------------- | :---- | :---------- |
| `model_id` | Hugging Face model prefixed with `hf:`<br>OpenAI model prefixed with `openai:`<br>Anthropic model prefixed with `anthropic:`<br>Google model prefixed with `google:` | Required | Model to evaluate |
| `split` | `dev` or `devtest` | Optional, default is `devtest` | FLORES-200 split to use |
| `output` | Filepath | Optional, default is `None` | Saves detailed results to CSV files |
| `languages` | FLORES-200 language-script codes or `"all"` | Optional, default is all languages | Subset of FLORES-200 languages to use |

### Example Usage (CLI):
Evaluate model on all FLORES-200 languages using `dev` split:\
`python src/evaluate_equity.py "hf:meta-llama/Llama-4-Scout-17B-16E-Instruct" --split "dev"`
 
Evaluate model on a subset of FLORES-200 languages:\
`python src/evaluate_equity.py "hf:Qwen/Qwen3.8-27B" --languages "eng_Latn" "zho_Hans"`

Evaluate model and output token counts by language to CSV file:\
`python src/evaluate_equity.py "openai:gpt-5" --output "my_folder/token_counts_by_language.csv"`

### Advanced Cases
For models requiring special arguments, call the functions directly in Python instead of using CLI.

Example: Moonshot AI's [Kimi K3](https://huggingface.co/moonshotai/Kimi-K3) requires `trust_remote_code = True`
