"""
Tokenizer equity evaluation script.

Computes the ratio between the most token-expensive and least
token-expensive language, using FLORES-200 (facebook/flores) as a fixed,
comparable set of segments across ~200 languages.

Usage:
    python evaluate_equity.py hf:meta-llama/Llama-3.1-8B
    python evaluate_equity.py openai:gpt-4o --split devtest --output results.csv
"""

# %% Setup
# Python standard library
import argparse
from dataclasses import dataclass
from typing import Protocol

# Third-party libraries
import pandas as pd
from datasets import load_dataset, get_dataset_config_names
from transformers import AutoTokenizer
from dotenv import load_dotenv

# API tokens
load_dotenv()  # no-op if no .env is present; picks up HF_TOKEN, ANTHROPIC_API_KEY, GEMINI_API_KEY, etc.

# FLORES-200 Hugging Face repo
FLORES_REPO = "facebook/flores"


# %% Data loading
def load_flores200(split: str, languages: list[str] | None = None) -> pd.DataFrame:
    """Load FLORES-200 segments from facebook/flores.
    split: 'dev' or 'devtest'.
    languages: FLORES language codes (e.g. 'eng_Latn') to load;
        loads all available languages if languages = "all" or language argument not provided 
    Returns columns: [lang, segment_id, text]
    """
    # facebook/flores provides the following:
    #   Single-language data files (e.g. "eng_Latn") in long format
    #   Paired translation data files (e.g. "eng_Latn-ary_Arab") in wide format
    #   An "all" data file in wide format
    # Special case: ISO 639-3 "ajp" has been deprecated (merged with "apc") since the release of FLORES-200, "ajp" excluded from this eval
    
    if (not languages) or (languages == "all"):
        df = load_dataset(FLORES_REPO, "all", split = split).to_pandas()
        df = df.drop(columns = ["URL", "domain", "topic", "has_image", "has_hyperlink", "sentence_ajp_Arab"])
        df = pd.wide_to_long(df = df, stubnames = "sentence", i = "id", j = "lang", sep = "_", suffix = ".+").reset_index(drop = False)

    else:
        df = pd.DataFrame()
        for lang in languages:
            df_i = load_dataset(FLORES_REPO, lang, split=split).to_pandas()
            df_i = df_i[["id", "sentence"]]
            df_i["lang"] = lang
            df = pd.concat(objs = [df, df_i], ignore_index = True)

    df = df.groupby("lang").agg(text = ("sentence", "\n".join))
    
    return df


# %% Token counting
class TokenCounter(Protocol):
    def count_batch(self, texts: list[str]) -> list[int]:
        """Token count per text, same order as input."""


class HFTokenCounter:
    """Loads a Hugging Face tokenizer and counts tokens locally."""
    def __init__(self, model_name: str, trust_remote_code: bool = False):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=trust_remote_code
        )

    def count_batch(self, texts: list[str]) -> list[int]:
        # add_special_tokens=False: count content tokens only, so tokenizers
        # with different BOS/EOS conventions are compared on the same basis.
        encoded = self.tokenizer(texts, add_special_tokens=False)
        return [len(ids) for ids in encoded["input_ids"]]


class OpenAITokenCounter:
    """Uses `tiktoken` to count tokens locally, without a completion call."""
    def __init__(self, model_name: str, **kwargs):
        import tiktoken

        self.encoding = tiktoken.encoding_for_model(model_name)

    def count_batch(self, texts: list[str]) -> list[int]:
        return [len(self.encoding.encode(text)) for text in texts]


class AnthropicTokenCounter:
    """Uses Anthropic's count_tokens endpoint (no completion is generated)."""
    def __init__(self, model_name: str, **kwargs):
        import anthropic

        self.model_name = model_name
        self.client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env

    def count_batch(self, texts: list[str]) -> list[int]:
        counts = []
        for text in texts:
            resp = self.client.messages.count_tokens(
                model=self.model_name,
                messages=[{"role": "user", "content": text}],
            )
            counts.append(resp.input_tokens)
        return counts


class GoogleTokenCounter:
    """Uses the Gemini API's count_tokens endpoint (no completion is generated)."""
    def __init__(self, model_name: str, **kwargs):
        from google import genai

        self.model_name = model_name
        self.client = genai.Client()  # reads GEMINI_API_KEY from env

    def count_batch(self, texts: list[str]) -> list[int]:
        counts = []
        for text in texts:
            resp = self.client.models.count_tokens(model=self.model_name, contents=text)
            counts.append(resp.total_tokens)
        return counts


def load_token_counter(model_id: str, **kwargs) -> TokenCounter:
    """model_id is prefixed, e.g. 'hf:meta-llama/Llama-3.1-8B', 'openai:gpt-4o',
    'anthropic:claude-sonnet-4-5', 'google:gemini-2.5-flash'."""
    provider, name = model_id.split(":", 1)
    if provider == "hf":
        return HFTokenCounter(name, **kwargs)
    elif provider == "openai":
        return OpenAITokenCounter(name, **kwargs)
    elif provider == "anthropic":
        return AnthropicTokenCounter(name, **kwargs)
    elif provider == "google":
        return GoogleTokenCounter(name, **kwargs)
    else:
        raise ValueError(
            f"Unsupported provider '{provider}'. Add a TokenCounter adapter "
            f"for it and register it in load_token_counter()."
        )


# %% Pipeline
def count_tokens_for_dataset(df: pd.DataFrame, counter: TokenCounter) -> pd.DataFrame:
    """Adds an n_tokens column, one value per segment."""
    df = df.copy()
    df["n_tokens"] = counter.count_batch(df["text"].tolist())
    df = df.reset_index(drop = False)
    return df


@dataclass
class EquityResult:
    ratio: float
    max_language: str
    max_tokens: float
    min_language: str
    min_tokens: float


def compute_equity_ratio(totals: pd.DataFrame) -> EquityResult:
    max_row = totals.loc[totals["n_tokens"].idxmax(), ["lang", "n_tokens"]]
    min_row = totals.loc[totals["n_tokens"].idxmin(), ["lang", "n_tokens"]]
    return EquityResult(
        ratio=max_row["n_tokens"] / min_row["n_tokens"],
        max_language=max_row["lang"], max_tokens=max_row["n_tokens"],
        min_language=min_row["lang"], min_tokens=min_row["n_tokens"],
    )


def run_evaluation(
    model_id: str,
    split: str = "devtest",
    save_path: str | None = None,
    languages: list[str] | None = None,
) -> EquityResult:
    df = load_flores200(split, languages)
    counter = load_token_counter(model_id)
    token_counts_by_language = count_tokens_for_dataset(df, counter)
    if save_path:
        token_counts_by_language.to_csv(save_path, index=False)
    return compute_equity_ratio(token_counts_by_language)


# %% Evaluation procedure
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_id")
    parser.add_argument("--split", default="devtest")
    parser.add_argument("--output", default=None)
    parser.add_argument("--languages", nargs="*", default=None)
    args = parser.parse_args()

    result = run_evaluation(
        args.model_id, split=args.split, save_path=args.output, languages=args.languages
    )
    print(result)
