#!/usr/bin/env python3

import os
import argparse
from pathlib import Path
from tokenizers import Tokenizer
from datasets import load_dataset


def tokenize_html(
    html_content: str,
    tokenizer_path: str,
    max_length: int = 512,
):
    """
    Tokenize HTML content using the trained tokenizer.

    Args:
        html_content: HTML content to tokenize
        tokenizer_path: Path to the tokenizer file
        max_length: Maximum length for tokenization

    Returns:
        Dictionary with input_ids, attention_mask
    """
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}")

    # Load the tokenizer
    tokenizer = Tokenizer.from_file(tokenizer_path)

    # Tokenize the input
    encoding = tokenizer.encode(html_content)

    # Get the input_ids and truncate if needed
    input_ids = encoding.ids[:max_length]

    # Create attention mask (1 for real tokens, 0 for padding)
    attention_mask = [1] * len(input_ids)

    # Pad to max_length
    padding_length = max_length - len(input_ids)
    if padding_length > 0:
        input_ids = input_ids + [tokenizer.token_to_id("<pad>")] * padding_length
        attention_mask = attention_mask + [0] * padding_length

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "tokens": encoding.tokens[:max_length],
    }


def process_sample_html(tokenizer_path: str):
    """Process a sample HTML from the WebSight dataset."""
    # Load a single example from the dataset
    dataset = load_dataset(
        "HuggingFaceM4/WebSight", "v0.2", data_files="train/*.parquet", streaming=True
    )

    # Get the first example
    for example in dataset["train"].take(1):
        html_content = example.get("html", "")
        if not html_content:
            print("No HTML content found in the first example")
            return

        # Display some basic info
        print(f"HTML content length: {len(html_content)} chars")
        print(f"HTML preview (first 200 chars):\n{html_content[:200]}...")

        # Tokenize the HTML
        encoding = tokenize_html(html_content, tokenizer_path)

        # Display token info
        print(f"\nTokenized to {len(encoding['tokens'])} tokens")
        print(f"First 20 tokens: {encoding['tokens'][:20]}")

        # Decode back to show roundtrip
        tokenizer = Tokenizer.from_file(tokenizer_path)
        decoded = tokenizer.decode(encoding["input_ids"])
        print(f"\nDecoded preview (first 200 chars):\n{decoded[:200]}...")


def main():
    parser = argparse.ArgumentParser(description="Use HTML BPE tokenizer")
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="html_tokenizer/output/tokenizer.json",
        help="Path to the tokenizer.json file",
    )
    parser.add_argument(
        "--html_file", type=str, help="Optional: Path to an HTML file to tokenize"
    )

    args = parser.parse_args()

    if args.html_file:
        # Process a specific HTML file
        if not os.path.exists(args.html_file):
            print(f"Error: HTML file not found at {args.html_file}")
            return

        with open(args.html_file, "r", encoding="utf-8") as f:
            html_content = f.read()

        encoding = tokenize_html(html_content, args.tokenizer_path)

        # Display token info
        print(f"HTML file: {args.html_file}")
        print(f"Tokenized to {len(encoding['tokens'])} tokens")
        print(f"First 20 tokens: {encoding['tokens']}")
    else:
        # Process a sample from the dataset
        process_sample_html(args.tokenizer_path)


if __name__ == "__main__":
    main()
