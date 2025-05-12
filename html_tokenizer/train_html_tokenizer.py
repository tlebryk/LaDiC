#!/usr/bin/env python3

import os
import argparse
import json
import glob
from pathlib import Path
from typing import Dict, List, Optional
import re
import numpy as np
from collections import Counter
from tqdm import tqdm
import tokenizers
from tokenizers import Tokenizer, normalizers, pre_tokenizers
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.processors import TemplateProcessing
from datasets import load_dataset, Dataset


class HTMLPreprocessor:
    """Preprocessor for HTML content."""

    def __init__(self, remove_comments=True, remove_js=True, remove_css=True):
        self.remove_comments = remove_comments
        self.remove_js = remove_js
        self.remove_css = remove_css

        # Regexes for cleaning HTML
        self.comment_pattern = re.compile(r"<!--.*?-->", re.DOTALL)
        self.script_pattern = re.compile(r"<script[^>]*>.*?</script>", re.DOTALL)
        self.style_pattern = re.compile(r"<style[^>]*>.*?</style>", re.DOTALL)

    def preprocess(self, html_content: str) -> str:
        """Clean HTML content by optionally removing comments, JavaScript, and CSS."""
        if not html_content:
            return ""

        if self.remove_comments:
            html_content = self.comment_pattern.sub("", html_content)

        if self.remove_js:
            html_content = self.script_pattern.sub("", html_content)

        if self.remove_css:
            html_content = self.style_pattern.sub("", html_content)

        return html_content


def load_local_html_files(
    directory_path: str, file_pattern: str = "*.html", max_samples: Optional[int] = None
) -> Dataset:
    """
    Load local HTML files from a directory.

    Args:
        directory_path: Path to directory containing HTML files
        file_pattern: Glob pattern to match HTML files (default: "*.html")
        max_samples: Maximum number of files to load

    Returns:
        HuggingFace Dataset containing the HTML content
    """
    html_files = glob.glob(os.path.join(directory_path, file_pattern))

    if max_samples:
        html_files = html_files[:max_samples]

    dataset_items = []
    print(f"Loading {len(html_files)} local HTML files...")

    for file_path in tqdm(html_files):
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                html_content = f.read()
                dataset_items.append(
                    {"html": html_content, "filename": os.path.basename(file_path)}
                )
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue

    print(f"Successfully loaded {len(dataset_items)} HTML files")
    return Dataset.from_list(dataset_items)


def train_bpe_tokenizer(
    dataset,
    output_dir: str,
    vocab_size: int = 50000,
    min_frequency: int = 2,
    batch_size: int = 1000,
    max_samples: Optional[int] = None,
    html_field: str = "html",
    special_tokens: List[str] = ["<pad>", "<s>", "</s>", "<mask>", "<unk>"],
):
    """
    Train a BPE tokenizer on HTML content from a dataset.

    Args:
        dataset: HuggingFace dataset object
        output_dir: Directory to save the tokenizer and vocab files
        vocab_size: Size of the vocabulary
        min_frequency: Minimum frequency for a token to be included
        batch_size: Batch size for processing
        max_samples: Maximum number of samples to process (for debugging/testing)
        html_field: Field name in the dataset containing HTML content
        special_tokens: Special tokens to add to the tokenizer
    """
    os.makedirs(output_dir, exist_ok=True)

    # Initialize the tokenizer
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.normalizer = normalizers.Sequence(
        [normalizers.Replace(r"\s+", " "), normalizers.NFKC(), normalizers.Lowercase()]
    )
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()

    # Initialize the trainer
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
        show_progress=True,
    )

    # HTML preprocessor
    html_preprocessor = HTMLPreprocessor()

    def batch_iterator():
        processed_samples = 0

        # Process dataset in batches
        for i in tqdm(range(0, len(dataset), batch_size)):
            if max_samples and processed_samples >= max_samples:
                break

            end_idx = min(i + batch_size, len(dataset))
            batch = dataset[i:end_idx]

            # Extract and preprocess HTML content
            html_contents = []
            for sample in batch["html"]:
                html_content = None
                # Handle different types of samples
                if isinstance(sample, dict) and html_field in sample:
                    # Dictionary-like samples (from HuggingFace datasets)
                    html_content = sample[html_field]
                elif isinstance(sample, str):
                    # Direct string content
                    html_content = sample

                if html_content:
                    processed_html = html_preprocessor.preprocess(html_content)
                    if processed_html:
                        html_contents.append(processed_html)

            processed_samples += len(html_contents)

            if html_contents:
                yield html_contents

    # Train the tokenizer
    print(f"Training BPE tokenizer with vocab size {vocab_size}...")
    tokenizer.train_from_iterator(batch_iterator(), trainer)

    # Add post-processor for full compatibility with transformers
    tokenizer.post_processor = TemplateProcessing(
        single="<s> $A </s>",
        pair="<s> $A </s> $B:1 </s>:1",
        special_tokens=[
            ("<s>", tokenizer.token_to_id("<s>")),
            ("</s>", tokenizer.token_to_id("</s>")),
        ],
    )

    # Save the tokenizer
    tokenizer_path = os.path.join(output_dir, "tokenizer.json")
    tokenizer.save(tokenizer_path)
    print(f"Tokenizer saved to {tokenizer_path}")

    # Save vocabulary
    vocab = tokenizer.get_vocab()
    vocab_path = os.path.join(output_dir, "vocab.json")

    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    print(f"Vocabulary saved to {vocab_path}")

    # Generate merges.txt file
    # Since tokenizer.model.merges is not accessible in newer versions,
    # we'll create a minimal merges.txt file that's compatible with the format
    merges_path = os.path.join(output_dir, "merges.txt")

    # Create an empty merges.txt file (will be populated by the tokenizer when loaded)
    with open(merges_path, "w", encoding="utf-8") as f:
        # Write the version header that BPE tokenizers expect
        f.write("#version: 0.2\n")
        # We can't access merges directly, but we have the vocab
        # Extract character pairs that might be in the vocab
        # This is a simplified approach and may not reflect the actual merges
        pairs = set()
        for token in vocab.keys():
            if len(token) > 1 and not token.startswith("<") and not token.endswith(">"):
                for i in range(len(token) - 1):
                    pair = (token[i], token[i + 1])
                    pairs.add(pair)

        # Write some of the possible merges
        for pair in list(pairs)[:1000]:  # Limit to avoid huge files
            f.write(f"{pair[0]} {pair[1]}\n")

    print(f"Merges saved to {merges_path}")

    return tokenizer


def process_websight_dataset(
    output_dir: str,
    vocab_size: int = 50000,
    min_frequency: int = 2,
    batch_size: int = 1000,
    max_samples: Optional[int] = None,
    subset_pattern: str = None,
):
    """
    Process WebSight dataset and train a BPE tokenizer.

    Args:
        output_dir: Directory to save the tokenizer files
        vocab_size: Size of the vocabulary
        min_frequency: Minimum frequency for a token to be included
        batch_size: Batch size for processing
        max_samples: Maximum number of samples to process
        subset_pattern: Pattern to select specific files from the dataset
    """
    # Load dataset with specific files if pattern is provided
    if subset_pattern:
        print(f"Loading WebSight dataset with pattern: {subset_pattern}")
        dataset = load_dataset(
            "HuggingFaceM4/WebSight", "v0.2", data_files=subset_pattern, streaming=True
        )
    else:
        print("Loading complete WebSight dataset (warning: this is 320GB)")
        dataset = load_dataset("HuggingFaceM4/WebSight", "v0.2", streaming=True)

    # Convert streaming dataset to a regular dataset with a limited size for training
    # This limits memory usage while still providing enough data
    limited_dataset = []
    print(
        f"Collecting up to {max_samples if max_samples else 'all'} samples for tokenizer training..."
    )

    for i, example in enumerate(tqdm(dataset["train"])):
        if max_samples and i >= max_samples:
            break
        if "html" in example and example["html"]:
            limited_dataset.append(example)

    # Convert to HuggingFace Dataset for easier batch processing
    train_dataset = Dataset.from_list(limited_dataset)
    print(f"Collected {len(train_dataset)} samples for tokenizer training")

    # Train tokenizer
    tokenizer = train_bpe_tokenizer(
        train_dataset,
        output_dir=output_dir,
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        batch_size=batch_size,
        max_samples=max_samples,
        html_field="html",
    )

    return tokenizer


def process_local_html_files(
    local_dir: str,
    output_dir: str,
    file_pattern: str = "*.html",
    vocab_size: int = 50000,
    min_frequency: int = 2,
    batch_size: int = 1000,
    max_samples: Optional[int] = None,
):
    """
    Process local HTML files and train a BPE tokenizer.

    Args:
        local_dir: Directory containing HTML files
        output_dir: Directory to save the tokenizer files
        file_pattern: Glob pattern to match HTML files
        vocab_size: Size of the vocabulary
        min_frequency: Minimum frequency for a token to be included
        batch_size: Batch size for processing
        max_samples: Maximum number of samples to process
    """
    # Load local HTML files
    dataset = load_local_html_files(local_dir, file_pattern, max_samples)

    if len(dataset) == 0:
        print(f"No HTML files found in {local_dir} matching pattern '{file_pattern}'")
        return None

    # Train tokenizer
    tokenizer = train_bpe_tokenizer(
        dataset,
        output_dir=output_dir,
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        batch_size=batch_size,
        max_samples=max_samples,
        html_field="html",
    )

    return tokenizer


def main():
    parser = argparse.ArgumentParser(
        description="Train BPE tokenizer on HTML data (WebSight or local files)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Directory to save tokenizer files",
    )
    parser.add_argument("--vocab_size", type=int, default=50000, help="Vocabulary size")
    parser.add_argument(
        "--min_frequency", type=int, default=2, help="Minimum frequency for a token"
    )
    parser.add_argument(
        "--batch_size", type=int, default=1000, help="Batch size for processing"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=100000,
        help="Maximum number of samples to process",
    )
    parser.add_argument(
        "--subset_pattern",
        type=str,
        default=None,
        help="Pattern to select specific files for WebSight (e.g., 'train/*.parquet')",
    )
    parser.add_argument(
        "--local_dir",
        type=str,
        default=None,
        help="Path to local directory containing HTML files (if not using WebSight)",
    )
    parser.add_argument(
        "--file_pattern",
        type=str,
        default="*.html",
        help="Glob pattern to match local HTML files (e.g., '*.html', '**/*.html')",
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Process dataset and train tokenizer
    if args.local_dir:
        # Use local HTML files
        process_local_html_files(
            local_dir=args.local_dir,
            output_dir=args.output_dir,
            file_pattern=args.file_pattern,
            vocab_size=args.vocab_size,
            min_frequency=args.min_frequency,
            batch_size=args.batch_size,
            max_samples=args.max_samples,
        )
    else:
        # Use WebSight dataset
        process_websight_dataset(
            output_dir=args.output_dir,
            vocab_size=args.vocab_size,
            min_frequency=args.min_frequency,
            batch_size=args.batch_size,
            max_samples=args.max_samples,
            subset_pattern=args.subset_pattern,
        )


if __name__ == "__main__":
    main()
