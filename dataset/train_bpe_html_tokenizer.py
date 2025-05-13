import argparse
import pandas as pd
from collections import Counter
import json
import os
from html_tokenizer import HTMLTailwindPreprocessor # Assuming html_tokenizer.py is in the same directory or PYTHONPATH
from bpe_tokenizer import BPETokenizer # Assuming bpe_tokenizer.py is in the same directory or PYTHONPATH

def load_html_from_parquet(dataset_path: str, html_column: str, sample_size: int = None) -> list[str]:
    """Loads HTML content from Parquet file(s)."""
    html_contents = []
    if os.path.isdir(dataset_path):
        file_paths = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.endswith('.parquet')]
    elif os.path.isfile(dataset_path) and dataset_path.endswith('.parquet'):
        file_paths = [dataset_path]
    else:
        raise ValueError(f"Invalid dataset_path: {dataset_path}. Must be a .parquet file or a directory of .parquet files.")

    if not file_paths:
        raise ValueError(f"No .parquet files found in {dataset_path}.")

    print(f"Found {len(file_paths)} parquet files to process.")

    for file_path in file_paths:
        print(f"Reading {file_path}...")
        df = pd.read_parquet(file_path)
        if html_column not in df.columns:
            raise ValueError(f"HTML column '{html_column}' not found in {file_path}. Available columns: {df.columns.tolist()}")
        # Drop rows where the HTML content is None or not a string
        df.dropna(subset=[html_column], inplace=True)
        df = df[df[html_column].apply(lambda x: isinstance(x, str))]
        html_contents.extend(df[html_column].tolist())
        if sample_size is not None and len(html_contents) >= sample_size:
            break
    
    if sample_size is not None:
        return html_contents[:sample_size]
    return html_contents

def main():
    parser = argparse.ArgumentParser(description="Train a BPE tokenizer for HTML content.")
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path to the WebSight Parquet file or directory of Parquet files.")
    parser.add_argument("--html_column", type=str, default="html_content",
                        help="Name of the column containing HTML content in the Parquet file.")
    parser.add_argument("--vocab_size", type=int, default=10000,
                        help="Desired BPE vocabulary size.")
    parser.add_argument("--output_dir", type=str, default="dataset/",
                        help="Directory to save the trained BPE model and vocabulary.")
    parser.add_argument("--sample_size", type=int, default=None,
                        help="Number of HTML documents to sample for training the tokenizer (default: use all).")
    parser.add_argument("--min_proto_token_freq", type=int, default=2,
                        help="Minimum frequency for a proto-token to be considered in BPE training.")


    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading HTML data...")
    try:
        html_documents = load_html_from_parquet(args.dataset_path, args.html_column, args.sample_size)
    except ValueError as e:
        print(f"Error loading data: {e}")
        return
    
    if not html_documents:
        print("No HTML documents loaded. Exiting.")
        return
    
    print(f"Loaded {len(html_documents)} HTML documents for tokenization training.")

    print("Preprocessing HTML and collecting proto-tokens...")
    html_preprocessor = HTMLTailwindPreprocessor()
    all_proto_tokens = []
    for i, html_doc in enumerate(html_documents):
        if i % 1000 == 0 and i > 0:
            print(f"  Pre-segmented {i}/{len(html_documents)} documents...")
        if not isinstance(html_doc, str):
            # print(f"Warning: Skipping non-string document at index {i}")
            continue
        proto_tokens = html_preprocessor.pre_segment(html_doc)
        all_proto_tokens.extend(proto_tokens)

    proto_token_counts = Counter(all_proto_tokens)
    
    # Filter proto-tokens by minimum frequency
    filtered_proto_token_counts = {
        token: count for token, count in proto_token_counts.items() if count >= args.min_proto_token_freq
    }
    if not filtered_proto_token_counts:
        print(f"No proto-tokens meet the minimum frequency of {args.min_proto_token_freq}. Exiting.")
        print(f"Total unique proto-tokens found before filtering: {len(proto_token_counts)}")
        if proto_token_counts:
             print(f"Most common proto-tokens before filtering: {proto_token_counts.most_common(10)}")
        return

    print(f"Collected {len(proto_token_counts)} unique proto-tokens.")
    print(f"Using {len(filtered_proto_token_counts)} unique proto-tokens (min_freq={args.min_proto_token_freq}) for BPE training.")


    print("Training BPE tokenizer...")
    bpe_trainer = BPETokenizer(vocab_size=args.vocab_size)
    bpe_trainer.train(filtered_proto_token_counts) # Train on filtered counts

    bpe_model_filename = f"html_bpe_model_v{args.vocab_size}.json"
    bpe_model_path = os.path.join(args.output_dir, bpe_model_filename)
    bpe_trainer.save_model(bpe_model_path)
    print(f"BPE model saved to {bpe_model_path}")

    # Define special tokens
    special_tokens = ['<PAD>', '<UNK>', '<START>', '<END>']
    
    # Create the final vocabulary for integer mapping
    # Start with special tokens, then add BPE vocab, ensuring no duplicates
    # and that BPE tokens don't override special tokens if they happen to be the same string.
    final_vocab_list = list(special_tokens)
    bpe_vocab_set = set(bpe_trainer.vocab) # Use the vocab from the trained BPE model

    for token in bpe_vocab_set:
        if token not in special_tokens:
            final_vocab_list.append(token)
    
    # Check if desired vocab_size was reached by BPE; if not, it means BPE stopped early.
    # The final_vocab_list might be smaller than vocab_size + num_special_tokens.
    print(f"BPE training resulted in {len(bpe_trainer.vocab)} mergeable tokens (excluding initial chars).")
    print(f"Final vocabulary size (including special tokens): {len(final_vocab_list)}")


    vocab_filename = f"html_vocab_v{args.vocab_size}.json"
    vocab_map_path = os.path.join(args.output_dir, vocab_filename)
    with open(vocab_map_path, 'w', encoding='utf-8') as f:
        json.dump(final_vocab_list, f, indent=2)
    print(f"Final vocabulary list saved to {vocab_map_path}")

if __name__ == "__main__":
    main() 