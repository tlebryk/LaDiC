import glob
import os
from collections import Counter
import json # For saving/loading raw pre-segmented corpus if needed, not directly for BPE model here.
import pandas as pd # Added for Parquet file reading

# Adjust import paths if your script is located elsewhere relative to these files.
# Assuming train_bpe_pipeline.py is in the same root as the dataset directory
from html_tokenizer import HTMLTailwindPreprocessor
from bpe_tokenizer import BPETokenizer

# --- Configuration ---
# HTML_FILES_DIR = "small-dataset/"  # Directory containing HTML files relative to this script's parent
# HTML_FILES_PATTERN = "*.html"
PARQUET_FILE_PATH = "train-00000-of-00738-80a58552f2fb3344.parquet" # Relative to this script's parent (dataset/)
HTML_COLUMN_NAME = "text" # Column in Parquet containing HTML strings

BPE_VOCAB_SIZE = 10000  # Target vocabulary size for BPE
BPE_MODEL_SAVE_PATH = "html_bpe_model_parquet.json" # Where to save the trained BPE model
PRESEGMENTED_CORPUS_SAVE_PATH = "pre_segmented_corpus_parquet.json" # Optional: save intermediate proto-tokens

def load_and_preprocess_data_from_parquet(parquet_path: str, html_column: str) -> list[str]:
    """
    Loads HTML data from a Parquet file, preprocesses them using HTMLTailwindPreprocessor,
    and returns a flat list of all proto-tokens.
    """
    preprocessor = HTMLTailwindPreprocessor()
    all_proto_tokens = []

    if not os.path.exists(parquet_path):
        print(f"Error: Parquet file not found at {parquet_path}")
        print("Please ensure the file name and path are correct relative to the script location.")
        return []

    try:
        print(f"Loading Parquet file from: {parquet_path}...")
        df = pd.read_parquet(parquet_path)
        print(f"Successfully loaded Parquet file. Shape: {df.shape}")
    except Exception as e:
        print(f"Error reading Parquet file: {e}")
        print("Make sure you have 'pyarrow' or 'fastparquet' installed (e.g., pip install pandas pyarrow)")
        return []

    if html_column not in df.columns:
        print(f"Error: HTML column '{html_column}' not found in the Parquet file.")
        print(f"Available columns: {df.columns.tolist()}")
        return []
    
    if df.empty:
        print("The DataFrame from Parquet is empty. No data to process.")
        return []

    print(f"Extracting HTML content from column '{html_column}' and pre-segmenting...")
    num_rows = len(df)
    for index, row in df.iterrows():
        html_content = row[html_column]
        if pd.isna(html_content) or not isinstance(html_content, str) or not html_content.strip():
            # print(f"Warning: Row {index} has missing, non-string, or empty HTML content. Skipping.")
            continue
        
        proto_tokens = preprocessor.pre_segment(html_content)
        all_proto_tokens.extend(proto_tokens)
        
        if (index + 1) % 1000 == 0 or (index + 1) == num_rows:
            print(f"Processed {index + 1}/{num_rows} rows...")
            
    print(f"Total proto-tokens extracted: {len(all_proto_tokens)}")
    return all_proto_tokens

def main():
    print("--- Starting BPE Tokenizer Training Pipeline (from Parquet) ---")

    # 1. Load and Preprocess HTML Data from Parquet
    base_data_dir = os.path.dirname(os.path.abspath(__file__)) # Directory of this script (dataset/)
    target_parquet_path = os.path.join(base_data_dir, PARQUET_FILE_PATH)
    
    print(f"Step 1: Loading and pre-segmenting HTML data from Parquet file '{target_parquet_path}' (column: '{HTML_COLUMN_NAME}')...")
    all_proto_tokens = load_and_preprocess_data_from_parquet(target_parquet_path, HTML_COLUMN_NAME)

    if not all_proto_tokens:
        print("No proto-tokens were generated from the Parquet file. Exiting pipeline.")
        return

    # Optional: Save the pre-segmented corpus for inspection or reuse
    # corpus_save_full_path = os.path.join(base_data_dir, PRESEGMENTED_CORPUS_SAVE_PATH)
    # try:
    #     with open(corpus_save_full_path, 'w', encoding='utf-8') as f:
    #         json.dump(all_proto_tokens, f) # indent=2 can make this very large
    #     print(f"Pre-segmented corpus saved to {corpus_save_full_path}")
    # except IOError as e:
    #     print(f"Error saving pre-segmented corpus: {e}")

    # 2. Calculate Proto-token Frequencies
    print("\nStep 2: Calculating proto-token frequencies...")
    proto_token_counts = Counter(all_proto_tokens)
    print(f"Unique proto-tokens: {len(proto_token_counts)}")
    # print("Most common proto-tokens:", proto_token_counts.most_common(20))

    # 3. Train BPE Tokenizer
    print(f"\nStep 3: Training BPE tokenizer with vocab_size={BPE_VOCAB_SIZE}...")
    bpe_tokenizer = BPETokenizer(vocab_size=BPE_VOCAB_SIZE)
    bpe_tokenizer.train(proto_token_counts) # proto_token_counts is dict[str, int]
    print("BPE training complete.")

    # 4. Save Trained BPE Model
    model_save_full_path = os.path.join(base_data_dir, BPE_MODEL_SAVE_PATH)
    print(f"\nStep 4: Saving trained BPE model to '{model_save_full_path}'...")
    bpe_tokenizer.save_model(model_save_full_path)
    print(f"BPE model saved successfully.")

    # 5. Example: Load and Test the Trained Tokenizer
    print("\nStep 5: Testing the loaded BPE model...")
    loaded_bpe_tokenizer = BPETokenizer()
    loaded_bpe_tokenizer.load_model(model_save_full_path)
    
    sample_html_for_testing = '''
    <body class="bg-gray-200 p-4">
        <h1 class="text-xl font-semibold">Test Page</h1>
        <p>This is a test paragraph with some <span class="text-red-500">red text</span>.</p>
    </body>
    '''
    print(f"\nOriginal test HTML snippet:\n{sample_html_for_testing}")
    
    preprocessor_for_test = HTMLTailwindPreprocessor()
    test_proto_tokens = preprocessor_for_test.pre_segment(sample_html_for_testing)
    print(f"Pre-segmented test tokens: {test_proto_tokens}")
    
    final_bpe_tokens = loaded_bpe_tokenizer.tokenize(test_proto_tokens)
    print(f"Final BPE tokens: {final_bpe_tokens}")

    print("\n--- BPE Tokenizer Training Pipeline Finished ---")

if __name__ == "__main__":
    main() 