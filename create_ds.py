import os
import json
import glob
import random
from math import floor


def create_train_val_test_split(
    input_dir,
    output_dir,
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    image_prefix="",
    seed=42,
):
    """
    Create train, validation, and test splits from paired .png and .html files.

    Args:
        input_dir: Directory containing the .png and .html files
        output_dir: Directory to save the output JSON files
        train_ratio: Proportion of data for training set (default: 0.8)
        val_ratio: Proportion of data for validation set (default: 0.1)
        test_ratio: Proportion of data for test set (default: 0.1)
        image_prefix: Optional prefix for image paths in the JSON
        seed: Random seed for reproducibility
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Set random seed for reproducibility
    random.seed(seed)

    # Find all paired files by looking for HTML files first
    paired_files_data = []
    html_files = glob.glob(os.path.join(input_dir, "*.html"))

    print(f"Found {len(html_files)} HTML files in {input_dir}")

    for html_file in html_files:
        # Get the base name without extension
        base_name = os.path.splitext(os.path.basename(html_file))[0]

        # Construct the path to the corresponding PNG file
        png_file = os.path.join(input_dir, f"{base_name}.png")

        # Check if the PNG file exists
        if os.path.exists(png_file):
            # Read the HTML content
            try:
                with open(html_file, "r", encoding="utf-8", errors="ignore") as f:
                    html_content = f.read().strip()
                # Store png path, html path, base name, and html content
                paired_files_data.append((png_file, html_file, base_name, html_content))
            except Exception as e:
                print(f"Error reading {html_file}: {e}")
                continue
        # else:
        #     print(f"Warning: No matching PNG found for {html_file}")


    print(f"Found {len(paired_files_data)} paired HTML/PNG files.")

    # Create a mapping of filenames to integer IDs
    # Sort the files first to ensure consistent IDs across runs
    sorted_file_list = sorted([base_name for _, _, base_name, _ in paired_files_data])
    id_mapping = {filename: i + 1 for i, filename in enumerate(sorted_file_list)}

    # Shuffle the data
    random.shuffle(paired_files_data)

    # Calculate split indices
    total_samples = len(paired_files_data)
    if total_samples == 0:
        print("No paired files found. Exiting.")
        return

    train_end = floor(total_samples * train_ratio)
    val_end = train_end + floor(total_samples * val_ratio)

    # Split the data
    train_files = paired_files_data[:train_end]
    val_files = paired_files_data[train_end:val_end]
    test_files = paired_files_data[val_end:]

    # Create and save splits
    # Assuming caption should always be a string for HTML content
    create_json_file(
        train_files,
        os.path.join(output_dir, "train.json"),
        image_prefix,
        id_mapping,
        caption_as_string=True, # Use HTML content as string caption
    )
    create_json_file(
        val_files,
        os.path.join(output_dir, "val.json"),
        image_prefix,
        id_mapping,
        caption_as_string=True, # Use HTML content as string caption
    )
    create_json_file(
        test_files,
        os.path.join(output_dir, "test.json"),
        image_prefix,
        id_mapping,
        caption_as_string=True, # Use HTML content as string caption
    )

    # Print summary
    print(
        f"Created splits with {len(train_files)} training samples, {len(val_files)} validation samples, and {len(test_files)} test samples in {output_dir}"
    )
    print(f"Generated {len(id_mapping)} unique integer IDs for images")


def create_json_file(
    file_pairs_data, output_path, image_prefix="", id_mapping=None, caption_as_string=False
):
    """
    Create a JSON file from a list of file pairs data.

    Args:
        file_pairs_data: List of tuples (png_file, html_file, base_name, html_content)
        output_path: Path to save the JSON file
        image_prefix: Optional prefix for image paths in the JSON
        id_mapping: Dictionary mapping filenames to integer IDs
        caption_as_string: If True, caption will be a string; otherwise, a list (forced to string for HTML)
    """
    data = []

    for _, _, base_name, html_content in file_pairs_data:

        # Get the integer ID for this image
        image_id = id_mapping.get(base_name, 0) if id_mapping else 0

        # Create an entry for this pair
        entry = {
            "image": f"{image_prefix}{base_name}.png", # Path relative to image_root
            "image_id": image_id,  # Now using an integer ID
            "caption": html_content # Use the HTML content directly as the caption string
        }
        # The caption_as_string flag is less relevant here as HTML is naturally a string,
        # but we adhere to the structure expected by the original call if needed.
        # If the downstream loader strictly needs a list, uncomment the next lines:
        # if not caption_as_string:
        #     entry["caption"] = [html_content]

        data.append(entry)

    # Write the data to the output JSON file
    with open(output_path, "w", encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Created {output_path} with {len(data)} entries")


# --- Updated Usage for WebSight ---
# Define the input directory where the extracted .html and .png files are
input_directory = "datasets/websight/all_data"
# Define the output directory where train.json, val.json, test.json will be saved
output_directory = "datasets/websight"
# Define the prefix if image paths in JSON should be relative to a different root later
# Usually empty if image_root in main.py points directly to input_directory
image_prefix = "" # Adjust if needed, e.g., "all_data/" if image_root becomes "datasets/websight"

print(f"Processing HTML/PNG pairs from: {input_directory}")
print(f"Saving JSON splits to: {output_directory}")

create_train_val_test_split(
    input_directory,
    output_directory,
    train_ratio=0.8, # Adjust ratios as needed
    val_ratio=0.1,
    test_ratio=0.1,
    image_prefix=image_prefix,
)
