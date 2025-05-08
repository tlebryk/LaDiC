import os
import json
import glob
import random
from math import floor

def create_train_val_test_split(input_dir, output_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, image_prefix="", seed=42):
    """
    Create train, validation, and test splits from paired .png and .gui files.
    
    Args:
        input_dir: Directory containing the .png and .gui files
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
    
    # Find all paired files
    paired_files = []
    gui_files = glob.glob(os.path.join(input_dir, "*.gui"))
    
    for gui_file in gui_files:
        # Get the base name without extension
        base_name = os.path.splitext(os.path.basename(gui_file))[0]
        
        # Construct the path to the corresponding PNG file
        png_file = os.path.join(input_dir, f"{base_name}.png")
        
        # Check if the PNG file exists
        if os.path.exists(png_file):
            paired_files.append((png_file, gui_file, base_name))
    
    # Shuffle the data
    random.shuffle(paired_files)
    
    # Calculate split indices
    total_samples = len(paired_files)
    train_end = floor(total_samples * train_ratio)
    val_end = train_end + floor(total_samples * val_ratio)
    
    # Split the data
    train_files = paired_files[:train_end]
    val_files = paired_files[train_end:val_end]
    test_files = paired_files[val_end:]
    
    # Create and save splits
    create_json_file(train_files, os.path.join(output_dir, "train.json"), image_prefix)
    create_json_file(val_files, os.path.join(output_dir, "val.json"), image_prefix)
    create_json_file(test_files, os.path.join(output_dir, "test.json"), image_prefix)
    
    # Print summary
    print(f"Created splits with {len(train_files)} training samples, {len(val_files)} validation samples, and {len(test_files)} test samples")

def create_json_file(file_pairs, output_path, image_prefix=""):
    """
    Create a JSON file from a list of file pairs.
    
    Args:
        file_pairs: List of tuples (png_file, gui_file, base_name)
        output_path: Path to save the JSON file
        image_prefix: Optional prefix for image paths in the JSON
    """
    data = []
    
    for _, gui_file, base_name in file_pairs:
        # Read the content of the .gui file
        with open(gui_file, 'r') as f:
            gui_content = f.read().strip()
        
        # Create an entry for this pair
        entry = {
            "image": f"{image_prefix}{base_name}.png",
            "caption": [gui_content]
        }
        
        data.append(entry)
    
    # Write the data to the output JSON file
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Created {output_path} with {len(data)} entries")

# Usage
input_directory = "datasets/COCO/web"
output_directory = "datasets/COCO"
image_prefix = ""  # No prefix by default

create_train_val_test_split(
    input_directory,
    output_directory,
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    image_prefix=image_prefix
)