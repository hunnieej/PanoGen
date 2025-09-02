#!/usr/bin/env python3
"""
Quick test and example usage for extract_outputs.py
"""

import os
from extract_outputs import extract_images, create_tar_archive, find_max_tile_idx

def test_extract_outputs():
    """Test the extract_outputs functionality with example paths."""
    
    # Example usage scenarios
    examples = [
        {
            "input_dir": "/home/mmai6k_jh/workspace/PanoGen/spherediff/MultiDiffusion/outputs/t5",
            "description": "Extract from t5 directory with default settings"
        },
        {
            "input_dir": "/home/mmai6k_jh/workspace/PanoGen/spherediff/MultiDiffusion/outputs/t6", 
            "description": "Extract from t6 directory with custom step number",
            "last_num": 25
        }
    ]
    
    print("Extract Outputs - Usage Examples")
    print("=" * 50)
    
    for i, example in enumerate(examples, 1):
        print(f"\nExample {i}: {example['description']}")
        print("-" * 30)
        
        input_dir = example["input_dir"]
        last_num = example.get("last_num", 49)
        
        # Show what would be extracted
        if os.path.exists(input_dir):
            max_tiles = find_max_tile_idx(input_dir, last_num)
            print(f"Input directory: {input_dir}")
            print(f"Last step number: {last_num}")
            print(f"Max tile index found: {max_tiles}")
            
            # List files that would be extracted
            files_to_extract = []
            
            # Check for final.png
            if os.path.exists(os.path.join(input_dir, "final.png")):
                files_to_extract.append("final.png")
            
            # Check for step file
            step_file = f"step_{last_num:03d}.png"
            if os.path.exists(os.path.join(input_dir, step_file)):
                files_to_extract.append(step_file)
            
            # Check for tile files
            tile_count = 0
            if max_tiles >= 0:
                for idx in range(max_tiles + 1):
                    tile_file = f"tile_{idx:03d}_{last_num:03d}.png"
                    if os.path.exists(os.path.join(input_dir, tile_file)):
                        tile_count += 1
            
            print(f"Files to extract: {len(files_to_extract)} + {tile_count} tiles")
            print(f"Tar filename: {os.path.basename(input_dir)}.tar")
            
        else:
            print(f"Directory does not exist: {input_dir}")
    
    print("\n" + "=" * 50)
    print("Command line usage examples:")
    print("=" * 50)
    
    print("\n1. Basic usage (auto-detect tiles, last step 49):")
    print("   python extract_outputs.py /path/to/output/directory")
    
    print("\n2. Custom step number:")
    print("   python extract_outputs.py /path/to/output/directory --last_num 25")
    
    print("\n3. Specify max tile index:")
    print("   python extract_outputs.py /path/to/output/directory --max_tile_idx 88")
    
    print("\n4. Custom tar name:")
    print("   python extract_outputs.py /path/to/output/directory --tar_name my_results")
    
    print("\n5. Keep extracted directory:")
    print("   python extract_outputs.py /path/to/output/directory --keep_extracted")
    
    print("\n6. Full example:")
    print("   python extract_outputs.py /home/mmai6k_jh/workspace/PanoGen/spherediff/MultiDiffusion/outputs/t5 --last_num 49 --max_tile_idx 88")

if __name__ == "__main__":
    test_extract_outputs()
