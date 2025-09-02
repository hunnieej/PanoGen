#!/usr/bin/env python3
"""
Extract specific output images from panorama generation results and create tar archive.

This script extracts:
1. final.png
2. step_{last_num}.png 
3. tile_{idx}_{last_num}.png (for idx from 0 to max_tile_idx)

Creates a tar file named after the directory's basename.
"""

import os
import sys
import argparse
import tarfile
import shutil
from pathlib import Path

def find_max_tile_idx(directory, last_num):
    """Find the maximum tile index for the given step number."""
    max_idx = -1
    for file in os.listdir(directory):
        if file.startswith(f"tile_") and file.endswith(f"_{last_num:03d}.png"):
            try:
                # Extract tile index from filename like "tile_005_049.png"
                parts = file.split("_")
                if len(parts) >= 3:
                    tile_idx = int(parts[1])
                    max_idx = max(max_idx, tile_idx)
            except (ValueError, IndexError):
                continue
    return max_idx

def extract_images(input_dir, output_dir, last_num=49, max_tile_idx=None):
    """
    Extract specific images from input directory to output directory.
    
    Args:
        input_dir: Source directory containing generated images
        output_dir: Destination directory for extracted images
        last_num: The step number to extract (default: 49)
        max_tile_idx: Maximum tile index to extract (if None, auto-detect)
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        print(f"Error: Input directory '{input_dir}' does not exist")
        return False
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    copied_files = []
    
    # 1. Copy final.png
    final_src = input_path / "final.png"
    if final_src.exists():
        final_dst = output_path / "final.png"
        shutil.copy2(final_src, final_dst)
        copied_files.append("final.png")
        print(f"Copied: final.png")
    else:
        print(f"Warning: final.png not found")
    
    # 2. Copy step_{last_num}.png
    step_src = input_path / f"step_{last_num:03d}.png"
    if step_src.exists():
        step_dst = output_path / f"step_{last_num:03d}.png"
        shutil.copy2(step_src, step_dst)
        copied_files.append(f"step_{last_num:03d}.png")
        print(f"Copied: step_{last_num:03d}.png")
    else:
        print(f"Warning: step_{last_num:03d}.png not found")
    
    # 3. Auto-detect max tile index if not provided
    if max_tile_idx is None:
        max_tile_idx = find_max_tile_idx(input_dir, last_num)
        if max_tile_idx == -1:
            print(f"Warning: No tile files found for step {last_num}")
            return len(copied_files) > 0
        print(f"Auto-detected max tile index: {max_tile_idx}")
    
    # 4. Copy tile_{idx}_{last_num}.png files
    tile_count = 0
    for idx in range(max_tile_idx + 1):
        tile_src = input_path / f"tile_{idx:03d}_{last_num:03d}.png"
        if tile_src.exists():
            tile_dst = output_path / f"tile_{idx:03d}_{last_num:03d}.png"
            shutil.copy2(tile_src, tile_dst)
            copied_files.append(f"tile_{idx:03d}_{last_num:03d}.png")
            tile_count += 1
        else:
            print(f"Warning: tile_{idx:03d}_{last_num:03d}.png not found")
    
    print(f"Copied {tile_count} tile files (indices 0-{max_tile_idx})")
    print(f"Total files copied: {len(copied_files)}")
    
    return len(copied_files) > 0

def create_tar_archive(source_dir, tar_path):
    """Create a tar archive from the source directory."""
    try:
        with tarfile.open(tar_path, 'w') as tar:
            # Add all files in the directory to the tar, but use relative paths
            for file_path in Path(source_dir).iterdir():
                if file_path.is_file():
                    tar.add(file_path, arcname=file_path.name)
        print(f"Created tar archive: {tar_path}")
        return True
    except Exception as e:
        print(f"Error creating tar archive: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Extract panorama output images and create tar archive")
    parser.add_argument("input_dir", help="Input directory containing generated images")
    parser.add_argument("--output_dir", help="Output directory for extracted images (default: input_dir + '_extracted')")
    parser.add_argument("--last_num", type=int, default=49, help="Step number to extract (default: 49)")
    parser.add_argument("--max_tile_idx", type=int, help="Maximum tile index to extract (default: auto-detect)")
    parser.add_argument("--tar_name", help="Name of tar file (default: directory basename)")
    parser.add_argument("--keep_extracted", action="store_true", help="Keep extracted directory after creating tar")
    
    args = parser.parse_args()
    
    input_dir = os.path.abspath(args.input_dir)
    
    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = input_dir + "_extracted"
    
    # Determine tar filename
    if args.tar_name:
        tar_name = args.tar_name
        if not tar_name.endswith('.tar'):
            tar_name += '.tar'
    else:
        # Use the last component of the input directory path
        dir_basename = os.path.basename(input_dir.rstrip('/'))
        tar_name = f"{dir_basename}.tar"
    
    tar_path = os.path.join(os.path.dirname(input_dir), tar_name)
    
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Tar file: {tar_path}")
    print(f"Last step number: {args.last_num}")
    if args.max_tile_idx is not None:
        print(f"Max tile index: {args.max_tile_idx}")
    print("-" * 50)
    
    # Extract images
    success = extract_images(input_dir, output_dir, args.last_num, args.max_tile_idx)
    
    if not success:
        print("No files were extracted. Exiting.")
        sys.exit(1)
    
    # Create tar archive
    if create_tar_archive(output_dir, tar_path):
        print(f"Successfully created: {tar_path}")
        
        # Clean up extracted directory unless --keep_extracted is specified
        if not args.keep_extracted:
            shutil.rmtree(output_dir)
            print(f"Cleaned up temporary directory: {output_dir}")
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()
