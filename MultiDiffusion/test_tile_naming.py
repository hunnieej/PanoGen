#!/usr/bin/env python3
"""
Test script to verify the tile naming convention
"""

def test_tile_naming():
    outfolder = "test_output"
    num_tiles = 5
    num_steps = 3
    
    print("Expected tile filenames:")
    for step_idx in range(num_steps):
        print(f"\nStep {step_idx}:")
        for tile_idx in range(num_tiles):
            filename = f"tile_{tile_idx:03d}_{step_idx:03d}.png"
            print(f"  {filename}")
    
    print(f"\nTotal files: {num_tiles * num_steps}")
    print(f"Final file: final.png")

if __name__ == "__main__":
    test_tile_naming()
