#!/usr/bin/env python3
"""
Test script to verify the new file naming convention
"""

def test_naming_convention():
    # Test the naming logic
    outfolder = "test_panorama"
    
    # Final file
    final_filename = f"{outfolder}_F.png"
    print(f"Final file: {final_filename}")
    
    # Intermediate files
    for step_idx in range(5):
        intermediate_filename = f"{outfolder}_{step_idx:03d}.png"
        print(f"Step {step_idx}: {intermediate_filename}")
    
    print("\nExpected output:")
    print("Final file: test_panorama_F.png")
    print("Step 0: test_panorama_000.png")
    print("Step 1: test_panorama_001.png")
    print("Step 2: test_panorama_002.png")
    print("Step 3: test_panorama_003.png") 
    print("Step 4: test_panorama_004.png")

if __name__ == "__main__":
    test_naming_convention()
