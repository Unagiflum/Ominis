
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tetrominoes import get_allowed_shapes, SHAPES

def test_frequencies():
    print("Testing Weighted Frequencies (Training Mode)...")
    weight_base = 2
    allowed_weighted = get_allowed_shapes(include_pentominoes=True, include_tetrominoes=True, include_ominis=True, max_size=5, weighted=True, weight_base=weight_base)
    
    counts = {}
    for shape in allowed_weighted:
        size = len(SHAPES[shape])
        counts[size] = counts.get(size, 0) + 1
        
    print(f"Counts per size: {counts}")
    
    # Validation
    # Monomino (1 shape) * 2^1 = 2
    # Domino (1 shape) * 2^2 = 4
    # Triomino (2 shapes) * 2^3 = 16
    # Tetromino (7 shapes) * 2^4 = 112
    # Pentomino (18 shapes) * 2^5 = 576  (12 pentominoes + 6 mirrors? Let's check SHAPES length)
    
    # Check shape counts in SHAPES
    shape_counts_by_size = {}
    for shape, blocks in SHAPES.items():
        s = len(blocks)
        shape_counts_by_size[s] = shape_counts_by_size.get(s, 0) + 1
        
    print(f"Unique shapes available per size: {shape_counts_by_size}")
    
    expected_counts = {}
    for size, count in shape_counts_by_size.items():
        if size <= 5:
            expected_counts[size] = count * (weight_base ** size)
            
    print(f"Expected logical counts: {expected_counts}")
    
    # Assert
    for size in expected_counts:
        if counts.get(size, 0) != expected_counts[size]:
            print(f"FAIL: Size {size} count mismatch. Got {counts.get(size, 0)}, expected {expected_counts[size]}")
        else:
            print(f"PASS: Size {size} count is correct ({counts.get(size, 0)})")
            
    # Verify unweighted
    print("\nTesting Unweighted (Standard Mode)...")
    allowed_unweighted = get_allowed_shapes(include_pentominoes=True, include_tetrominoes=True, include_ominis=True, max_size=5, weighted=False)
    unweighted_counts = {}
    for shape in allowed_unweighted:
        size = len(SHAPES[shape])
        unweighted_counts[size] = unweighted_counts.get(size, 0) + 1
        
    for size in shape_counts_by_size:
        if size <= 5:
            if unweighted_counts.get(size, 0) != shape_counts_by_size[size]:
                 print(f"FAIL: Unweighted size {size} count mismatch by size. Got {unweighted_counts.get(size, 0)}, expected {shape_counts_by_size[size]}")
            else:
                 print(f"PASS: Unweighted size {size} matches unique shape count.")

if __name__ == "__main__":
    test_frequencies()
