
import pygame
from game import Game
from grid import Grid
from tetrominoes import Pentomino

# Mock the agent to access verify logic if needed, 
# but mostly we need to run the logic that is inside game.step_ai_training
# The logic is embedded in step_ai_training, so we might need to extract it or simulate it.

def check_blocks_over_holes_logic():
    print("Initializing Game...")
    pygame.init() # Needed for game init
    game = Game()
    
    # Setup a grid with a deep hole
    # Column 0:
    # Row 23: Block
    # Row 22: Empty (Hole)
    # Row 21: Block
    # ...
    
    print("Setting up grid with a deep hole at (0, 22)...")
    game.grid.grid = [[(0,0,0) for _ in range(game.grid_width)] for _ in range(game.grid_height)]
    
    # Ground at bottom
    for x in range(game.grid_width):
        game.grid.grid[23][x] = (100, 100, 100)
        
    # Create a hole in col 0
    game.grid.grid[22][0] = (0, 0, 0) # Hole
    game.grid.grid[21][0] = (100, 100, 100) # Cap
    
    # Verify hole count
    h, holes = game.get_grid_stats()
    print(f"Initial Holes: {holes}")
    # Should be 1 hole at (0, 22) because (0, 21) is block.
    
    # Place a piece on TOP of column 0
    # Say at row 15
    print("Placing a piece at (0, 15)...")
    # We can just simulate the logic from game.py
    
    # Mock piece placement
    # Let's say we placed a single block at (0, 15)
    # Just manual logic check since the code in game.py is:
    # for px, py in piece.shape...
    
    # Let's create a dummy piece object
    class DummyPiece:
        def __init__(self, x, y):
            self.x = x
            self.y = y
            self.shape = [(0,0)] # Single block
            
    piece_before = DummyPiece(0, 15)
    
    # Run the logic from game.py (Step Id 14, Lines 913-925)
    # New logic: Calculate net holes using get_post_clear_grid_stats
    
    # 1. Determine clearing lines (None in this case)
    current_clearing_lines = []
    # (Simple check for full lines - none in this setup)
    
    # 2. Get holes before
    h_before, holes_before = game.get_grid_stats()
    
    # 3. Apply the piece temporarily to the grid to check 'after' state
    # Simplified application for this test
    for px, py in piece_before.shape:
        game.grid.grid[piece_before.y + py][piece_before.x + px] = (255, 0, 0)
        
    # 4. Get holes after (simulated post-clear)
    _, holes_final = game.get_post_clear_grid_stats(current_clearing_lines)
    
    net_holes_created = (holes_final > holes_before)

    print(f"holes_before: {holes_before}")
    print(f"holes_final: {holes_final}")
    print(f"net_holes_created: {net_holes_created}")
    
    if net_holes_created:
        print("FAIL: Flagged as creating holes despite only having a deep existing hole.")
    else:
        print("PASS: Correctly ignored deep hole.")

if __name__ == "__main__":
    check_blocks_over_holes_logic()
