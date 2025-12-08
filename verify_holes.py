
import pygame
from game import Game

def verify_hole_definition():
    pygame.init()
    game = Game()
    
    # Setup a grid with 2 contiguous holes
    # Col 0:
    # Row 20: Block
    # Row 21: Empty
    # Row 22: Empty
    # Row 23: Block
    
    game.grid.grid = [[(0,0,0) for _ in range(game.grid_width)] for _ in range(game.grid_height)]
    
    game.grid.grid[20][0] = (100, 100, 100)
    game.grid.grid[23][0] = (100, 100, 100)
    
    # Check count
    _, holes = game.get_grid_stats()
    print(f"Stack: Block, Empty, Empty, Block")
    print(f"Holes counted: {holes}")
    
    # User definition: "immediately adjacent and below a block"
    # Row 21 is empty and below Block (20). -> Hole.
    # Row 22 is empty and below Empty (21). -> Not a hole directly below a block?
    # Wait, "immediately adjacent and below A block".
    # If the definition means checking each empty cell: "Is there a block at y-1?"
    # Then Row 22 is NOT a hole.
    # So expected count = 1.
    
    if holes == 2:
        print("FAIL: Still counts ALL voids (Total holes).")
    elif holes == 1:
        print("PASS: Counts only immediate voids.")
    else:
        print(f"FAIL: Unexpected count {holes}")

if __name__ == "__main__":
    verify_hole_definition()
