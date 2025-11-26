import pygame
from game import Game
from agent import MonteCarloAgent
import sys

# Initialize Pygame headless-ish
pygame.init()
pygame.font.init()
screen = pygame.display.set_mode((100, 100)) # Tiny window

print("Initializing Game...")
game = Game()
game.screen = screen

# Setup Agent
print("Setting up Agent...")
game.agent = MonteCarloAgent(game.train_params)
game.state = "TRAINING"

# 1. Test Trajectory Clearing
print("\n--- Test 1: Trajectory Clearing ---")
game.reset()
print(f"Trajectory after reset (should be empty): {len(game.current_trajectory)}")

# Manually add dummy data
game.current_trajectory.append(("state", "action", "next_state"))
print(f"Trajectory after manual add: {len(game.current_trajectory)}")

# Reset again
game.reset()
print(f"Trajectory after second reset (should be empty): {len(game.current_trajectory)}")

if len(game.current_trajectory) == 0:
    print("PASS: Trajectory cleared successfully.")
else:
    print("FAIL: Trajectory NOT cleared.")

# 2. Test Game Loop with Agent
print("\n--- Test 2: Game Loop Execution ---")
try:
    # Run a few updates
    for i in range(10):
        game.update()
        # Force a step if needed (since update depends on time)
        game.step_ai_training()
        
    print("PASS: Game loop ran for 10 steps without crashing.")
except Exception as e:
    print(f"FAIL: Game loop crashed: {e}")
    import traceback
    traceback.print_exc()

pygame.quit()
sys.exit()
