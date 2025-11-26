import pygame
from game import Game

pygame.init()
pygame.font.init()
screen = pygame.display.set_mode((1280, 720))

game = Game()
game.screen = screen
game.screen_width = 1280
game.screen_height = 720

print(f"Initial State: {game.state}")
print(f"Agent: {game.agent}")

# Simulate clicking Start in Train Menu
game.state = "TRAIN_MENU"
game.train_params['visual_mode'] = True # Visual mode ON

print("--- Clicking Start (Visual Mode) ---")
# Manually trigger the start logic
game.reset()
from agent import MonteCarloAgent
game.agent = MonteCarloAgent(game.train_params)
game.state = "TRAINING"

print(f"State after start: {game.state}")
print(f"Agent after start: {game.agent}")

# Simulate one update loop
print("--- Simulating Update ---")
game.update()
print(f"State after update: {game.state}")

# Check if reset clears agent?
print("--- Calling Reset ---")
game.reset()
print(f"State after reset: {game.state}")
print(f"Agent after reset: {game.agent}")
