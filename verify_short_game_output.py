import sys
from unittest.mock import MagicMock

# Mock torch and agent before importing game
sys.modules['torch'] = MagicMock()
sys.modules['torch.nn'] = MagicMock()
sys.modules['torch.optim'] = MagicMock()
sys.modules['agent'] = MagicMock()

import pygame
from game import Game
import io

# Mock stdout to capture print statements
class Capturing(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = io.StringIO()
        return self
    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio
        sys.stdout = self._stdout

def test_short_game_output():
    pygame.init()
    game = Game()
    
    # Setup for Short Game Training
    game.train_params['short_games'] = True
    game.train_params['visual_mode'] = False
    game.train_params['pieces_tracked'] = 2 
    
    # Initialize Agent (Mocked)
    game.agent = MagicMock()
    game.state = "TRAINING"
    # We don't call reset() fully because it might try to use real agent logic if we aren't careful,
    # but with mocked agent module it should be fine.
    # Actually, Game imports MonteCarloAgent from agent.
    # Since we mocked sys.modules['agent'], Game.agent will be a Mock.
    
    print("Test 1: Short Game with 0 lines (Should print NOTHING)")
    with Capturing() as output:
        game.lines_cleared_total = 0
        game.finish_training_round()
        
    if not output:
        print("PASS: No output for 0 lines.")
    else:
        print(f"FAIL: Output was: {output}")

    print("\nTest 2: Short Game with >0 lines (Should print 'Short game...')")
    with Capturing() as output:
        game.lines_cleared_total = 5
        game.finish_training_round()
        
    expected = "Short game, Lines: 5"
    if output and output[0] == expected:
        print(f"PASS: Output matched expected: '{expected}'")
    else:
        print(f"FAIL: Output was: {output}, Expected: '{expected}'")

    print("\nTest 3: Normal Game (Should print 'Game Over...')")
    game.train_params['short_games'] = False
    with Capturing() as output:
        game.lines_cleared_total = 10
        game.pieces_locked = 50
        game.finish_training_round()
        
    expected_prefix = "Game Over, Pieces: 50, Lines: 10"
    if output and output[0] == expected_prefix:
        print(f"PASS: Output matched expected: '{expected_prefix}'")
    else:
        print(f"FAIL: Output was: {output}, Expected: '{expected_prefix}'")

    pygame.quit()

if __name__ == "__main__":
    test_short_game_output()
