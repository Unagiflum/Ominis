
import sys
import numpy as np
from collections import deque
import torch

# Mock the specific parts we need from agent.py
from agent import MonteCarloAgent

class MockAgent(MonteCarloAgent):
    def __init__(self, train_params):
        super().__init__(train_params)
        
    def replay(self):
        pass

def test_output():
    train_params = {
        'epsilon_min_percent': 5,
        'learning_rate': 0.001,
        'hl_size_idx': 0,
        'hl_count': 1
    }
    
    agent = MockAgent(train_params)
    agent.batch_size = 0
    
    # 1. Normal Case
    # Window: 5000 moves, 20 lines, 4 game overs
    agent.total_samples_since_train = 5000
    agent.lines_since_train = 20
    agent.gameovers_since_train = 4
    agent.inference_moves_since_train = 5000
    
    with open("verify_result.txt", "w") as f:
        sys.stdout = f
        
        print("--- Test 1 (Initial) ---")
        agent._maybe_train()
        # Expect: Moves/Line = 250.0 (ave: 250.0); Lines/Game = 5.0 (ave: 5.0)

        # 2. Zero-lines case (Agent surviving but not clearing)
        # Window: 5000 moves, 0 lines, 0 game overs
        # Moves/Line: N/A, Lines/Game: N/A
        # Average should absorb this: 
        # Total moves: 10000; Total lines: 20 -> Ave MPL: 500.0
        # Total gameovers: 4 -> Ave LPG: 20/4 = 5.0
        agent.total_samples_since_train = 5000
        agent.lines_since_train = 0
        agent.gameovers_since_train = 0
        agent.inference_moves_since_train = 5000
        
        print("\n--- Test 2 (Zero Lines/GO) ---")
        agent._maybe_train()
        # Expect: 
        # M/L: N/A (ave: 500.0) -> (5000+5000) / (20+0)
        # L/G: N/A (ave: 5.000) -> (20+0) / (4+0)
        
        # 3. Third Case - Poor performance
        # Window: 5000 moves, 10 lines, 10 game overs
        agent.total_samples_since_train = 5000
        agent.lines_since_train = 10
        agent.gameovers_since_train = 10
        agent.inference_moves_since_train = 5000
        
        print("\n--- Test 3 (Poor Performance) ---")
        agent._maybe_train()
        # Expect:
        # Agg: Moves=15000, Lines=30, GO=14
        # M/L: 500.0 (ave: 500.0) -> 15000/30
        # L/G: 1.0 (ave: 2.143) -> 30/14
    
if __name__ == "__main__":
    test_output()
