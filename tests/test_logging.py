import sys
import os
import shutil

# Add parent directory to path to import agent
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import MonteCarloAgent

def test_logging():
    print("Testing logging...")
    
    # Setup params
    params = {
        'hl_size_idx': 1, # 256
        'hl_count': 2,
        'gamma': 0.99,
        'epsilon_min_percent': 10,
        'learning_rate': 0.001
    }
    
    # clean up previous test runs
    progress_dir = "progress"
    model_name = "model-256-2.csv"
    csv_path = os.path.join(progress_dir, model_name)
    
    if os.path.exists(csv_path):
        os.remove(csv_path)
    
    # 1. Initialize Agent and check file creation
    agent = MonteCarloAgent(params)
    
    if not os.path.exists(csv_path):
        print("FAIL: CSV file not created on init")
        return
        
    with open(csv_path, 'r') as f:
        header = f.readline().strip()
        if header != "Batch, Moves per Line, Lines per Game":
            print(f"FAIL: Incorrect header: {header}")
            return
    print("PASS: File created with correct header")
    
    # 2. Add some history and log
    agent.history.append((100, 10, 1)) # 10 MPL, 10 LPG
    agent.history.append((200, 20, 1)) # 10 MPL, 20 LPG (Total: 300, 30, 2 -> 10 MPL, 15 LPG)
    
    agent.training_steps = 1000
    agent._log_progress_to_csv()
    
    with open(csv_path, 'r') as f:
        lines = f.readlines()
        if len(lines) != 2:
            print(f"FAIL: Expected 2 lines, got {len(lines)}")
            return
        entry = lines[1].strip()
        print(f"Log Entry: {entry}")
        # Expect: 1000, 10.0, 15.000
        if "1000, 10.0, 15.000" not in entry:
             print("FAIL: Log entry content mismatch")
             return
             
    print("PASS: Logging correct calculations")
    
    # 3. Test Resume
    print("Testing resume...")
    agent2 = MonteCarloAgent(params)
    if agent2.training_steps != 1000:
        print(f"FAIL: Expected resume at 1000, got {agent2.training_steps}")
        return
    print("PASS: Resumed correctly")

    print("All tests passed!")

if __name__ == "__main__":
    test_logging()
