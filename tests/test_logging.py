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
        'hl_size_idx': 4, # 256
        'hl_count': 2,
        'gamma': 0.99,
        'epsilon_min_percent': 10,
        'learning_rate_start': 0.001,
        'learning_rate_end': 0.001,
        'learning_rate_current': 0.001
    }
    
    # Clean up
    progress_dir = "progress"
    model_name = "model-256-2.csv"
    csv_path = os.path.join(progress_dir, model_name)
    model_file = "models/model-256-2.pth"
    
    if os.path.exists(csv_path):
        os.remove(csv_path)
    if os.path.exists(model_file):
        os.remove(model_file)
    
    # Ensure models dir exists
    if not os.path.exists("models"):
        os.makedirs("models")

    # 1. Initialize Agent (Fresh Start)
    print("\n--- Test 1: Fresh Start ---")
    agent = MonteCarloAgent(params)
    
    if not os.path.exists(csv_path):
        print("FAIL: CSV file not created on init")
        return
        
    with open(csv_path, 'r') as f:
        header = f.readline().strip()
        lines = f.readlines()
        if header != "Batch, Lines per Piece, Lines per Game":
            print(f"FAIL: Incorrect header: {header}")
            return
        if len(lines) > 0:
             print("FAIL: Expected empty file (header only)")
             return
    print("PASS: File created with correct header")
    
    # 2. Add some history and log
    print("\n--- Test 2: Logging Data ---")
    agent.history.append((100, 20, 10, 1))
    agent.training_steps = 1000
    agent._log_progress_to_csv()
    
    with open(csv_path, 'r') as f:
        lines = f.readlines()
        if len(lines) != 2:
            print(f"FAIL: Expected 2 lines, got {len(lines)}")
            return
        entry = lines[1].strip()
        print(f"Log Entry: {entry}")
        if "1000, 0.5000, 10.000" not in entry:
             print("FAIL: Log entry content mismatch")
             return
    print("PASS: Logging correct")
    
    # 3. Simulate Model Save (Create dummy .pth)
    print("\n--- Test 3: Resume with Model ---")
    with open(model_file, 'w') as f:
        f.write("dummy model content")
        
    agent2 = MonteCarloAgent(params)
    if agent2.training_steps != 1000:
        print(f"FAIL: Expected resume at 1000, got {agent2.training_steps}")
        return
    print("PASS: Resumed correctly with model file present")

    # 4. Simulate Fresh Start (Model deleted, CSV remains)
    print("\n--- Test 4: Wipe on Missing Model ---")
    os.remove(model_file)
    # CSV still has data from Test 2
    
    agent3 = MonteCarloAgent(params)
    if agent3.training_steps != 0:
         print(f"FAIL: Expected training_steps reset to 0, got {agent3.training_steps}")
         return
         
    with open(csv_path, 'r') as f:
        lines = f.readlines()
        if len(lines) != 1:
             print(f"FAIL: Expected CSV wipe (1 line header), got {len(lines)} lines")
             print(f"Content: {lines}")
             return
    print("PASS: CSV wiped correctly when model missing")

    print("\nAll tests passed!")

if __name__ == "__main__":
    test_logging()
