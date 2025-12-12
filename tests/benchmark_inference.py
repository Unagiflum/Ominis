import sys
import os
import time
import torch
import numpy as np

# Add parent directory to path to import model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import OminisNet

def benchmark_config(hidden_size, hidden_count, device, iterations=1000):
    print(f"\nBenchmarking Configuration: Hidden Size={hidden_size}, Hidden Count={hidden_count}")
    
    try:
        model = OminisNet(hidden_size=hidden_size, hidden_count=hidden_count).to(device)
        model.eval()
    except Exception as e:
        print(f"Failed to create model: {e}")
        return None

    # Dummy inputs matching agent.py get_state
    # Grid: (1, 3, 34, 12)
    grid_input = torch.randn(1, 3, 34, 12).to(device)
    # Next piece: (1, 100)
    next_piece_input = torch.randn(1, 100).to(device)

    # Warmup
    print("Warming up...")
    with torch.no_grad():
        for _ in range(100):
            _ = model(grid_input, next_piece_input)
            
    if device.type == 'cuda':
        torch.cuda.synchronize()

    print(f"Running {iterations} iterations...")
    times = []
    
    with torch.no_grad():
        start_global = time.perf_counter()
        for _ in range(iterations):
            t0 = time.perf_counter()
            _ = model(grid_input, next_piece_input)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000) # Convert to ms
            
        end_global = time.perf_counter()

    avg_time = np.mean(times)
    max_time = np.max(times)
    min_time = np.min(times)
    p95_time = np.percentile(times, 95)
    p99_time = np.percentile(times, 99)
    
    print(f"Results (ms): Avg={avg_time:.4f}, Min={min_time:.4f}, Max={max_time:.4f}, P95={p95_time:.4f}, P99={p99_time:.4f}")
    
    return {
        "avg": avg_time,
        "max": max_time,
        "p99": p99_time
    }

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Configurations to test
    # (hidden_size, hidden_count)
    configs = [
        # Current like configs (guessing roughly based on agent.py [128, 256, 512])
        (128, 2),
        (256, 2),
        (512, 2),
        
        # Scaling up - Deeper
        (512, 4),
        (512, 8),
        
        # Scaling up - Wider
        (1024, 2),
        (1024, 4),
        (2048, 2),
        (2048, 4),
        
        # Large
        (4096, 2),
        (4096, 4)
    ]

    print("-" * 60)
    
    passed_configs = []

    for size, count in configs:
        result = benchmark_config(size, count, device)
        if result:
            # Check if it passes the 50ms budget (using P99 for safety)
            if result['p99'] < 50.0:
                 passed_configs.append(f"Size={size}, Count={count}: {result['p99']:.2f}ms (P99)")
            else:
                 print(f"--> EXCEEDED BUDGET: Size={size}, Count={count}")

    print("\n" + "=" * 60)
    print("Configurations under 50ms (P99):")
    
    with open("benchmark_final_results.txt", "w") as f:
        f.write("Configurations under 50ms (P99):\n")
        for msg in passed_configs:
            print(msg)
            f.write(msg + "\n")
            
    print("=" * 60)
    print("Results written to benchmark_final_results.txt")

if __name__ == "__main__":
    main()
