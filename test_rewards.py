"""
Test script to verify reward calculations.
This helps diagnose if there's an issue with the reward logic.
"""

# Simulate different scenarios
test_params = {
    'height_penalty': 50,
    'overhang_penalty': 50
}

def calculate_reward_test(lines_cleared, current_height, start_height, current_holes, start_holes, game_over, piece_in_upper_half):
    reward = 0
    
    # 1. Line Clears
    if lines_cleared > 0:
        reward += (lines_cleared ** 2) * 100
        
    # 2. Game Over Penalty
    if game_over:
        reward -= 500
        
    # 3. Height Change
    height_change = current_height - start_height
    if height_change > 0:
        h_factor = test_params['height_penalty'] / 100.0
        reward -= height_change * h_factor * 10
        
    # 4. Holes Change
    holes_change = current_holes - start_holes
    if holes_change > 0:
        o_factor = test_params['overhang_penalty'] / 100.0
        reward -= holes_change * o_factor * 20
        
    # 5. High Stacking Penalty
    if piece_in_upper_half:
        reward -= 10
        
    return reward

# Test scenarios
print("=" * 60)
print("REWARD CALCULATION TESTS")
print("=" * 60)

print("\nScenario 1: Clear 1 line, height stays same")
reward = calculate_reward_test(
    lines_cleared=1, current_height=10, start_height=10,
    current_holes=0, start_holes=0, game_over=False, piece_in_upper_half=False
)
print(f"Reward: {reward} (Expected: +100)")

print("\nScenario 2: Clear 4 lines, height decreases by 4")
reward = calculate_reward_test(
    lines_cleared=4, current_height=10, start_height=14,
    current_holes=0, start_holes=0, game_over=False, piece_in_upper_half=False
)
print(f"Reward: {reward} (Expected: +1600, no penalty for decrease)")

print("\nScenario 3: Place piece, height increases by 3, no lines cleared")
reward = calculate_reward_test(
    lines_cleared=0, current_height=13, start_height=10,
    current_holes=0, start_holes=0, game_over=False, piece_in_upper_half=False
)
print(f"Reward: {reward} (Expected: -15)")

print("\nScenario 4: Stack high (upper half), height +3, create 2 holes")
reward = calculate_reward_test(
    lines_cleared=0, current_height=13, start_height=10,
    current_holes=2, start_holes=0, game_over=False, piece_in_upper_half=True
)
print(f"Reward: {reward} (Expected: -15 (height) -20 (holes) -10 (upper half) = -45)")

print("\nScenario 5: Clear 1 line in upper half, creates no holes")
reward = calculate_reward_test(
    lines_cleared=1, current_height=10, start_height=11,
    current_holes=0, start_holes=0, game_over=False, piece_in_upper_half=True
)
print(f"Reward: {reward} (Expected: +100 -10 = +90)")

print("\nScenario 6: Game Over")
reward = calculate_reward_test(
    lines_cleared=0, current_height=24, start_height=20,
    current_holes=0, start_holes=0, game_over=True, piece_in_upper_half=True
)
print(f"Reward: {reward} (Expected: -500 -20 -10 = -530)")

print("\n" + "=" * 60)
print("\nKey Insight:")
print("- Clearing lines gives BIG positive rewards (100-1600)")
print("- Stacking high WITHOUT clearing only gives small penalties (-15 to -45)")
print("- If the agent clears lines while stacked high, it still gets +90 to +1590")
print("\n⚠️  POTENTIAL ISSUE: Agent may learn that stacking high is OK")
print("   as long as it clears lines occasionally!")
print("=" * 60)
