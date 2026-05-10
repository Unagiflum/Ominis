import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agent import MonteCarloAgent


def test_floor_restart_bumps_epsilon_and_learning_rate_after_delay():
    params = {
        "hl_size_idx": 0,
        "hl_count": 1,
        "epsilon_min": 0.01,
        "epsilon_start": 0.01,
        "learning_rate_start": 0.001,
        "learning_rate_end": 0.0001,
        "learning_rate_current": 0.0001,
    }

    csv_path = os.path.join("progress", "model-16-1.csv")
    csv_existed = os.path.exists(csv_path)

    try:
        agent = MonteCarloAgent(params)
        agent.floor_restart_batches_at_floor = agent.FLOOR_RESTART_BATCHES - 1

        agent._apply_floor_restart_if_due()

        expected_epsilon = params["epsilon_min"] * agent.FLOOR_RESTART_MULTIPLIER
        expected_lr = params["learning_rate_end"] * agent.FLOOR_RESTART_MULTIPLIER

        assert agent.epsilon == expected_epsilon
        assert agent.learning_rate == expected_lr
        assert agent.optimizer.param_groups[0]["lr"] == expected_lr
        assert agent.floor_restart_batches_at_floor == 0
    finally:
        if not csv_existed and os.path.exists(csv_path):
            os.remove(csv_path)


def test_floor_restart_clamps_zero_epsilon_min_to_configured_floor():
    params = {
        "hl_size_idx": 0,
        "hl_count": 1,
        "epsilon_min": 0.0,
        "epsilon_start": 0.0,
        "learning_rate_start": 0.001,
        "learning_rate_end": 0.0001,
        "learning_rate_current": 0.0001,
    }

    csv_path = os.path.join("progress", "model-16-1.csv")
    csv_existed = os.path.exists(csv_path)

    try:
        agent = MonteCarloAgent(params)
        assert agent.epsilon_min == agent.EPSILON_MIN_NONZERO
        assert agent.epsilon == agent.EPSILON_MIN_NONZERO

        agent.floor_restart_batches_at_floor = agent.FLOOR_RESTART_BATCHES - 1

        agent._apply_floor_restart_if_due()

        expected_epsilon = agent.EPSILON_MIN_NONZERO * agent.FLOOR_RESTART_MULTIPLIER
        assert agent.epsilon == expected_epsilon
    finally:
        if not csv_existed and os.path.exists(csv_path):
            os.remove(csv_path)


if __name__ == "__main__":
    test_floor_restart_bumps_epsilon_and_learning_rate_after_delay()
    test_floor_restart_clamps_zero_epsilon_min_to_configured_floor()
    print("PASS: floor restart bumps epsilon and learning rate")
