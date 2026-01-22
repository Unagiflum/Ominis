import os
import sys
import tempfile
from types import SimpleNamespace

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from grid import Grid
from tetrominoes import Pentomino, SHAPES


def test_action_mask():
    try:
        from agent import MonteCarloAgent
    except ModuleNotFoundError as e:
        if getattr(e, "name", None) == "torch":
            print("SKIP: torch is not installed; action mask test requires torch for agent import")
            return
        raise

    params = {
        "hl_size_idx": 0,  # 16
        "hl_count": 1,
        "gamma": 0.7,
        "epsilon_min_percent": 5,
        "learning_rate_start": 0.0001,
        "learning_rate_end": 0.0001,
        "learning_rate_current": 0.0001,
    }

    with tempfile.TemporaryDirectory() as tmp:
        old_cwd = os.getcwd()
        os.chdir(tmp)
        try:
            agent = MonteCarloAgent(params)

            grid = Grid(width=12, height=24, cell_size=1)

            # Place a vertical domino at (5, 5) and block (6, 5) so RIGHT and ROTATE_CW are invalid.
            piece = Pentomino(x=5, y=5, allowed_shapes=["Domino"])
            piece.shape = list(SHAPES["Domino"])
            piece.rotation = 0

            grid.grid[5][6] = (255, 0, 0)
            game = SimpleNamespace(current_piece=piece, grid=grid)

            mask = agent.get_action_mask(game)
            if len(mask) != 9:
                print(f"FAIL: Expected mask length 9, got {len(mask)}")
                return

            stay_stay = agent.encode_action(1, 1)
            left_stay = agent.encode_action(0, 1)
            right_stay = agent.encode_action(2, 1)
            stay_ccw = agent.encode_action(1, 0)
            stay_cw = agent.encode_action(1, 2)
            left_cw = agent.encode_action(0, 2)
            right_ccw = agent.encode_action(2, 0)

            checks = [
                ("stay_stay", stay_stay, True),
                ("left_stay", left_stay, True),
                ("stay_ccw", stay_ccw, True),
                ("right_stay", right_stay, False),
                ("stay_cw", stay_cw, False),
                ("left_cw", left_cw, True),
                ("right_ccw", right_ccw, False),
            ]

            for name, idx, expected in checks:
                actual = bool(mask[idx])
                if actual != expected:
                    print(f"FAIL: {name} (idx={idx}) expected {expected}, got {actual}")
                    return

            print("PASS: Action masking behaves as expected")
        finally:
            os.chdir(old_cwd)


if __name__ == "__main__":
    test_action_mask()
