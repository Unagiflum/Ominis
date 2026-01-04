import torch
import torch.optim as optim
import torch.nn.functional as F
import copy
import random
import numpy as np
import math
from collections import deque
from model import OminisNet

class MonteCarloAgent:
    """
    Agent using Monte Carlo supervised learning.
    
    Instead of TD bootstrapping with a target network, this agent:
    1. Collects full trajectories for each falling piece
    2. Computes a single scalar Monte Carlo return R_piece at the end
    3. Trains Q(s, a) to predict R_piece for all (state, action) pairs in the trajectory
    """
    EPSILON_MAX_PERCENT = 25.0
    EPSILON_STEP_PERCENT = 0.5
    EPSILON_HALF_LIFE_MIN_EXP = 2.0
    EPSILON_HALF_LIFE_MAX_EXP = 7.0
    EPSILON_HALF_LIFE_STEP_EXP = 0.5
    EPSILON_HALF_LIFE_STEPS = int((EPSILON_HALF_LIFE_MAX_EXP - EPSILON_HALF_LIFE_MIN_EXP) / EPSILON_HALF_LIFE_STEP_EXP)

    def __init__(self, train_params):
        self.params = train_params
        
        # Hyperparameters
        self.batch_size = 5000 # Batch size for training (doubled with mirroring)
        self.gamma = 1.0 # Discount factor (fixed)
        self.params['gamma'] = 1.0
        self.epsilon = 0.7 # Initial exploration rate
        epsilon_min_percent = self._snap_epsilon_percent(self.params.get('epsilon_min_percent', 5))
        self.params['epsilon_min_percent'] = epsilon_min_percent
        self.epsilon_min = epsilon_min_percent / 100.0 # Minimum exploration rate (Default 5%)
        epsilon_current_percent = self.params.get('epsilon_current_percent')
        if epsilon_current_percent is not None:
            epsilon_current_percent = self._snap_epsilon_percent(epsilon_current_percent)
            self.params['epsilon_current_percent'] = epsilon_current_percent
            self.epsilon = epsilon_current_percent / 100.0
        self.epsilon = max(self.epsilon, self.epsilon_min)
        half_life = self._snap_epsilon_half_life(self.params.get('epsilon_half_life_batches', 10 ** 4.5))
        self.params['epsilon_half_life_batches'] = half_life
        self.epsilon_decay = self._compute_epsilon_decay(half_life)
        lr = self.params.get('learning_rate', 0.0001)
        # Clamp to UI-supported range
        lr = max(0.0001, min(0.005, lr))
        self.learning_rate = lr
        self.memory = deque(maxlen=10000) # Single replay memory for all trajectories
        self.total_samples_since_train = 0
        self.train_trigger_interval = 5000  # Train more frequently with smaller batches
        self.lines_since_train = 0
        self.gameovers_since_train = 0
        self.inference_moves_since_train = 0
        self.pieces_since_train = 0
        
        # Benchmarking history
        # Stores tuples of (inference_moves, pieces_locked, lines_cleared, game_overs)
        self.history = deque(maxlen=100)
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Model
        self.HL_SIZES = [128, 256, 512, 1024, 2048]
        hl_size_idx = int(self.params.get('hl_size_idx', 1))
        hl_size_idx = max(0, min(len(self.HL_SIZES) - 1, hl_size_idx))
        # Keep the shared params in-range so UI/agent stay aligned
        self.params['hl_size_idx'] = hl_size_idx
        hidden_size = self.HL_SIZES[hl_size_idx]
        hidden_count = self.params['hl_count']
        
        # Main network (updated every training step with Monte Carlo targets)
        self.model = OminisNet(hidden_size=hidden_size, hidden_count=hidden_count).to(self.device)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Action Space: 9 joint actions (3 lateral × 3 rotation)
        # Lateral: 0=Left, 1=Stay, 2=Right
        # Rotation: 0=CCW, 1=Stay, 2=CW 
        # Encoding: action_idx = lateral_idx * 3 + rotation_idx
        # Encoding: action_idx = lateral_idx * 3 + rotation_idx
        self.num_actions = 9

        # Logging initialization
        self.training_steps = 0
        self._init_logging()

    def _init_logging(self):
        """Initialize CSV logging for training progress."""
        import os
        
        # Determine filename
        # Model name format: model-{size}-{count}.pth
        # We want: model-{size}-{count}.csv
        hidden_size = self.HL_SIZES[self.params['hl_size_idx']]
        hidden_count = self.params['hl_count']
        model_name = f"model-{hidden_size}-{hidden_count}"
        
        # Ensure progress directory exists
        progress_dir = "progress"
        if not os.path.exists(progress_dir):
            try:
                os.makedirs(progress_dir)
            except OSError as e:
                print(f"Error creating progress directory: {e}")
                return
                
        self.csv_path = os.path.join(progress_dir, f"{model_name}.csv")
        
        # Check for model existence to determine if we should resume or wipe
        # Model path is assumed to be in "models/" folder relative to CWD, consistent with game.py
        model_path = os.path.join("models", f"{model_name}.pth")
        
        resume_logging = False
        if os.path.exists(self.csv_path) and os.path.exists(model_path):
            resume_logging = True
            
        if resume_logging:
            try:
                with open(self.csv_path, 'r') as f:
                    lines = f.readlines()
                    if len(lines) > 1:
                        last_line = lines[-1].strip()
                        if last_line:
                            try:
                                # Format: Batch, Lines per Piece, Lines per Game
                                # 1000, 0.0600, 4.235
                                last_batch = int(last_line.split(',')[0])
                                self.training_steps = last_batch
                                print(f"Resuming logging from batch {self.training_steps}")
                            except ValueError:
                                print("Could not parse last batch from CSV, starting at 0")
            except Exception as e:
                print(f"Error reading existing CSV: {e}")
        else:
            # Create new file with header (wiping if exists)
            try:
                with open(self.csv_path, 'w') as f:
                    f.write("Batch, Lines per Piece, Lines per Game\n")
            except Exception as e:
                print(f"Error creating CSV file: {e}")

    def _snap_epsilon_percent(self, value):
        value = max(0.0, min(self.EPSILON_MAX_PERCENT, float(value)))
        step_count = int(round(value / self.EPSILON_STEP_PERCENT))
        snapped = step_count * self.EPSILON_STEP_PERCENT
        return max(0.0, min(self.EPSILON_MAX_PERCENT, snapped))

    def _snap_epsilon_half_life(self, value):
        default_exp = 4.5
        try:
            value = float(value)
        except (TypeError, ValueError):
            value = 10 ** default_exp
        min_value = 10 ** self.EPSILON_HALF_LIFE_MIN_EXP
        max_value = 10 ** self.EPSILON_HALF_LIFE_MAX_EXP
        value = max(min_value, min(max_value, value))
        exp = math.log10(value)
        exp = max(self.EPSILON_HALF_LIFE_MIN_EXP, min(self.EPSILON_HALF_LIFE_MAX_EXP, exp))
        step_count = int(round((exp - self.EPSILON_HALF_LIFE_MIN_EXP) / self.EPSILON_HALF_LIFE_STEP_EXP))
        step_count = max(0, min(self.EPSILON_HALF_LIFE_STEPS, step_count))
        snapped_exp = self.EPSILON_HALF_LIFE_MIN_EXP + step_count * self.EPSILON_HALF_LIFE_STEP_EXP
        return int(round(10 ** snapped_exp))

    def _compute_epsilon_decay(self, half_life_batches):
        half_life_batches = max(1.0, float(half_life_batches))
        return 10 ** (math.log10(0.5) / half_life_batches)

    def get_state(self, game):
        buffer_rows = 10

        # 1. Board Channel (locked blocks only)
        # Cache the locked board until the grid changes (piece locks / line clears / reset).
        board_key = (id(game.grid), getattr(game, "pieces_locked", 0), getattr(game, "lines_cleared_total", 0))
        if getattr(self, "_cached_board_key", None) != board_key:
            board = np.zeros((34, 12), dtype=np.float32)
            for y in range(24):
                for x in range(12):
                    if game.grid.grid[y][x] != (0, 0, 0):
                        board[y + buffer_rows][x] = 1.0
            self._cached_board_key = board_key
            self._cached_board = board
        else:
            board = self._cached_board
                    
        # 2. Piece Channel
        piece_grid = np.zeros((34, 12), dtype=np.float32)
        if game.current_piece:
            for x, y in game.current_piece.shape:
                # Piece y is relative to board top (0).
                # So in our 34-height grid, it is y + 10.
                # Note: piece.y can be negative (spawning).
                py = game.current_piece.y + y + buffer_rows
                px = game.current_piece.x + x
                if 0 <= py < 34 and 0 <= px < 12:
                    piece_grid[py][px] = 1.0
                    
        # 3. Ghost Channel
        ghost_grid = np.zeros((34, 12), dtype=np.float32)
        ghost = game.get_ghost_piece()
        if ghost:
            for x, y in ghost.shape:
                py = ghost.y + y + buffer_rows
                px = ghost.x + x
                if 0 <= py < 34 and 0 <= px < 12:
                    ghost_grid[py][px] = 1.0
                    
        # Stack channels
        grid_input = np.stack([board, piece_grid, ghost_grid]) # (3, 34, 12)
        
        # 4. Next Piece (Flattened 10x10)
        # Cache until next_piece changes (only updates when a piece locks/spawns or on reset).
        next_piece_id = id(getattr(game, "next_piece", None))
        if getattr(self, "_cached_next_piece_id", None) != next_piece_id:
            next_piece_vec = np.zeros(100, dtype=np.float32)
            if game.next_piece:
                for x, y in game.next_piece.shape:
                    nx = x + 5
                    ny = y + 5
                    if 0 <= nx < 10 and 0 <= ny < 10:
                        next_piece_vec[ny * 10 + nx] = 1.0
            self._cached_next_piece_id = next_piece_id
            self._cached_next_piece_vec = next_piece_vec
        else:
            next_piece_vec = self._cached_next_piece_vec
                    
        return grid_input, next_piece_vec

    def get_action_mask(self, game):
        """
        Compute a boolean mask of valid joint actions for the current game state.

        An action is considered valid if executing its lateral/rotation commands would
        actually produce that same (lateral_idx, rotation_idx) outcome (i.e., no
        collision/no-op for a commanded move).
        """
        mask = np.zeros(self.num_actions, dtype=np.bool_)

        # Always allow "do nothing" (stay + stay)
        stay_action = self.encode_action(1, 1)
        mask[stay_action] = True

        if not getattr(game, "current_piece", None):
            return mask

        for action_idx in range(self.num_actions):
            lat_idx, rot_idx = self.decode_action(action_idx)
            test_piece = copy.deepcopy(game.current_piece)

            actual_lateral_idx = 1
            actual_rotation_idx = 1

            # Lateral move (applied first, matching step_ai_training)
            if lat_idx == 0:
                prev_x = test_piece.x
                if not game.grid.check_collision(test_piece, offset_x=-1):
                    test_piece.x -= 1
                if test_piece.x < prev_x:
                    actual_lateral_idx = 0
            elif lat_idx == 2:
                prev_x = test_piece.x
                if not game.grid.check_collision(test_piece, offset_x=1):
                    test_piece.x += 1
                if test_piece.x > prev_x:
                    actual_lateral_idx = 2

            # Rotation move (applied after lateral, matching step_ai_training)
            if rot_idx == 2:
                prev_shape = set(test_piece.shape)
                test_piece.rotate_right()
                if game.grid.check_collision(test_piece):
                    test_piece.rotate_left()
                elif set(test_piece.shape) != prev_shape:
                    actual_rotation_idx = 2
            elif rot_idx == 0:
                prev_shape = set(test_piece.shape)
                test_piece.rotate_left()
                if game.grid.check_collision(test_piece):
                    test_piece.rotate_right()
                elif set(test_piece.shape) != prev_shape:
                    actual_rotation_idx = 0

            mask[action_idx] = (actual_lateral_idx == lat_idx and actual_rotation_idx == rot_idx)

        # Ensure stay action remains valid even if logic changes above
        mask[stay_action] = True
        return mask

    def select_action(self, state, action_mask=None):
        """Select action using epsilon-greedy with joint action space.
        
        Returns:
            action_idx: Integer 0-8 representing joint action
                       (lateral_idx * 3 + rotation_idx)
        """
        grid_input, next_piece_input = state

        valid_action_indices = None
        if action_mask is not None:
            action_mask = np.asarray(action_mask, dtype=np.bool_)
            valid_action_indices = np.flatnonzero(action_mask)
            if valid_action_indices.size == 0:
                valid_action_indices = np.array([self.encode_action(1, 1)], dtype=np.int64)
        
        self.model.eval()
        if random.random() <= self.epsilon:
            if valid_action_indices is None:
                # Random joint action (unmasked)
                return random.randint(0, self.num_actions - 1)
            # Random joint action (masked)
            return int(random.choice(valid_action_indices))
        
        # Predict Q-values for all 9 joint actions
        grid_tensor = torch.FloatTensor(grid_input).unsqueeze(0).to(self.device)
        next_tensor = torch.FloatTensor(next_piece_input).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            q_values = self.model(grid_tensor, next_tensor)

        if action_mask is not None:
            mask_tensor = torch.tensor(action_mask, dtype=torch.bool, device=q_values.device).unsqueeze(0)
            q_values = q_values.masked_fill(~mask_tensor, -1.0e9)
            
        # Greedy action selection (argmax) - pick highest Q-value
        action_idx = q_values.argmax(dim=-1).item()
        
        return action_idx
    
    def decode_action(self, action_idx):
        """Decode joint action index to (lateral_idx, rotation_idx)."""
        lateral_idx = action_idx // 3
        rotation_idx = action_idx % 3
        return lateral_idx, rotation_idx
    
    def encode_action(self, lateral_idx, rotation_idx):
        """Encode (lateral_idx, rotation_idx) to joint action index."""
        return lateral_idx * 3 + rotation_idx

    def calculate_reward(self, lines_cleared, hole_delta, jaggedness_delta):
        """
        Calculate reward for a piece placement using post-clear board state.

        Args:
            lines_cleared: Lines cleared by the placement
            hole_delta: holes_after - holes_before (post-clear vs pre-lock)
            jaggedness_delta: jaggedness_after - jaggedness_before (post-clear vs pre-lock)

        Returns:
            Total reward for this piece placement (game over penalty applied elsewhere)

        Game over penalty of 2.0 is applied in game.py. Scale rewards here accordingly            

        """
        reward = 0.0

        reward += 0.25 * lines_cleared

        if hole_delta < 0:
            reward -= 0.45 * hole_delta
        else:
            if hole_delta > 0:
                reward -= 0.50 * hole_delta

        if jaggedness_delta < 0:
            reward -= 0.05 * jaggedness_delta
        else:
            if jaggedness_delta > 0:
                reward -= 0.075 * jaggedness_delta

        return reward

    def add_trajectory_with_done(self, trajectory, final_reward):
        """
        Add a complete piece trajectory to memory with Monte Carlo return.
        
        Args:
            trajectory: List of (state, action_idx) tuples for one piece
            final_reward: R_piece - scalar Monte Carlo return for this piece
        """
        for i in range(len(trajectory)):
            state, action_idx = trajectory[i]
            
            # Monte Carlo: Apply gamma discount to the final reward
            # The last move (index len-1) gets final_reward * gamma^0 = final_reward
            # The move before that gets final_reward * gamma^1, etc.
            steps_from_end = len(trajectory) - 1 - i
            R_piece = final_reward * (self.gamma ** steps_from_end)
            
            # Store simplified tuple: (state, action_idx, R_piece)
            self.remember(state, action_idx, R_piece)


    def remember(self, state, action_idx, R_piece):
        """
        Store a (state, action_idx, Monte Carlo return) tuple in replay memory.
        
        Args:
            state: (grid_input, next_piece_input) tuple
            action_idx: Joint action index (0-8)
            R_piece: Scalar Monte Carlo return for the piece trajectory
        """
        self.memory.append((state, action_idx, R_piece))

    def record_training_stats(self, samples_added, lines_cleared, is_game_over, pieces_locked=1):
        """
        Track stats for when to trigger training.
        """
        self.total_samples_since_train += samples_added
        self.lines_since_train += lines_cleared
        self.pieces_since_train += pieces_locked
        if is_game_over:
            self.gameovers_since_train += 1
        self._maybe_train()

    def record_inference_step(self, moves=1):
        """Count inference decisions made since the last training update."""
        self.inference_moves_since_train += moves

    def _maybe_train(self):
        """
        Trigger training when enough new samples have been collected.
        """
        if self.total_samples_since_train < self.train_trigger_interval:
            return
        if len(self.memory) < self.batch_size:
            return
        
        # Capture stats for logging before reset
        lines = self.lines_since_train
        inference_moves = self.inference_moves_since_train
        pieces = self.pieces_since_train

        self.replay()
        # Per-window stats
        gameovers = self.gameovers_since_train
        
        # Add current window to history
        self.history.append((inference_moves, pieces, lines, gameovers))
        
        self.training_steps += 1
        
        # Log to CSV every 300 batches
        if self.training_steps > 0 and self.training_steps % 300 == 0:
            self._log_progress_to_csv()
        
        if pieces > 0:
            lines_per_piece = lines / pieces
            lpp_str = f"{lines_per_piece:.4f}"
        else:
            lpp_str = "N/A"
            
        # Running average: Sum of lines / Sum of pieces
        total_pieces_hist = sum(h[1] for h in self.history)
        total_lines_hist = sum(h[2] for h in self.history)
        
        if total_pieces_hist > 0:
            avg_lpp = total_lines_hist / total_pieces_hist
            lpp_str += f" (ave: {avg_lpp:.4f})"
        else:
            lpp_str += " (ave: N/A)"

        if gameovers > 0:
            lines_per_game = lines / gameovers
            lines_per_game_str = f"{lines_per_game:.3f}"
        else:
            lines_per_game_str = "N/A"
            
        # Running average: Sum of lines / Sum of gameovers
        total_gameovers_hist = sum(h[3] for h in self.history)
            
        if total_gameovers_hist > 0:
            avg_lpg = total_lines_hist / total_gameovers_hist
            lines_per_game_str += f" (ave: {avg_lpg:.3f})"
        else:
            lines_per_game_str += " (ave: N/A)"

        epsilon_str = f"{self.epsilon:.3f}"
        print(f"{inference_moves} moves, {pieces} pieces, {lines} lines, {gameovers} Game Overs || Lines / Piece = {lpp_str} || Lines / Game = {lines_per_game_str} || Epsilon = {epsilon_str}")

        self.total_samples_since_train = 0
        self.lines_since_train = 0
        self.gameovers_since_train = 0
        self.inference_moves_since_train = 0
        self.pieces_since_train = 0

    def _mirror_state_action(self, state, action_idx):
        """
        Horizontally mirror the state (board channels + next piece) and action.
        
        Args:
            state: (grid_input, next_piece_input) tuple
            action_idx: Joint action index 0-8
        
        Returns:
            mirrored_state: (mirrored_grid_input, mirrored_next_piece_input)
            mirrored_action_idx: Mirrored joint action index
        """
        grid_input, next_piece_input = state

        # Flip spatial channels horizontally (width axis)
        mirrored_grid = np.flip(np.array(grid_input), axis=2).copy()

        # Next piece is flattened 10x10; flip around the x=5 origin (not the 4.5 grid center).
        next_piece_grid = np.array(next_piece_input).reshape(10, 10)
        flipped_next_piece = np.flip(next_piece_grid, axis=1)
        mirrored_next_piece_grid = np.zeros_like(flipped_next_piece)
        mirrored_next_piece_grid[:, 1:] = flipped_next_piece[:, :-1]
        mirrored_next_piece = mirrored_next_piece_grid.reshape(-1).astype(np.float32)

        # Decode action, mirror, and re-encode
        lateral_idx, rotation_idx = self.decode_action(action_idx)
        # Map: Left <-> Right (0<->2, 1 stays), CCW <-> CW (0<->2, 1 stays)
        mirrored_lateral = 2 - lateral_idx
        mirrored_rotation = 2 - rotation_idx
        mirrored_action_idx = self.encode_action(mirrored_lateral, mirrored_rotation)

        return (mirrored_grid, mirrored_next_piece), mirrored_action_idx

    def replay(self):
        """
        Train the network on a batch of experiences using Monte Carlo returns.
        
        Loss: MSE between Q(state, action) and R_piece for chosen actions.
        No TD bootstrapping or target network involved.
        """
        if len(self.memory) < self.batch_size:
            return

        self.model.train()
        batch = random.sample(self.memory, self.batch_size)
        
        # Prepare batches
        grid_batch = []
        next_piece_batch = []
        action_batch = []  # Now stores single action indices (0-8)
        R_piece_batch = []
        
        for state, action_idx, R_piece in batch:
            grid_batch.append(state[0])
            next_piece_batch.append(state[1])
            action_batch.append(action_idx)
            R_piece_batch.append(R_piece)

            # Add mirrored counterpart to enforce left-right symmetry
            mirrored_state, mirrored_action_idx = self._mirror_state_action(state, action_idx)
            grid_batch.append(mirrored_state[0])
            next_piece_batch.append(mirrored_state[1])
            action_batch.append(mirrored_action_idx)
            R_piece_batch.append(R_piece)
            
        # Convert to tensors
        grid_tensor = torch.FloatTensor(np.array(grid_batch)).to(self.device)
        next_piece_tensor = torch.FloatTensor(np.array(next_piece_batch)).to(self.device)
        R_piece_tensor = torch.FloatTensor(R_piece_batch).to(self.device)
        action_tensor = torch.LongTensor(action_batch).unsqueeze(1).to(self.device)  # Shape: (batch, 1)
        
        # Forward pass through main network - now returns single Q-value tensor
        q_values = self.model(grid_tensor, next_piece_tensor)  # Shape: (batch, 9)
        
        # Extract Q-value for chosen action using gather
        q_chosen = q_values.gather(1, action_tensor).squeeze(1)  # Shape: (batch,)
        
        # Monte Carlo supervised loss: Q(s,a) should predict R_piece
        criterion = torch.nn.MSELoss()
        loss = criterion(q_chosen, R_piece_tensor)
        
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
        
        self.optimizer.step()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        # Enforce minimum exploration rate so printed epsilon matches behavior
        self.epsilon = max(self.epsilon, self.epsilon_min)





    def _log_progress_to_csv(self):
        """Append progress stats to CSV file."""
        # Calculate running averages from history
        total_lines = sum(h[2] for h in self.history)
        total_pieces = sum(h[1] for h in self.history)
        total_gameovers = sum(h[3] for h in self.history)
        
        avg_lpp = 0.0
        if total_pieces > 0:
            avg_lpp = total_lines / total_pieces
            
        avg_lpg = 0.0
        if total_gameovers > 0:
            avg_lpg = total_lines / total_gameovers
            
        import os
        
        # Check if file exists before opening in append mode
        # If it doesn't exist (e.g. user deleted it), we need to write the header
        write_header = not os.path.exists(self.csv_path)
            
        try:
            with open(self.csv_path, 'a') as f:
                if write_header:
                    f.write("Batch, Lines per Piece, Lines per Game\n")
                
                # Format: Batch, Lines_per_Piece, Lines_per_Game
                f.write(f"{self.training_steps}, {avg_lpp:.4f}, {avg_lpg:.3f}\n")
        except PermissionError:
            print(f"Skipping log for batch {self.training_steps}: File is locked.")
        except Exception as e:
            print(f"Error writing to CSV: {e}")

    def update_hyperparameters(self):
        """Update hyperparameters from self.params (which are shared with UI)."""
        self.gamma = 1.0
        self.params['gamma'] = 1.0
        epsilon_min_percent = self._snap_epsilon_percent(self.params.get('epsilon_min_percent', 5))
        self.params['epsilon_min_percent'] = epsilon_min_percent
        self.epsilon_min = epsilon_min_percent / 100.0
        half_life = self._snap_epsilon_half_life(self.params.get('epsilon_half_life_batches', 10 ** 4.5))
        self.params['epsilon_half_life_batches'] = half_life
        self.epsilon_decay = self._compute_epsilon_decay(half_life)
        lr = self.params.get('learning_rate', 0.0001)
        lr = max(0.0001, min(0.005, lr))
        self.learning_rate = lr
        # If the floor increases, ensure current epsilon respects it
        self.epsilon = max(self.epsilon, self.epsilon_min)
        
        # Update optimizer learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.learning_rate

    def save(self, path):
        """
        Save model checkpoint.
        
        Saves:
        - model_state_dict: Main network weights
        - optimizer_state_dict: Optimizer state
        - epsilon: Current exploration rate
        - params: Architecture parameters
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'params': self.params
        }, path)

    def load(self, path):
        """
        Load model checkpoint.
        
        Loads main network weights, optimizer state, and epsilon.
        Architecture params are taken from current UI settings, not from file.
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        
        # Note: We don't load params from file, we use the current UI params
        # This ensures the slider values control the architecture
