import torch
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
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
    def __init__(self, train_params):
        self.params = train_params
        
        # Hyperparameters
        self.batch_size = 2500 # Batch size for training (doubled with mirroring)
        self.gamma = 0.7 #(discount factor for Monte Carlo return, prioritizes most recent moves)
        self.epsilon = 1.00 # Initial exploration rate
        self.epsilon_min = self.params.get('epsilon_min_percent', 5) / 100.0 # Minimum exploration rate (Default 5%)
        self.epsilon_decay = 0.999 # Decay per training step
        lr = self.params.get('learning_rate', 0.0001)
        # Clamp to UI-supported range
        lr = max(0.0001, min(0.005, lr))
        self.learning_rate = lr
        self.memory = deque(maxlen=5000) # Single replay memory for all trajectories
        self.total_samples_since_train = 0
        self.train_trigger_interval = 2500  # Train more frequently with smaller batches
        self.lines_since_train = 0
        self.gameovers_since_train = 0
        self.inference_moves_since_train = 0
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Model
        self.HL_SIZES = [128, 256, 512]
        hidden_size = self.HL_SIZES[self.params['hl_size_idx']]
        hidden_count = self.params['hl_count']
        
        # Main network (updated every training step with Monte Carlo targets)
        self.model = OminisNet(hidden_size=hidden_size, hidden_count=hidden_count).to(self.device)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Action Space: 9 joint actions (3 lateral × 3 rotation)
        # Lateral: 0=Left, 1=Stay, 2=Right
        # Rotation: 0=CCW, 1=Stay, 2=CW 
        # Encoding: action_idx = lateral_idx * 3 + rotation_idx
        self.num_actions = 9

    def get_state(self, game):
        # 1. Board Channel (12 x 34)
        # Game grid is list of lists of colors. Convert to 1s and 0s.
        board = np.zeros((34, 12), dtype=np.float32)
        # Grid is 24 high, but we want 10 buffer.
        # game.grid.grid is 24 rows.
        # We need to pad top with 10 empty rows.
        
        # Actually, let's just use the visible grid + buffer if available?
        # The prompt said "game board + 10 blocks above".
        # game.grid.grid is 24x12.
        # Let's assume the input is 34x12.
        # Rows 0-9 are buffer (empty for now), 10-33 are game grid.
        
        for y in range(24):
            for x in range(12):
                if game.grid.grid[y][x] != (0, 0, 0):
                    board[y + 10][x] = 1.0
                    
        # 2. Piece Channel
        piece_grid = np.zeros((34, 12), dtype=np.float32)
        if game.current_piece:
            for x, y in game.current_piece.shape:
                # Piece y is relative to board top (0).
                # So in our 34-height grid, it is y + 10.
                # Note: piece.y can be negative (spawning).
                py = game.current_piece.y + y + 10
                px = game.current_piece.x + x
                if 0 <= py < 34 and 0 <= px < 12:
                    piece_grid[py][px] = 1.0
                    
        # 3. Ghost Channel
        ghost_grid = np.zeros((34, 12), dtype=np.float32)
        ghost = game.get_ghost_piece()
        if ghost:
            for x, y in ghost.shape:
                py = ghost.y + y + 10
                px = ghost.x + x
                if 0 <= py < 34 and 0 <= px < 12:
                    ghost_grid[py][px] = 1.0
                    
        # Stack channels
        grid_input = np.stack([board, piece_grid, ghost_grid]) # (3, 34, 12)
        
        # 4. Next Piece (Flattened 10x10)
        # We need to represent the next piece shape.
        # Let's center it in a 10x10 grid.
        next_piece_vec = np.zeros(100, dtype=np.float32)
        if game.next_piece:
            # Center is 5, 5
            for x, y in game.next_piece.shape:
                # Normalize shape coordinates? They are usually small (0-4).
                # Let's just place them at offset 5,5
                nx = x + 5
                ny = y + 5
                if 0 <= nx < 10 and 0 <= ny < 10:
                    next_piece_vec[ny * 10 + nx] = 1.0
                    
        return grid_input, next_piece_vec

    def select_action(self, state):
        """Select action using epsilon-greedy with joint action space.
        
        Returns:
            action_idx: Integer 0-8 representing joint action
                       (lateral_idx * 3 + rotation_idx)
        """
        grid_input, next_piece_input = state
        
        self.model.eval()
        if random.random() <= self.epsilon:
            # Random joint action
            return random.randint(0, self.num_actions - 1)
        
        # Predict Q-values for all 9 joint actions
        grid_tensor = torch.FloatTensor(grid_input).unsqueeze(0).to(self.device)
        next_tensor = torch.FloatTensor(next_piece_input).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            q_values = self.model(grid_tensor, next_tensor)
            
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

    def calculate_reward(self, lines_cleared, game_over, height_increased, blocks_over_holes, 
                         placement_height_delta, holes_before, holes_after):
        """
        Calculate reward for a piece placement.
        
        Args:
            lines_cleared: Number of lines cleared (0-4+)
            game_over: Whether the game ended
            height_increased: True if max height rose after placement
            blocks_over_holes: True if any new blocks are above blank spaces
            placement_height_delta: How many rows above the lowest column the piece's lowest block was placed
            holes_before: Total holes before placement
            holes_after: Total holes after placement (post line-clear)
        
        Returns:
            Total reward for this piece placement
        """
        reward = 0

        # Lines: 350 * L^2
        if lines_cleared > 0:
            reward += 50#350 * (lines_cleared ** 2)

        # Game over: -1000
        if game_over:
            reward -= 50#1000

        # Good placement reward (+50) - only if ALL THREE conditions are met:
        # 1. No holes are created
        # 2. Max height is not increased
        # 3. Lowest block of placement is no more than 2 places higher than lowest column
        if not height_increased and not blocks_over_holes and placement_height_delta <= 2:
            reward += 50
        else:
            # Bad placement: one of the conditions failed
            reward -= 20
        
        # Bonus for decreasing net holes (+20)
        # This rewards sliding under overhangs or clearing lines with holes beneath
        holes_reduced = holes_before - holes_after
        if holes_reduced > 0:
            reward += 20

        return reward



    def add_trajectory_with_done(self, trajectory, final_reward):
        """
        Add a complete piece trajectory to memory with Monte Carlo return.
        
        Args:
            trajectory: List of (state, action, next_state) tuples for one piece
            final_reward: R_piece - scalar Monte Carlo return for this piece
        """
        for i in range(len(trajectory)):
            state, action, next_state = trajectory[i]
            
            # Monte Carlo: Apply gamma discount to the final reward
            # The last move (index len-1) gets final_reward * gamma^0 = final_reward
            # The move before that gets final_reward * gamma^1, etc.
            steps_from_end = len(trajectory) - 1 - i
            R_piece = final_reward * (self.gamma ** steps_from_end)
            
            # Store simplified tuple: (state, action, R_piece)
            self.remember(state, action, R_piece)


    def remember(self, state, action, R_piece):
        """
        Store a (state, action, Monte Carlo return) tuple in replay memory.
        
        Args:
            state: (grid_input, next_piece_input) tuple
            action: [lateral_idx, rotation_idx] list
            R_piece: Scalar Monte Carlo return for the piece trajectory
        """
        self.memory.append((state, action, R_piece))

    def record_training_stats(self, samples_added, lines_cleared, is_game_over):
        """
        Track stats for when to trigger training.
        """
        self.total_samples_since_train += samples_added
        self.lines_since_train += lines_cleared
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
        moves = self.total_samples_since_train
        lines = self.lines_since_train
        inference_moves = self.inference_moves_since_train

        self.replay()
        # Per-window stats
        gameovers = self.gameovers_since_train
        if lines > 0:
            moves_per_line = inference_moves / lines
            mpline_str = f"{moves_per_line:.1f}"
            go_per_line = gameovers / lines
            goline_str = f"{go_per_line:.3f}"
        else:
            mpline_str = "N/A"
            goline_str = "N/A"

        epsilon_str = f"{self.epsilon:.3f}"
        print(f"Trained on {moves} moves; {lines} lines, {gameovers} Game Overs; Moves/Line = {mpline_str}; Game Overs/Line = {goline_str}; Epsilon: {epsilon_str}")

        self.total_samples_since_train = 0
        self.lines_since_train = 0
        self.gameovers_since_train = 0
        self.inference_moves_since_train = 0

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

        # Next piece is flattened 10x10; flip horizontally and flatten again
        next_piece_grid = np.array(next_piece_input).reshape(10, 10)
        mirrored_next_piece = np.flip(next_piece_grid, axis=1).reshape(-1).astype(np.float32)

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





    def update_hyperparameters(self):
        """Update hyperparameters from self.params (which are shared with UI)."""
        self.epsilon_min = self.params.get('epsilon_min_percent', 5) / 100.0
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
