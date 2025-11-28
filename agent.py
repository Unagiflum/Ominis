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
        self.batch_size = 128
        self.gamma = 0.6
        self.epsilon = 1.0
        self.epsilon_min = self.params.get('epsilon_min_percent', 15) / 100.0
        self.epsilon_decay = 0.999
        self.learning_rate = self.params.get('learning_rate', 0.0005)
        self.memory = deque(maxlen=30000)
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Model
        self.HL_SIZES = [128, 256, 512]
        hidden_size = self.HL_SIZES[self.params['hl_size_idx']]
        hidden_count = self.params['hl_count']
        
        # Main network (updated every training step with Monte Carlo targets)
        self.model = OminisNet(hidden_size=hidden_size, hidden_count=hidden_count).to(self.device)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Action Space
        # Lateral: 0=Left, 1=Stay, 2=Right
        # Rotate: 0=CCW, 1=Stay, 2=CW
        self.action_space = [3, 3]

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
        grid_input, next_piece_input = state
        
        self.model.eval()
        if random.random() <= self.epsilon:
            # Random actions
            lat = random.randint(0, 2)
            rot = random.randint(0, 2)
            return [lat, rot]
        
        # Predict
        grid_tensor = torch.FloatTensor(grid_input).unsqueeze(0).to(self.device)
        next_tensor = torch.FloatTensor(next_piece_input).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            lat_q, rot_q = self.model(grid_tensor, next_tensor)
            
        lat = torch.argmax(lat_q).item()
        rot = torch.argmax(rot_q).item()
        
        return [lat, rot]

    def calculate_reward(self, lines_cleared, game_over):
        """Monte Carlo reward: only line clears (+) and game over (-)."""
        reward = 0

        if lines_cleared > 0:
            reward += 350 * (lines_cleared ** 2)

        if game_over:
            reward -= 1000

        return reward



    def add_trajectory_with_done(self, trajectory, final_reward, game_over):
        """
        Add a complete piece trajectory to memory with Monte Carlo return.
        
        Args:
            trajectory: List of (state, action, next_state) tuples for one piece
            final_reward: R_piece - scalar Monte Carlo return for this piece
            game_over: Whether the game ended (unused in current implementation)
        """
        for i in range(len(trajectory)):
            state, action, next_state = trajectory[i]
            
            # Monte Carlo: Apply gamma discount to the final reward
            # The last move (index len-1) gets final_reward * gamma^0 = final_reward
            # The move before that gets final_reward * gamma^1, etc.
            steps_from_end = len(trajectory) - 1 - i
            R_piece = final_reward * (self.gamma ** steps_from_end)
            
            # Store simplified tuple: (state, action, R_piece)
            # We don't need next_state since we're not doing TD bootstrapping.
            # The 'done' flag is also not stored because R_piece is the full return.
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

    def _mirror_state_action(self, state, action):
        """
        Horizontally mirror the state (board channels + next piece) and action.
        
        Returns:
            mirrored_state: (mirrored_grid_input, mirrored_next_piece_input)
            mirrored_action: [mirrored_lateral, mirrored_rotation]
        """
        grid_input, next_piece_input = state

        # Flip spatial channels horizontally (width axis)
        mirrored_grid = np.flip(np.array(grid_input), axis=2).copy()

        # Next piece is flattened 10x10; flip horizontally and flatten again
        next_piece_grid = np.array(next_piece_input).reshape(10, 10)
        mirrored_next_piece = np.flip(next_piece_grid, axis=1).reshape(-1).astype(np.float32)

        # Map actions: Left <-> Right, CCW <-> CW, Stay stays
        mirrored_action = [
            2 - action[0],  # lateral mapping: 0<->2, 1 stays
            2 - action[1]   # rotation mapping: 0<->2, 1 stays
        ]

        return (mirrored_grid, mirrored_next_piece), mirrored_action

    def replay(self):
        """
        Train the network on a batch of experiences using Monte Carlo returns.
        
        Loss: MSE between Q(state, action) and R_piece for chosen actions.
        No TD bootstrapping or target network involved.
        """
        if len(self.memory) < self.batch_size:
            return
            
        self.model.train()
        minibatch = random.sample(self.memory, self.batch_size)
        
        # Prepare batches
        grid_batch = []
        next_piece_batch = []
        action_batch = []
        R_piece_batch = []
        
        for state, action, R_piece in minibatch:
            grid_batch.append(state[0])
            next_piece_batch.append(state[1])
            action_batch.append(action)
            R_piece_batch.append(R_piece)

            # Add mirrored counterpart to enforce left-right symmetry
            mirrored_state, mirrored_action = self._mirror_state_action(state, action)
            grid_batch.append(mirrored_state[0])
            next_piece_batch.append(mirrored_state[1])
            action_batch.append(mirrored_action)
            R_piece_batch.append(R_piece)
            
        # Convert to tensors
        grid_tensor = torch.FloatTensor(np.array(grid_batch)).to(self.device)
        next_piece_tensor = torch.FloatTensor(np.array(next_piece_batch)).to(self.device)
        R_piece_tensor = torch.FloatTensor(R_piece_batch).to(self.device)
        
        # Forward pass through main network
        lat_q, rot_q = self.model(grid_tensor, next_piece_tensor)
        
        # Compute loss using only the Q-values for the chosen actions
        # We use gather to select Q(s, a) for the action taken
        action_tensor = torch.LongTensor(action_batch).to(self.device)
        
        # Extract Q-values for chosen actions from each head
        lat_q_chosen = lat_q.gather(1, action_tensor[:, 0:1]).squeeze(1)  # Shape: (batch_size,)
        rot_q_chosen = rot_q.gather(1, action_tensor[:, 1:2]).squeeze(1)
        
        # Monte Carlo supervised loss: All heads predict the same R_piece
        criterion = torch.nn.MSELoss()
        loss_lat = criterion(lat_q_chosen, R_piece_tensor)
        loss_rot = criterion(rot_q_chosen, R_piece_tensor)
        
        # Total loss is the sum of all head losses
        loss = loss_lat + loss_rot
        
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
        
        self.optimizer.step()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay





    def update_hyperparameters(self):
        """Update hyperparameters from self.params (which are shared with UI)."""
        self.epsilon_min = self.params.get('epsilon_min_percent', 15) / 100.0
        self.learning_rate = self.params.get('learning_rate', 0.0005)
        
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

