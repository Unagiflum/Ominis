import torch
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
from collections import deque
from model import OminisNet

class DQNAgent:
    def __init__(self, train_params):
        self.params = train_params
        
        # Hyperparameters
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.memory = deque(maxlen=10000)
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Model
        hl_sizes = [128, 256, 512]
        hidden_size = hl_sizes[self.params['hl_size_idx']]
        hidden_count = self.params['hl_count']
        
        self.model = OminisNet(hidden_size=hidden_size, hidden_count=hidden_count).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Action Space
        # Lateral: 0=Left, 1=Stay, 2=Right
        # Rotate: 0=CCW, 1=Stay, 2=CW
        # Vertical: 0=Down, 1=Stay
        self.action_space = [3, 3, 2]

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
        
        if random.random() <= self.epsilon:
            # Random actions
            lat = random.randint(0, 2)
            rot = random.randint(0, 2)
            vert = random.randint(0, 1)
            return [lat, rot, vert]
        
        # Predict
        grid_tensor = torch.FloatTensor(grid_input).unsqueeze(0).to(self.device)
        next_tensor = torch.FloatTensor(next_piece_input).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            lat_q, rot_q, vert_q = self.model(grid_tensor, next_tensor)
            
        lat = torch.argmax(lat_q).item()
        rot = torch.argmax(rot_q).item()
        vert = torch.argmax(vert_q).item()
        
        return [lat, rot, vert]

    def calculate_reward(self, game, lines_cleared, game_over):
        reward = 0
        
        # 1. Line Clears
        if lines_cleared > 0:
            # Reward exponential to lines?
            reward += (lines_cleared ** 2) * 100
            
        # 2. Game Over Penalty
        if game_over:
            reward -= 500
            
        # 3. Height Penalty
        # Calculate max height
        max_height = 0
        for y in range(24):
            if any(c != (0,0,0) for c in game.grid.grid[y]):
                max_height = 24 - y
                break
        
        # Penalty % (0-100) -> Factor 0.0 - 1.0
        # Let's say max penalty is -10 per block height?
        h_factor = self.params['height_penalty'] / 100.0
        reward -= max_height * h_factor * 2
        
        # 4. Overhang Penalty
        # Count holes (empty blocks with filled blocks above them)
        holes = 0
        for x in range(12):
            found_block = False
            for y in range(24):
                if game.grid.grid[y][x] != (0,0,0):
                    found_block = True
                elif found_block:
                    holes += 1
                    
        o_factor = self.params['overhang_penalty'] / 100.0
        reward -= holes * o_factor * 5
        
        return reward

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
            
        minibatch = random.sample(self.memory, self.batch_size)
        
        # Prepare batches
        grid_batch = []
        next_piece_batch = []
        action_batch = []
        reward_batch = []
        next_grid_batch = []
        next_next_piece_batch = []
        done_batch = []
        
        for state, action, reward, next_state, done in minibatch:
            grid_batch.append(state[0])
            next_piece_batch.append(state[1])
            action_batch.append(action)
            reward_batch.append(reward)
            next_grid_batch.append(next_state[0])
            next_next_piece_batch.append(next_state[1])
            done_batch.append(done)
            
        grid_tensor = torch.FloatTensor(np.array(grid_batch)).to(self.device)
        next_piece_tensor = torch.FloatTensor(np.array(next_piece_batch)).to(self.device)
        next_grid_tensor = torch.FloatTensor(np.array(next_grid_batch)).to(self.device)
        next_next_piece_tensor = torch.FloatTensor(np.array(next_next_piece_batch)).to(self.device)
        rewards = torch.FloatTensor(reward_batch).to(self.device)
        dones = torch.FloatTensor(done_batch).to(self.device)
        
        # Current Q values
        lat_q, rot_q, vert_q = self.model(grid_tensor, next_piece_tensor)
        
        # Target Q values
        with torch.no_grad():
            next_lat_q, next_rot_q, next_vert_q = self.model(next_grid_tensor, next_next_piece_tensor)
            
        # Update Q values for taken actions
        loss = 0
        criterion = torch.nn.MSELoss()
        
        for i in range(self.batch_size):
            lat_act, rot_act, vert_act = action_batch[i]
            
            # Target = Reward + Gamma * Max(Next Q)
            target_val = rewards[i]
            if not dones[i]:
                # Combined max Q? Or separate?
                # BDQ usually treats them independently or sums them.
                # Let's treat independently for simplicity.
                target_val += self.gamma * (torch.max(next_lat_q[i]) + torch.max(next_rot_q[i]) + torch.max(next_vert_q[i])) / 3.0
            
            # We want each head to predict this target? 
            # Or should we split reward?
            # Standard BDQ: Q_global = V(s) + A(s, a).
            # Here we have 3 independent heads. Let's train them to maximize the common reward.
            
            target_lat = lat_q[i].clone()
            target_lat[lat_act] = target_val
            
            target_rot = rot_q[i].clone()
            target_rot[rot_act] = target_val
            
            target_vert = vert_q[i].clone()
            target_vert[vert_act] = target_val
            
            loss += criterion(lat_q[i], target_lat)
            loss += criterion(rot_q[i], target_rot)
            loss += criterion(vert_q[i], target_vert)
            
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
