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
        self.HL_SIZES = [128, 256, 512]
        hidden_size = self.HL_SIZES[self.params['hl_size_idx']]
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

    def calculate_reward(self, game, lines_cleared, game_over, start_stats, piece_in_upper_half):
        # Calculate reward based on the CHANGE in state
        reward = 0
        
        # 1. Line Clears (Big reward)
        if lines_cleared > 0:
            reward += (lines_cleared ** 2) * 100
            
        # 2. Game Over Penalty
        if game_over:
            reward -= 500
            
        # Get current stats
        current_height, current_holes = game.get_grid_stats()
        start_height, start_holes = start_stats
        
        # 3. Height Change
        # If height increased, penalty.
        height_change = current_height - start_height
        if height_change > 0:
            h_factor = self.params['height_penalty'] / 100.0
            reward -= height_change * h_factor * 10 # Multiplier for impact
            
        # 4. Holes Change
        # If holes increased, penalty.
        holes_change = current_holes - start_holes
        if holes_change > 0:
            o_factor = self.params['overhang_penalty'] / 100.0
            reward -= holes_change * o_factor * 20 # Multiplier for impact
            
        # 5. High Stacking Penalty
        # Penalize if THIS piece was placed in the upper half
        if piece_in_upper_half:
            reward -= 10
            
        return reward

    def add_trajectory(self, trajectory, final_reward):
        """
        trajectory: List of (state, action, next_state) tuples
        final_reward: The reward calculated at the end of the trajectory
        """
        # Distribute reward to all steps
        # Optional: Discount factor backwards? 
        # User said "distributed to the entire series". 
        # Usually this means each step gets the final reward (or discounted).
        # Let's use a slight discount to encourage faster placement if possible, 
        # or just flat reward if we want them to value all steps equally.
        # Let's use gamma decay from the end.
        
        current_reward = final_reward
        
        # Iterate backwards
        for i in range(len(trajectory) - 1, -1, -1):
            state, action, next_state = trajectory[i]
            
            # The last step gets the full final_reward.
            # Previous steps get discounted version?
            # Actually, standard Q-learning does this via Bellman update.
            # But here we are assigning the IMMEDIATE reward for the transition.
            # If we assign the final outcome as the immediate reward for ALL steps,
            # it's like Monte Carlo return assignment.
            
            # Let's assign the final_reward to the LAST step.
            # And 0 (or small step penalty) to others?
            # User said "reward is distributed to the entire series".
            # This might mean: Reward for step i = Final Reward / N ? No, that dilutes it.
            # It likely means: Each step contributed to this outcome.
            # Let's give the final_reward to EVERY step.
            # This tells the AI: "This move led to this good/bad outcome".
            
            # However, if we give +100 to every step, the total return is N * 100.
            # If we have 10 steps, that's huge.
            # Maybe we should just give it to the last step, and let Q-propagation handle the rest?
            # BUT, the user specifically asked for "distributed".
            # "distributed to the entire series of moves, calculating output and prediction at each individual step."
            
            # Interpretation: The user wants to avoid the "sparse reward" problem where only the last move sees the reward.
            # So let's assign the final reward to ALL steps in the trajectory.
            # This is a form of Reward Shaping.
            
            step_reward = final_reward
            
            # Mark done only on the last step?
            # If the game is over, the last step is done.
            # If just piece placed, it's not "done" for the episode, but it is a terminal state for the piece.
            # In infinite Tetris, "done" usually means Game Over.
            # So done=False for all except Game Over.
            
            is_last_step = (i == len(trajectory) - 1)
            # We need to know if the game ended. 
            # We can infer from final_reward if it's the penalty? No.
            # We should pass 'done' status.
            
            # Let's update the signature to include done.
            pass

    def add_trajectory_with_done(self, trajectory, final_reward, game_over):
        for i in range(len(trajectory)):
            state, action, next_state = trajectory[i]
            
            # Assign final reward to all steps
            # Maybe slightly discounted by distance to end?
            # reward = final_reward * (self.gamma ** (len(trajectory) - 1 - i))
            # Let's try flat reward first as it's more robust for "this sequence was good".
            
            reward = final_reward
            
            # Only the very last step of the GAME has done=True.
            # Intermediate piece placements have done=False.
            is_terminal = game_over if (i == len(trajectory) - 1) else False
            
            self.remember(state, action, reward, next_state, is_terminal)


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

    def save(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'params': self.params
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        # We don't overwrite params from file, we use the current UI params
        # But we could check if they match? For now, trust the UI.
