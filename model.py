import torch
import torch.nn as nn
import torch.nn.functional as F

class OminisNet(nn.Module):
    def __init__(self, input_channels=3, hidden_size=256, hidden_count=2, output_heads=[3, 3]):
        super(OminisNet, self).__init__()
        
        # CNN for spatial features (Board, Piece, Ghost)
        # Input: (Batch, 3, 34, 12)
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        
        # Flatten size calculation:
        # 34 * 12 * 64 = 26112
        self.flatten_size = 34 * 12 * 64
        
        # Next Piece Input (10x10 flattened = 100)
        self.next_piece_size = 100
        
        # Dense Layers
        self.fc_input_size = self.flatten_size + self.next_piece_size
        
        layers = []
        # First layer
        layers.append(nn.Linear(self.fc_input_size, hidden_size))
        layers.append(nn.ReLU())
        
        # Additional hidden layers
        for _ in range(hidden_count - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
            
        self.feature_extractor = nn.Sequential(*layers)
        
        # Output Heads
        # Lateral: Left, Stay, Right
        self.lateral_head = nn.Linear(hidden_size, output_heads[0])
        
        # Rotation: CCW, Stay, CW
        self.rotate_head = nn.Linear(hidden_size, output_heads[1])
        
    def forward(self, grid_input, next_piece_input):
        # grid_input: (B, 3, 34, 12)
        x = F.relu(self.conv1(grid_input))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        x = x.view(x.size(0), -1) # Flatten
        
        # Concatenate with next piece info
        combined = torch.cat([x, next_piece_input], dim=1)
        
        features = self.feature_extractor(combined)
        
        lateral = self.lateral_head(features)
        rotate = self.rotate_head(features)
        
        return lateral, rotate
