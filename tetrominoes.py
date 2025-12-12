import random

# Colors (R, G, B) - Neon/Pastel Palette
COLORS = [
    (255, 105, 180), # Hot Pink
    (255, 160, 122), # Light Salmon
    (255, 255, 0),   # Yellow
    (173, 255, 47),  # Green Yellow
    (0, 255, 255),   # Cyan
    (135, 206, 235), # Sky Blue
    (147, 112, 219), # Medium Purple
    (221, 160, 221), # Plum
    (255, 20, 147),  # Deep Pink
    (255, 69, 0),    # Orange Red
    (50, 205, 50),   # Lime Green
    (64, 224, 208),  # Turquoise
    (30, 144, 255),  # Dodger Blue
    (138, 43, 226),  # Blue Violet
    (255, 0, 255),   # Magenta
    (255, 215, 0),   # Gold
    (0, 250, 154),   # Medium Spring Green
    (123, 104, 238), # Medium Slate Blue
    # Tetromino Colors (Standard-ish)
    (0, 255, 255),   # Cyan (I) - Duplicate but fine
    (0, 0, 255),     # Blue (J)
    (255, 165, 0),   # Orange (L)
    (255, 255, 0),   # Yellow (O) - Duplicate but fine
    (0, 128, 0),     # Green (S)
    (128, 0, 128),   # Purple (T)
    (255, 0, 0),     # Red (Z)
    # Ominis Colors
    (200, 200, 200), # Silver (Domino)
    (255, 215, 0),   # Gold (Triomino I)
    (255, 140, 0),   # Dark Orange (Triomino L)
]

# Pentomino Shapes (One-sided)
# Defined as list of (x, y) coordinates relative to a center (0,0) or top-left
# We will use a grid system where (0,0) is the pivot if possible, or just a list of blocks.
# 18 shapes: F, I, L, P, N, T, U, V, W, X, Y, Z (plus mirrors for chiral ones)
# Chiral: F, L, N, P, Y, Z.  Symmetric: I, T, U, V, W, X.
# Total = 6 symmetric + 2 * 6 chiral = 18.

SHAPES = {
    'I': [(0, -2), (0, -1), (0, 0), (0, 1), (0, 2)],
    'T': [(-1, -1), (0, -1), (1, -1), (0, 0), (0, 1)],
    'U': [(-1, -1), (-1, 0), (0, 0), (1, 0), (1, -1)],
    'V': [(-1, -2), (-1, -1), (-1, 0), (0, 0), (1, 0)],
    'W': [(-1, -1), (-1, 0), (0, 0), (0, 1), (1, 1)],
    'X': [(0, -1), (-1, 0), (0, 0), (1, 0), (0, 1)],
    
    'F': [(0, -1), (1, -1), (-1, 0), (0, 0), (0, 1)],
    'F_mirror': [(0, -1), (-1, -1), (1, 0), (0, 0), (0, 1)],
    
    'L': [(-1, -2), (-1, -1), (-1, 0), (-1, 1), (0, 1)],
    'L_mirror': [(0, -2), (0, -1), (0, 0), (0, 1), (-1, 1)], # J-like
    
    'N': [(-1, 0), (0, 0), (0, -1), (1, -1), (2, -1)],
    'N_mirror': [(1, 0), (0, 0), (0, -1), (-1, -1), (-2, -1)],
    
    'P': [(-1, -1), (0, -1), (-1, 0), (0, 0), (-1, 1)],
    'P_mirror': [(0, -1), (1, -1), (0, 0), (1, 0), (1, 1)],
    
    'Y': [(0, -2), (0, -1), (0, 0), (0, 1), (-1, 0)],
    'Y_mirror': [(0, -2), (0, -1), (0, 0), (0, 1), (1, 0)],
    
    'Z': [(-1, -1), (0, -1), (0, 0), (0, 1), (1, 1)],
    'Z_mirror': [(1, -1), (0, -1), (0, 0), (0, 1), (-1, 1)],

    # Tetrominoes
    'Tet_I': [(0, -1), (0, 0), (0, 1), (0, 2)],
    'Tet_J': [(0, -1), (0, 0), (0, 1), (-1, 1)],
    'Tet_L': [(0, -1), (0, 0), (0, 1), (1, 1)],
    'Tet_O': [(0, 0), (1, 0), (0, 1), (1, 1)],
    'Tet_S': [(0, 0), (1, 0), (0, 1), (-1, 1)], # S-shape
    'Tet_T': [(-1, 0), (0, 0), (1, 0), (0, 1)],
    'Tet_Z': [(-1, 0), (0, 0), (0, 1), (1, 1)], # Z-shape
    
    # Ominis
    'Domino': [(0, 0), (0, 1)],
    'Triomino_I': [(0, -1), (0, 0), (0, 1)],
    'Triomino_L': [(0, 0), (0, 1), (1, 0)],
    'Monomino': [(0, 0)],
}

# Shape to Color mapping (explicit to avoid index issues)
SHAPE_COLORS = {
    'I': (0, 255, 255),      # Cyan
    'T': (128, 0, 128),      # Purple
    'U': (255, 165, 0),      # Orange
    'V': (255, 0, 0),        # Red
    'W': (0, 255, 0),        # Lime Green
    'X': (255, 255, 0),      # Yellow
    'F': (255, 105, 180),    # Hot Pink
    'F_mirror': (255, 20, 147),  # Deep Pink
    'L': (30, 144, 255),     # Dodger Blue
    'L_mirror': (135, 206, 235), # Sky Blue
    'N': (173, 255, 47),     # Green Yellow
    'N_mirror': (0, 250, 154),   # Medium Spring Green
    'P': (147, 112, 219),    # Medium Purple
    'P_mirror': (138, 43, 226),  # Blue Violet
    'Y': (255, 160, 122),    # Light Salmon
    'Y_mirror': (255, 69, 0),    # Orange Red
    'Z': (221, 160, 221),    # Plum
    'Z_mirror': (255, 0, 255),   # Magenta
    'Tet_I': (0, 255, 255),  # Cyan
    'Tet_J': (0, 0, 255),    # Blue
    'Tet_L': (255, 165, 0),  # Orange
    'Tet_O': (255, 255, 0),  # Yellow
    'Tet_S': (0, 128, 0),    # Green
    'Tet_T': (128, 0, 128),  # Purple
    'Tet_Z': (255, 0, 0),    # Red
    'Domino': (200, 200, 200),   # Silver
    'Triomino_I': (255, 215, 0), # Gold
    'Triomino_L': (255, 140, 0), # Dark Orange
    'Monomino': (192, 192, 192), # Silver/Gray
}

def get_allowed_shapes(include_pentominoes=True, include_tetrominoes=False, include_ominis=False, max_size=5, weighted=False):
    allowed = []
    for shape, blocks in SHAPES.items():
        count = len(blocks)
        if count > max_size:
            continue
            
        should_add = False
        if count == 5 and include_pentominoes:
            should_add = True
        elif count == 4 and include_tetrominoes:
            should_add = True
        elif count < 4 and include_ominis:
            should_add = True
            
        if should_add:
            if weighted:
                # Add 2^n copies
                # Size 1: 2^1 = 2
                # Size 2: 2^2 = 4
                # Size 3: 2^3 = 8
                # Size 4: 2^4 = 16
                # Size 5: 2^5 = 32
                # Note: User request wording: "for every 1 1-omino, there would be 2 dominoes, 8 triominoes, and 56 tetrominoes."
                # User example breakdown:
                # "if size 4 enabled... for every 1-omino, there should be 2 dominoes, 4 of each type of triomino... and 8 of each of the 7 tetrominoes"
                # This implies 2^(size-1) * base_weight? Or just 2^size?
                # User says: "polyominoes of size n are 2^n represented"
                # Let's check the example math:
                # 1-omino (size 1): 2^1 = 2 per shape. (1 shape: Monomino) -> Total 2.
                # Domino (size 2): 2^2 = 4 per shape. (1 shape: Domino) -> Total 4. Ratio 4:2 = 2:1. Correct ("2 dominoes per 1-omino").
                # Triomino (size 3): 2^3 = 8 per shape. (2 shapes: I, L). Total 16. Ratio 16:2 = 8:1. Correct ("8 triominoes per 1-omino").
                # Tetromino (size 4): 2^4 = 16 per shape. (7 shapes). Total 16*7 = 112. ratio 112:2 = 56:1. Correct ("56 tetrominoes").
                
                weight = 2 ** count
                allowed.extend([shape] * weight)
            else:
                allowed.append(shape)
    return allowed

class Pentomino:
    def __init__(self, x, y, allowed_shapes=None):
        self.x = x
        self.y = y
        
        if allowed_shapes:
            self.type = random.choice(allowed_shapes)
        else:
            self.type = random.choice(list(SHAPES.keys()))
            
        self.shape = SHAPES[self.type]
        self.color = SHAPE_COLORS[self.type]
        self.rotation = 0
        
        # Random Rotation
        for _ in range(random.randint(0, 3)):
            self.rotate_right()

    def rotate_right(self):
        self.rotation = (self.rotation + 1) % 4
        self.shape = [(y, -x) for x, y in self.shape]

    def rotate_left(self):
        self.rotation = (self.rotation - 1) % 4
        self.shape = [(-y, x) for x, y in self.shape]
