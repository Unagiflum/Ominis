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
    
    'N': [(-1, 0), (0, 0), (0, -1), (1, -1), (1, -2)],
    'N_mirror': [(1, 0), (0, 0), (0, -1), (-1, -1), (-1, -2)],
    
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
}

class Pentomino:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.type = random.choice(list(SHAPES.keys()))
        self.shape = SHAPES[self.type]
        self.color = SHAPE_COLORS[self.type]
        self.rotation = 0

    def rotate_right(self):
        self.rotation = (self.rotation + 1) % 4
        self.shape = [(y, -x) for x, y in self.shape]

    def rotate_left(self):
        self.rotation = (self.rotation - 1) % 4
        self.shape = [(-y, x) for x, y in self.shape]
