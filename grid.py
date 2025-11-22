class Grid:
    def __init__(self, width, height, cell_size):
        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.grid = [[(0, 0, 0) for _ in range(width)] for _ in range(height)]

    def check_collision(self, pentomino, offset_x=0, offset_y=0):
        for x, y in pentomino.shape:
            nx = pentomino.x + x + offset_x
            ny = pentomino.y + y + offset_y
            
            if nx < 0 or nx >= self.width or ny >= self.height:
                return True
            if ny >= 0 and self.grid[ny][nx] != (0, 0, 0):
                return True
        return False

    def lock_shape(self, pentomino):
        for x, y in pentomino.shape:
            nx = pentomino.x + x
            ny = pentomino.y + y
            if 0 <= ny < self.height and 0 <= nx < self.width:
                self.grid[ny][nx] = pentomino.color

    def clear_lines(self):
        lines_cleared = 0
        new_grid = [row for row in self.grid if (0, 0, 0) in row]
        lines_cleared = self.height - len(new_grid)
        for _ in range(lines_cleared):
            new_grid.insert(0, [(0, 0, 0) for _ in range(self.width)])
        self.grid = new_grid
        return lines_cleared
