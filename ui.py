import pygame

class UI:
    def __init__(self, screen, font_path=None):
        self.screen = screen
        self.font = pygame.font.Font(font_path, 24) if font_path else pygame.font.SysFont("Arial", 24)
        self.large_font = pygame.font.Font(font_path, 48) if font_path else pygame.font.SysFont("Arial", 48)
        self.bg_color = (20, 20, 25) # Dark background
        self.grid_bg_color = (30, 30, 35)
        self.text_color = (220, 220, 220)

    def draw_block(self, x, y, size, color):
        rect = pygame.Rect(x, y, size, size)
        pygame.draw.rect(self.screen, color, rect)
        
        # Subtle texture/bevel effect
        highlight = (min(color[0] + 30, 255), min(color[1] + 30, 255), min(color[2] + 30, 255))
        shadow = (max(color[0] - 30, 0), max(color[1] - 30, 0), max(color[2] - 30, 0))
        
        # Top-Left Highlight
        pygame.draw.line(self.screen, highlight, (x, y), (x + size, y), 2)
        pygame.draw.line(self.screen, highlight, (x, y), (x, y + size), 2)
        
        # Bottom-Right Shadow
        pygame.draw.line(self.screen, shadow, (x, y + size), (x + size, y + size), 2)
        pygame.draw.line(self.screen, shadow, (x + size, y), (x + size, y + size), 2)
        
        # Inner square for "texture" feel
        inner_rect = pygame.Rect(x + 4, y + 4, size - 8, size - 8)
        pygame.draw.rect(self.screen, (color[0], color[1], color[2], 100), inner_rect, 1)

    def draw_grid(self, grid, offset_x, offset_y):
        # Draw grid background
        pygame.draw.rect(self.screen, self.grid_bg_color, 
                         (offset_x, offset_y, grid.width * grid.cell_size, grid.height * grid.cell_size))
        
        # Draw locked blocks
        for y, row in enumerate(grid.grid):
            for x, color in enumerate(row):
                if color != (0, 0, 0):
                    self.draw_block(offset_x + x * grid.cell_size, 
                                    offset_y + y * grid.cell_size, 
                                    grid.cell_size, color)

    def draw_pentomino(self, pentomino, offset_x, offset_y, cell_size):
        for x, y in pentomino.shape:
            self.draw_block(offset_x + (pentomino.x + x) * cell_size, 
                            offset_y + (pentomino.y + y) * cell_size, 
                            cell_size, pentomino.color)
            
    def draw_preview(self, pentomino, x, y, cell_size):
        # Draw label
        label = self.font.render("NEXT", True, self.text_color)
        self.screen.blit(label, (x, y - 30))
        
        # Draw piece centered in a box
        # Approximate center of 5x5 box
        center_x = x + 2.5 * cell_size
        center_y = y + 2.5 * cell_size
        
        for px, py in pentomino.shape:
            self.draw_block(center_x + px * cell_size, 
                            center_y + py * cell_size, 
                            cell_size, pentomino.color)

    def draw_score(self, score, level, lines, x, y):
        score_text = self.font.render(f"SCORE: {score}", True, self.text_color)
        level_text = self.font.render(f"LEVEL: {level}", True, self.text_color)
        lines_text = self.font.render(f"LINES: {lines}", True, self.text_color)
        
        self.screen.blit(score_text, (x, y))
        self.screen.blit(level_text, (x, y + 40))
        self.screen.blit(lines_text, (x, y + 80))

    def draw_game_over(self, screen_width, screen_height):
        overlay = pygame.Surface((screen_width, screen_height))
        overlay.set_alpha(200)
        overlay.fill((0, 0, 0))
        self.screen.blit(overlay, (0, 0))
        
        text = self.large_font.render("GAME OVER", True, (255, 50, 50))
        rect = text.get_rect(center=(screen_width // 2, screen_height // 2))
        self.screen.blit(text, rect)
        
        restart_text = self.font.render("Press SPACE to Restart", True, self.text_color)
        restart_rect = restart_text.get_rect(center=(screen_width // 2, screen_height // 2 + 60))
        self.screen.blit(restart_text, restart_rect)
