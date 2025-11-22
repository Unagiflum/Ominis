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
        
        # Bevel Effect
        # Lighter on Top and Left
        highlight = (min(color[0] + 50, 255), min(color[1] + 50, 255), min(color[2] + 50, 255))
        # Darker on Bottom and Right
        shadow = (max(color[0] - 50, 0), max(color[1] - 50, 0), max(color[2] - 50, 0))
        
        border_width = 3
        
        # Top
        pygame.draw.polygon(self.screen, highlight, [(x, y), (x + size, y), (x + size - border_width, y + border_width), (x + border_width, y + border_width)])
        # Left
        pygame.draw.polygon(self.screen, highlight, [(x, y), (x + border_width, y + border_width), (x + border_width, y + size - border_width), (x, y + size)])
        
        # Bottom
        pygame.draw.polygon(self.screen, shadow, [(x, y + size), (x + size, y + size), (x + size - border_width, y + size - border_width), (x + border_width, y + size - border_width)])
        # Right
        pygame.draw.polygon(self.screen, shadow, [(x + size, y), (x + size, y + size), (x + size - border_width, y + size - border_width), (x + size - border_width, y + border_width)])
        
        # No inner square, just the bevel for a clean look

    def draw_grid(self, grid, offset_x, offset_y, row_offsets=None, flash_lines=None):
        # Draw grid background
        pygame.draw.rect(self.screen, self.grid_bg_color, 
                         (offset_x, offset_y, grid.width * grid.cell_size, grid.height * grid.cell_size))
        
        # Clip to grid area
        clip_rect = pygame.Rect(offset_x, offset_y, grid.width * grid.cell_size, grid.height * grid.cell_size)
        self.screen.set_clip(clip_rect)
        
        # Draw locked blocks
        for y, row in enumerate(grid.grid):
            # Calculate visual Y position
            visual_y = offset_y + y * grid.cell_size
            if row_offsets and y < len(row_offsets):
                visual_y += row_offsets[y]
                
            for x, color in enumerate(row):
                if color != (0, 0, 0):
                    self.draw_block(offset_x + x * grid.cell_size, 
                                    visual_y, 
                                    grid.cell_size, color)
                                    
        # Draw Flash Effect
        if flash_lines:
            s = pygame.Surface((grid.width * grid.cell_size, grid.cell_size))
            s.set_alpha(150)
            s.fill((255, 255, 255))
            for line in flash_lines:
                self.screen.blit(s, (offset_x, offset_y + line * grid.cell_size))
                
        self.screen.set_clip(None) # Reset clip

    def draw_pentomino(self, pentomino, offset_x, offset_y, cell_size):
        # Clip to grid area for current piece too
        # Assuming grid dimensions are known or passed. 
        # For simplicity, we'll just rely on the fact that draw_block clips if we set clip? 
        # No, we unset clip above. Let's re-clip or just check bounds.
        
        # Better: check bounds manually to avoid drawing outside
        # But we need the grid dimensions. 
        # Let's just use the same clip rect logic if possible, or pass it in.
        # For now, let's just draw. If it's above the board, it might draw over UI?
        # Yes, we should clip.
        
        clip_rect = pygame.Rect(offset_x, offset_y, 12 * cell_size, 24 * cell_size) # Hardcoded grid size for now or pass it
        self.screen.set_clip(clip_rect)
        
        for x, y in pentomino.shape:
            self.draw_block(offset_x + (pentomino.x + x) * cell_size, 
                            offset_y + (pentomino.y + y) * cell_size, 
                            cell_size, pentomino.color)
        
        self.screen.set_clip(None)

    def draw_ghost_pentomino(self, pentomino, offset_x, offset_y, cell_size):
        clip_rect = pygame.Rect(offset_x, offset_y, 12 * cell_size, 24 * cell_size)
        self.screen.set_clip(clip_rect)
        
        for x, y in pentomino.shape:
            px = offset_x + (pentomino.x + x) * cell_size
            py = offset_y + (pentomino.y + y) * cell_size
            
            rect = pygame.Rect(px, py, cell_size, cell_size)
            # Draw outline only
            pygame.draw.rect(self.screen, pentomino.color, rect, 2) # 2px border
            
        self.screen.set_clip(None)
            
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
