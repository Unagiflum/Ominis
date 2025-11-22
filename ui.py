import pygame

class UI:
    def __init__(self, screen, font_path=None):
        self.screen = screen
        # Use Consolas for digital look
        self.font = pygame.font.SysFont("Consolas", 20, bold=True)
        self.large_font = pygame.font.SysFont("Consolas", 40, bold=True)
        self.score_font = pygame.font.SysFont("Consolas", 28, bold=True)
        
        self.bg_color = (20, 20, 25) # Dark background
        self.grid_bg_color = (10, 10, 15)
        self.text_color = (220, 220, 220)
        self.border_color = (100, 100, 100) # Grey border

    def draw_block(self, x, y, size, color):
        rect = pygame.Rect(x, y, size, size)
        pygame.draw.rect(self.screen, color, rect)
        
        # Bevel Effect
        highlight = (min(color[0] + 50, 255), min(color[1] + 50, 255), min(color[2] + 50, 255))
        shadow = (max(color[0] - 50, 0), max(color[1] - 50, 0), max(color[2] - 50, 0))
        border_width = 3
        
        pygame.draw.polygon(self.screen, highlight, [(x, y), (x + size, y), (x + size - border_width, y + border_width), (x + border_width, y + border_width)])
        pygame.draw.polygon(self.screen, highlight, [(x, y), (x + border_width, y + border_width), (x + border_width, y + size - border_width), (x, y + size)])
        pygame.draw.polygon(self.screen, shadow, [(x, y + size), (x + size, y + size), (x + size - border_width, y + size - border_width), (x + border_width, y + size - border_width)])
        pygame.draw.polygon(self.screen, shadow, [(x + size, y), (x + size, y + size), (x + size - border_width, y + size - border_width), (x + size - border_width, y + border_width)])

    def draw_grid(self, grid, offset_x, offset_y, row_offsets=None, flash_lines=None):
        # Draw Border
        border_rect = pygame.Rect(offset_x - 5, offset_y - 5, grid.width * grid.cell_size + 10, grid.height * grid.cell_size + 10)
        pygame.draw.rect(self.screen, self.border_color, border_rect, 2)
        
        # Draw grid background
        pygame.draw.rect(self.screen, self.grid_bg_color, 
                         (offset_x, offset_y, grid.width * grid.cell_size, grid.height * grid.cell_size))
        
        # Clip to grid area
        clip_rect = pygame.Rect(offset_x, offset_y, grid.width * grid.cell_size, grid.height * grid.cell_size)
        self.screen.set_clip(clip_rect)
        
        # Draw locked blocks
        for y, row in enumerate(grid.grid):
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
                
        self.screen.set_clip(None)

    def draw_pentomino(self, pentomino, offset_x, offset_y, cell_size):
        clip_rect = pygame.Rect(offset_x, offset_y, 12 * cell_size, 24 * cell_size)
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
            pygame.draw.rect(self.screen, pentomino.color, rect, 2)
            
        self.screen.set_clip(None)
            
    def draw_preview(self, pentomino, x, y, cell_size):
        # Draw Border Box
        box_size = 7 * cell_size
        border_rect = pygame.Rect(x, y, box_size, box_size)
        pygame.draw.rect(self.screen, self.border_color, border_rect, 2)
        
        # Label
        label = self.font.render("NEXT", True, self.text_color)
        self.screen.blit(label, (x + box_size//2 - label.get_width()//2, y + 10))
        
        # Draw piece centered
        center_x = x + box_size // 2
        center_y = y + box_size // 2 + 10
        
        for px, py in pentomino.shape:
            self.draw_block(center_x + px * cell_size, 
                            center_y + py * cell_size, 
                            cell_size, pentomino.color)

    def draw_score(self, score, level, lines, x, y):
        # Draw Border Box
        width = 210
        height = 150
        border_rect = pygame.Rect(x, y, width, height)
        pygame.draw.rect(self.screen, self.border_color, border_rect, 2)
        
        # Text
        score_lbl = self.font.render("SCORE", True, self.border_color)
        score_val = self.score_font.render(str(score), True, self.text_color)
        
        level_lbl = self.font.render("LEVEL", True, self.border_color)
        level_val = self.score_font.render(str(level), True, self.text_color)
        
        lines_lbl = self.font.render("LINES", True, self.border_color)
        lines_val = self.score_font.render(str(lines), True, self.text_color)
        
        # Positioning
        padding = 10
        current_y = y + padding
        
        self.screen.blit(score_lbl, (x + padding, current_y))
        self.screen.blit(score_val, (x + width - score_val.get_width() - padding, current_y))
        current_y += 40
        
        self.screen.blit(level_lbl, (x + padding, current_y))
        self.screen.blit(level_val, (x + width - level_val.get_width() - padding, current_y))
        current_y += 40
        
        self.screen.blit(lines_lbl, (x + padding, current_y))
        self.screen.blit(lines_val, (x + width - lines_val.get_width() - padding, current_y))

    def draw_instructions(self, x, y):
        instructions = [
            "CONTROLS:",
            "Arrows: Move",
            "Space: Hard Drop",
            ", / . : Rotate",
            "F1: Pause",
            "ESC: Menu"
        ]
        
        current_y = y
        for line in instructions:
            text = self.font.render(line, True, (150, 150, 150))
            self.screen.blit(text, (x, current_y))
            current_y += 25

    def draw_game_over(self, screen_width, screen_height):
        overlay = pygame.Surface((screen_width, screen_height))
        overlay.set_alpha(200)
        overlay.fill((0, 0, 0))
        self.screen.blit(overlay, (0, 0))
        
        text = self.large_font.render("GAME OVER", True, (255, 50, 50))
        rect = text.get_rect(center=(screen_width // 2, screen_height // 2))
        self.screen.blit(text, rect)
        
        restart_text = self.font.render("Press ENTER to Restart", True, self.text_color)
        restart_rect = restart_text.get_rect(center=(screen_width // 2, screen_height // 2 + 60))
        self.screen.blit(restart_text, restart_rect)

    def draw_pause_screen(self, screen_width, screen_height):
        overlay = pygame.Surface((screen_width, screen_height))
        overlay.set_alpha(150)
        overlay.fill((0, 0, 0))
        self.screen.blit(overlay, (0, 0))
        
        text = self.large_font.render("PAUSED", True, self.text_color)
        rect = text.get_rect(center=(screen_width // 2, screen_height // 2))
        self.screen.blit(text, rect)
        
        resume_text = self.font.render("Press F1 to Resume", True, self.text_color)
        resume_rect = resume_text.get_rect(center=(screen_width // 2, screen_height // 2 + 60))
        self.screen.blit(resume_text, resume_rect)

    def draw_checkbox(self, x, y, checked, label):
        # Box
        rect = pygame.Rect(x, y, 20, 20)
        pygame.draw.rect(self.screen, self.text_color, rect, 2)
        if checked:
            pygame.draw.rect(self.screen, self.text_color, (x + 4, y + 4, 12, 12))
            
        # Label
        text = self.font.render(label, True, self.text_color)
        self.screen.blit(text, (x + 30, y))
        
        return rect

    def draw_button(self, x, y, width, height, label, active):
        rect = pygame.Rect(x, y, width, height)
        color = self.text_color if active else (100, 100, 100)
        pygame.draw.rect(self.screen, color, rect, 2)
        
        text = self.font.render(label, True, color)
        text_rect = text.get_rect(center=rect.center)
        self.screen.blit(text, text_rect)
        
        return rect

    def draw_slider(self, x, y, width, value, label):
        # Label
        text = self.font.render(label, True, self.text_color)
        self.screen.blit(text, (x, y - 25))
        
        # Bar
        bar_rect = pygame.Rect(x, y, width, 10)
        pygame.draw.rect(self.screen, (100, 100, 100), bar_rect)
        
        # Handle
        handle_x = x + int(value * width)
        handle_rect = pygame.Rect(handle_x - 5, y - 5, 10, 20)
        pygame.draw.rect(self.screen, self.text_color, handle_rect)
        
        return bar_rect
