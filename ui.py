import pygame
import math
import time

class UI:
    def __init__(self, screen, font_path=None):
        self.screen = screen
        # Use Consolas for digital look
        self.font = pygame.font.SysFont("Consolas", 20, bold=True)
        self.large_font = pygame.font.SysFont("Consolas", 40, bold=True)
        self.score_font = pygame.font.SysFont("Consolas", 28, bold=True)
        
        self.bg_color = (10, 10, 20) # Deep dark blue/black
        self.grid_bg_color = (5, 5, 10)
        self.text_color = (200, 240, 255) # Cyan-ish white
        self.border_color = (0, 255, 255) # Cyan neon
        self.accent_color = (255, 0, 255) # Magenta neon
        
        self.start_time = time.time()

    def get_pulse(self, speed=2.0):
        t = time.time() - self.start_time
        return (math.sin(t * speed) + 1) / 2 # 0.0 to 1.0

    def draw_background(self):
        # Draw scrolling grid
        t = time.time() - self.start_time
        scroll_y = (t * 20) % 40
        scroll_x = (t * 10) % 40
        
        self.screen.fill(self.bg_color)
        
        # Draw grid lines
        grid_color = (20, 20, 40)
        for x in range(0, self.screen.get_width(), 40):
            pygame.draw.line(self.screen, grid_color, (x, 0), (x, self.screen.get_height()))
        for y in range(0, self.screen.get_height(), 40):
            draw_y = y + scroll_y
            if draw_y < self.screen.get_height():
                pygame.draw.line(self.screen, grid_color, (0, draw_y), (self.screen.get_width(), draw_y))

    def draw_block(self, x, y, size, color):
        rect = pygame.Rect(x, y, size, size)
        
        # 1. Main Block (slightly smaller for gap)
        block_rect = rect.inflate(-2, -2)
        pygame.draw.rect(self.screen, color, block_rect, border_radius=5)
        
        # 2. Inner Highlight (Top-Left)
        highlight_color = (min(color[0] + 100, 255), min(color[1] + 100, 255), min(color[2] + 100, 255))
        highlight_rect = pygame.Rect(x + 4, y + 4, size//2, size//2)
        # Draw a curve or just a rect? Let's do a simple rect for the "shine"
        # pygame.draw.rect(self.screen, highlight_color, highlight_rect, border_radius=3)
        
        # Better shine: L shape
        pygame.draw.line(self.screen, highlight_color, (x + 5, y + 5), (x + 5, y + size - 8), 2)
        pygame.draw.line(self.screen, highlight_color, (x + 5, y + 5), (x + size - 8, y + 5), 2)

        # 3. Dark Border (Bottom-Right)
        shadow_color = (max(color[0] - 50, 0), max(color[1] - 50, 0), max(color[2] - 50, 0))
        # pygame.draw.line(self.screen, shadow_color, (x + size - 5, y + 5), (x + size - 5, y + size - 5), 2)
        # pygame.draw.line(self.screen, shadow_color, (x + 5, y + size - 5), (x + size - 5, y + size - 5), 2)

    def draw_grid(self, grid, offset_x, offset_y, row_offsets=None, flash_lines=None):
        # Draw Border with Glow
        border_rect = pygame.Rect(offset_x - 5, offset_y - 5, grid.width * grid.cell_size + 10, grid.height * grid.cell_size + 10)
        
        # Pulse effect for border
        pulse = self.get_pulse(3.0)
        glow_color = (
            int(self.border_color[0] * (0.5 + 0.5 * pulse)),
            int(self.border_color[1] * (0.5 + 0.5 * pulse)),
            int(self.border_color[2] * (0.5 + 0.5 * pulse))
        )
        pygame.draw.rect(self.screen, glow_color, border_rect, 2, border_radius=8)
        
        # Draw grid background
        pygame.draw.rect(self.screen, self.grid_bg_color, 
                         (offset_x, offset_y, grid.width * grid.cell_size, grid.height * grid.cell_size),
                         border_radius=4)
        
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
            s.set_alpha(150 + int(50 * pulse))
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
            # Outline only for ghost
            pygame.draw.rect(self.screen, pentomino.color, rect.inflate(-2, -2), 2, border_radius=5)
            
        self.screen.set_clip(None)
            
    def draw_preview(self, pentomino, x, y, cell_size):
        # Draw Border Box
        box_size = 7 * cell_size
        border_rect = pygame.Rect(x, y, box_size, box_size)
        
        # Draw solid background first
        pygame.draw.rect(self.screen, self.bg_color, border_rect, border_radius=8)
        
        pygame.draw.rect(self.screen, self.border_color, border_rect, 2, border_radius=8)
        
        # Label
        label = self.font.render("NEXT", True, self.text_color)
        self.screen.blit(label, (x + box_size//2 - label.get_width()//2, y + 10))
        
        # Calculate shape dimensions
        xs = [x for x, y in pentomino.shape]
        ys = [y for x, y in pentomino.shape]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        
        shape_width = (max_x - min_x + 1) * cell_size
        shape_height = (max_y - min_y + 1) * cell_size
        
        # Center of the box
        center_x = x + box_size // 2
        center_y = y + box_size // 2 + 10 # +10 for label offset
        
        # Offset to center the shape
        start_x = center_x - shape_width // 2 - min_x * cell_size
        start_y = center_y - shape_height // 2 - min_y * cell_size
        
        for px, py in pentomino.shape:
            self.draw_block(start_x + px * cell_size, 
                            start_y + py * cell_size, 
                            cell_size, pentomino.color)

    def draw_score(self, score, level, lines, x, y):
        # Draw Border Box
        width = 210
        height = 150
        border_rect = pygame.Rect(x, y, width, height)
        
        # Draw solid background first
        pygame.draw.rect(self.screen, self.bg_color, border_rect, border_radius=8)
        
        pygame.draw.rect(self.screen, self.border_color, border_rect, 2, border_radius=8)
        
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

    def draw_instructions(self, x, y, mode="PLAYING"):
        # Draw Border Box (Same width as Preview: 7 * 30 = 210)
        box_width = 210
        # Calculate height based on lines
        line_height = 25
        padding = 10
        
        if mode == "WATCH_AI":
            instructions = [
                ("AI PLAYING", ""),
                ("F1", "Pause")
            ]
        else:
            instructions = [
                ("Arrows", "Move"),
                ("Space", "Hard Drop"),
                (", / .", "Rotate"),
                ("F1", "Pause")
            ]
            
        box_height = len(instructions) * line_height + padding * 2 + 20 # +20 for title
        
        border_rect = pygame.Rect(x, y, box_width, box_height)
        
        # Draw solid background first
        pygame.draw.rect(self.screen, self.bg_color, border_rect, border_radius=8)
        
        pygame.draw.rect(self.screen, self.border_color, border_rect, 2, border_radius=8)
        
        # Title
        title = self.font.render("CONTROLS", True, self.text_color)
        self.screen.blit(title, (x + box_width//2 - title.get_width()//2, y + padding))
        
        current_y = y + padding + 25
        
        for key, func in instructions:
            color = (150, 150, 150)
            if key == "AI PLAYING":
                color = self.accent_color
                # Center this one
                text = self.font.render(key, True, color)
                self.screen.blit(text, (x + box_width//2 - text.get_width()//2, current_y))
            else:
                # Key Left Aligned
                key_text = self.font.render(key, True, self.border_color) # Different color for keys
                self.screen.blit(key_text, (x + padding, current_y))
                
                # Function Right Aligned
                func_text = self.font.render(func, True, color)
                self.screen.blit(func_text, (x + box_width - func_text.get_width() - padding, current_y))
                
            current_y += line_height

    def draw_train_menu(self, screen_width, screen_height, params, mouse_pos, grid):
        self.draw_background()
        
        # Standard Padding
        padding = 20
        left_pane_width = 350 # Wider for sliders
        
        # Title
        title = self.large_font.render("TRAIN AI", True, self.text_color)
        self.screen.blit(title, (padding, padding))
        
        current_y = padding + 60
        
        # --- Architecture Group ---
        arch_height = 160
        self.draw_group_box(padding, current_y, left_pane_width, arch_height, "Architecture")
        
        # Hidden Layer Size (128, 256, 512)
        # Map 0.0-1.0 to index 0-2
        hl_sizes = [128, 256, 512]
        hl_size_idx = int(params['hl_size_idx'])
        hl_size_val = hl_sizes[hl_size_idx]
        
        # Slider logic handled in Game, here we just draw based on param 0.0-1.0 equivalent
        # But params passed are likely the actual values or indices? 
        # Let's assume params is a dict of values, and we might need rects for interaction.
        # Actually, for sliders, we need to return rects so Game can update them.
        
        # Let's define the slider rects here and return them in a dict
        slider_rects = {}
        
        sy = current_y + 40
        # Hidden Layer Size
        s_rect = self.draw_slider(padding + 20, sy, 200, params['hl_size_idx'] / 2.0, f"Hidden Size: {hl_size_val}", mouse_pos)
        slider_rects['hl_size'] = s_rect
        sy += 50
        
        # Hidden Layer Count (1, 2, 3, 4)
        s_rect = self.draw_slider(padding + 20, sy, 200, (params['hl_count'] - 1) / 3.0, f"Hidden Layers: {params['hl_count']}", mouse_pos)
        slider_rects['hl_count'] = s_rect
        
        current_y += arch_height + 20
        
        # --- Reward / Curriculum Group ---
        reward_height = 220
        self.draw_group_box(padding, current_y, left_pane_width, reward_height, "Reward & Curriculum")
        
        sy = current_y + 40
        # Height Penalty
        s_rect = self.draw_slider(padding + 20, sy, 200, params['height_penalty'] / 100.0, f"Height Penalty: {params['height_penalty']}%", mouse_pos)
        slider_rects['height_penalty'] = s_rect
        sy += 50
        
        # Overhang Penalty
        s_rect = self.draw_slider(padding + 20, sy, 200, params['overhang_penalty'] / 100.0, f"Overhang Penalty: {params['overhang_penalty']}%", mouse_pos)
        slider_rects['overhang_penalty'] = s_rect
        sy += 50
        
        # Max Polyomino Size (2, 3, 4, 5)
        # Map 0.0-1.0 to 2-5
        s_rect = self.draw_slider(padding + 20, sy, 200, (params['max_size'] - 2) / 3.0, f"Max Piece Size: {params['max_size']}", mouse_pos)
        slider_rects['max_size'] = s_rect
        
        
        # Bottom Controls
        bottom_y = screen_height - padding - 50 
        
        # Back Button
        btn_back_rect = self.draw_button(padding, bottom_y, 150, 40, "BACK", True, mouse_pos)
        bottom_y -= 50
        
        # Start Button
        btn_start_rect = self.draw_button(padding, bottom_y, 150, 40, "START", True, mouse_pos)
        bottom_y -= 50
        
        # Visual Mode Checkbox
        chk_rect = self.draw_checkbox(padding, bottom_y, params['visual_mode'], "Visual Mode", mouse_pos)
        
        # Draw Game Board on Right
        offset_x = left_pane_width + padding * 2
        offset_y = padding + 5
        
        self.draw_grid(grid, offset_x, offset_y)
        
        return btn_back_rect, chk_rect, btn_start_rect, slider_rects

    def draw_group_box(self, x, y, width, height, title):
        rect = pygame.Rect(x, y, width, height)
        pygame.draw.rect(self.screen, self.bg_color, rect, border_radius=8)
        pygame.draw.rect(self.screen, self.border_color, rect, 2, border_radius=8)
        
        # Title on top border
        title_surf = self.font.render(f" {title} ", True, self.accent_color)
        # Clear background behind title
        title_rect = title_surf.get_rect(topleft=(x + 20, y - 10))
        pygame.draw.rect(self.screen, self.bg_color, title_rect)
        self.screen.blit(title_surf, title_rect)

    def draw_game_over(self, x, y, width, height):
        overlay = pygame.Surface((width, height))
        overlay.set_alpha(200)
        overlay.fill((0, 0, 0))
        self.screen.blit(overlay, (x, y))
        
        # Pulsing Game Over text
        pulse = self.get_pulse(5.0)
        color_val = 150 + int(105 * pulse)
        text_color = (color_val, 50, 50)
        
        text = self.large_font.render("GAME OVER", True, text_color)
        rect = text.get_rect(center=(x + width // 2, y + height // 2))
        self.screen.blit(text, rect)

    def draw_pause_screen(self, x, y, width, height):
        overlay = pygame.Surface((width, height))
        overlay.set_alpha(150)
        overlay.fill((0, 0, 0))
        self.screen.blit(overlay, (x, y))
        
        text = self.large_font.render("PAUSED", True, self.text_color)
        rect = text.get_rect(center=(x + width // 2, y + height // 2))
        self.screen.blit(text, rect)
        
        resume_text = self.font.render("Press F1 to Resume", True, self.text_color)
        resume_rect = resume_text.get_rect(center=(x + width // 2, y + height // 2 + 60))
        self.screen.blit(resume_text, resume_rect)

    def draw_checkbox(self, x, y, checked, label, mouse_pos=None):
        # Box
        rect = pygame.Rect(x, y, 20, 20)
        
        # Hover effect
        color = self.text_color
        if mouse_pos and rect.collidepoint(mouse_pos):
            color = self.accent_color
            
        pygame.draw.rect(self.screen, color, rect, 2, border_radius=4)
        if checked:
            pygame.draw.rect(self.screen, color, (x + 4, y + 4, 12, 12), border_radius=2)
            
        # Label
        text = self.font.render(label, True, color)
        self.screen.blit(text, (x + 30, y))
        
        return rect

    def draw_button(self, x, y, width, height, label, active, mouse_pos=None):
        rect = pygame.Rect(x, y, width, height)
        
        # Draw solid background first
        pygame.draw.rect(self.screen, self.bg_color, rect, border_radius=10)
        
        color = self.text_color if active else (100, 100, 100)
        
        # Hover effect
        if active and mouse_pos and rect.collidepoint(mouse_pos):
            color = self.accent_color
            # Fill slightly
            s = pygame.Surface((width, height))
            s.set_alpha(50)
            s.fill(color)
            self.screen.blit(s, (x, y))
            
        pygame.draw.rect(self.screen, color, rect, 2, border_radius=10)
        
        text = self.font.render(label, True, color)
        text_rect = text.get_rect(center=rect.center)
        self.screen.blit(text, text_rect)
        
        return rect

    def draw_slider(self, x, y, width, value, label, mouse_pos=None):
        # Label
        text = self.font.render(label, True, self.text_color)
        self.screen.blit(text, (x + width//2 - text.get_width()//2, y - 25))
        
        # Bar
        bar_rect = pygame.Rect(x, y, width, 10)
        pygame.draw.rect(self.screen, (50, 50, 50), bar_rect, border_radius=5)
        
        # Active part of bar
        active_width = int(value * width)
        active_rect = pygame.Rect(x, y, active_width, 10)
        pygame.draw.rect(self.screen, self.border_color, active_rect, border_radius=5)
        
        # Handle
        handle_x = x + active_width
        handle_rect = pygame.Rect(handle_x - 8, y - 8, 16, 26)
        
        color = self.text_color
        if mouse_pos and handle_rect.collidepoint(mouse_pos):
            color = self.accent_color
            
        pygame.draw.rect(self.screen, color, handle_rect, border_radius=4)
        
        return bar_rect
