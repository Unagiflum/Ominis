import pygame
import math
import time

class UI:
    def __init__(self, screen, font_path=None):
        self.screen = screen
        # Use Consolas for digital look
        self.font = pygame.font.SysFont("Consolas", 20, bold=True)
        self.small_font = pygame.font.SysFont("Consolas", 14, bold=True)
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

    def _truncate_text(self, text, max_width, font):
        if font.size(text)[0] <= max_width:
            return text
        trimmed = text
        while trimmed and font.size(trimmed + "...")[0] > max_width:
            trimmed = trimmed[:-1]
        return (trimmed + "...") if trimmed else "..."

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

    def draw_train_menu(self, screen_width, screen_height, params, mouse_pos, grid, is_training=False, volume=0.5, epsilon_bump_val=0.2):

        self.draw_background()
        
        # Standard Padding
        padding = 20
        left_pane_width = 210 # Standard width (same as score/preview)
        
        # Title
        title = self.large_font.render("TRAIN AI", True, self.text_color)
        self.screen.blit(title, (padding, padding))
        
        current_y = padding + 60
        
        # --- Architecture Group ---
        arch_height = 160
        self.draw_group_box(padding, current_y, left_pane_width, arch_height, "Architecture")
        
        # Hidden Layer Size (128, 256, 512, 1024, 2048)
        hl_sizes = [128, 256, 512, 1024, 2048]
        max_hl_idx = max(1, len(hl_sizes) - 1)
        hl_size_idx = max(0, min(max_hl_idx, int(params.get('hl_size_idx', 1))))
        hl_size_val = hl_sizes[hl_size_idx]
        
        slider_rects = {}
        slider_width = 170
        slider_x = padding + (left_pane_width - slider_width) // 2
        slider_bar_offset = -3
        
        sy = current_y + 40
        # Hidden Layer Size
        s_rect = self.draw_slider(slider_x, sy, slider_width, hl_size_idx / max_hl_idx, f"Hidden Size: {hl_size_val}", mouse_pos, self.small_font, active=not is_training, bar_offset_y=slider_bar_offset)
        if not is_training: slider_rects['hl_size'] = s_rect
        sy += 50
        
        # Hidden Layer Count (1, 2, 3, 4)
        s_rect = self.draw_slider(slider_x, sy, slider_width, (params['hl_count'] - 1) / 3.0, f"Hidden Layers: {params['hl_count']}", mouse_pos, self.small_font, active=not is_training, bar_offset_y=slider_bar_offset)
        if not is_training: slider_rects['hl_count'] = s_rect
        
        # Epsilon Reset Button
        # Moved up (sy + 30 instead of 45) and smaller height (25 instead of 30)
        btn_eps_reset_rect = self.draw_button(slider_x, sy + 30, slider_width, 25, f"Eps {epsilon_bump_val:.2f}", not is_training, mouse_pos, font=self.small_font)



        
        current_y += arch_height + 20
        
        # --- Reward / Curriculum Group ---
        reward_height = 270
        self.draw_group_box(padding, current_y, left_pane_width, reward_height, "Reward & Curriculum")
        
        sy = current_y + 35
        # Current / Min Rand. % (0% - 20%)
        min_rand_raw = params.get('epsilon_min_percent', 5)
        current_rand_raw = params.get('epsilon_current_percent', 20)
        min_rand = max(0, min(20, min_rand_raw))
        current_rand = max(0, min(20, current_rand_raw))
        floor_rand = min(min_rand, current_rand)
        current_rand = max(min_rand, current_rand)
        label = f"Randomness: {current_rand}%->{floor_rand}%"
        s_rect = self.draw_range_slider(slider_x, sy, slider_width, floor_rand / 20.0, current_rand / 20.0, label, mouse_pos, self.small_font, active=not is_training, bar_offset_y=slider_bar_offset)
        if not is_training: slider_rects['epsilon_range_percent'] = s_rect
        sy += 45
        
        # Learning Rate (0.0001 - 0.0050)
        lr = params.get('learning_rate', 0.001)
        lr = max(0.0001, min(0.005, lr))
        s_rect = self.draw_slider(slider_x, sy, slider_width, (lr - 0.0001) / 0.0049, f"Learning Rate: {lr:.4f}", mouse_pos, self.small_font, active=not is_training, bar_offset_y=slider_bar_offset)
        if not is_training: slider_rects['learning_rate'] = s_rect
        sy += 45
        
        # Piece Size Range (1-5)
        min_size = params.get('min_size', 1)
        max_size = params.get('max_size', 5)
        min_size = max(1, min(5, int(min_size)))
        max_size = max(1, min(5, int(max_size)))
        floor_size = min(min_size, max_size)
        ceil_size = max(min_size, max_size)
        label = f"Piece Size: {floor_size}->{ceil_size}"
        s_rect = self.draw_range_slider(slider_x, sy, slider_width, (floor_size - 1) / 4.0, (ceil_size - 1) / 4.0, label, mouse_pos, self.small_font, active=not is_training, bar_offset_y=slider_bar_offset)
        if not is_training: slider_rects['piece_size_range'] = s_rect
        sy += 45

        # Big Piece Weight (1 - 4)
        big_weight = params.get('big_piece_weight', 1)
        big_weight = max(1, min(4, int(big_weight)))
        s_rect = self.draw_slider(slider_x, sy, slider_width, (big_weight - 1) / 3.0, f"Big Piece Weight: {big_weight}", mouse_pos, self.small_font, active=not is_training, bar_offset_y=slider_bar_offset)
        if not is_training: slider_rects['big_piece_weight'] = s_rect
        sy += 45
        
        # Short Game Length (how many pieces before auto-restart in short games mode)
        short_game_length = max(1, min(20, params.get('pieces_tracked', 10)))
        s_rect = self.draw_slider(slider_x, sy, slider_width, (short_game_length - 1) / 19.0, f"Piece History: {short_game_length}", mouse_pos, self.small_font, active=not is_training, bar_offset_y=slider_bar_offset)
        if not is_training: slider_rects['pieces_tracked'] = s_rect
        sy += 25
        
        # Short Games Checkbox
        # Use smaller font for this one to fit
        short_games_chk_rect = self.draw_checkbox(slider_x, sy, params.get('short_games', False), "Short Games", mouse_pos, active=not is_training, font=self.small_font)
        
        # Bottom Controls
        bottom_y = screen_height - padding - 50 
        
        # Back Button
        btn_back_rect = self.draw_button(padding, bottom_y, 150, 40, "BACK", True, mouse_pos)
        bottom_y -= 50
        
        # Start/Stop Button
        label = "STOP" if is_training else "START"
        btn_start_rect = self.draw_button(padding, bottom_y, 150, 40, label, True, mouse_pos)
        bottom_y -= 50
        
        # Visual Mode Checkbox
        chk_rect = self.draw_checkbox(padding, bottom_y, params['visual_mode'], "Visual Mode", mouse_pos)
        bottom_y -= 30

        # Volume Slider
        # Only relevant if visual mode is on? Or always show? User said "need not do anything when visual mode is off".
        # Let's show it always for consistency.
        vol_slider_rect = self.draw_slider(padding + 10, bottom_y, 150, volume, "Volume", mouse_pos, self.small_font, bar_offset_y=slider_bar_offset)
        
        # Draw Game Board on Right
        offset_x = left_pane_width + padding * 2
        offset_y = padding + 5
        
        self.draw_grid(grid, offset_x, offset_y)
        
        return btn_back_rect, chk_rect, btn_start_rect, slider_rects, vol_slider_rect, short_games_chk_rect, btn_eps_reset_rect


    def draw_group_box(self, x, y, width, height, title):
        rect = pygame.Rect(x, y, width, height)
        pygame.draw.rect(self.screen, self.bg_color, rect, border_radius=8)
        pygame.draw.rect(self.screen, self.border_color, rect, 2, border_radius=8)
        
        # Title on top border
        title_surf = self.small_font.render(f" {title} ", True, self.accent_color)
        # Clear background behind title
        title_rect = title_surf.get_rect(topleft=(x + 10, y - 8))
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

    def draw_checkbox(self, x, y, checked, label, mouse_pos=None, active=True, font=None):
        # Box
        rect = pygame.Rect(x, y, 20, 20)
        
        # Hover effect
        base_color = self.text_color if active else (100, 100, 100)
        color = base_color
        if active and mouse_pos and rect.collidepoint(mouse_pos):
            color = self.accent_color
            
        pygame.draw.rect(self.screen, color, rect, 2, border_radius=4)
        if checked:
            pygame.draw.rect(self.screen, color, (x + 4, y + 4, 12, 12), border_radius=2)
            
        # Label
        use_font = font if font else self.font
        text = use_font.render(label, True, color)
        self.screen.blit(text, (x + 30, y))
        
        return rect

    def draw_button(self, x, y, width, height, label, active, mouse_pos=None, font=None):
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
        
        use_font = font if font else self.font
        text = use_font.render(label, True, color)
        text_rect = text.get_rect(center=rect.center)
        self.screen.blit(text, text_rect)
        
        return rect


    def draw_dropdown(self, x, y, width, height, label, options, selected_idx=None, is_open=False, mouse_pos=None, font=None, option_width=None, list_options=None):
        if font is None:
            font = self.font
        rect = pygame.Rect(x, y, width, height)

        # Base box
        pygame.draw.rect(self.screen, self.bg_color, rect, border_radius=10)
        color = self.text_color
        if mouse_pos and rect.collidepoint(mouse_pos):
            color = self.accent_color
        pygame.draw.rect(self.screen, color, rect, 2, border_radius=10)

        display_text = label
        if selected_idx is not None and 0 <= selected_idx < len(options):
            display_text = options[selected_idx]
        display_text = self._truncate_text(display_text, width - 26, font)
        text = font.render(display_text, True, color)
        self.screen.blit(text, (x + 8, y + height // 2 - text.get_height() // 2))

        # Dropdown arrow
        arrow_x = x + width - 16
        arrow_y = y + height // 2 - 3
        pygame.draw.polygon(self.screen, color, [
            (arrow_x - 6, arrow_y),
            (arrow_x + 6, arrow_y),
            (arrow_x, arrow_y + 6)
        ])

        option_rects = []
        list_items = list_options if list_options is not None else options
        list_width = option_width if option_width is not None else width
        if is_open:
            if not list_items:
                empty_rect = pygame.Rect(x, y + height, list_width, height)
                pygame.draw.rect(self.screen, self.bg_color, empty_rect, border_radius=8)
                pygame.draw.rect(self.screen, (100, 100, 100), empty_rect, 2, border_radius=8)
                empty_text = font.render("No models found", True, (120, 120, 120))
                self.screen.blit(empty_text, (empty_rect.x + 8, empty_rect.y + height // 2 - empty_text.get_height() // 2))
            else:
                for idx, option in enumerate(list_items):
                    opt_rect = pygame.Rect(x, y + height + idx * height, list_width, height)
                    pygame.draw.rect(self.screen, self.bg_color, opt_rect, border_radius=8)
                    opt_color = self.text_color
                    if mouse_pos and opt_rect.collidepoint(mouse_pos):
                        opt_color = self.accent_color
                        s = pygame.Surface((list_width, height))
                        s.set_alpha(50)
                        s.fill(opt_color)
                        self.screen.blit(s, (opt_rect.x, opt_rect.y))
                    pygame.draw.rect(self.screen, opt_color, opt_rect, 2, border_radius=8)
                    opt_surf = font.render(option, True, opt_color)
                    self.screen.blit(opt_surf, (opt_rect.x + 8, opt_rect.y + height // 2 - opt_surf.get_height() // 2))
                    option_rects.append(opt_rect)

        return rect, option_rects

    def draw_slider(self, x, y, width, value, label, mouse_pos=None, font=None, active=True, bar_offset_y=0):
        if font is None:
            font = self.font
        # Label
        label_color = self.text_color if active else (120, 120, 120)
        text = font.render(label, True, label_color)
        self.screen.blit(text, (x + width//2 - text.get_width()//2, y - 25))
        
        # Bar
        bar_y = y + bar_offset_y
        bar_rect = pygame.Rect(x, bar_y, width, 10)
        bar_color = (50, 50, 50) if active else (35, 35, 35)
        pygame.draw.rect(self.screen, bar_color, bar_rect, border_radius=5)
        
        # Active part of bar
        active_width = int(value * width)
        active_rect = pygame.Rect(x, bar_y, active_width, 10)
        active_color = self.border_color if active else (80, 80, 80)
        pygame.draw.rect(self.screen, active_color, active_rect, border_radius=5)
        
        # Handle
        handle_x = x + active_width
        handle_rect = pygame.Rect(handle_x - 8, bar_y - 8, 16, 26)
        
        color = self.text_color if active else (100, 100, 100)
        if active and mouse_pos and handle_rect.collidepoint(mouse_pos):
            color = self.accent_color
            
        pygame.draw.rect(self.screen, color, handle_rect, border_radius=4)
        
        return bar_rect

    def draw_range_slider(self, x, y, width, min_value, max_value, label, mouse_pos=None, font=None, active=True, bar_offset_y=0):
        if font is None:
            font = self.font
        # Label
        label_color = self.text_color if active else (120, 120, 120)
        text = font.render(label, True, label_color)
        self.screen.blit(text, (x + width//2 - text.get_width()//2, y - 25))
        
        # Bar
        bar_y = y + bar_offset_y
        bar_rect = pygame.Rect(x, bar_y, width, 10)
        bar_color = (50, 50, 50) if active else (35, 35, 35)
        pygame.draw.rect(self.screen, bar_color, bar_rect, border_radius=5)
        
        # Active range
        start = int(min(min_value, max_value) * width)
        end = int(max(min_value, max_value) * width)
        active_color = self.border_color if active else (80, 80, 80)
        if end > start:
            active_rect = pygame.Rect(x + start, bar_y, end - start, 10)
            pygame.draw.rect(self.screen, active_color, active_rect, border_radius=5)
        
        # Handles
        min_handle_x = x + start
        max_handle_x = x + end
        min_handle_rect = pygame.Rect(min_handle_x - 8, bar_y - 8, 16, 26)
        max_handle_rect = pygame.Rect(max_handle_x - 8, bar_y - 8, 16, 26)
        
        color = self.text_color if active else (100, 100, 100)
        min_color = color
        max_color = color
        if active and mouse_pos:
            if min_handle_rect.collidepoint(mouse_pos):
                min_color = self.accent_color
            if max_handle_rect.collidepoint(mouse_pos):
                max_color = self.accent_color
        
        pygame.draw.rect(self.screen, min_color, min_handle_rect, border_radius=4)
        pygame.draw.rect(self.screen, max_color, max_handle_rect, border_radius=4)
        
        return bar_rect
