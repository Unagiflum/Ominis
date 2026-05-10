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
        
        self.bg_color = (30, 30, 30) # Plain dark gray
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
        self.screen.fill(self.bg_color)

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

        if not pentomino:
            return
        
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

    def draw_train_menu(self, screen_width, screen_height, params, mouse_pos, grid, is_training=False, volume=0.5, save_active=True, reward_active=None, reward_input_text=""):

        self.draw_background()
        
        # Standard Padding
        padding = 20
        left_pane_width = 210 # Standard width (same as score/preview)
        slider_bar_offset = -3
        
        # Title
        title = self.large_font.render("TRAIN AI", True, self.text_color)
        self.screen.blit(title, (padding, padding))
        
        current_y = padding + 45

        slider_rects = {}
        short_games_chk_rect = None
        dropdown_rect = None
        reward_input_rects = {}
        reward_lines_squared_rect = None
        reset_defaults_rect = None

        show_training_board = is_training and params.get('visual_mode', False)
        show_settings = not show_training_board

        if show_settings:
            # --- Architecture Group ---
            arch_height = 145
            self.draw_group_box(padding, current_y, left_pane_width, arch_height, "Architecture")
            
            # Hidden Layer Size (16, 32, 64, 128, 256, 512, 1024, 2048)
            hl_sizes = [16, 32, 64, 128, 256, 512, 1024, 2048]
            max_hl_idx = max(1, len(hl_sizes) - 1)
            hl_size_idx = max(0, min(max_hl_idx, int(params.get('hl_size_idx', 1))))
            hl_size_val = hl_sizes[hl_size_idx]
            
            slider_width = 170
            slider_x = padding + (left_pane_width - slider_width) // 2
            
            sy = current_y + 40
            # Hidden Layer Size
            s_rect = self.draw_slider(slider_x, sy, slider_width, hl_size_idx / max_hl_idx, f"Hidden Size: {hl_size_val}", mouse_pos, self.small_font, active=not is_training, bar_offset_y=slider_bar_offset)
            if not is_training: slider_rects['hl_size'] = s_rect
            sy += 40
            
            # Hidden Layer Count (1, 2, 3, 4)
            s_rect = self.draw_slider(slider_x, sy, slider_width, (params['hl_count'] - 1) / 3.0, f"Hidden Layers: {params['hl_count']}", mouse_pos, self.small_font, active=not is_training, bar_offset_y=slider_bar_offset)
            if not is_training: slider_rects['hl_count'] = s_rect
            
            dropdown_rect = pygame.Rect(slider_x, sy + 25, slider_width, 25)

            # --- Rewards Group ---
            reward_x = padding + left_pane_width + padding
            reward_width = screen_width - reward_x - padding
            reward_y = current_y
            reward_enabled = not is_training

            reward_rows = [
                {"label": "Holes", "left": "reward_hole_decrease", "right": "reward_hole_increase"},
                {"label": "Jaggedness", "left": "reward_jaggedness_decrease", "right": "reward_jaggedness_increase"},
                {"label": "Pits", "left": "reward_pits_decrease", "right": "reward_pits_increase"},
                {"label": "Max Height", "left": None, "right": "reward_max_height_increase"},
                {"label": "Height St.dev", "left": "reward_height_std_decrease", "right": "reward_height_std_increase"},
                {"separator": True},
                {"label": "Game Over", "left": "reward_game_over", "right": None, "center": True},
                {"label": "Lines Cleared", "left": "reward_lines_cleared", "right": None, "center": True},
                {"label": "High line mult.", "left": "reward_high_line_mult", "right": None, "center": True},
                {"label": "Scale line reward per lines squared", "checkbox": True},
            ]

            def format_reward(value):
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    value = 0.0
                if abs(value) < 0.0005:
                    value = 0.0
                return f"{value:.3f}"

            def reward_display(key):
                if key == reward_active:
                    return reward_input_text
                return format_reward(params.get(key, 0.0))

            reward_padding = 12
            label_gap = 8
            col_gap = 8
            input_width = 96
            input_height = 24
            checkbox_label_offset = 24
            input_labels = [row["label"] for row in reward_rows if row.get("left") or row.get("right")]
            max_label_width = max(self.small_font.size(label)[0] for label in input_labels) if input_labels else 0
            label_width = max_label_width
            separator_height = 12

            header_y = reward_y + 28
            header_h = self.small_font.get_height()
            first_row_y = header_y + header_h + 6
            row_step = input_height + 8
            row_y = first_row_y
            last_bottom = row_y
            for row in reward_rows:
                if row.get("separator"):
                    last_bottom = row_y + separator_height
                    row_y += separator_height
                else:
                    last_bottom = row_y + input_height
                    row_y += row_step
            reward_height = max(arch_height, (last_bottom + reward_padding) - reward_y)

            self.draw_group_box(reward_x, reward_y, reward_width, reward_height, "Rewards")

            label_x = reward_x + reward_padding
            col1_x = label_x + label_width + label_gap
            col2_x = col1_x + input_width + col_gap
            header_color = self.text_color if reward_enabled else (120, 120, 120)
            dec_surf = self.small_font.render("Decrease", True, header_color)
            inc_surf = self.small_font.render("Increase", True, header_color)
            self.screen.blit(dec_surf, (col1_x + input_width // 2 - dec_surf.get_width() // 2, header_y))
            self.screen.blit(inc_surf, (col2_x + input_width // 2 - inc_surf.get_width() // 2, header_y))

            row_y = first_row_y
            label_color = self.text_color if reward_enabled else (120, 120, 120)
            for row in reward_rows:
                if row.get("separator"):
                    line_y = row_y + separator_height // 2
                    line_start = label_x
                    line_end = reward_x + reward_width - reward_padding
                    pygame.draw.line(self.screen, self.border_color, (line_start, line_y), (line_end, line_y), 1)
                    row_y += separator_height
                    continue

                label = row["label"]
                if row.get("checkbox"):
                    checkbox_x = label_x
                    checkbox_y = row_y + (input_height - 20) // 2
                    reward_lines_squared_rect = self.draw_checkbox(
                        checkbox_x,
                        checkbox_y,
                        params.get('reward_lines_squared', False),
                        label,
                        mouse_pos,
                        active=reward_enabled,
                        font=self.small_font,
                        label_offset=checkbox_label_offset
                    )
                    row_y += row_step
                    continue

                label_surf = self.small_font.render(label, True, label_color)
                self.screen.blit(label_surf, (label_x, row_y + (input_height - label_surf.get_height()) // 2))

                left_key = row.get("left")
                right_key = row.get("right")

                input_center_x = col1_x + ((col2_x + input_width) - col1_x - input_width) // 2
                if left_key:
                    left_text = reward_display(left_key)
                    left_input_x = input_center_x if row.get("center") else col1_x
                    left_rect = self.draw_text_input(
                        left_input_x,
                        row_y,
                        input_width,
                        input_height,
                        left_text,
                        active=reward_enabled,
                        focused=(reward_active == left_key),
                        mouse_pos=mouse_pos,
                        font=self.small_font
                    )
                    reward_input_rects[left_key] = left_rect

                if right_key:
                    right_text = reward_display(right_key)
                    right_input_x = input_center_x if row.get("center") else col2_x
                    right_rect = self.draw_text_input(
                        right_input_x,
                        row_y,
                        input_width,
                        input_height,
                        right_text,
                        active=reward_enabled,
                        focused=(reward_active == right_key),
                        mouse_pos=mouse_pos,
                        font=self.small_font
                    )
                    reward_input_rects[right_key] = right_rect

                row_y += row_step

            reset_button_height = 34
            reset_button_y = reward_y + reward_height + 15
            reset_defaults_rect = self.draw_button(
                reward_x,
                reset_button_y,
                reward_width,
                reset_button_height,
                "Reset All to Defaults",
                reward_enabled,
                mouse_pos
            )

            current_y += arch_height + 15
            
            # --- Curriculum Group ---
            curriculum_height = 315
            self.draw_group_box(padding, current_y, left_pane_width, curriculum_height, "Curriculum")
            
            sy = current_y + 35
            # Current / Min Rand. (0.0005 - 0.5 on a log scale)
            epsilon_min = 0.0005
            epsilon_max = 0.5
            def clamp_epsilon(val, default=0.0):
                try:
                    val = float(val)
                except (TypeError, ValueError):
                    val = default
                return max(epsilon_min, min(epsilon_max, val))
            def epsilon_to_norm(val):
                val = clamp_epsilon(val)
                log_min = math.log10(epsilon_min)
                log_max = math.log10(epsilon_max)
                return (math.log10(val) - log_min) / (log_max - log_min)
            sig_threshold = 1e-4
            def format_two_sig(val):
                try:
                    val = float(val)
                except (TypeError, ValueError):
                    return "0.00"
                if val == 0.0:
                    return "0.00"
                abs_val = abs(val)
                if abs_val < sig_threshold:
                    return f"{val:.1e}"
                exponent = math.floor(math.log10(abs_val))
                decimals = max(0, 1 - exponent)
                return f"{val:.{decimals}f}"
            def format_epsilon(val):
                return format_two_sig(val)
            min_rand_raw = params.get('epsilon_min', epsilon_min)
            current_rand_raw = params.get('epsilon_start', 0.1)
            min_rand = clamp_epsilon(min_rand_raw)
            current_rand = clamp_epsilon(current_rand_raw)
            floor_rand = min(min_rand, current_rand)
            current_rand = max(min_rand, current_rand)
            label = f"Rand.: {format_epsilon(current_rand)}->{format_epsilon(floor_rand)}"
            s_rect = self.draw_range_slider(slider_x, sy, slider_width, epsilon_to_norm(floor_rand), epsilon_to_norm(current_rand), label, mouse_pos, self.small_font, active=not is_training, bar_offset_y=slider_bar_offset)
            if not is_training: slider_rects['epsilon_range'] = s_rect
            sy += 45

            # Learning Rate Range (0.00005 - 1e-2, logarithmic)
            lr_min = 0.00005
            lr_max = 1e-2
            def clamp_lr(val, default=0.0025):
                try:
                    val = float(val)
                except (TypeError, ValueError):
                    val = default
                return max(lr_min, min(lr_max, val))
            def lr_to_norm(val):
                val = clamp_lr(val)
                lr_log_min = math.log10(lr_min)
                lr_log_max = math.log10(lr_max)
                return (math.log10(val) - lr_log_min) / (lr_log_max - lr_log_min)
            def format_lr(val):
                return format_two_sig(val)
            lr_start_raw = params.get('learning_rate_start', params.get('learning_rate', 0.0025))
            lr_end_raw = params.get('learning_rate_end', lr_start_raw)
            lr_start = clamp_lr(lr_start_raw)
            lr_end = clamp_lr(lr_end_raw)
            if lr_start < lr_end:
                lr_start, lr_end = lr_end, lr_start
            label = f"L.Rate: {format_lr(lr_start)}->{format_lr(lr_end)}"
            s_rect = self.draw_range_slider(slider_x, sy, slider_width, lr_to_norm(lr_end), lr_to_norm(lr_start), label, mouse_pos, self.small_font, active=not is_training, bar_offset_y=slider_bar_offset)
            if not is_training: slider_rects['learning_rate_range'] = s_rect
            sy += 45

            # R&L Half-life (10^2 - 10^7 in half-powers)
            half_life_raw = params.get('epsilon_half_life_batches', 10 ** 4)
            try:
                half_life = float(half_life_raw)
            except (TypeError, ValueError):
                half_life = 10 ** 4
            half_life = max(1e2, min(1e7, half_life))
            exp = math.log10(half_life)
            exp = max(2.0, min(7.0, exp))
            step_count = int(round((exp - 2.0) / 0.5))
            step_count = max(0, min(10, step_count))
            snapped_exp = 2.0 + step_count * 0.5
            exp_label = f"{snapped_exp:.1f}"
            label = f"R&L Half-life: 10^{exp_label}"
            s_rect = self.draw_slider(slider_x, sy, slider_width, step_count / 10.0, label, mouse_pos, self.small_font, active=not is_training, bar_offset_y=slider_bar_offset)
            if not is_training: slider_rects['epsilon_half_life_batches'] = s_rect
            sy += 45
            
            # Piece Size Range (1-5)
            min_size = params.get('min_size', 1)
            max_size = params.get('max_size', 4)
            min_size = max(1, min(5, int(min_size)))
            max_size = max(1, min(5, int(max_size)))
            floor_size = min(min_size, max_size)
            ceil_size = max(min_size, max_size)
            label = f"Piece Size: {floor_size}->{ceil_size}"
            s_rect = self.draw_range_slider(slider_x, sy, slider_width, (floor_size - 1) / 4.0, (ceil_size - 1) / 4.0, label, mouse_pos, self.small_font, active=not is_training, bar_offset_y=slider_bar_offset)
            if not is_training: slider_rects['piece_size_range'] = s_rect
            sy += 45

            # Big Piece Weight (1 - 4)
            big_weight = params.get('big_piece_weight', 4)
            big_weight = max(1, min(4, int(big_weight)))
            s_rect = self.draw_slider(slider_x, sy, slider_width, (big_weight - 1) / 3.0, f"Big Piece Weight: {big_weight}", mouse_pos, self.small_font, active=not is_training, bar_offset_y=slider_bar_offset)
            if not is_training: slider_rects['big_piece_weight'] = s_rect
            sy += 45
            
            # Short Game Length (how many pieces before auto-restart in short games mode)
            short_game_length = max(1, min(20, params.get('pieces_tracked', 1)))
            s_rect = self.draw_slider(slider_x, sy, slider_width, (short_game_length - 1) / 19.0, f"Piece History: {short_game_length}", mouse_pos, self.small_font, active=not is_training, bar_offset_y=slider_bar_offset)
            if not is_training: slider_rects['pieces_tracked'] = s_rect
            sy += 25
            
            # Short Games Checkbox
            # Use smaller font for this one to fit
            short_games_chk_rect = self.draw_checkbox(slider_x, sy, params.get('short_games', False), "Short Games", mouse_pos, active=not is_training, font=self.small_font)
        
        # Bottom Controls
        button_width = 150
        button_height = 34
        button_gap = 8
        bottom_y = screen_height - padding - button_height 
        
        # Main Menu Button
        btn_back_rect = self.draw_button(padding, bottom_y, button_width, button_height, "Main Menu", True, mouse_pos)
        bottom_y -= button_height + button_gap
        
        # Start/Stop Button
        label = "STOP" if is_training else "START"
        btn_start_rect = self.draw_button(padding, bottom_y, button_width, button_height, label, True, mouse_pos)
        bottom_y -= button_height + button_gap

        # Save Model Button
        btn_save_rect = self.draw_button(padding, bottom_y, button_width, button_height, "SAVE MODEL", save_active and not is_training, mouse_pos)
        bottom_y -= button_height + button_gap
        
        # View Training Checkbox
        chk_rect = self.draw_checkbox(padding, bottom_y + 10, params['visual_mode'], "View Training", mouse_pos)
        bottom_y -= 30

        # Volume Slider
        # Only relevant if visual mode is on? Or always show? User said "need not do anything when visual mode is off".
        # Let's show it always for consistency.
        vol_slider_rect = self.draw_slider(padding + 10, bottom_y + 15, 150, volume, "Volume", mouse_pos, self.small_font, bar_offset_y=slider_bar_offset)
        
        # Draw Game Board on Right
        if show_training_board:
            offset_x = left_pane_width + padding * 2
            offset_y = padding + 5
            
            self.draw_grid(grid, offset_x, offset_y)
        
        return (btn_back_rect, chk_rect, btn_start_rect, btn_save_rect, slider_rects, vol_slider_rect,
                short_games_chk_rect, dropdown_rect, reward_input_rects, reward_lines_squared_rect,
                reset_defaults_rect)


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

    def draw_choice_dialog(self, screen_width, screen_height, message_lines, choices, mouse_pos=None, layout="row"):
        overlay = pygame.Surface((screen_width, screen_height))
        overlay.set_alpha(180)
        overlay.fill((0, 0, 0))
        self.screen.blit(overlay, (0, 0))

        text_font = self.font
        button_font = self.small_font
        padding = 20
        line_gap = 6

        line_surfaces = []
        max_line_width = 0
        total_text_height = 0
        for line in message_lines:
            surf = text_font.render(line, True, self.text_color)
            line_surfaces.append(surf)
            max_line_width = max(max_line_width, surf.get_width())
            total_text_height += surf.get_height()
        if line_surfaces:
            total_text_height += line_gap * (len(line_surfaces) - 1)

        button_height = 32
        button_gap = 10
        label_widths = [button_font.size(label)[0] for _, label in choices]
        widest_label = max(label_widths) if label_widths else 0
        uniform_width = max(120, widest_label + 20)
        button_widths = []
        for width in label_widths:
            button_widths.append(max(120, width + 20))

        total_buttons_width = sum(button_widths)
        if button_widths:
            total_buttons_width += button_gap * (len(button_widths) - 1)
        total_buttons_height = button_height * len(button_widths)
        if button_widths:
            total_buttons_height += button_gap * (len(button_widths) - 1)

        if layout == "column":
            panel_width = max(360, max_line_width + padding * 2, uniform_width + padding * 2)
            panel_height = padding * 2 + total_text_height + 20 + total_buttons_height
        else:
            panel_width = max(360, max_line_width + padding * 2, total_buttons_width + padding * 2)
            panel_height = padding * 2 + total_text_height + 20 + button_height

        panel_x = (screen_width - panel_width) // 2
        panel_y = (screen_height - panel_height) // 2
        panel_rect = pygame.Rect(panel_x, panel_y, panel_width, panel_height)

        pygame.draw.rect(self.screen, self.bg_color, panel_rect, border_radius=10)
        pygame.draw.rect(self.screen, self.border_color, panel_rect, 2, border_radius=10)

        current_y = panel_y + padding
        for surf in line_surfaces:
            self.screen.blit(surf, (panel_x + panel_width // 2 - surf.get_width() // 2, current_y))
            current_y += surf.get_height() + line_gap

        rects = {}
        if layout == "column":
            button_y = panel_y + panel_height - padding - total_buttons_height
            x = panel_x + (panel_width - uniform_width) // 2
            for key, label in choices:
                rect = self.draw_button(
                    x,
                    button_y,
                    uniform_width,
                    button_height,
                    label,
                    True,
                    mouse_pos,
                    font=button_font
                )
                rects[key] = rect
                button_y += button_height + button_gap
        else:
            button_y = panel_y + panel_height - padding - button_height
            start_x = panel_x + (panel_width - total_buttons_width) // 2
            x = start_x
            for (key, label), btn_width in zip(choices, button_widths):
                rect = self.draw_button(
                    x,
                    button_y,
                    btn_width,
                    button_height,
                    label,
                    True,
                    mouse_pos,
                    font=button_font
                )
                rects[key] = rect
                x += btn_width + button_gap

        return rects

    def draw_checkbox(self, x, y, checked, label, mouse_pos=None, active=True, font=None, label_offset=30):
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
        self.screen.blit(text, (x + label_offset, y))
        
        return rect

    def draw_button(self, x, y, width, height, label, active, mouse_pos=None, font=None, fill_color=None, text_color=None, border_color=None, hover_color=None, hover_text_color=None):
        rect = pygame.Rect(x, y, width, height)
        
        # Draw solid background first
        background = fill_color if fill_color is not None else self.bg_color
        pygame.draw.rect(self.screen, background, rect, border_radius=10)
        
        if active:
            base_text_color = text_color if text_color is not None else self.text_color
            base_border_color = border_color if border_color is not None else base_text_color
            hover_tint = self.accent_color if hover_color is None else hover_color
            if hover_text_color is None:
                hover_text_color = self.accent_color if text_color is None else base_text_color
        else:
            base_text_color = text_color if text_color is not None else (100, 100, 100)
            base_border_color = border_color if border_color is not None else base_text_color
            hover_tint = None
            hover_text_color = base_text_color

        draw_border_color = base_border_color
        draw_text_color = base_text_color

        # Hover effect
        if active and mouse_pos and rect.collidepoint(mouse_pos):
            if hover_tint is not None:
                s = pygame.Surface((width, height))
                s.set_alpha(50)
                s.fill(hover_tint)
                self.screen.blit(s, (x, y))
                draw_border_color = hover_tint
            draw_text_color = hover_text_color
            
        pygame.draw.rect(self.screen, draw_border_color, rect, 2, border_radius=10)
        
        use_font = font if font else self.font
        text = use_font.render(label, True, draw_text_color)
        text_rect = text.get_rect(center=rect.center)
        self.screen.blit(text, text_rect)
        
        return rect

    def draw_text_input(self, x, y, width, height, text, placeholder="", active=True, mouse_pos=None, font=None, focused=None):
        if font is None:
            font = self.font
        rect = pygame.Rect(x, y, width, height)

        if focused is None:
            focused = active

        pygame.draw.rect(self.screen, self.bg_color, rect, border_radius=8)

        color = self.text_color if active else (100, 100, 100)
        if active:
            if focused:
                color = self.accent_color
            elif mouse_pos and rect.collidepoint(mouse_pos):
                color = self.accent_color

        pygame.draw.rect(self.screen, color, rect, 2, border_radius=8)

        display_text = text if text else placeholder
        if active:
            display_color = self.text_color if text else (120, 120, 120)
        else:
            display_color = (120, 120, 120) if text else (90, 90, 90)
        display_text = self._truncate_text(display_text, width - 16, font)
        text_surf = font.render(display_text, True, display_color)
        self.screen.blit(text_surf, (x + 8, y + height // 2 - text_surf.get_height() // 2))

        if active and focused:
            caret_text = self._truncate_text(text, width - 16, font) if text else ""
            caret_x = x + 8 + font.size(caret_text)[0] + 2
            caret_y = y + 6
            caret_h = height - 12
            caret_x = min(caret_x, x + width - 6)
            caret_x = max(caret_x, x + 6)
            pygame.draw.line(self.screen, color, (caret_x, caret_y), (caret_x, caret_y + caret_h), 2)

        return rect


    def draw_dropdown(self, x, y, width, height, label, options, selected_idx=None, is_open=False, mouse_pos=None, font=None, option_width=None, list_options=None, active=True, scroll_offset=0, max_visible=None):
        if font is None:
            font = self.font
        rect = pygame.Rect(x, y, width, height)
        is_open = is_open and active

        # Base box
        pygame.draw.rect(self.screen, self.bg_color, rect, border_radius=10)
        color = self.text_color if active else (100, 100, 100)
        if active and mouse_pos and rect.collidepoint(mouse_pos):
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
                total_items = len(list_items)
                visible_count = total_items
                if max_visible is not None:
                    visible_count = max(1, min(total_items, int(max_visible)))
                max_scroll = max(0, total_items - visible_count)
                scroll_offset = max(0, min(int(scroll_offset), max_scroll))

                start_idx = scroll_offset
                end_idx = start_idx + visible_count
                visible_items = list_items[start_idx:end_idx]

                for idx, option in enumerate(visible_items):
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
                    display_option = self._truncate_text(option, list_width - 16, font)
                    opt_surf = font.render(display_option, True, opt_color)
                    self.screen.blit(opt_surf, (opt_rect.x + 8, opt_rect.y + height // 2 - opt_surf.get_height() // 2))
                    option_rects.append(opt_rect)

                if total_items > visible_count:
                    list_rect = pygame.Rect(x, y + height, list_width, height * visible_count)
                    fade_height = min(12, height // 2)
                    if fade_height > 0:
                        if scroll_offset > 0:
                            fade = pygame.Surface((list_width, fade_height), pygame.SRCALPHA)
                            for i in range(fade_height):
                                alpha = int(200 * (1 - (i / float(fade_height))))
                                pygame.draw.line(fade, (*self.bg_color, alpha), (0, i), (list_width, i))
                            self.screen.blit(fade, (list_rect.x, list_rect.y))
                        if end_idx < total_items:
                            fade = pygame.Surface((list_width, fade_height), pygame.SRCALPHA)
                            for i in range(fade_height):
                                alpha = int(200 * (i / float(fade_height)))
                                pygame.draw.line(fade, (*self.bg_color, alpha), (0, i), (list_width, i))
                            self.screen.blit(fade, (list_rect.x, list_rect.y + list_rect.height - fade_height))

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
