import pygame
import math
from grid import Grid
from tetrominoes import Pentomino
from input_manager import InputManager
from ui import UI
from audio import AudioPlayer

class Game:
    def __init__(self):
        self.screen_width = 635
        self.screen_height = 770
        self.grid_width = 12 
        self.grid_height = 24 
        self.cell_size = 30
        
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Ominis")
        
        self.clock = pygame.time.Clock()
        self.grid = Grid(self.grid_width, self.grid_height, self.cell_size)
        self.input_manager = InputManager()
        self.ui = UI(self.screen)
        self.audio = AudioPlayer("assets/music", "assets/sounds")
        
        self.state = "MENU" # MENU, PLAYING, GAMEOVER, ANIMATING_CLEAR, ANIMATING_DROP, PAUSED
        self.score = 0
        self.level = 1
        self.lines_cleared_total = 0
        
        self.current_piece = None
        self.next_piece = None
        
        self.fall_time = 0
        self.fall_speed = 1000
        self.right_held_time = 0
        self.last_move_time = 0

        # Menu State
        self.include_pentominoes = True
        self.include_tetrominoes = False
        self.include_ominis = False
        self.volume = 0.5
        self.allowed_shapes = []
        
        # UI Rects (for click detection)
        self.chk_pent_rect = None
        self.chk_tet_rect = None
        self.chk_omi_rect = None
        self.btn_start_rect = None
        self.btn_train_rect = None
        self.btn_start_rect = None
        self.btn_train_rect = None
        self.btn_watch_rect = None
        self.btn_back_rect = None
        self.train_chk_rect = None
        self.btn_train_start_rect = None
        self.train_visual_mode = True
        self.btn_quit_rect = None
        self.slider_rect = None
        self.dragging_slider = False

        # Input handling state (DAS - Delayed Auto Shift)
        self.das_delay = 200 # ms before repeat starts
        self.das_repeat = 50 # ms between repeats
        self.left_held_time = 0

    def spawn_piece(self):
        p = Pentomino(self.grid_width // 2, 0, self.allowed_shapes)
        # Adjust y to ensure the shape starts just above the board
        # We want the lowest block (max_y) to be at y = -1
        max_y = max(y for x, y in p.shape)
        p.y = -max_y - 1
        return p

    def reset(self):
        self.grid = Grid(self.grid_width, self.grid_height, self.cell_size)
        self.score = 0
        self.level = 1
        self.lines_cleared_total = 0
        self.lines_cleared_total = 0
        self.fall_speed = 1000
        
        # Determine allowed shapes based on selection
        from tetrominoes import get_allowed_shapes
        self.allowed_shapes = get_allowed_shapes(self.include_pentominoes, self.include_tetrominoes, self.include_ominis)
        
        # Fallback if nothing selected (should be prevented by UI, but safety first)
        if not self.allowed_shapes:
             self.allowed_shapes = get_allowed_shapes(True, False, False)
        self.current_piece = self.spawn_piece()
        self.next_piece = self.spawn_piece()
        self.state = "PLAYING"
        self.left_held_time = 0
        self.right_held_time = 0
        self.audio.reset_sequence()
        self.audio.start()

    def update_score(self, lines):
        if lines > 0:
            points = [0, 100, 300, 500, 800, 1200]
            self.score += points[min(lines, 5)] * self.level
            self.lines_cleared_total += lines
            
            if self.lines_cleared_total >= self.level * 5:
                self.level += 1
                self.fall_speed = max(50, self.fall_speed - 50)

    def handle_input(self):
        current_time = pygame.time.get_ticks()
        
        # Continuous Input (DAS)
        if self.state == "PLAYING":
            # Left
            if self.input_manager.is_left_held():
                if self.left_held_time == 0: # Just pressed
                    self.left_held_time = current_time
                    self.last_move_time = current_time
                    pass 
                else:
                    # Check DAS
                    held_duration = current_time - self.left_held_time
                    if held_duration > self.das_delay:
                        if current_time - self.last_move_time > self.das_repeat:
                            if not self.grid.check_collision(self.current_piece, offset_x=-1):
                                self.current_piece.x -= 1
                            self.last_move_time = current_time
            else:
                self.left_held_time = 0

            # Right
            if self.input_manager.is_right_held():
                if self.right_held_time == 0:
                    self.right_held_time = current_time
                    self.last_move_time = current_time
                    pass
                else:
                    held_duration = current_time - self.right_held_time
                    if held_duration > self.das_delay:
                        if current_time - self.last_move_time > self.das_repeat:
                            if not self.grid.check_collision(self.current_piece, offset_x=1):
                                self.current_piece.x += 1
                            self.last_move_time = current_time
            else:
                self.right_held_time = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            
            action = self.input_manager.get_action(event)
            
            if action == "EXIT":
                if self.state == "PLAYING" or self.state == "GAMEOVER" or self.state == "WATCH_AI" or self.state == "TRAIN_MENU" or self.state == "TRAINING":
                    self.state = "MENU"
                    self.audio.stop()
            
            if self.state == "MENU":
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1: # Left click
                        mouse_pos = event.pos
                        if self.chk_pent_rect and self.chk_pent_rect.collidepoint(mouse_pos):
                            self.include_pentominoes = not self.include_pentominoes
                        elif self.chk_tet_rect and self.chk_tet_rect.collidepoint(mouse_pos):
                            self.include_tetrominoes = not self.include_tetrominoes
                        elif self.chk_omi_rect and self.chk_omi_rect.collidepoint(mouse_pos):
                            self.include_ominis = not self.include_ominis
                        elif self.btn_start_rect and self.btn_start_rect.collidepoint(mouse_pos):
                            if self.include_pentominoes or self.include_tetrominoes or self.include_ominis:
                                self.reset()
                        elif self.btn_train_rect and self.btn_train_rect.collidepoint(mouse_pos):
                            self.state = "TRAIN_MENU"
                        elif self.btn_watch_rect and self.btn_watch_rect.collidepoint(mouse_pos):
                            if self.include_pentominoes or self.include_tetrominoes or self.include_ominis:
                                self.reset()
                                self.state = "WATCH_AI"

            elif self.state == "TRAIN_MENU":
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        if self.btn_back_rect and self.btn_back_rect.collidepoint(event.pos):
                            self.state = "MENU"
                        elif self.train_chk_rect and self.train_chk_rect.collidepoint(event.pos):
                            self.train_visual_mode = not self.train_visual_mode
                        elif self.btn_train_start_rect and self.btn_train_start_rect.collidepoint(event.pos):
                            self.reset()
                            self.state = "TRAINING"

            elif self.state == "WATCH_AI" or (self.state == "PAUSED" and hasattr(self, 'last_state') and self.last_state == "WATCH_AI") or (self.state == "GAMEOVER" and hasattr(self, 'last_state') and self.last_state == "WATCH_AI"):
                 if event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        if self.btn_back_rect and self.btn_back_rect.collidepoint(event.pos):
                            self.state = "MENU"
                            self.audio.stop()

            if self.state == "PLAYING" or self.state == "PAUSED" or self.state == "GAMEOVER" or self.state == "TRAINING":
                 if event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        if self.btn_quit_rect and self.btn_quit_rect.collidepoint(event.pos):
                            self.state = "MENU"
                            self.audio.stop()
                        elif self.state == "TRAINING" and self.train_chk_rect and self.train_chk_rect.collidepoint(event.pos):
                            self.train_visual_mode = not self.train_visual_mode

            if self.state == "PLAYING" or self.state == "PAUSED" or self.state == "WATCH_AI" or self.state == "GAMEOVER" or self.state == "TRAINING":
                 if event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        if self.slider_rect and self.slider_rect.collidepoint(event.pos):
                            self.dragging_slider = True
                            self.update_volume(event.pos[0])
                 elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1:
                        self.dragging_slider = False
                 elif event.type == pygame.MOUSEMOTION:
                    if self.dragging_slider:
                        self.update_volume(event.pos[0])
                        
            if self.state == "GAMEOVER":
                pass # No special input for now, just buttons above
            
            # Pause Toggle
            if event.type == pygame.KEYDOWN and event.key == pygame.K_F1:
                if self.state == "PLAYING" or self.state == "WATCH_AI":
                    self.last_state = self.state # Remember state to resume to
                    self.state = "PAUSED"
                    self.audio.pause()
                elif self.state == "PAUSED":
                    self.state = self.last_state
                    self.audio.unpause()
            
            elif self.state == "PLAYING":
                if action == "LEFT":
                    if not self.grid.check_collision(self.current_piece, offset_x=-1):
                        self.current_piece.x -= 1
                    self.left_held_time = current_time # Start DAS timer
                    self.last_move_time = current_time
                    
                elif action == "RIGHT":
                    if not self.grid.check_collision(self.current_piece, offset_x=1):
                        self.current_piece.x += 1
                    self.right_held_time = current_time # Start DAS timer
                    self.last_move_time = current_time
                    
                elif action == "DOWN":
                    if not self.grid.check_collision(self.current_piece, offset_y=1):
                        self.current_piece.y += 1
                        self.score += 1 
                        self.fall_time = pygame.time.get_ticks() 
                elif action == "ROTATE_CW" or action == "ROTATE_CW_ALT":
                    self.current_piece.rotate_right()
                    if self.grid.check_collision(self.current_piece):
                        self.current_piece.rotate_left() 
                elif action == "ROTATE_CCW":
                    self.current_piece.rotate_left()
                    if self.grid.check_collision(self.current_piece):
                        self.current_piece.rotate_right() 
                elif action == "HARD_DROP":
                    while not self.grid.check_collision(self.current_piece, offset_y=1):
                        self.current_piece.y += 1
                        self.score += 2 # Bonus for hard drop
                    # Force immediate lock
                    self.fall_time = 0 # Trigger update immediately

        return True

    def get_ghost_piece(self):
        if not self.current_piece:
            return None
        
        ghost = Pentomino(self.current_piece.x, self.current_piece.y)
        ghost.shape = self.current_piece.shape # Copy shape
        ghost.color = self.current_piece.color
        ghost.type = self.current_piece.type
        
        # Move down until collision
        while not self.grid.check_collision(ghost, offset_y=1):
            ghost.y += 1
            
        return ghost

    def check_and_clear_lines(self):
        # Identify lines to clear
        lines_to_clear = []
        for y, row in enumerate(self.grid.grid):
            if (0, 0, 0) not in row:
                lines_to_clear.append(y)
        
        if lines_to_clear:
            self.clearing_lines = lines_to_clear
            self.pre_anim_state = self.state # Save state to return to
            self.state = "ANIMATING_CLEAR"
            self.animation_timer = pygame.time.get_ticks()
            self.audio.play_clear()
            return True
        return False

    def apply_line_clears(self):
        # Calculate offsets for drop animation
        # For each row, calculate how many cleared lines are below it
        self.row_offsets = [0] * self.grid_height
        
        # Actually clear the grid
        # We need to do this carefully to track where rows come from
        
        # Create new grid
        new_grid = []
        # Rows that are NOT cleared
        kept_rows = []
        for y in range(self.grid_height):
            if y not in self.clearing_lines:
                kept_rows.append(self.grid.grid[y])
        
        # Number of new empty rows needed
        num_new = self.grid_height - len(kept_rows)
        for _ in range(num_new):
            new_grid.append([(0, 0, 0) for _ in range(self.grid_width)])
        new_grid.extend(kept_rows)
        
        self.grid.grid = new_grid
        self.update_score(len(self.clearing_lines))
        
        # Calculate visual offsets
        # A row at index 'y' in the NEW grid came from 'src_y' in the OLD grid
        # src_y = y - num_new (roughly, but we need to account for gaps)
        
        # Simpler approach for offsets:
        # Rows 0 to num_new-1 are new, so they fade in or drop from top? 
        # Let's say they drop from -num_new * cell_size
        
        # Rows starting at num_new correspond to the top-most kept row.
        # If we cleared lines 20, 21, 22.
        # Rows 0-19 are kept. They shift down by 3.
        # So in the new grid, row 3 (was 0) should start at visual position of row 0.
        # Offset = (Old Y - New Y) * cell_size
        
        # Let's map new index to old index
        old_indices = []
        current_old = 0
        # First num_new rows are "new" (index -1 or something)
        for _ in range(num_new):
            old_indices.append(-1)
            
        for y in range(self.grid_height):
            if y not in self.clearing_lines:
                old_indices.append(y)
                
        # Now populate row_offsets
        # self.row_offsets[new_y] = (old_y - new_y) * cell_size
        for new_y in range(self.grid_height):
            old_y = old_indices[new_y]
            if old_y == -1:
                self.row_offsets[new_y] = -num_new * self.cell_size # Drop from above
            else:
                self.row_offsets[new_y] = (old_y - new_y) * self.cell_size

    def game_tick(self):
        """Advances the game by one 'tick' (one step of gravity)."""
        if not self.grid.check_collision(self.current_piece, offset_y=1):
            self.current_piece.y += 1
            if self.state == "PLAYING" and self.input_manager.is_down_held():
                self.score += 1
        else:
            self.grid.lock_shape(self.current_piece)
            
            # Check for Game Over (Locked piece extends above grid)
            game_over = False
            for x, y in self.current_piece.shape:
                if self.current_piece.y + y < 0:
                    self.state = "GAMEOVER"
                    self.audio.stop()
                    game_over = True
                    break
            
            if not game_over:
                if self.check_and_clear_lines():
                    pass # State changed to ANIMATING_CLEAR
                else:
                    self.current_piece = self.next_piece
                    self.next_piece = self.spawn_piece()
                    if self.grid.check_collision(self.current_piece):
                        self.state = "GAMEOVER"
                        self.audio.stop()

    def step_ai(self, moves):
        """
        Executes moves based on budget: 3 Lateral, 3 Vertical, 3 Rotation.
        Then advances the game by one tick.
        moves: List of strings, e.g., ["LEFT", "ROTATE_CW", "DOWN"]
        """
        if self.state != "PLAYING" and self.state != "TRAINING":
             return

        # Move Budgets
        budget = {
            "LATERAL": 3,
            "VERTICAL": 3,
            "ROTATION": 3
        }

        for move in moves:
            if move == "LEFT":
                if budget["LATERAL"] > 0:
                    if not self.grid.check_collision(self.current_piece, offset_x=-1):
                        self.current_piece.x -= 1
                    budget["LATERAL"] -= 1
            elif move == "RIGHT":
                if budget["LATERAL"] > 0:
                    if not self.grid.check_collision(self.current_piece, offset_x=1):
                        self.current_piece.x += 1
                    budget["LATERAL"] -= 1
            elif move == "DOWN":
                if budget["VERTICAL"] > 0:
                    if not self.grid.check_collision(self.current_piece, offset_y=1):
                        self.current_piece.y += 1
                        self.score += 1
                    budget["VERTICAL"] -= 1
            elif move == "ROTATE_CW":
                if budget["ROTATION"] > 0:
                    self.current_piece.rotate_right()
                    if self.grid.check_collision(self.current_piece):
                        self.current_piece.rotate_left()
                    budget["ROTATION"] -= 1
            elif move == "ROTATE_CCW":
                if budget["ROTATION"] > 0:
                    self.current_piece.rotate_left()
                    if self.grid.check_collision(self.current_piece):
                        self.current_piece.rotate_right()
                    budget["ROTATION"] -= 1
            elif move == "HARD_DROP":
                 # Hard drop consumes all vertical budget? Or just happens?
                 # User said "3 down presses". Hard drop is usually separate.
                 # Let's allow it if we have vertical budget, and it consumes 1?
                 # Or maybe it's a "free" action that ends the move phase?
                 # For safety, let's count it as 1 Vertical.
                 if budget["VERTICAL"] > 0:
                     while not self.grid.check_collision(self.current_piece, offset_y=1):
                        self.current_piece.y += 1
                        self.score += 2
                     budget["VERTICAL"] -= 1
                     # Hard drop usually forces a lock immediately.
                     # But here we just end the move phase and let game_tick lock it.
                     break # Stop processing further moves after hard drop

        # Advance game state
        self.game_tick()

    def update(self):
        current_time = pygame.time.get_ticks()
        
        if self.state == "PAUSED":
            return

        self.audio.update()
        
        # Update music speed based on fall speed
        ratio = (1000 / max(50, self.fall_speed)) ** 0.25
        self.audio.set_speed(ratio)

        if self.state == "PLAYING" or self.state == "WATCH_AI":
            # Determine fall speed
            current_fall_speed = self.fall_speed
            if self.state == "PLAYING" and self.input_manager.is_down_held():
                current_fall_speed = 80 # Slower fast drop (was 50)
            
            if current_time - self.fall_time > current_fall_speed:
                self.game_tick()
                self.fall_time = current_time

        elif self.state == "TRAINING":
            if self.train_visual_mode:
                # Visual Mode: Sync with fall speed
                if current_time - self.fall_time > self.fall_speed:
                    self.step_ai([]) # Pass dummy moves for now
                    self.fall_time = current_time
            else:
                # Headless Mode: Run as fast as possible (called every frame)
                # We might want to run multiple steps per frame if possible, 
                # but for now one step per frame (uncapped FPS) is good.
                self.step_ai([]) # Pass dummy moves for now
                
        elif self.state == "ANIMATING_CLEAR":
            if current_time - self.animation_timer > 200: # 200ms flash
                self.apply_line_clears()
                self.state = "ANIMATING_DROP"
                self.animation_timer = current_time
                
        elif self.state == "ANIMATING_DROP":
            # Reduce offsets
            all_done = True
            drop_speed = 2 # Pixels per frame
            for i in range(self.grid_height):
                if self.row_offsets[i] < 0:
                    self.row_offsets[i] += drop_speed
                    if self.row_offsets[i] > 0:
                        self.row_offsets[i] = 0
                    all_done = False
                elif self.row_offsets[i] > 0: # Should not happen with drop
                    self.row_offsets[i] -= drop_speed
                    if self.row_offsets[i] < 0:
                        self.row_offsets[i] = 0
                    all_done = False
            
            if all_done:
                # Restore state
                if self.state == "ANIMATING_DROP": # Double check
                     if hasattr(self, 'pre_anim_state'):
                         self.state = self.pre_anim_state
                     else:
                         self.state = "PLAYING" if self.state != "WATCH_AI" else "WATCH_AI"
                
                self.current_piece = self.next_piece
                self.next_piece = self.spawn_piece()
                if self.grid.check_collision(self.current_piece):
                    self.state = "GAMEOVER"
                    self.audio.stop()

    def draw(self):
        self.ui.draw_background()
        mouse_pos = pygame.mouse.get_pos()
        
        if self.state == "MENU":
            title = self.ui.large_font.render("OMINIS", True, self.ui.text_color)
            self.screen.blit(title, (self.screen_width // 2 - title.get_width() // 2, 100))
            
            # Checkboxes Group
            group_width = 260
            group_x = (self.screen_width - group_width) // 2
            group_rect = pygame.Rect(group_x, 240, group_width, 160)
            
            # Draw solid background first
            pygame.draw.rect(self.screen, self.ui.bg_color, group_rect, border_radius=10)
            
            pygame.draw.rect(self.screen, self.ui.border_color, group_rect, 2, border_radius=10)
            
            # Instructions (Select Group) - Inside border
            inst = self.ui.font.render("Select Groups:", True, self.ui.text_color)
            self.screen.blit(inst, (group_rect.centerx - inst.get_width() // 2, 250))
            
            start_y = 280
            chk_x = group_x + 20 # Padding inside box
            self.chk_pent_rect = self.ui.draw_checkbox(chk_x, start_y, self.include_pentominoes, "Pentominoes (5)", mouse_pos)
            self.chk_tet_rect = self.ui.draw_checkbox(chk_x, start_y + 40, self.include_tetrominoes, "Tetrominoes (4)", mouse_pos)
            self.chk_omi_rect = self.ui.draw_checkbox(chk_x, start_y + 80, self.include_ominis, "Twos and Threes", mouse_pos)
            
            # Start Button
            active = self.include_pentominoes or self.include_tetrominoes or self.include_ominis
            self.btn_start_rect = self.ui.draw_button(self.screen_width // 2 - 100, 450, 200, 50, "START GAME", active, mouse_pos)
            
            # AI Buttons
            self.btn_watch_rect = self.ui.draw_button(self.screen_width // 2 - 100, 520, 200, 50, "WATCH AI PLAY", active, mouse_pos)
            self.btn_train_rect = self.ui.draw_button(self.screen_width // 2 - 100, 590, 200, 50, "TRAIN AI", True, mouse_pos)
            
        elif self.state == "TRAIN_MENU":
            self.btn_back_rect, self.train_chk_rect, self.btn_train_start_rect = self.ui.draw_train_menu(self.screen_width, self.screen_height, self.train_visual_mode, mouse_pos)
            
        else:
            # Standard Padding
            padding = 20
            
            # Left Pane Layout
            left_pane_x = padding
            current_y = padding
            
            # Scoreboard (Height 150)
            self.ui.draw_score(self.score, self.level, self.lines_cleared_total, left_pane_x, current_y)
            current_y += 150 + padding
            
            # Preview (Height 210)
            self.ui.draw_preview(self.next_piece, left_pane_x, current_y, self.cell_size)
            current_y += 210 + padding
            
            # Instructions (Variable Height)
            # Calculate height: lines * 25 + 20 + 20
            # Play: 4 lines -> 100 + 40 = 140
            # Watch: 2 lines -> 50 + 40 = 90
            # Add extra padding to be safe against overlap
            # Determine mode for instructions - check if we came from WATCH_AI when paused
            instruction_mode = self.state
            if self.state == "PAUSED" and hasattr(self, 'last_state') and self.last_state == "WATCH_AI":
                instruction_mode = "WATCH_AI"
            
            inst_height = 180 if instruction_mode != "WATCH_AI" else 130
            self.ui.draw_instructions(left_pane_x, current_y, mode=instruction_mode)
            current_y += inst_height + padding
            
            # Volume Slider (Height ~40)
            # Center in 210 width: 210 - 150 = 60 -> x + 30
            self.slider_rect = self.ui.draw_slider(left_pane_x + 30, current_y, 150, self.volume, "Volume", mouse_pos)
            current_y += 40 + padding
            
            # Buttons (Height 40)
            btn_y = current_y
            
            # Grid Offset
            offset_x = left_pane_x + 210 + padding
            # Grid border is drawn at offset_y - 5. We want border at 'padding'.
            # So offset_y - 5 = padding => offset_y = padding + 5
            offset_y = padding + 5
            
            # Pass offsets and flash lines if animating
            offsets = self.row_offsets if self.state == "ANIMATING_DROP" else None
            flash = self.clearing_lines if self.state == "ANIMATING_CLEAR" else None
            
            self.ui.draw_grid(self.grid, offset_x, offset_y, offsets, flash)
            
            if self.state == "PLAYING" or self.state == "WATCH_AI" or self.state == "PAUSED" or self.state == "GAMEOVER":
                # Draw Ghost Piece
                ghost = self.get_ghost_piece()
                if ghost and ghost.y != self.current_piece.y:
                    self.ui.draw_ghost_pentomino(ghost, offset_x, offset_y, self.cell_size)
                
                self.ui.draw_pentomino(self.current_piece, offset_x, offset_y, self.cell_size)
            
            # Back Button for Watch AI (Moved down to avoid overlap)
            # Also show if PAUSED from WATCH_AI, or GAMEOVER from WATCH_AI
            if self.state == "WATCH_AI" or (self.state == "PAUSED" and hasattr(self, 'last_state') and self.last_state == "WATCH_AI") or (self.state == "GAMEOVER" and hasattr(self, 'last_state') and self.last_state == "WATCH_AI"):
                self.btn_back_rect = self.ui.draw_button(left_pane_x + 30, btn_y, 150, 40, "BACK", True, mouse_pos)
            
            # Quit Button for Playing
            # Show if PLAYING or (PAUSED and NOT from WATCH_AI) or (GAMEOVER and NOT from WATCH_AI) or TRAINING
            elif self.state == "PLAYING" or (self.state == "PAUSED" and (not hasattr(self, 'last_state') or self.last_state != "WATCH_AI")) or (self.state == "GAMEOVER" and (not hasattr(self, 'last_state') or self.last_state != "WATCH_AI")) or self.state == "TRAINING":
                self.btn_quit_rect = self.ui.draw_button(left_pane_x + 30, btn_y, 150, 40, "QUIT GAME", True, mouse_pos)
                
                if self.state == "TRAINING":
                     # Draw Checkbox above Quit button
                     self.train_chk_rect = self.ui.draw_checkbox(left_pane_x + 30, btn_y - 40, self.train_visual_mode, "Visual Mode", mouse_pos)
            
            # Calculate grid rect for overlays
            grid_rect_x = offset_x
            grid_rect_y = offset_y
            grid_rect_w = self.grid_width * self.cell_size
            grid_rect_h = self.grid_height * self.cell_size
            
            if self.state == "GAMEOVER":
                self.ui.draw_game_over(grid_rect_x, grid_rect_y, grid_rect_w, grid_rect_h)
                
            if self.state == "PAUSED":
                self.ui.draw_pause_screen(grid_rect_x, grid_rect_y, grid_rect_w, grid_rect_h)
        
        pygame.display.flip()

    def update_volume(self, mouse_x):
        if self.slider_rect:
            # Calculate volume from mouse position relative to slider
            rel_x = mouse_x - self.slider_rect.x
            vol = rel_x / self.slider_rect.width
            self.volume = max(0.0, min(1.0, vol))
            self.volume = max(0.0, min(1.0, vol))
            self.audio.set_volume(self.volume)

    def run(self):
        running = True
        headless_frame_count = 0
        while running:
            running = self.handle_input()
            self.update()
            
            if self.state == "TRAINING" and not self.train_visual_mode:
                # Headless Mode
                headless_frame_count += 1
                if headless_frame_count % 60 == 0: # Draw every 60 frames (approx 1 sec if running at 60fps, but here it's uncapped)
                    self.draw()
                    pygame.display.flip() # Ensure flip happens
                
                self.clock.tick(0) # Uncapped FPS
            else:
                self.draw()
                self.clock.tick(60)
        
        self.audio.cleanup()
        pygame.quit()
