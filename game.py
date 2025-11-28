import pygame
import math
from collections import deque
from grid import Grid
from tetrominoes import Pentomino
from input_manager import InputManager
from ui import UI
from audio import AudioPlayer
from agent import MonteCarloAgent

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
        self.pieces_locked = 0
        
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
        self.btn_watch_rect = None
        self.btn_back_rect = None
        self.train_chk_rect = None
        self.btn_train_start_rect = None
        self.train_visual_mode = True
        self.btn_quit_rect = None
        self.slider_rect = None
        self.train_slider_rect = None
        self.dragging_slider = False
        
        # Training Parameters
        self.train_params = {
            'visual_mode': True,
            'hl_size_idx': 1, # 0=128, 1=256, 2=512
            'hl_count': 2,
            'hl_count': 2,
            'epsilon_min_percent': 15,
            'learning_rate': 0.0005,
            'max_size': 5,
            'pieces_tracked': 10,
            'short_games': False
        }
        self.train_slider_rects = {}
        self.active_slider = None

        # Input handling state (DAS - Delayed Auto Shift)
        self.das_delay = 200 # ms before repeat starts
        self.das_repeat = 50 # ms between repeats
        self.left_held_time = 0
        self.left_held_time = 0
        self.agent = None
        self.training_step_count = 0
        self.current_trajectory = [] # Buffer for trajectory-based training
        self.trajectory_buffer = deque(maxlen=self.train_params['pieces_tracked'])
        self.start_stats = (0, 0) # (Height, Holes) at start of piece
        self.last_save_time = 0 # Track last auto-save time
        
        # Pending reward application for visual-mode line clear animation
        self.pending_reward_event = None
        
        # Short games tracking
        self.short_games_move_count = 0
        
        self.load_settings()
        # Ensure buffers respect any loaded settings (e.g., pieces_tracked)
        self.trajectory_buffer = deque(maxlen=self.get_pieces_tracked_limit())

    def get_model_filename(self):
        """Generates filename based on current architecture params."""
        # We need the sizes from Agent or define them here.
        # Let's define them here to match Agent.
        hl_sizes = [128, 256, 512]
        size = hl_sizes[self.train_params['hl_size_idx']]
        count = self.train_params['hl_count']
        
        # Ensure models dir exists
        import os
        if not os.path.exists("models"):
            os.makedirs("models")
            
        return os.path.join("models", f"model-{size}-{count}.pth")

    def load_settings(self):
        import json
        import os
        if os.path.exists("settings.json"):
            try:
                with open("settings.json", "r") as f:
                    saved_params = json.load(f)
                    # Update params
                    for k, v in saved_params.items():
                        if k in self.train_params:
                            self.train_params[k] = v
                print("Loaded settings from settings.json")
            except Exception as e:
                print(f"Failed to load settings: {e}")

    def save_settings(self):
        import json
        try:
            with open("settings.json", "w") as f:
                json.dump(self.train_params, f)
            print("Saved settings to settings.json")
        except Exception as e:
            print(f"Failed to save settings: {e}")

    def get_pieces_tracked_limit(self):
        """Clamp the pieces_tracked curriculum value to a sane range."""
        return max(1, min(20, self.train_params.get('pieces_tracked', 10)))

    def get_grid_stats(self):
        """Returns (max_height, holes) for the current grid."""
        max_height = 0
        for y in range(self.grid_height):
            if any(c != (0,0,0) for c in self.grid.grid[y]):
                max_height = self.grid_height - y
                break
        
        holes = 0
        for x in range(self.grid_width):
            found_block = False
            for y in range(self.grid_height):
                if self.grid.grid[y][x] != (0,0,0):
                    found_block = True
                elif found_block:
                    holes += 1
        return max_height, holes
    
    def get_lowest_column_height(self):
        """Returns the height of the lowest column (lowest max height across all columns).
        This is used as the baseline for height penalties."""
        lowest_height = self.grid_height
        
        for x in range(self.grid_width):
            # Find the highest block in this column
            column_height = 0
            for y in range(self.grid_height):
                if self.grid.grid[y][x] != (0, 0, 0):
                    column_height = self.grid_height - y
                    break
            
            # Track the lowest column height found
            if column_height < lowest_height:
                lowest_height = column_height
        
        return lowest_height


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
        self.pieces_locked = 0
        self.fall_speed = 1000
        self.current_trajectory = []
        self.trajectory_buffer = deque(maxlen=self.get_pieces_tracked_limit())
        self.pending_reward_event = None
        self.short_games_move_count = 0
        
        # Save original state before it might be changed
        original_state = self.state
        
        # Determine allowed shapes based on selection
        from tetrominoes import get_allowed_shapes
        
        max_size = 5
        if self.state == "TRAINING" or self.state == "TRAIN_MENU": # Use training params
             max_size = self.train_params['max_size']
             # In training, maybe we force all types enabled? Or respect checkboxes?
             # User said "Max size polyomino to include". Implies we include everything up to that size.
             # So let's enable all flags if in training, but filter by size.
             self.allowed_shapes = get_allowed_shapes(True, True, True, max_size)
        else:
             self.allowed_shapes = get_allowed_shapes(self.include_pentominoes, self.include_tetrominoes, self.include_ominis)
        
        # Fallback if nothing selected (should be prevented by UI, but safety first)
        if not self.allowed_shapes:
             self.allowed_shapes = get_allowed_shapes(True, False, False)
        self.current_piece = self.spawn_piece()
        self.start_stats = self.get_grid_stats()
        self.next_piece = self.spawn_piece()
        
        # Only set to PLAYING if not already in TRAINING (to avoid overwriting state during training reset)
        if self.state != "TRAINING":
            self.state = "PLAYING"
            
        self.left_held_time = 0
        self.right_held_time = 0
        self.audio.reset_sequence()
        
        # Only start music if not in headless training
        if self.state == "TRAINING" and not self.train_params['visual_mode']:
            pass
        else:
            self.audio.start()
        
        # Short games: Initialize with random rows filled, each 75% randomly filled with at least 1 missing
        # Use original_state to check, since state may have been changed to PLAYING above
        if (original_state == "TRAINING" or original_state == "TRAIN_MENU") and self.train_params.get('short_games', False):
            import random
            from tetrominoes import SHAPE_COLORS
            
            # Get list of available colors
            colors = list(SHAPE_COLORS.values())
            
            # Fill random number of rows (1-16) from the bottom
            num_rows = random.randint(1, 16)
            for row_idx in range(self.grid_height - num_rows, self.grid_height):
                # Fill 75% of blocks randomly
                filled_count = 0
                for col_idx in range(self.grid_width):
                    if random.random() < 0.75:
                        # Pick a random color
                        color = random.choice(colors)
                        self.grid.grid[row_idx][col_idx] = color
                        filled_count += 1
                
                # Ensure at least one block is missing
                if filled_count == self.grid_width:
                    # All filled, remove one randomly
                    empty_col = random.randint(0, self.grid_width - 1)
                    self.grid.grid[row_idx][empty_col] = (0, 0, 0)
            
            # Reset move counter for short games
            self.short_games_move_count = 0
        

    def update_score(self, lines):
        if lines > 0:
            # Quadratic scoring: 100 * lines^2 * level
            self.score += 100 * (lines ** 2) * self.level
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
                if self.state == "TRAINING" and self.agent:
                    self.agent.save(self.get_model_filename())
                    print(f"Model saved ({self.get_model_filename()}).")
                    self.save_settings()

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
                            # Settings are already loaded in __init__
                            self.state = "TRAIN_MENU"
                            # Stop music if visual mode is off (from saved params or default)
                            if not self.train_params['visual_mode']:
                                self.audio.stop()
                        elif self.btn_watch_rect and self.btn_watch_rect.collidepoint(mouse_pos):
                            if self.include_pentominoes or self.include_tetrominoes or self.include_ominis:
                                self.reset()
                                # Initialize agent for watching
                                # We need some default params for the agent structure
                                # Let's use the current train_params
                                self.agent = MonteCarloAgent(self.train_params)
                                import os
                                model_file = self.get_model_filename()
                                if os.path.exists(model_file):
                                    try:
                                        self.agent.load(model_file)
                                        print(f"Loaded model {model_file} for watching.")
                                    except Exception as e:
                                        print(f"Failed to load model {model_file}: {e}")
                                    except Exception as e:
                                        print(f"Failed to load model for watching: {e}")
                                
                                self.state = "WATCH_AI"

            elif self.state == "TRAIN_MENU":
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        if self.btn_back_rect and self.btn_back_rect.collidepoint(event.pos):
                            self.state = "MENU"
                        elif self.train_chk_rect and self.train_chk_rect.collidepoint(event.pos):
                            self.train_params['visual_mode'] = not self.train_params['visual_mode']
                        elif hasattr(self, 'short_games_chk_rect') and self.short_games_chk_rect and self.short_games_chk_rect.collidepoint(event.pos):
                            self.train_params['short_games'] = not self.train_params.get('short_games', False)
                        elif self.btn_train_start_rect and self.btn_train_start_rect.collidepoint(event.pos):
                            self.reset()
                            self.agent = MonteCarloAgent(self.train_params)
                            # Try to load existing model
                            import os
                            model_file = self.get_model_filename()
                            if os.path.exists(model_file):
                                try:
                                    self.agent.load(model_file)
                                    print(f"Loaded existing model {model_file}.")
                                except Exception as e:
                                    print(f"Failed to load model {model_file}: {e}")
                            else:
                                print(f"No existing model found for {model_file}, starting fresh.")
                            
                            self.save_settings()
                            
                            self.state = "TRAINING"
                            if not self.train_params['visual_mode']:
                                self.audio.stop() # No music in headless
                            
                            # Initialize auto-save timer
                            self.last_save_time = pygame.time.get_ticks()
                        
                        # Check sliders
                        for name, rect in self.train_slider_rects.items():
                            if rect.collidepoint(event.pos):
                                self.active_slider = name
                                self.update_train_slider(event.pos[0])
                        
                        if self.train_slider_rect and self.train_slider_rect.collidepoint(event.pos):
                            self.dragging_slider = True
                            self.update_volume(event.pos[0], is_train_slider=True)
                                
                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1:
                        self.active_slider = None
                        self.dragging_slider = False
                        
                elif event.type == pygame.MOUSEMOTION:
                    if self.active_slider:
                        self.update_train_slider(event.pos[0])
                    elif self.dragging_slider:
                        self.update_volume(event.pos[0], is_train_slider=True)

            elif self.state == "WATCH_AI" or (self.state == "PAUSED" and hasattr(self, 'last_state') and self.last_state == "WATCH_AI") or (self.state == "GAMEOVER" and hasattr(self, 'last_state') and self.last_state == "WATCH_AI"):
                 if event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        if self.btn_back_rect and self.btn_back_rect.collidepoint(event.pos):
                            self.state = "MENU"
                            self.audio.stop()
                        elif self.slider_rect and self.slider_rect.collidepoint(event.pos):
                            self.dragging_slider = True
                            self.update_volume(event.pos[0])
                 elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1:
                        self.dragging_slider = False
                 elif event.type == pygame.MOUSEMOTION:
                    if self.dragging_slider:
                        self.update_volume(event.pos[0])

            elif self.state == "PLAYING" or self.state == "PAUSED" or self.state == "GAMEOVER" or self.state == "TRAINING":
                 if event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        if self.btn_quit_rect and self.btn_quit_rect.collidepoint(event.pos):
                            self.state = "MENU"
                            self.audio.stop()
                        elif self.state == "TRAINING" and self.train_chk_rect and self.train_chk_rect.collidepoint(event.pos):
                            self.train_params['visual_mode'] = not self.train_params['visual_mode']
                            if not self.train_params['visual_mode']:
                                self.audio.stop()
                                print("Visual mode OFF - Audio stopped")
                            else:
                                self.audio.start()
                                print("Visual mode ON - Audio started")
                        elif self.state == "TRAINING" and self.btn_train_start_rect and self.btn_train_start_rect.collidepoint(event.pos):
                            # STOP button pressed
                            if self.agent:
                                self.agent.save(self.get_model_filename())
                                print(f"Model saved ({self.get_model_filename()}).")
                                self.save_settings()
                            self.state = "TRAIN_MENU"
                            # Stop music when returning to Train Menu
                            self.audio.stop()
                        elif self.state == "TRAINING" and self.btn_back_rect and self.btn_back_rect.collidepoint(event.pos):
                            # BACK button pressed
                            if self.agent:
                                self.agent.save(self.get_model_filename())
                                print(f"Model saved ({self.get_model_filename()}).")
                                self.save_settings()
                            self.state = "MENU"
                            # Main menu has no music
                            self.audio.stop()
                        elif self.state != "TRAINING" and self.slider_rect and self.slider_rect.collidepoint(event.pos):
                            self.dragging_slider = True
                            self.update_volume(event.pos[0])
                        elif self.state == "TRAINING" and self.train_slider_rect and self.train_slider_rect.collidepoint(event.pos):
                            self.dragging_slider = True
                            self.update_volume(event.pos[0], is_train_slider=True)
                 elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1:
                        self.dragging_slider = False
                 elif event.type == pygame.MOUSEMOTION:
                    if self.dragging_slider:
                        self.update_volume(event.pos[0], is_train_slider=(self.state == "TRAINING"))
                        
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
            # In headless training mode, skip animation and sound
            if self.state == "TRAINING" and not self.train_params['visual_mode']:
                # Directly apply clears without animation
                self.clearing_lines = lines_to_clear
                self.apply_line_clears()
                
                # CRITICAL FIX: Spawn new piece and update start_stats
                self.current_piece = self.next_piece
                self.start_stats = self.get_grid_stats()  # Update for new piece
                self.next_piece = self.spawn_piece()
                
                # Check for game over on spawn
                if self.grid.check_collision(self.current_piece):
                    self.handle_game_over()
                
                return True  # Line clear handled, piece spawned
            else:
                # Normal animation flow
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
            self.pieces_locked += 1
            
            # Check for Game Over (Locked piece extends above grid)
            game_over = False
            for x, y in self.current_piece.shape:
                if self.current_piece.y + y < 0:
                    self.handle_game_over()
                    game_over = True
                    break
            
            if not game_over:
                if self.check_and_clear_lines():
                    pass # State changed to ANIMATING_CLEAR
                else:
                    self.current_piece = self.next_piece
                    self.start_stats = self.get_grid_stats()
                    self.next_piece = self.spawn_piece()
                    if self.grid.check_collision(self.current_piece):
                        self.handle_game_over()


    def handle_game_over(self):
        self.state = "GAMEOVER"
        self.audio.stop()
        # Only print if NOT in training (finish_training_round handles it for training)
        if self.state != "TRAINING" and self.state != "TRAIN_MENU": 
             print(f"Game Over, Pieces: {self.pieces_locked}, Lines: {self.lines_cleared_total}")

    def queue_current_trajectory(self):
        """Move the current piece trajectory into the rolling buffer."""
        if self.current_trajectory:
            self.trajectory_buffer.append(self.current_trajectory[:])
        self.current_trajectory = []

    def apply_reward_to_buffer(self, reward_value, done):
        """Apply a reward to every step in the buffered trajectories."""
        if not self.agent:
            self.trajectory_buffer.clear()
            return
        if not self.trajectory_buffer:
            return

        for trajectory in self.trajectory_buffer:
            self.agent.add_trajectory_with_done(trajectory, reward_value, done)

        self.trajectory_buffer.clear()
        self.agent.replay()

    def finish_training_round(self):
        """Handle end-of-game bookkeeping and restart training."""
        if self.train_params.get('short_games', False):
            if self.lines_cleared_total > 0:
                print(f"Short game, Lines: {self.lines_cleared_total}")
        else:
            print(f"Game Over, Pieces: {self.pieces_locked}, Lines: {self.lines_cleared_total}")
        self.state = "TRAINING" # Restore state BEFORE reset so it knows to use training params
        self.reset() # Auto-restart during training
        if not self.train_params['visual_mode']:
            self.audio.stop()

    def step_ai(self, moves):
        """
        Executes moves based on budget: 3 Lateral, 3 Rotation (vertical/hard drops only if provided).
        Then advances the game by one tick.
        moves: List of strings, e.g., ["LEFT", "ROTATE_CW"]
        """
        if self.state != "PLAYING" and self.state != "TRAINING":
             return

        # Move Budgets
        budget = {
            "LATERAL": 1,
            "VERTICAL": 3,
            "ROTATION": 1
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

    def step_ai_training(self):
        if not self.agent:
            return

        # 1. Get Current State
        state = self.agent.get_state(self)
        
        # 2. Select Action
        action = self.agent.select_action(state)
        lat_idx, rot_idx = action
        
        # Map indices to moves
        moves = []
        
        # Lateral: 0=Left, 1=Stay, 2=Right
        if lat_idx == 0: moves.append("LEFT")
        elif lat_idx == 2: moves.append("RIGHT")
        
        # Rotate: 0=CCW, 1=Stay, 2=CW
        if rot_idx == 0: moves.append("ROTATE_CCW")
        elif rot_idx == 2: moves.append("ROTATE_CW")
        
        # 3. Execute Moves (and advance tick)
        # We need to track if the piece was locked/placed during this step.
        piece_before = self.current_piece
        lines_before = self.lines_cleared_total
        
        # Execute the moves
        self.step_ai(moves)
        
        lines_after = self.lines_cleared_total
        lines_cleared = lines_after - lines_before
        
        # Check Game Over
        done = (self.state == "GAMEOVER")
        
        # Check if piece was locked
        # Piece is locked if:
        # 1. Lines were cleared (implies lock)
        # 2. Game Over (implies lock)
        # 3. Current piece object changed (implies lock and spawn new)
        # 4. State changed to ANIMATING_CLEAR (Visual mode lock)
        
        if self.state == "ANIMATING_CLEAR":
            lines_cleared = len(self.clearing_lines)
            piece_locked = True
        else:
            piece_locked = (lines_cleared > 0) or done or (self.current_piece is not piece_before)
        
        # 4. Get New State
        next_state = self.agent.get_state(self)
        
        # 5. Buffer Step
        # We store (state, action, next_state)
        # We don't calculate reward yet.
        self.current_trajectory.append((state, action, next_state))
        
        # 6. If Piece Locked, Distribute Reward and Train
        if piece_locked:
            piece_limit_reached = False

            # Short games: Increment move counter
            if self.train_params.get('short_games', False):
                self.short_games_move_count += 1

            # Move the locked piece trajectory into the rolling buffer (last N pieces)
            self.queue_current_trajectory()

            # Short games: Check if we've reached the piece limit (N)
            pieces_limit = self.get_pieces_tracked_limit()
            if self.train_params.get('short_games', False) and self.short_games_move_count >= pieces_limit:
                done = True  # Mark as done without triggering game over
                piece_limit_reached = True

            # Decide if we should pay out rewards now (line clear or game end)
            reward_event = None
            
            # Check for Game Over first (highest priority/impact)
            is_game_over = (self.state == "GAMEOVER")
            
            # Calculate reward
            # If we cleared lines AND died, we should probably just count the death?
            # Or sum them? Let's sum them to be safe, but Game Over is -1000 and Clear is +350*N^2.
            # If I clear 1 line (+350) and die (-1000), result is -650. This seems fair.
            
            if lines_cleared > 0 or (done and not piece_limit_reached):
                 reward_event = self.agent.calculate_reward(lines_cleared, is_game_over)

            if reward_event is not None:
                # Apply reward after the clear animation finishes (visual mode)
                if self.state == "ANIMATING_CLEAR":
                    self.pending_reward_event = (reward_event, done)
                else:
                    self.apply_reward_to_buffer(reward_event, done)
                    if done:
                        self.finish_training_round()
            elif piece_limit_reached:
                # Piece cap reached: skip training to avoid noisy targets, just restart
                self.current_trajectory = []
                self.trajectory_buffer.clear()
                self.finish_training_round()
        
        
        # Auto-save every 5 minutes
        current_time = pygame.time.get_ticks()
        if current_time - self.last_save_time > 5 * 60 * 1000:  # 5 minutes in milliseconds
            if self.agent:
                self.agent.save(self.get_model_filename())
                print(f"Auto-saved model ({self.get_model_filename()}) at {current_time // 1000}s")
                self.save_settings()
                self.last_save_time = current_time

    def step_ai_watch(self):
        """Executes AI moves for watching mode (no training)."""
        if not self.agent or not self.current_piece:
            return

        # 1. Get Current State
        state = self.agent.get_state(self)
        
        # 2. Select Action (No epsilon exploration usually, but agent handles it)
        # We might want to force epsilon=0 for watching?
        old_eps = self.agent.epsilon
        self.agent.epsilon = 0 # Force greedy
        action = self.agent.select_action(state)
        self.agent.epsilon = old_eps
        
        lat_idx, rot_idx = action
        
        # Map indices to moves
        moves = []
        if lat_idx == 0: moves.append("LEFT")
        elif lat_idx == 2: moves.append("RIGHT")
        
        if rot_idx == 0: moves.append("ROTATE_CCW")
        elif rot_idx == 2: moves.append("ROTATE_CW")
        
        # 3. Execute Moves
        self.step_ai(moves)

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

        elif self.state == "WATCH_AI":
             # AI plays at a readable speed
             watch_speed = 250 # ms
             if current_time - self.fall_time > watch_speed:
                 self.step_ai_watch()
                 self.fall_time = current_time

        elif self.state == "TRAINING":
            if self.train_params['visual_mode']:
                # Visual Mode: Run at specific speed (e.g. 250ms)
                # User said: "screen shows the agent playing at some speed... Perhaps 250ms per piece advance"
                # Wait, "per piece advance" usually means per step (gravity).
                train_speed = 10 # ms
                if current_time - self.fall_time > train_speed:
                    self.step_ai_training()
                    self.fall_time = current_time
            else:

                # Headless Mode: Run as fast as possible
                # We loop until we detect user input or a refresh timer expires
                
                # Initialize refresh timer if needed
                if not hasattr(self, 'last_headless_draw'):
                    self.last_headless_draw = current_time
                
                # If we just drew (or it's been > 1000ms), let's loop
                # But if we just returned to let draw happen, we need to reset the timer?
                # Actually, the logic should be:
                # 1. If input pending, return immediately.
                # 2. If time to draw (1s elapsed), return immediately.
                # 3. Otherwise, run a batch of steps.
                
                # Check for forced refresh (e.g. to show stats)
                if current_time - self.last_headless_draw > 1000:
                    self.last_headless_draw = current_time
                    return # Return to allow draw()
                
                steps_per_check = 20
                while True:
                    # Run a batch of steps
                    for _ in range(steps_per_check):
                        self.step_ai_training()
                    
                    # Check for user input (responsiveness)
                    # peek returns True if any of the types are in the queue
                    if pygame.event.peek([pygame.QUIT, pygame.MOUSEBUTTONDOWN, pygame.KEYDOWN]):
                        # Input detected! Return to let handle_input process it next frame
                        # We also reset the draw timer so we draw immediately to show feedback
                        self.last_headless_draw = 0 
                        return
                    
                    # Check refresh timer
                    if pygame.time.get_ticks() - self.last_headless_draw > 1000:
                        return
                
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
                self.start_stats = self.get_grid_stats()  # Update for new piece
                self.next_piece = self.spawn_piece()
                if self.grid.check_collision(self.current_piece):
                    self.state = "GAMEOVER"
                    self.audio.stop()
                
                # Process pending reward if in training mode
                if self.state == "TRAINING" and self.pending_reward_event is not None:
                    reward_value, done_flag = self.pending_reward_event
                    self.apply_reward_to_buffer(reward_value, done_flag)
                    self.pending_reward_event = None
                    if done_flag:
                        self.finish_training_round()

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
            self.chk_omi_rect = self.ui.draw_checkbox(chk_x, start_y + 80, self.include_ominis, "Oligominoes (<4)", mouse_pos)
            
            # Start Button
            active = self.include_pentominoes or self.include_tetrominoes or self.include_ominis
            self.btn_start_rect = self.ui.draw_button(self.screen_width // 2 - 100, 450, 200, 50, "START GAME", active, mouse_pos)
            
            # AI Buttons
            self.btn_watch_rect = self.ui.draw_button(self.screen_width // 2 - 100, 520, 200, 50, "WATCH AI PLAY", active, mouse_pos)
            self.btn_train_rect = self.ui.draw_button(self.screen_width // 2 - 100, 590, 200, 50, "TRAIN AI", True, mouse_pos)
            
        elif self.state == "TRAIN_MENU" or self.state == "TRAINING" or (self.state in ["ANIMATING_CLEAR", "ANIMATING_DROP"] and hasattr(self, 'pre_anim_state') and self.pre_anim_state == "TRAINING"):
            # In TRAINING, we still want to show the menu controls, but maybe disable some?
            # User said "left panel should retain the Train AI controls".
            # So we use the same draw function.
            # But we need to make sure the game board is drawn with the current state.
            
            # If Headless (visual_mode=False), we might not want to draw the board updates every frame?
            # User said: "If visual mode is not selected, the game board stays blank"
            # So we pass a dummy grid or just don't draw the grid content in draw_train_menu if headless?
            # Actually, draw_train_menu calls draw_grid.
            
            # Let's handle the "Blank Board" logic in ui.py or here by passing a clean grid if headless.
            display_grid = self.grid
            if self.state == "TRAINING" and not self.train_params['visual_mode']:
                # Create a dummy empty grid for display
                display_grid = Grid(self.grid_width, self.grid_height, self.cell_size)
            
            is_training = (self.state == "TRAINING" or (self.state in ["ANIMATING_CLEAR", "ANIMATING_DROP"] and hasattr(self, 'pre_anim_state') and self.pre_anim_state == "TRAINING"))
            self.btn_back_rect, self.train_chk_rect, self.btn_train_start_rect, self.train_slider_rects, self.train_slider_rect, self.short_games_chk_rect = self.ui.draw_train_menu(self.screen_width, self.screen_height, self.train_params, mouse_pos, display_grid, is_training=is_training, volume=self.volume)
            
            # If TRAINING, we also need to draw the falling piece if Visual Mode is ON
            if is_training and self.train_params['visual_mode']:
                 # We need to manually draw the piece on top because draw_train_menu draws the grid background/locked blocks
                 # But draw_train_menu calls draw_grid which clips.
                 # We need to calculate offsets again.
                 # draw_train_menu uses:
                 # offset_x = left_pane_width + padding * 2
                 # offset_y = padding + 5
                 # left_pane_width = 210
                 padding = 20
                 offset_x = 210 + padding * 2
                 offset_y = padding + 5
                 
                 if self.current_piece:
                     self.ui.draw_pentomino(self.current_piece, offset_x, offset_y, self.cell_size)
            
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
            # Show if PLAYING or (PAUSED and NOT from WATCH_AI) or (GAMEOVER and NOT from WATCH_AI)
            elif self.state == "PLAYING" or (self.state == "PAUSED" and (not hasattr(self, 'last_state') or self.last_state != "WATCH_AI")) or (self.state == "GAMEOVER" and (not hasattr(self, 'last_state') or self.last_state != "WATCH_AI")):
                self.btn_quit_rect = self.ui.draw_button(left_pane_x + 30, btn_y, 150, 40, "QUIT GAME", True, mouse_pos)
            
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

    def update_volume(self, mouse_x, is_train_slider=False):
        rect = self.train_slider_rect if is_train_slider else self.slider_rect
        if rect:
            # Calculate volume from mouse position relative to slider
            rel_x = mouse_x - rect.x
            vol = rel_x / rect.width
            self.volume = max(0.0, min(1.0, vol))
            self.audio.set_volume(self.volume)

    def update_train_slider(self, mouse_x):
        if not self.active_slider:
            return
            
        rect = self.train_slider_rects[self.active_slider]
        rel_x = mouse_x - rect.x
        val = max(0.0, min(1.0, rel_x / rect.width))
        
        if self.active_slider == 'hl_size':
            # Map 0-1 to 0, 1, 2
            idx = int(val * 2.99)
            self.train_params['hl_size_idx'] = idx
        elif self.active_slider == 'hl_count':
            # Map 0-1 to 1, 2, 3, 4
            count = 1 + int(val * 3.99)
            self.train_params['hl_count'] = count
        elif self.active_slider == 'epsilon_min_percent':
            # Map 0-1 to 1-50
            val_percent = 1 + int(val * 49.99)
            self.train_params['epsilon_min_percent'] = val_percent
            if self.agent: self.agent.update_hyperparameters()
        elif self.active_slider == 'learning_rate':
            # Map 0-1 to 0.0001 - 0.001
            # Range 0.0009
            lr = 0.0001 + (val * 0.0009)
            self.train_params['learning_rate'] = round(lr, 5)
            if self.agent: self.agent.update_hyperparameters()
        elif self.active_slider == 'max_size':
            # Map 0-1 to 1, 2, 3, 4, 5
            size = 1 + int(val * 4.99)
            self.train_params['max_size'] = size
        elif self.active_slider == 'pieces_tracked':
            # Map 0-1 to 1-20
            count = 1 + int(val * 19)
            self.train_params['pieces_tracked'] = max(1, min(20, count))

    def run(self):
        running = True
        headless_frame_count = 0
        while running:
            running = self.handle_input()
            self.update()
            
            if self.state == "TRAINING" and not self.train_params['visual_mode']:
                # Headless Mode
                # Draw is called either when input is detected or every ~1s
                self.draw()
                # Do NOT tick clock, we want max speed
                # self.clock.tick(60) 
            else:
                self.draw()
                self.clock.tick(60)
        
        self.audio.cleanup()
        pygame.quit()
