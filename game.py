import pygame
import math
from collections import deque
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
        self.watch_dropdown_rect = None
        self.watch_play_rect = None
        self.watch_option_rects = []
        self.watch_dropdown_open = False
        self.watch_models = []
        self.watch_model_selected = None
        
        # Architecture UI
        self.epsilon_bump_value = 0.2
        self.btn_eps_reset_rect = None
        self.hl_sizes = [128, 256, 512, 1024, 2048]

        # Training Parameters
        self.train_params = {
            'visual_mode': True,
            'hl_size_idx': 1, # index into self.hl_sizes (0=128 ... 4=2048)
            'hl_count': 2,
            'epsilon_min_percent': 5,
            'learning_rate': 0.001,
            'max_size': 5,
            'pieces_tracked': 10,
            'gamma': 0.70,
            'short_games': False
        }
        self.train_slider_rects = {}
        self.active_slider = None

        # Input handling state (DAS - Delayed Auto Shift)
        self.das_delay = 200 # ms before repeat starts
        self.das_repeat = 50 # ms between repeats
        self.left_held_time = 0
        self.right_held_time = 0
        self.agent = None
        self.training_step_count = 0
        self.current_trajectory = [] # Buffer for current piece's moves
        self.start_stats = (0, 0, 0) # (Height, Holes, Jaggedness) at start of piece
        self.last_save_time = 0 # Track last auto-save time
        
        # Pending reward application for visual-mode line clear animation
        self.pending_reward_event = None
        
        # Short games tracking
        self.short_games_move_count = 0
        
        self.load_settings()
        # Ensure hidden size index stays within the available architecture options
        self.train_params['hl_size_idx'] = max(0, min(len(self.hl_sizes) - 1, int(self.train_params.get('hl_size_idx', 1))))

    def get_model_filename(self):
        """Generates filename based on current architecture params."""
        hl_size_idx = max(0, min(len(self.hl_sizes) - 1, int(self.train_params['hl_size_idx'])))
        size = self.hl_sizes[hl_size_idx]
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
                # Clamp learning rate to supported slider range
                self.train_params['learning_rate'] = max(0.0001, min(0.005, self.train_params.get('learning_rate', 0.001)))
                print("Loaded settings from settings.json")
            except Exception as e:
                print(f"Failed to load settings: {e}")

    def create_agent(self, params_override=None):
        """Lazily import and build the AI agent so the game can start without PyTorch installed."""
        try:
            from agent import MonteCarloAgent
        except ImportError as e:
            print(f"AI features unavailable (missing dependency): {e}")
            return None
        except Exception as e:
            print(f"Failed to import AI agent: {e}")
            return None

        try:
            params = params_override if params_override is not None else self.train_params
            return MonteCarloAgent(params)
        except Exception as e:
            print(f"Failed to initialize AI agent: {e}")
            return None

    def refresh_watch_models(self):
        import os
        models_dir = "models"
        if not os.path.isdir(models_dir):
            self.watch_models = []
        else:
            files = [f for f in os.listdir(models_dir) if f.lower().endswith(".pth")]
            files.sort()
            self.watch_models = files
        if self.watch_model_selected and self.watch_model_selected not in self.watch_models:
            self.watch_model_selected = None

    def _parse_model_arch(self, model_file):
        import os
        base = os.path.splitext(os.path.basename(model_file))[0]
        parts = base.split("-")
        if len(parts) < 3:
            return None
        try:
            size = int(parts[-2])
            count = int(parts[-1])
        except ValueError:
            return None
        if size not in self.hl_sizes:
            return None
        count = max(1, min(4, count))
        return self.hl_sizes.index(size), count

    def build_watch_agent(self, model_file):
        import os
        params = dict(self.train_params)
        arch = self._parse_model_arch(model_file)
        if arch:
            params['hl_size_idx'], params['hl_count'] = arch
        agent = self.create_agent(params_override=params)
        if not agent:
            return None
        model_path = os.path.join("models", model_file)
        if not os.path.exists(model_path):
            print(f"Model not found: {model_path}")
            return None
        try:
            agent.load(model_path)
            print(f"Loaded model {model_path} for watching.")
        except Exception as e:
            print(f"Failed to load model {model_path}: {e}")
            return None
        return agent

    def start_watch_ai(self):
        self.refresh_watch_models()
        if not self.watch_model_selected:
            return
        self.watch_dropdown_open = False
        agent = self.build_watch_agent(self.watch_model_selected)
        if agent:
            self.agent = agent
            self.audio.stop()
            self.reset(start_music=True)
            self.state = "WATCH_AI"
        else:
            self.agent = None

    def save_settings(self):
        import json
        try:
            with open("settings.json", "w") as f:
                json.dump(self.train_params, f)
            print("Saved settings to settings.json")
        except Exception as e:
            print(f"Failed to save settings: {e}")

    def get_short_game_length(self):
        """Get the number of pieces for a short game (1-20)."""
        return max(1, min(20, self.train_params.get('pieces_tracked', 10)))

    def get_watch_ai_fall_speed(self):
        """
        Watch AI uses a capped gravity speed:
        - Levels 1-16: 250ms per step
        - Level 17: 200ms, level 18: 150ms, level 19: 100ms, level 20+: 50ms minimum
        """
        if self.level <= 16:
            return 250
        # Drop 50ms per level starting at 17, but never below 50ms
        return max(50, 250 - 50 * (self.level - 16))

    def _analyze_grid(self, grid_cells):
        """Return (max_height, holes, jaggedness) for a grid cell matrix."""
        heights = [0] * self.grid_width
        holes = 0

        for x in range(self.grid_width):
            top_y = None
            for y in range(self.grid_height):
                if grid_cells[y][x] != (0, 0, 0):
                    top_y = y
                    heights[x] = self.grid_height - y
                    break

            if top_y is None:
                continue

            # Holes are all empty cells below the top filled cell in a column.
            for y in range(top_y + 1, self.grid_height):
                if grid_cells[y][x] == (0, 0, 0):
                    holes += 1

        jaggedness = 0
        for x in range(self.grid_width - 1):
            jaggedness += abs(heights[x] - heights[x + 1])

        max_height = max(heights) if heights else 0
        return max_height, holes, jaggedness

    def get_grid_stats(self):
        """Returns (max_height, holes, jaggedness) for the current grid."""
        return self._analyze_grid(self.grid.grid)
    
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

    def get_post_clear_grid_stats(self, clearing_lines):
        """
        Simulates line clearing on a temporary grid to calculate post-clear stats.
        Returns: (simulated_height, simulated_holes, simulated_jaggedness)
        """
        if not clearing_lines:
            return self.get_grid_stats()

        # Create simulated grid structure
        new_grid = []
        kept_rows = []
        for y in range(self.grid_height):
            if y not in clearing_lines:
                kept_rows.append(self.grid.grid[y]) # Reference is fine since we won't mutate inner lists
        
        num_new = self.grid_height - len(kept_rows)
        for _ in range(num_new):
            new_grid.append([(0, 0, 0) for _ in range(self.grid_width)])
        new_grid.extend(kept_rows)
        
        return self._analyze_grid(new_grid)

    def get_piece_lowest_height_after_clear(self, piece, clearing_lines):
        """
        Returns the height (from bottom) of the lowest surviving block of `piece`
        after removing rows in clearing_lines and applying the resulting drop.
        If all blocks are cleared, returns 0.
        """
        if not piece:
            return 0

        # Pre-sort for efficient counting
        clearing_set = set(clearing_lines or [])
        sorted_clears = sorted(clearing_set)

        lowest_block_height = 0
        for px, py in piece.shape:
            abs_y = piece.y + py
            # Skip blocks that land on a cleared line
            if abs_y in clearing_set:
                continue
            # Blocks drop by the number of cleared lines below them
            cleared_below = sum(1 for cl in sorted_clears if cl > abs_y)
            new_y = abs_y + cleared_below
            if new_y >= self.grid_height:
                continue
            block_height = self.grid_height - new_y
            lowest_block_height = max(lowest_block_height, block_height)

        return lowest_block_height


    def spawn_piece(self):
        """Spawn a piece centered by its bounding box with an unbiased tie-break."""
        p = Pentomino(0, 0, self.allowed_shapes)

        # Bounding box in the current (already randomized) rotation
        min_x = min(x for x, _ in p.shape)
        max_x = max(x for x, _ in p.shape)
        width = max_x - min_x + 1

        ideal_left = (self.grid_width - width) / 2  # may be .5 on even widths

        # If perfectly half-cell, alternate left/right to avoid bias
        frac = ideal_left - math.floor(ideal_left)
        if abs(frac - 0.5) < 1e-6:
            if not hasattr(self, "spawn_parity"):
                self.spawn_parity = 0
            ideal_left = math.floor(ideal_left) + self.spawn_parity
            self.spawn_parity ^= 1
        else:
            ideal_left = round(ideal_left)

        p.x = int(ideal_left - min_x)

        # Start just above the board; keep lowest block at y = -1
        max_y = max(y for _, y in p.shape)
        p.y = -max_y - 1
        return p

    def reset(self, start_music=True):
        self.grid = Grid(self.grid_width, self.grid_height, self.cell_size)
        self.score = 0
        self.level = 1
        self.lines_cleared_total = 0
        self.pieces_locked = 0
        self.fall_speed = 1000
        self.current_trajectory = []
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
             self.allowed_shapes = get_allowed_shapes(True, True, True, max_size, weighted=True)
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
        elif start_music:
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
                                self.reset(start_music=False)
                                self.state = "WATCH_AI"
                                self.agent = None
                                self.watch_model_selected = None
                                self.watch_dropdown_open = False
                                self.refresh_watch_models()

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
                            agent = self.create_agent()
                            if agent:
                                self.agent = agent
                                self.reset()
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
                            else:
                                self.agent = None
                                print("Training requires the AI dependencies; staying in the Train menu.")
                        
                        # Check sliders
                        for name, rect in self.train_slider_rects.items():
                            if rect.collidepoint(event.pos):
                                self.active_slider = name
                                self.update_train_slider(event.pos[0])
                        
                        # Check Epsilon Reset Button
                        if self.btn_eps_reset_rect and self.btn_eps_reset_rect.collidepoint(event.pos):
                            if self.agent:
                                self.agent.epsilon = self.epsilon_bump_value
                                self.agent.save(self.get_model_filename())
                                print(f"Epsilon manually reset to {self.epsilon_bump_value} and saved.")
                            else:
                                print("Agent not loaded. Start training first.")

                        
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
                        mouse_pos = event.pos
                        dropdown_handled = False
                        if self.watch_dropdown_open:
                            for rect, model_name in self.watch_option_rects:
                                if rect.collidepoint(mouse_pos):
                                    self.watch_model_selected = model_name
                                    self.watch_dropdown_open = False
                                    dropdown_handled = True
                                    break
                            if not dropdown_handled:
                                if self.watch_dropdown_rect and self.watch_dropdown_rect.collidepoint(mouse_pos):
                                    self.watch_dropdown_open = False
                                else:
                                    self.watch_dropdown_open = False
                                dropdown_handled = True
                        if dropdown_handled:
                            continue

                        if self.btn_back_rect and self.btn_back_rect.collidepoint(mouse_pos):
                            self.state = "MENU"
                            self.audio.stop()
                            self.watch_dropdown_open = False
                        elif self.watch_play_rect and self.watch_play_rect.collidepoint(mouse_pos):
                            if self.watch_model_selected:
                                self.start_watch_ai()
                        elif self.watch_dropdown_rect and self.watch_dropdown_rect.collidepoint(mouse_pos):
                            self.watch_dropdown_open = not self.watch_dropdown_open
                            if self.watch_dropdown_open:
                                self.refresh_watch_models()
                        elif self.slider_rect and self.slider_rect.collidepoint(mouse_pos):
                            self.dragging_slider = True
                            self.update_volume(mouse_pos[0])
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
                # Deferred application: We Just mark lines to clear and return True.
                # The actual clearing and spawning happens in step_ai_training AFTER reward calculation.
                self.clearing_lines = lines_to_clear
                return True

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
        if self.state in ["WATCH_AI", "PLAYING"]:
            self.last_state = self.state
        self.state = "GAMEOVER"
        self.audio.stop()

    def apply_trajectory_reward(self, trajectory, reward_value, lines_cleared, is_game_over):
        """Apply a reward to the given trajectory and push to agent memory."""
        if not self.agent or not trajectory:
            return

        if is_game_over:
            reward_value -= 2.0

        self.agent.add_trajectory_with_done(trajectory, reward_value)
        self.agent.record_training_stats(len(trajectory), lines_cleared, is_game_over)

    def finish_training_round(self):
        """Handle end-of-game bookkeeping and restart training."""
        self.state = "TRAINING" # Restore state BEFORE reset so it knows to use training params
        self.reset() # Auto-restart during training
        if not self.train_params['visual_mode']:
            self.audio.stop()

    def step_ai(self, moves):
        """
        Executes moves based on budget: 3 Lateral, 3 Rotation (vertical/hard drops only if provided).
        Then advances the game by one tick.
        moves: List of strings, e.g., ["LEFT", "ROTATE_CW"]
        Returns:
            (lateral_idx, rotation_idx) describing what actually changed the piece
            (0=Left/CCW, 1=No change, 2=Right/CW). If nothing changed, returns (1, 1).
        """
        if self.state not in ["PLAYING", "TRAINING", "WATCH_AI"] or not self.current_piece:
             return (1, 1)

        # Move Budgets
        budget = {
            "LATERAL": 1,
            "VERTICAL": 3,
            "ROTATION": 1
        }

        actual_lateral_idx = 1
        actual_rotation_idx = 1

        for move in moves:
            if move == "LEFT":
                if budget["LATERAL"] > 0:
                    prev_x = self.current_piece.x
                    if not self.grid.check_collision(self.current_piece, offset_x=-1):
                        self.current_piece.x -= 1
                    if self.current_piece.x < prev_x:
                        actual_lateral_idx = 0
                    budget["LATERAL"] -= 1
            elif move == "RIGHT":
                if budget["LATERAL"] > 0:
                    prev_x = self.current_piece.x
                    if not self.grid.check_collision(self.current_piece, offset_x=1):
                        self.current_piece.x += 1
                    if self.current_piece.x > prev_x:
                        actual_lateral_idx = 2
                    budget["LATERAL"] -= 1
            elif move == "DOWN":
                if budget["VERTICAL"] > 0:
                    if not self.grid.check_collision(self.current_piece, offset_y=1):
                        self.current_piece.y += 1
                        self.score += 1
                    budget["VERTICAL"] -= 1
            elif move == "ROTATE_CW":
                if budget["ROTATION"] > 0:
                    prev_shape = set(self.current_piece.shape)
                    self.current_piece.rotate_right()
                    if self.grid.check_collision(self.current_piece):
                        self.current_piece.rotate_left()
                    elif set(self.current_piece.shape) != prev_shape:
                        actual_rotation_idx = 2
                    budget["ROTATION"] -= 1
            elif move == "ROTATE_CCW":
                if budget["ROTATION"] > 0:
                    prev_shape = set(self.current_piece.shape)
                    self.current_piece.rotate_left()
                    if self.grid.check_collision(self.current_piece):
                        self.current_piece.rotate_right()
                    elif set(self.current_piece.shape) != prev_shape:
                        actual_rotation_idx = 0
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
        return (actual_lateral_idx, actual_rotation_idx)

    def step_ai_training(self):
        if not self.agent:
            return

        # 1. Get Current State
        state = self.agent.get_state(self)
        
        # 2. Select Action (returns joint action index 0-8)
        action_mask = self.agent.get_action_mask(self)
        selected_action = self.agent.select_action(state, action_mask=action_mask)
        # Count every inference step, even if it is later discarded from memory
        self.agent.record_inference_step()
        lat_idx, rot_idx = self.agent.decode_action(selected_action)
        
        # Map indices to moves
        moves = []
        
        # Lateral: 0=Left, 1=Stay, 2=Right
        if lat_idx == 0: moves.append("LEFT")
        elif lat_idx == 2: moves.append("RIGHT")
        
        # Rotate: 0=CCW, 1=Stay, 2=CW
        if rot_idx == 0: moves.append("ROTATE_CCW")
        elif rot_idx == 2: moves.append("ROTATE_CW")
        
        # 3. Execute Moves (and advance tick)
        # 3. Execute Moves (and advance tick)
        # Track state before move executes
        piece_before = self.current_piece
        lines_before = self.lines_cleared_total
        # Capture stats locally to ensure fresh data for reward calculation
        _, holes_before_step, jaggedness_before = self.get_grid_stats()
        
        # Reset clearing lines flag for Headless detection
        self.clearing_lines = []

        # Execute the moves
        actual_lateral_idx, actual_rotation_idx = self.step_ai(moves)
        
        lines_after = self.lines_cleared_total
        lines_cleared = lines_after - lines_before
        
        # Check Game Over
        done = (self.state == "GAMEOVER")
        
        # Check if piece was locked
        # Extended check for Headless mode deferred clearing
        is_headless_clearing = (self.state == "TRAINING" and not self.train_params['visual_mode'] and self.clearing_lines)
        
        if self.state == "ANIMATING_CLEAR" or is_headless_clearing:
            lines_cleared = len(self.clearing_lines)
            piece_locked = True
        else:
            piece_locked = (lines_cleared > 0) or done or (self.current_piece is not piece_before)
        
        # 4. Buffer Step
        actual_action = self.agent.encode_action(actual_lateral_idx, actual_rotation_idx)
        self.current_trajectory.append((state, actual_action))
        
        # 5. If Piece Locked, Calculate Reward and Train
        if piece_locked:
            piece_limit_reached = False

            # Short games: Increment move counter
            if self.train_params.get('short_games', False):
                self.short_games_move_count += 1

            # Capture the trajectory for this piece before clearing
            trajectory = self.current_trajectory[:]
            self.current_trajectory = []

            # Short games: Check if we've reached the piece limit
            if self.train_params.get('short_games', False) and self.short_games_move_count >= self.get_short_game_length():
                done = True
                piece_limit_reached = True

            # Determine which lines would be cleared
            clearing_lines = self.clearing_lines[:] if self.clearing_lines else []
            if not clearing_lines:
                for y, row in enumerate(self.grid.grid):
                    if (0, 0, 0) not in row:
                        clearing_lines.append(y)

            # Post-clear stats on simulated grid
            _, holes_after, jaggedness_after = self.get_post_clear_grid_stats(clearing_lines)
            hole_delta = holes_after - holes_before_step
            jaggedness_delta = jaggedness_after - jaggedness_before

            is_game_over = (self.state == "GAMEOVER")
            
            # Calculate reward for this placement
            reward = self.agent.calculate_reward(
                lines_cleared,
                hole_delta,
                jaggedness_delta
            )

            # Apply reward to this piece's trajectory
            if self.state == "ANIMATING_CLEAR":
                # Store trajectory with reward for after animation completes
                self.pending_reward_event = (trajectory, reward, lines_cleared, is_game_over)
            elif is_headless_clearing:
                pending_reward = reward
            else:
                self.apply_trajectory_reward(trajectory, reward, lines_cleared, is_game_over)

            # HEADLESS DEFERRED ACTIONS
            # If we had deferred line clearing, execute it now
            if is_headless_clearing:
                self.apply_line_clears()
                # Reset clearing list
                self.clearing_lines = []

                self.current_piece = self.next_piece
                self.start_stats = self.get_grid_stats() # Update for new piece
                self.next_piece = self.spawn_piece()

                # Check collision for new piece
                if self.grid.check_collision(self.current_piece):
                    self.handle_game_over()

                is_game_over = (self.state == "GAMEOVER")
                if is_game_over:
                    done = True

                self.apply_trajectory_reward(trajectory, pending_reward, lines_cleared, is_game_over)

            if done:
                self.finish_training_round()
        
        # Auto-save every 5 minutes
        current_time = pygame.time.get_ticks()
        if current_time - self.last_save_time > 5 * 60 * 1000:
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
        action_mask = self.agent.get_action_mask(self)
        action = self.agent.select_action(state, action_mask=action_mask)
        self.agent.epsilon = old_eps
        
        lat_idx, rot_idx = self.agent.decode_action(action)
        
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
        
        # Update music speed based on the active fall speed (human or watch AI)
        effective_fall_speed = self.get_watch_ai_fall_speed() if self.state == "WATCH_AI" else self.fall_speed
        ratio = (1000 / max(50, effective_fall_speed)) ** 0.25
        self.audio.set_speed(ratio)

        if self.state == "WATCH_AI":
            # Watch AI uses its own capped gravity schedule (fast 250ms until level 17+)
            current_fall_speed = self.get_watch_ai_fall_speed()
            if current_time - self.fall_time > current_fall_speed:
                self.step_ai_watch()
                self.fall_time = current_time

        elif self.state == "PLAYING":
            # Determine fall speed
            current_fall_speed = self.fall_speed
            if self.input_manager.is_down_held():
                current_fall_speed = 80 # Slower fast drop (was 50)
            
            if current_time - self.fall_time > current_fall_speed:
                self.game_tick()
                self.fall_time = current_time

        elif self.state == "TRAINING":
            if self.train_params['visual_mode']:
                # Visual Mode: Run at specific speed (e.g. 250ms)
                # User said: "screen shows the agent playing at some speed... Perhaps 250ms per piece advance"
                # Wait, "per piece advance" usually means per step (gravity).
                train_speed = 50 # ms
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
                
                steps_per_check = 50  # Smaller batch for responsive UI
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
                
                was_training = (self.state == "TRAINING")
                
                self.current_piece = self.next_piece
                self.start_stats = self.get_grid_stats()  # Update for new piece
                self.next_piece = self.spawn_piece()
                game_over_after_spawn = False
                if self.grid.check_collision(self.current_piece):
                    game_over_after_spawn = True
                    if self.state in ["WATCH_AI", "PLAYING"]:
                        self.last_state = self.state
                    self.state = "GAMEOVER"
                    self.audio.stop()
                
                # Process pending reward even if game over happened after the clear
                if was_training and self.pending_reward_event is not None:
                    trajectory, reward_value, pending_lines, pending_game_over = self.pending_reward_event
                    reward_game_over = pending_game_over or game_over_after_spawn
                    self.apply_trajectory_reward(trajectory, reward_value, pending_lines, reward_game_over)
                    self.pending_reward_event = None
                    if reward_game_over:
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
            self.btn_back_rect, self.train_chk_rect, self.btn_train_start_rect, self.train_slider_rects, self.train_slider_rect, self.short_games_chk_rect, self.btn_eps_reset_rect = self.ui.draw_train_menu(self.screen_width, self.screen_height, self.train_params, mouse_pos, display_grid, is_training=is_training, volume=self.volume, epsilon_bump_val=self.epsilon_bump_value)

            
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
            watch_ui = self.state == "WATCH_AI" or (self.state == "PAUSED" and hasattr(self, 'last_state') and self.last_state == "WATCH_AI") or (self.state == "GAMEOVER" and hasattr(self, 'last_state') and self.last_state == "WATCH_AI")
            
            # Scoreboard (Height 150)
            self.ui.draw_score(self.score, self.level, self.lines_cleared_total, left_pane_x, current_y)
            current_y += 150 + padding
            
            # Preview (Height 210)
            self.ui.draw_preview(self.next_piece, left_pane_x, current_y, self.cell_size)
            current_y += 210 + padding

            watch_dropdown_layout = None
            if watch_ui:
                dropdown_height = 40
                dropdown_width = 140
                play_width = 60
                gap = 10
                import os
                options = [os.path.splitext(name)[0] for name in self.watch_models]
                list_options = list(self.watch_models)
                option_width = None
                if self.watch_dropdown_open:
                    text_font = self.ui.small_font
                    max_text_width = text_font.size("No models found")[0]
                    for option in list_options:
                        max_text_width = max(max_text_width, text_font.size(option)[0])
                    needed_width = max_text_width + 16
                    max_option_width = self.screen_width - left_pane_x - padding
                    option_width = max(dropdown_width, min(needed_width, max_option_width))
                selected_idx = None
                if self.watch_model_selected in self.watch_models:
                    selected_idx = self.watch_models.index(self.watch_model_selected)
                watch_dropdown_layout = (left_pane_x, current_y, dropdown_width, dropdown_height, play_width, gap, options, list_options, option_width, selected_idx)
                current_y += dropdown_height + padding
            else:
                self.watch_dropdown_rect = None
                self.watch_play_rect = None
                self.watch_option_rects = []
                self.watch_dropdown_open = False
            
            # Instructions (Variable Height)
            # Calculate height: lines * 25 + 20 + 20
            # Play: 4 lines -> 100 + 40 = 140
            # Watch: 2 lines -> 50 + 40 = 90
            # Add extra padding to be safe against overlap
            # Determine mode for instructions - check if we came from WATCH_AI when paused
            instruction_mode = "WATCH_AI" if watch_ui else self.state
            
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

            if watch_dropdown_layout:
                (drop_x, drop_y, dropdown_width, dropdown_height, play_width, gap,
                 options, list_options, option_width, selected_idx) = watch_dropdown_layout
                self.watch_dropdown_rect, option_rects = self.ui.draw_dropdown(
                    drop_x,
                    drop_y,
                    dropdown_width,
                    dropdown_height,
                    "Select model",
                    options,
                    selected_idx,
                    self.watch_dropdown_open,
                    mouse_pos,
                    font=self.ui.small_font,
                    option_width=option_width,
                    list_options=list_options
                )
                self.watch_option_rects = list(zip(option_rects, self.watch_models))
                play_x = drop_x + dropdown_width + gap
                play_active = self.watch_model_selected in self.watch_models
                self.watch_play_rect = self.ui.draw_button(play_x, drop_y, play_width, dropdown_height, "PLAY", play_active, mouse_pos, font=self.ui.small_font)
        
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
            max_idx = len(self.hl_sizes) - 1
            # Map 0-1 to available hidden size options
            idx = int(val * (max_idx + 0.99))
            self.train_params['hl_size_idx'] = max(0, min(max_idx, idx))
        elif self.active_slider == 'hl_count':
            # Map 0-1 to 1, 2, 3, 4
            count = 1 + int(val * 3.99)
            self.train_params['hl_count'] = count
        elif self.active_slider == 'epsilon_min_percent':
            # Map 0-1 to 0-10
            val_percent = int(val * 10)
            self.train_params['epsilon_min_percent'] = max(0, min(10, val_percent))
            if self.agent: self.agent.update_hyperparameters()
        elif self.active_slider == 'learning_rate':
            # Map 0-1 to 0.0001 - 0.0050
            lr = 0.0001 + (val * 0.0049)
            self.train_params['learning_rate'] = round(lr, 5)
            if self.agent: self.agent.update_hyperparameters()
        elif self.active_slider == 'gamma':
            # Map 0-1 to 0.1 - 1.0
            raw_gamma = 0.1 + (val * 0.9)
            # Snap to 0.05 increments
            gamma = round(raw_gamma * 20) / 20.0
            self.train_params['gamma'] = max(0.1, min(1.0, gamma))
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
