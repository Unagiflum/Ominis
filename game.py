import pygame
import math
from collections import deque
from grid import Grid
from tetrominoes import Pentomino
from input_manager import InputManager
from ui import UI
from audio import AudioPlayer

class Game:
    EPSILON_MAX_PERCENT = 25.0
    EPSILON_STEP_PERCENT = 0.5
    EPSILON_HALF_LIFE_MIN_EXP = 2.0
    EPSILON_HALF_LIFE_MAX_EXP = 7.0
    EPSILON_HALF_LIFE_STEP_EXP = 0.5
    EPSILON_HALF_LIFE_STEPS = int((EPSILON_HALF_LIFE_MAX_EXP - EPSILON_HALF_LIFE_MIN_EXP) / EPSILON_HALF_LIFE_STEP_EXP)

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
        
        self.state = "MENU" # MENU, PLAY_MENU, WATCH_MENU, PLAYING, WATCH_AI, GAMEOVER, TRAIN_MENU, TRAINING, ANIMATING_CLEAR, ANIMATING_DROP, PAUSED
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
        self.allowed_shape_weights = None
        self.active_min_size = 1
        self.active_max_size = 5
        self.play_min_size = 1
        self.play_max_size = 5
        self.watch_min_size = 1
        self.watch_max_size = 5
        
        # UI Rects (for click detection)
        self.chk_pent_rect = None
        self.chk_tet_rect = None
        self.chk_omi_rect = None
        self.btn_start_rect = None
        self.btn_train_rect = None
        self.btn_watch_rect = None
        self.btn_back_rect = None
        self.btn_play_start_rect = None
        self.btn_watch_start_rect = None
        self.btn_main_menu_rect = None
        self.train_chk_rect = None
        self.btn_train_start_rect = None
        self.btn_train_save_rect = None
        self.train_visual_mode = True
        self.btn_quit_rect = None
        self.slider_rect = None
        self.train_slider_rect = None
        self.play_slider_rect = None
        self.watch_slider_rect = None
        self.dragging_slider = False
        self.dragging_size_slider = False
        self.active_size_slider = None
        self.active_size_handle = None
        self.watch_dropdown_rect = None
        self.watch_dropdown_open = False
        self.watch_dropdown_scroll = 0
        self.watch_dropdown_visible = 0
        self.watch_dropdown_list_width = 0
        self.watch_option_rects = []
        self.watch_models = []
        self.watch_model_selected = None

        self.train_dropdown_rect = None
        self.train_dropdown_open = False
        self.train_dropdown_scroll = 0
        self.train_dropdown_visible = 0
        self.train_dropdown_list_width = 0
        self.train_option_rects = []
        self.train_models = []
        self.train_model_selected = None
        self.train_model_source = None
        self.train_model_arch = None

        self.save_name_active = False
        self.save_name_text = ""
        self.save_name_rect = None
        self.save_name_limit = 32

        self.record_name_active = False
        self.record_name_text = ""
        self.record_name_rect = None
        self.record_name_limit = 32
        self.recording_temp_path = None
        self.recorder = None
        self.is_recording = False
        self.record_failed = False
        self.btn_record_rect = None
        self.record_audio_device = None
        self.record_audio_device_name = None
        self.record_audio_loopback = True
        
        # Architecture UI
        self.hl_sizes = [128, 256, 512, 1024, 2048]

        # Training Parameters
        self.train_params = {
            'visual_mode': True,
            'hl_size_idx': 1, # index into self.hl_sizes (0=128 ... 4=2048)
            'hl_count': 2,
            'epsilon_min_percent': 5,
            'epsilon_current_percent': 20,
            'epsilon_half_life_batches': int(round(10 ** 4.5)),
            'learning_rate': 0.001,
            'min_size': 1,
            'max_size': 5,
            'big_piece_weight': 1,
            'pieces_tracked': 10,
            'gamma': 1.0,
            'short_games': False
        }
        self.train_slider_rects = {}
        self.active_slider = None
        self.active_slider_handle = None
        self.epsilon_override_pending = False

        # Input handling state (DAS - Delayed Auto Shift)
        self.das_delay = 200 # ms before repeat starts
        self.das_repeat = 50 # ms between repeats
        self.left_held_time = 0
        self.right_held_time = 0
        self.agent = None
        self.training_step_count = 0
        self.current_trajectory = [] # Buffer for current piece's moves
        self.piece_history = deque() # Pending piece trajectories for delayed rewards
        self.start_stats = (0, 0, 0, 0, 0.0) # (Height, Holes, Jaggedness, Valleys, HeightStd) at start of piece
        self.last_save_time = 0 # Track last auto-save time
        
        # Pending reward application for visual-mode line clear animation
        self.pending_reward_event = None
        
        # Short games tracking
        self.short_games_move_count = 0
        
        self.load_settings()
        self.load_recording_settings()
        # Ensure hidden size index stays within the available architecture options
        self.train_params['hl_size_idx'] = max(0, min(len(self.hl_sizes) - 1, int(self.train_params.get('hl_size_idx', 1))))
        self.train_params['gamma'] = 1.0
        self.train_params['big_piece_weight'] = max(1, min(4, int(self.train_params.get('big_piece_weight', 1))))
        self._apply_piece_size_range(
            self.train_params.get('min_size', 1),
            self.train_params.get('max_size', 5)
        )

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
                if self._apply_loaded_train_params(saved_params):
                    print("Loaded settings from settings.json")
            except Exception as e:
                print(f"Failed to load settings: {e}")

    def load_recording_settings(self):
        import json
        import os
        self.record_audio_device = None
        self.record_audio_device_name = None
        self.record_audio_loopback = True
        if os.path.exists("recording.json"):
            try:
                with open("recording.json", "r") as f:
                    data = json.load(f)
                device = data.get("audio_device")
                if device is not None and device != "":
                    try:
                        self.record_audio_device = int(device)
                    except (TypeError, ValueError):
                        print("Recording setting audio_device is not a number.")
                name = data.get("audio_device_name")
                if isinstance(name, str) and name.strip():
                    self.record_audio_device_name = name.strip()
                loopback = data.get("audio_loopback")
                if loopback is not None:
                    if isinstance(loopback, str):
                        lowered = loopback.strip().lower()
                        if lowered in ("true", "1", "yes", "y"):
                            self.record_audio_loopback = True
                        elif lowered in ("false", "0", "no", "n"):
                            self.record_audio_loopback = False
                        else:
                            print("Recording setting audio_loopback is not valid.")
                    else:
                        self.record_audio_loopback = bool(loopback)
            except Exception as e:
                print(f"Failed to load recording settings: {e}")

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

    def refresh_train_models(self):
        import os
        models_dir = "models"
        if not os.path.isdir(models_dir):
            self.train_models = []
        else:
            files = [f for f in os.listdir(models_dir) if f.lower().endswith(".pth")]
            files.sort()
            self.train_models = files
        if self.train_model_selected and self.train_model_selected not in self.train_models:
            self.train_model_selected = None
            self.train_model_source = None
            self.train_model_arch = None

    def _load_model_checkpoint(self, model_path):
        try:
            import torch
        except ImportError as e:
            print(f"AI features unavailable (missing dependency): {e}")
            return None
        except Exception as e:
            print(f"Failed to import torch: {e}")
            return None
        try:
            return torch.load(model_path, map_location="cpu")
        except Exception as e:
            print(f"Failed to load model {model_path}: {e}")
            return None

    def _apply_loaded_train_params(self, loaded_params):
        if not isinstance(loaded_params, dict):
            return False
        for k, v in loaded_params.items():
            if k in self.train_params:
                self.train_params[k] = v

        self.train_params['hl_size_idx'] = max(0, min(len(self.hl_sizes) - 1, int(self.train_params.get('hl_size_idx', 1))))
        self.train_params['hl_count'] = max(1, min(4, int(self.train_params.get('hl_count', 2))))
        self.train_params['learning_rate'] = max(0.0001, min(0.005, float(self.train_params.get('learning_rate', 0.001))))
        self.train_params['gamma'] = 1.0
        self.train_params['big_piece_weight'] = max(1, min(4, int(self.train_params.get('big_piece_weight', 1))))
        self.train_params['pieces_tracked'] = max(1, min(20, int(self.train_params.get('pieces_tracked', 10))))
        self._apply_piece_size_range(
            self.train_params.get('min_size', 1),
            self.train_params.get('max_size', 5)
        )
        self._apply_epsilon_range(
            self.train_params.get('epsilon_min_percent', 5),
            self.train_params.get('epsilon_current_percent', 20),
            apply_current=False
        )
        self.train_params['epsilon_half_life_batches'] = self._snap_epsilon_half_life(
            self.train_params.get('epsilon_half_life_batches', 10 ** 4.5)
        )
        return True

    def select_train_model(self, model_name):
        import os
        self.train_model_selected = model_name
        self.train_model_source = model_name
        model_path = os.path.join("models", model_name)
        checkpoint = self._load_model_checkpoint(model_path)
        if checkpoint:
            loaded_params = checkpoint.get('params')
            if not self._apply_loaded_train_params(loaded_params):
                arch = self._parse_model_arch(model_name)
                if arch:
                    self.train_params['hl_size_idx'], self.train_params['hl_count'] = arch
                    self.train_params['hl_count'] = max(1, min(4, int(self.train_params['hl_count'])))
        self.train_model_arch = (
            int(self.train_params.get('hl_size_idx', 1)),
            int(self.train_params.get('hl_count', 2))
        )
        self.epsilon_override_pending = False

    def initialize_train_menu(self):
        import os
        self.load_settings()
        self.refresh_train_models()
        self.train_dropdown_open = False
        self.train_option_rects = []
        self.train_model_selected = None
        self.train_model_source = None
        self.train_model_arch = None
        self.epsilon_override_pending = False

        model_path = self.get_model_filename()
        if os.path.exists(model_path):
            model_name = os.path.basename(model_path)
            self.select_train_model(model_name)

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

    def _is_standard_model_name(self, model_file):
        import os
        base = os.path.splitext(os.path.basename(model_file))[0]
        if not base.startswith("model-"):
            return False
        return self._parse_model_arch(model_file) is not None

    def _get_standard_model_base(self):
        hl_size_idx = int(self.train_params.get('hl_size_idx', 1))
        hl_size_idx = max(0, min(len(self.hl_sizes) - 1, hl_size_idx))
        size = self.hl_sizes[hl_size_idx]
        count = max(1, min(4, int(self.train_params.get('hl_count', 2))))
        return f"model-{size}-{count}"

    def _get_train_dropdown_label(self):
        import os
        current_arch = (
            int(self.train_params.get('hl_size_idx', 1)),
            int(self.train_params.get('hl_count', 2))
        )
        if self.train_model_selected and self.train_model_arch == current_arch:
            return os.path.splitext(self.train_model_selected)[0]
        return self._get_standard_model_base()

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
            self.reset(start_music=True, state_override="WATCH_AI")
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

    def _is_valid_model_char(self, char):
        return char.isascii() and (char.isalnum() or char in ("-", "_", ".", " "))

    def _normalize_model_filename(self, name):
        if not name:
            return None
        trimmed = name.strip()
        if not trimmed:
            return None
        safe_chars = [ch for ch in trimmed if self._is_valid_model_char(ch)]
        safe_name = "".join(safe_chars).strip(" .")
        if not safe_name:
            return None
        if not safe_name.lower().endswith(".pth"):
            safe_name += ".pth"
        return safe_name

    def save_named_model(self, name):
        import os
        if not self.agent:
            print("No model loaded to save.")
            return False
        filename = self._normalize_model_filename(name)
        if not filename:
            print("Model name is empty.")
            return False
        if not os.path.exists("models"):
            os.makedirs("models")
        path = os.path.join("models", filename)
        try:
            self.agent.save(path)
            print(f"Model saved ({path}).")
            self.refresh_train_models()
            return True
        except Exception as e:
            print(f"Failed to save model {path}: {e}")
            return False

    def _ensure_video_dir(self):
        import os
        video_dir = "Video"
        if not os.path.exists(video_dir):
            os.makedirs(video_dir)
        return video_dir

    def _unique_path(self, path):
        import os
        if not os.path.exists(path):
            return path
        base, ext = os.path.splitext(path)
        idx = 1
        while True:
            candidate = f"{base}-{idx}{ext}"
            if not os.path.exists(candidate):
                return candidate
            idx += 1

    def _normalize_recording_filename(self, name):
        if not name:
            return None
        trimmed = name.strip()
        if not trimmed:
            return None
        safe_chars = [ch for ch in trimmed if self._is_valid_model_char(ch)]
        safe_name = "".join(safe_chars).strip(" .")
        if not safe_name:
            return None
        if not safe_name.lower().endswith(".mp4"):
            safe_name += ".mp4"
        return safe_name

    def _is_watch_game_active(self):
        if self.state == "WATCH_AI":
            return True
        if self.state == "PAUSED" and getattr(self, "last_state", None) == "WATCH_AI":
            return True
        if self.state in ("ANIMATING_CLEAR", "ANIMATING_DROP") and getattr(self, "pre_anim_state", None) == "WATCH_AI":
            return True
        return False

    def start_recording(self):
        if self.is_recording or not self._is_watch_game_active():
            return False
        try:
            from recorder import VideoRecorder, RecorderError
        except Exception as exc:
            print(f"Recording unavailable: {exc}")
            return False

        import time
        video_dir = self._ensure_video_dir()
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        temp_name = f"recording-{timestamp}.mp4"
        import os
        temp_path = self._unique_path(os.path.join(video_dir, temp_name))

        self.load_recording_settings()
        recorder = VideoRecorder(
            self.screen_width,
            self.screen_height,
            60,
            want_audio=True,
            audio_device=self.record_audio_device,
            audio_device_name=self.record_audio_device_name,
            audio_loopback=self.record_audio_loopback,
        )
        try:
            recorder.start(temp_path)
        except RecorderError as exc:
            print(f"Recording failed: {exc}")
            return False
        except Exception as exc:
            print(f"Recording failed: {exc}")
            return False

        self.recorder = recorder
        self.is_recording = True
        self.record_failed = False
        self.recording_temp_path = temp_path
        if not recorder.audio_enabled:
            detail = recorder.audio_error or "audio capture unavailable"
            print(f"Recording started (video only): {detail}")
        else:
            if recorder.audio_device_label:
                mode = "loopback" if recorder.audio_loopback_active else "input"
                print(f"Recording started: {temp_path} (audio {mode}: {recorder.audio_device_label})")
            else:
                print(f"Recording started: {temp_path}")
        return True

    def _discard_recording_temp(self, reason=None):
        import os
        if self.recording_temp_path:
            try:
                os.remove(self.recording_temp_path)
            except Exception:
                pass
            if reason:
                print(reason)
        self.recording_temp_path = None
        self.record_name_active = False
        self.record_name_text = ""
        self.record_name_rect = None
        self.record_failed = False

    def stop_recording(self, prompt_name=True):
        if not self.is_recording or not self.recorder:
            return
        finalize_failed = False
        try:
            self.recorder.stop()
        except Exception as exc:
            print(f"Failed to finalize recording: {exc}")
            finalize_failed = True
        self.recorder = None
        self.is_recording = False

        if finalize_failed:
            self.record_failed = True

        if self.record_failed:
            self._discard_recording_temp("Recording discarded due to errors.")
            return

        if prompt_name and self.recording_temp_path:
            self.record_name_active = True
            self.record_name_text = ""
        else:
            if self.recording_temp_path:
                print(f"Recording saved ({self.recording_temp_path}).")
            self.recording_temp_path = None
            self.record_name_active = False
            self.record_name_text = ""
            self.record_name_rect = None

    def save_named_recording(self, name):
        import os
        if not self.recording_temp_path:
            print("No recording to save.")
            return False
        filename = self._normalize_recording_filename(name)
        if not filename:
            print("Recording name is empty.")
            return False
        video_dir = self._ensure_video_dir()
        path = self._unique_path(os.path.join(video_dir, filename))
        try:
            os.replace(self.recording_temp_path, path)
            print(f"Recording saved ({path}).")
            self.recording_temp_path = None
            self.record_failed = False
            return True
        except Exception as exc:
            print(f"Failed to save recording {path}: {exc}")
            return False

    def cancel_recording_name(self):
        if self.recording_temp_path:
            print(f"Recording kept ({self.recording_temp_path}).")
        self.recording_temp_path = None
        self.record_name_active = False
        self.record_name_text = ""
        self.record_name_rect = None
        self.record_failed = False

    def capture_recording_frame(self):
        if not self.is_recording or not self.recorder:
            return
        try:
            frame = pygame.surfarray.array3d(self.screen)
            self.recorder.write_frame(frame)
        except Exception as exc:
            print(f"Recording failed: {exc}")
            self.record_failed = True
            self.stop_recording(prompt_name=False)

    def get_short_game_length(self):
        """Get the number of pieces for a short game (1-20)."""
        return self.get_piece_history_length()

    def get_piece_history_length(self):
        """Get the number of pieces to track for reward history (1-20)."""
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

    def _get_advance_interval_ms(self, state_override=None):
        state = state_override or self.state
        if state == "WATCH_AI":
            return self.get_watch_ai_fall_speed()
        if state == "TRAINING":
            return 50
        if state == "PLAYING":
            if self.input_manager.is_down_held():
                return 80
            return self.fall_speed
        return self.fall_speed

    def _analyze_grid(self, grid_cells):
        """Return (max_height, holes, jaggedness, valleys, height_std) for a grid cell matrix."""
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

        valleys = 0
        if self.grid_width > 1:
            for x in range(self.grid_width):
                if x == 0:
                    min_neighbor = heights[1]
                elif x == self.grid_width - 1:
                    min_neighbor = heights[self.grid_width - 2]
                else:
                    min_neighbor = min(heights[x - 1], heights[x + 1])
                depth = min_neighbor - heights[x]
                if depth >= 3:
                    valleys += depth - 2

        max_height = max(heights) if heights else 0
        height_std = 0.0
        if heights:
            mean_height = sum(heights) / len(heights)
            variance = sum((height - mean_height) ** 2 for height in heights) / len(heights)
            height_std = math.sqrt(variance)
        return max_height, holes, jaggedness, valleys, height_std

    def get_grid_stats(self):
        """Returns (max_height, holes, jaggedness, valleys, height_std) for the current grid."""
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
        Returns: (simulated_height, simulated_holes, simulated_jaggedness, simulated_valleys, simulated_height_std)
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
        p = Pentomino(0, 0, self.allowed_shapes, self.allowed_shape_weights)

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

    def reset(self, start_music=True, state_override=None):
        self.grid = Grid(self.grid_width, self.grid_height, self.cell_size)
        self.score = 0
        self.level = 1
        self.lines_cleared_total = 0
        self.pieces_locked = 0
        self.fall_speed = 1000
        self.current_trajectory = []
        self.piece_history = deque()
        self.pending_reward_event = None
        self.short_games_move_count = 0
        
        # Save original state before it might be changed
        original_state = self.state
        
        # Determine allowed shapes based on selection
        from tetrominoes import get_allowed_shapes, get_shape_weights
        
        self.allowed_shape_weights = None
        if self.state == "TRAINING" or self.state == "TRAIN_MENU": # Use training params
             min_size = self.train_params.get('min_size', 1)
             max_size = self.train_params.get('max_size', 5)
             big_weight = max(1, min(4, int(self.train_params.get('big_piece_weight', 1))))
             # In training, include all groups but filter by size range.
             self.allowed_shapes = get_allowed_shapes(True, True, True, max_size, min_size=min_size)
             self.allowed_shape_weights = get_shape_weights(self.allowed_shapes, weight_base=big_weight)
        else:
             min_size, max_size = self._clamp_piece_size_range(self.active_min_size, self.active_max_size)
             self.active_min_size = min_size
             self.active_max_size = max_size
             self.allowed_shapes = get_allowed_shapes(True, True, True, max_size, min_size=min_size)
        
        # Fallback if nothing selected (should be prevented by UI, but safety first)
        if not self.allowed_shapes:
             self.allowed_shapes = get_allowed_shapes(True, False, False)
             self.allowed_shape_weights = None
        self.current_piece = self.spawn_piece()
        self.start_stats = self.get_grid_stats()
        self.next_piece = self.spawn_piece()
        
        # Only set to PLAYING if not already in TRAINING (to avoid overwriting state during training reset)
        if self.state != "TRAINING":
            self.state = state_override if state_override is not None else "PLAYING"
            
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

            if self.record_name_active:
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RETURN:
                        if self.save_named_recording(self.record_name_text):
                            self.record_name_active = False
                            self.record_name_text = ""
                    elif event.key == pygame.K_ESCAPE:
                        self.cancel_recording_name()
                    elif event.key == pygame.K_BACKSPACE:
                        self.record_name_text = self.record_name_text[:-1]
                    elif event.unicode and self._is_valid_model_char(event.unicode):
                        if len(self.record_name_text) < self.record_name_limit:
                            self.record_name_text += event.unicode
                    continue
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if self.record_name_rect and not self.record_name_rect.collidepoint(event.pos):
                        self.cancel_recording_name()
                    continue
                continue

            if self.save_name_active:
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RETURN:
                        if self.save_named_model(self.save_name_text):
                            self.save_name_active = False
                            self.save_name_text = ""
                    elif event.key == pygame.K_ESCAPE:
                        self.save_name_active = False
                        self.save_name_text = ""
                    elif event.key == pygame.K_BACKSPACE:
                        self.save_name_text = self.save_name_text[:-1]
                    elif event.unicode and self._is_valid_model_char(event.unicode):
                        if len(self.save_name_text) < self.save_name_limit:
                            self.save_name_text += event.unicode
                    continue
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if self.save_name_rect and not self.save_name_rect.collidepoint(event.pos):
                        self.save_name_active = False
                        self.save_name_text = ""
                    continue
                continue
            
            action = self.input_manager.get_action(event)
            
            if action == "EXIT":
                if self.state in ["TRAINING", "TRAIN_MENU"] and self.agent:
                    self.agent.save(self.get_model_filename())
                    print(f"Model saved ({self.get_model_filename()}).")
                    self.save_settings()

                if self.state in ["PLAYING", "PLAY_MENU", "GAMEOVER", "WATCH_AI", "WATCH_MENU", "TRAIN_MENU", "TRAINING"]:
                    if self.is_recording:
                        self.stop_recording(prompt_name=True)
                    if self.state in ["PLAYING", "PLAY_MENU", "GAMEOVER", "WATCH_AI", "WATCH_MENU"]:
                        self.clear_game_state()
                    self.state = "MENU"
                    self.audio.stop()
                    self.watch_dropdown_open = False
            
            if self.state == "MENU":
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1: # Left click
                        mouse_pos = event.pos
                        if self.btn_start_rect and self.btn_start_rect.collidepoint(mouse_pos):
                            self.clear_game_state()
                            self.state = "PLAY_MENU"
                            self.audio.stop()
                        elif self.btn_train_rect and self.btn_train_rect.collidepoint(mouse_pos):
                            self.state = "TRAIN_MENU"
                            self.initialize_train_menu()
                            # Stop music if visual mode is off (from saved params or default)
                            if not self.train_params['visual_mode']:
                                self.audio.stop()
                        elif self.btn_watch_rect and self.btn_watch_rect.collidepoint(mouse_pos):
                            self.clear_game_state()
                            self.state = "WATCH_MENU"
                            self.audio.stop()
                            self.agent = None
                            self.watch_dropdown_open = False
                            self.refresh_watch_models()

            elif self.state == "TRAIN_MENU":
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        mouse_pos = event.pos
                        dropdown_handled = False
                        if self.train_dropdown_open:
                            for rect, model_name in self.train_option_rects:
                                if rect.collidepoint(mouse_pos):
                                    self.select_train_model(model_name)
                                    self.train_dropdown_open = False
                                    dropdown_handled = True
                                    break
                            if not dropdown_handled:
                                if self.train_dropdown_rect and self.train_dropdown_rect.collidepoint(mouse_pos):
                                    self.train_dropdown_open = False
                                else:
                                    self.train_dropdown_open = False
                                dropdown_handled = True
                        if dropdown_handled:
                            continue

                        if self.train_dropdown_rect and self.train_dropdown_rect.collidepoint(mouse_pos):
                            self.train_dropdown_open = not self.train_dropdown_open
                            if self.train_dropdown_open:
                                self.refresh_train_models()
                                self.train_dropdown_scroll = 0
                            continue

                        if self.btn_back_rect and self.btn_back_rect.collidepoint(event.pos):
                            if self.agent:
                                self.agent.save(self.get_model_filename())
                                print(f"Model saved ({self.get_model_filename()}).")
                                self.save_settings()
                            self.state = "MENU"
                            self.audio.stop()
                        elif self.train_chk_rect and self.train_chk_rect.collidepoint(event.pos):
                            self.train_params['visual_mode'] = not self.train_params['visual_mode']
                        elif self.agent and self.btn_train_save_rect and self.btn_train_save_rect.collidepoint(event.pos):
                            self.save_name_active = True
                            self.save_name_text = ""
                            self.save_name_rect = None
                            self.train_dropdown_open = False
                        elif hasattr(self, 'short_games_chk_rect') and self.short_games_chk_rect and self.short_games_chk_rect.collidepoint(event.pos):
                            self.train_params['short_games'] = not self.train_params.get('short_games', False)
                        elif self.btn_train_start_rect and self.btn_train_start_rect.collidepoint(event.pos):
                            agent = self.create_agent()
                            if agent:
                                self.train_dropdown_open = False
                                self.agent = agent
                                self.reset()
                                # Try to load existing model
                                import os
                                current_arch = (
                                    int(self.train_params.get('hl_size_idx', 1)),
                                    int(self.train_params.get('hl_count', 2))
                                )
                                selected_arch = self.train_model_arch
                                selected_path = None
                                use_selected = False
                                if self.train_model_source:
                                    selected_path = os.path.join("models", self.train_model_source)
                                    use_selected = selected_arch == current_arch and os.path.exists(selected_path)

                                standard_path = self.get_model_filename()
                                model_file = selected_path if use_selected else standard_path

                                if not use_selected:
                                    self.train_model_source = os.path.basename(standard_path)
                                    self.train_model_selected = self.train_model_source
                                    self.train_model_arch = current_arch
                                    self.refresh_train_models()

                                if model_file and os.path.exists(model_file):
                                    try:
                                        self.agent.load(model_file)
                                        print(f"Loaded existing model {model_file}.")
                                        if self.train_model_source and not self._is_standard_model_name(self.train_model_source):
                                            standard_path = self.get_model_filename()
                                            try:
                                                self.agent.save(standard_path)
                                                self.train_model_source = os.path.basename(standard_path)
                                                self.train_model_selected = self.train_model_source
                                                self.train_model_arch = current_arch
                                                self.refresh_train_models()
                                                print(f"Saved model to standard filename ({standard_path}).")
                                            except Exception as e:
                                                print(f"Failed to save standard model copy {standard_path}: {e}")
                                    except Exception as e:
                                        print(f"Failed to load model {model_file}: {e}")
                                else:
                                    print(f"No existing model found for {standard_path}, starting fresh.")
                                    if model_file == standard_path:
                                        try:
                                            self.agent.save(standard_path)
                                            self.refresh_train_models()
                                            print(f"Created new model file ({standard_path}).")
                                        except Exception as e:
                                            print(f"Failed to create new model file {standard_path}: {e}")

                                self._apply_epsilon_range(
                                    self.train_params.get('epsilon_min_percent', 5),
                                    self.train_params.get('epsilon_current_percent', 20),
                                    apply_current=self.epsilon_override_pending
                                )
                                self.epsilon_override_pending = False
                                
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
                                if name == 'epsilon_range_percent':
                                    self.active_slider_handle = self._select_epsilon_handle(event.pos[0], rect)
                                elif name == 'piece_size_range':
                                    self.active_slider_handle = self._select_piece_size_handle(event.pos[0], rect)
                                else:
                                    self.active_slider_handle = None
                                self.update_train_slider(event.pos[0])
                        
                        if self.train_slider_rect and self.train_slider_rect.collidepoint(event.pos):

                            self.dragging_slider = True
                            self.update_volume(event.pos[0], is_train_slider=True)
                    elif event.button in (4, 5):
                        if self.train_dropdown_open:
                            delta = -1 if event.button == 4 else 1
                            self._scroll_dropdown("train", delta)
                                
                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1:
                        self.active_slider = None
                        self.active_slider_handle = None
                        self.dragging_slider = False
                        
                elif event.type == pygame.MOUSEMOTION:
                    if self.active_slider:
                        self.update_train_slider(event.pos[0])
                    elif self.dragging_slider:
                        self.update_volume(event.pos[0], is_train_slider=True)
                elif event.type == pygame.MOUSEWHEEL:
                    if self.train_dropdown_open:
                        self._scroll_dropdown("train", -event.y)

            elif self.state == "WATCH_MENU" or self.state == "WATCH_AI" or (self.state == "PAUSED" and hasattr(self, 'last_state') and self.last_state == "WATCH_AI") or (self.state == "GAMEOVER" and hasattr(self, 'last_state') and self.last_state == "WATCH_AI"):
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

                        if self.btn_main_menu_rect and self.btn_main_menu_rect.collidepoint(mouse_pos):
                            if self.is_recording:
                                self.stop_recording(prompt_name=True)
                            self.clear_game_state()
                            self.state = "MENU"
                            self.audio.stop()
                            self.watch_dropdown_open = False
                        elif self.btn_record_rect and self.btn_record_rect.collidepoint(mouse_pos):
                            record_enabled = self._is_watch_game_active() or self.is_recording
                            if record_enabled:
                                if self.is_recording:
                                    self.stop_recording(prompt_name=True)
                                else:
                                    self.start_recording()
                        elif self.btn_watch_start_rect and self.btn_watch_start_rect.collidepoint(mouse_pos):
                            start_active = self.watch_model_selected in self.watch_models
                            if start_active:
                                self.active_min_size, self.active_max_size = self._clamp_piece_size_range(self.watch_min_size, self.watch_max_size)
                                self.start_watch_ai()
                        elif self.watch_dropdown_rect and self.watch_dropdown_rect.collidepoint(mouse_pos):
                            self.watch_dropdown_open = not self.watch_dropdown_open
                            if self.watch_dropdown_open:
                                self.refresh_watch_models()
                                self.watch_dropdown_scroll = 0
                        elif self.watch_slider_rect and self.watch_slider_rect.collidepoint(mouse_pos):
                            self.active_size_slider = "WATCH"
                            self.active_size_handle = self._select_size_handle(mouse_pos[0], self.watch_slider_rect, self.watch_min_size, self.watch_max_size)
                            self.dragging_size_slider = True
                            self.dragging_slider = False
                            self.update_size_slider(mouse_pos[0])
                        elif self.slider_rect and self.slider_rect.collidepoint(mouse_pos):
                            self.dragging_slider = True
                            self.dragging_size_slider = False
                            self.update_volume(mouse_pos[0])
                    elif event.button in (4, 5):
                        if self.watch_dropdown_open:
                            delta = -1 if event.button == 4 else 1
                            self._scroll_dropdown("watch", delta)
                 elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1:
                        self.dragging_slider = False
                        self.dragging_size_slider = False
                        self.active_size_slider = None
                        self.active_size_handle = None
                 elif event.type == pygame.MOUSEMOTION:
                    if self.dragging_size_slider:
                        self.update_size_slider(event.pos[0])
                    elif self.dragging_slider:
                        self.update_volume(event.pos[0])
                 elif event.type == pygame.MOUSEWHEEL:
                    if self.watch_dropdown_open:
                        self._scroll_dropdown("watch", -event.y)

            elif self.state == "PLAY_MENU" or self.state == "PLAYING" or (self.state == "PAUSED" and hasattr(self, 'last_state') and self.last_state == "PLAYING") or (self.state == "GAMEOVER" and hasattr(self, 'last_state') and self.last_state == "PLAYING"):
                 if event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        mouse_pos = event.pos
                        if self.btn_main_menu_rect and self.btn_main_menu_rect.collidepoint(mouse_pos):
                            self.clear_game_state()
                            self.state = "MENU"
                            self.audio.stop()
                        elif self.btn_play_start_rect and self.btn_play_start_rect.collidepoint(mouse_pos):
                            self.active_min_size, self.active_max_size = self._clamp_piece_size_range(self.play_min_size, self.play_max_size)
                            self.reset(start_music=True, state_override="PLAYING")
                        elif self.play_slider_rect and self.play_slider_rect.collidepoint(mouse_pos):
                            self.active_size_slider = "PLAY"
                            self.active_size_handle = self._select_size_handle(mouse_pos[0], self.play_slider_rect, self.play_min_size, self.play_max_size)
                            self.dragging_size_slider = True
                            self.dragging_slider = False
                            self.update_size_slider(mouse_pos[0])
                        elif self.slider_rect and self.slider_rect.collidepoint(mouse_pos):
                            self.dragging_slider = True
                            self.dragging_size_slider = False
                            self.update_volume(event.pos[0])
                 elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1:
                        self.dragging_slider = False
                        self.dragging_size_slider = False
                        self.active_size_slider = None
                        self.active_size_handle = None
                 elif event.type == pygame.MOUSEMOTION:
                    if self.dragging_size_slider:
                        self.update_size_slider(event.pos[0])
                    elif self.dragging_slider:
                        self.update_volume(event.pos[0])

            elif self.state == "TRAINING":
                 if event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        if self.train_chk_rect and self.train_chk_rect.collidepoint(event.pos):
                            self.train_params['visual_mode'] = not self.train_params['visual_mode']
                            if not self.train_params['visual_mode']:
                                self.audio.stop()
                                print("Visual mode OFF - Audio stopped")
                            else:
                                self.audio.start()
                                print("Visual mode ON - Audio started")
                        elif self.btn_train_start_rect and self.btn_train_start_rect.collidepoint(event.pos):
                            # STOP button pressed
                            if self.agent:
                                self.agent.save(self.get_model_filename())
                                print(f"Model saved ({self.get_model_filename()}).")
                                self.save_settings()
                            self.state = "TRAIN_MENU"
                            # Stop music when returning to Train Menu
                            self.audio.stop()
                        elif self.btn_back_rect and self.btn_back_rect.collidepoint(event.pos):
                            # MAIN MENU button pressed
                            if self.agent:
                                self.agent.save(self.get_model_filename())
                                print(f"Model saved ({self.get_model_filename()}).")
                                self.save_settings()
                            self.state = "MENU"
                            # Main menu has no music
                            self.audio.stop()
                        elif self.train_slider_rect and self.train_slider_rect.collidepoint(event.pos):
                            self.dragging_slider = True
                            self.update_volume(event.pos[0], is_train_slider=True)
                 elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1:
                        self.dragging_slider = False
                 elif event.type == pygame.MOUSEMOTION:
                    if self.dragging_slider:
                        self.update_volume(event.pos[0], is_train_slider=True)
                        
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
                # Scale animation timing to the current advance rate (baseline ~2x faster).
                advance_ms = max(1.0, float(self._get_advance_interval_ms(self.pre_anim_state)))
                self.anim_clear_ms = advance_ms * 0.1
                drop_ms_per_line = advance_ms * 0.125
                self.anim_drop_speed_px_per_ms = self.cell_size / max(1.0, drop_ms_per_line)
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
        if self.is_recording:
            self.stop_recording(prompt_name=True)

    def _flush_piece_entry(self, entry):
        self.agent.add_trajectory_with_done(entry['trajectory'], entry['reward'])
        self.agent.record_training_stats(len(entry['trajectory']), entry['lines'], entry['game_over'])

    def _flush_oldest_piece_history(self):
        entry = self.piece_history.popleft()
        self._flush_piece_entry(entry)

    def _flush_piece_history(self):
        while self.piece_history:
            self._flush_oldest_piece_history()

    def apply_trajectory_reward(self, trajectory, reward_value, lines_cleared, is_game_over, flush_all=False):
        """Accumulate reward over recent pieces and push once they age out."""
        if not self.agent or not trajectory:
            return

        if is_game_over:
            reward_value -= 2.0

        history_len = self.get_piece_history_length()
        entry = {
            'trajectory': trajectory,
            'reward': 0.0,
            'lines': lines_cleared,
            'game_over': is_game_over
        }
        self.piece_history.append(entry)

        if history_len <= 1:
            entry['reward'] += reward_value
        else:
            gamma = 0.1 ** (1.0 / (history_len - 1))
            age = 0
            for piece_entry in reversed(self.piece_history):
                if age >= history_len:
                    break
                piece_entry['reward'] += reward_value * (gamma ** age)
                age += 1

        while len(self.piece_history) >= history_len:
            self._flush_oldest_piece_history()

        if flush_all:
            self._flush_piece_history()

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
        max_height_before, holes_before_step, jaggedness_before, valleys_before, height_std_before = self.get_grid_stats()
        
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
            max_height_after, holes_after, jaggedness_after, valleys_after, height_std_after = self.get_post_clear_grid_stats(clearing_lines)
            hole_delta = holes_after - holes_before_step
            jaggedness_delta = jaggedness_after - jaggedness_before
            valley_delta = valleys_after - valleys_before
            max_height_delta = max_height_after - max_height_before
            height_std_delta = height_std_after - height_std_before

            is_game_over = (self.state == "GAMEOVER")
            
            # Calculate reward for this placement
            reward = self.agent.calculate_reward(
                lines_cleared,
                hole_delta,
                jaggedness_delta,
                valley_delta,
                max_height_delta,
                height_std_delta
            )

            # Apply reward to this piece's trajectory
            if self.state == "ANIMATING_CLEAR":
                # Store trajectory with reward for after animation completes
                self.pending_reward_event = (trajectory, reward, lines_cleared, is_game_over, piece_limit_reached)
            elif is_headless_clearing:
                pending_reward = reward
            else:
                self.apply_trajectory_reward(trajectory, reward, lines_cleared, is_game_over, flush_all=done)

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

                self.apply_trajectory_reward(trajectory, pending_reward, lines_cleared, is_game_over, flush_all=done)

            if done and self.state != "ANIMATING_CLEAR":
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
        
        # Update music speed based on the active fall speed (human or watch AI).
        # Preserve the source state during line-clear animations so watch tempo doesn't drop.
        tempo_state = self.state
        if self.state in ("ANIMATING_CLEAR", "ANIMATING_DROP") and hasattr(self, 'pre_anim_state'):
            tempo_state = self.pre_anim_state
        effective_fall_speed = self.get_watch_ai_fall_speed() if tempo_state == "WATCH_AI" else self.fall_speed
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
            clear_ms = getattr(self, "anim_clear_ms", 200)
            if current_time - self.animation_timer > clear_ms:
                self.apply_line_clears()
                self.state = "ANIMATING_DROP"
                self.animation_timer = current_time
                self.drop_anim_last_time = current_time
                
        elif self.state == "ANIMATING_DROP":
            # Reduce offsets
            all_done = True
            drop_speed = getattr(self, "anim_drop_speed_px_per_ms", (self.cell_size / 125.0))
            last_time = getattr(self, "drop_anim_last_time", current_time)
            dt_ms = max(0, current_time - last_time)
            self.drop_anim_last_time = current_time
            delta = drop_speed * dt_ms
            for i in range(self.grid_height):
                if self.row_offsets[i] < 0:
                    self.row_offsets[i] = min(0, self.row_offsets[i] + delta)
                    all_done = False
                elif self.row_offsets[i] > 0: # Should not happen with drop
                    self.row_offsets[i] = max(0, self.row_offsets[i] - delta)
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
                    if self.is_recording:
                        self.stop_recording(prompt_name=True)
                
                # Process pending reward even if game over happened after the clear
                if was_training and self.pending_reward_event is not None:
                    trajectory, reward_value, pending_lines, pending_game_over, pending_end_episode = self.pending_reward_event
                    reward_game_over = pending_game_over or game_over_after_spawn
                    end_episode = pending_end_episode or reward_game_over
                    self.apply_trajectory_reward(trajectory, reward_value, pending_lines, reward_game_over, flush_all=end_episode)
                    self.pending_reward_event = None
                    if end_episode:
                        self.finish_training_round()

    def draw(self):
        self.ui.draw_background()
        mouse_pos = pygame.mouse.get_pos()
        
        if self.state == "MENU":
            title = self.ui.large_font.render("OMINIS", True, self.ui.text_color)
            self.screen.blit(title, (self.screen_width // 2 - title.get_width() // 2, 100))
            self.chk_pent_rect = None
            self.chk_tet_rect = None
            self.chk_omi_rect = None

            button_width = 220
            button_height = 50
            button_gap = 20
            start_y = 320
            btn_x = self.screen_width // 2 - button_width // 2

            self.btn_start_rect = self.ui.draw_button(btn_x, start_y, button_width, button_height, "Play Game", True, mouse_pos)
            self.btn_watch_rect = self.ui.draw_button(btn_x, start_y + button_height + button_gap, button_width, button_height, "Watch AI Play", True, mouse_pos)
            self.btn_train_rect = self.ui.draw_button(btn_x, start_y + (button_height + button_gap) * 2, button_width, button_height, "Train AI", True, mouse_pos)
            
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
            dropdown_active = not is_training
            dropdown_open = self.train_dropdown_open and dropdown_active
            save_active = self.agent is not None
            self.btn_back_rect, self.train_chk_rect, self.btn_train_start_rect, self.btn_train_save_rect, self.train_slider_rects, self.train_slider_rect, self.short_games_chk_rect, train_dropdown_anchor = self.ui.draw_train_menu(self.screen_width, self.screen_height, self.train_params, mouse_pos, display_grid, is_training=is_training, volume=self.volume, save_active=save_active)

            train_dropdown_layout = None
            if train_dropdown_anchor:
                import os
                padding = 20
                options = [os.path.splitext(name)[0] for name in self.train_models]
                list_options = list(self.train_models)
                option_width = None
                if dropdown_open:
                    text_font = self.ui.small_font
                    max_text_width = text_font.size("No models found")[0]
                    for option in list_options:
                        max_text_width = max(max_text_width, text_font.size(option)[0])
                    needed_width = max_text_width + 16
                    max_option_width = self.screen_width - train_dropdown_anchor.x - padding
                    option_width = max(train_dropdown_anchor.width, min(needed_width, max_option_width))
                selected_idx = None
                if self.train_model_selected in self.train_models:
                    selected_idx = self.train_models.index(self.train_model_selected)
                available_height = self.screen_height - (train_dropdown_anchor.y + train_dropdown_anchor.height) - padding
                total_items = len(list_options)
                visible_count = 0
                if total_items > 0:
                    visible_count = max(1, min(total_items, available_height // train_dropdown_anchor.height))
                self.train_dropdown_visible = visible_count
                max_scroll = max(0, total_items - visible_count)
                self.train_dropdown_scroll = max(0, min(self.train_dropdown_scroll, max_scroll))
                list_width = option_width if option_width is not None else train_dropdown_anchor.width
                self.train_dropdown_list_width = list_width
                train_dropdown_layout = (
                    train_dropdown_anchor.x,
                    train_dropdown_anchor.y,
                    train_dropdown_anchor.width,
                    train_dropdown_anchor.height,
                    options,
                    list_options,
                    option_width,
                    selected_idx,
                    visible_count,
                    self.train_dropdown_scroll
                )
            else:
                self.train_dropdown_rect = None
                self.train_option_rects = []
                self.train_dropdown_open = False
            
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

            if train_dropdown_layout:
                (drop_x, drop_y, drop_w, drop_h, options, list_options, option_width, selected_idx, visible_count, scroll_offset) = train_dropdown_layout
                self.train_dropdown_rect, option_rects = self.ui.draw_dropdown(
                    drop_x,
                    drop_y,
                    drop_w,
                    drop_h,
                    self._get_train_dropdown_label(),
                    options,
                    selected_idx,
                    dropdown_open,
                    mouse_pos,
                    font=self.ui.small_font,
                    option_width=option_width,
                    list_options=list_options,
                    active=dropdown_active,
                    scroll_offset=scroll_offset,
                    max_visible=visible_count
                )
                start_idx = scroll_offset
                visible_models = self.train_models[start_idx:start_idx + len(option_rects)]
                self.train_option_rects = list(zip(option_rects, visible_models))

            if self.save_name_active:
                overlay = pygame.Surface((self.screen_width, self.screen_height))
                overlay.set_alpha(180)
                overlay.fill((0, 0, 0))
                self.screen.blit(overlay, (0, 0))

                panel_width = 320
                panel_height = 140
                panel_x = (self.screen_width - panel_width) // 2
                panel_y = (self.screen_height - panel_height) // 2
                panel_rect = pygame.Rect(panel_x, panel_y, panel_width, panel_height)

                pygame.draw.rect(self.screen, self.ui.bg_color, panel_rect, border_radius=10)
                pygame.draw.rect(self.screen, self.ui.border_color, panel_rect, 2, border_radius=10)

                title = self.ui.font.render("SAVE MODEL", True, self.ui.text_color)
                self.screen.blit(title, (panel_x + panel_width // 2 - title.get_width() // 2, panel_y + 12))

                name_label = self.ui.small_font.render("Name", True, self.ui.text_color)
                self.screen.blit(name_label, (panel_x + 20, panel_y + 52))

                self.save_name_rect = self.ui.draw_text_input(
                    panel_x + 20,
                    panel_y + 70,
                    panel_width - 40,
                    28,
                    self.save_name_text,
                    "model-name",
                    active=True,
                    mouse_pos=mouse_pos,
                    font=self.ui.small_font
                )

                hint = self.ui.small_font.render("Press Enter to save", True, (120, 120, 120))
                self.screen.blit(hint, (panel_x + panel_width // 2 - hint.get_width() // 2, panel_y + panel_height - 26))
            else:
                self.save_name_rect = None
            
        else:
            self.train_dropdown_rect = None
            self.train_option_rects = []
            self.train_dropdown_open = False

            # Standard Padding
            padding = 20
            section_gap = 12
            
            # Left Pane Layout
            left_pane_x = padding
            current_y = padding
            watch_ui = self.state in ["WATCH_MENU", "WATCH_AI"] or (self.state == "PAUSED" and hasattr(self, 'last_state') and self.last_state == "WATCH_AI") or (self.state == "GAMEOVER" and hasattr(self, 'last_state') and self.last_state == "WATCH_AI")
            if not watch_ui and self.state in ["ANIMATING_CLEAR", "ANIMATING_DROP"] and hasattr(self, 'pre_anim_state'):
                watch_ui = (self.pre_anim_state == "WATCH_AI")
            
            # Scoreboard (Height 150)
            self.ui.draw_score(self.score, self.level, self.lines_cleared_total, left_pane_x, current_y)
            current_y += 150 + padding
            
            # Preview (Height 210)
            self.ui.draw_preview(self.next_piece, left_pane_x, current_y, self.cell_size)
            current_y += 210 + padding

            # Instructions (Variable Height)
            instruction_mode = "WATCH_AI" if watch_ui else "PLAYING"
            inst_height = 100 if watch_ui else 140
            self.ui.draw_instructions(left_pane_x, current_y, mode=instruction_mode)
            current_y += inst_height + section_gap

            layout_offset = 10 if watch_ui else 20
            current_y += layout_offset

            # Volume Slider (Height ~40)
            self.slider_rect = self.ui.draw_slider(left_pane_x + 30, current_y, 150, self.volume, "Volume", mouse_pos)
            current_y += 40 + section_gap

            watch_dropdown_layout = None
            if watch_ui:
                dropdown_height = 30
                dropdown_width = 210
                options = list(self.watch_models)
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
                dropdown_y = current_y - 15
                available_height = self.screen_height - (dropdown_y + dropdown_height) - padding
                total_items = len(list_options)
                visible_count = 0
                if total_items > 0:
                    visible_count = max(1, min(total_items, available_height // dropdown_height))
                self.watch_dropdown_visible = visible_count
                max_scroll = max(0, total_items - visible_count)
                self.watch_dropdown_scroll = max(0, min(self.watch_dropdown_scroll, max_scroll))
                list_width = option_width if option_width is not None else dropdown_width
                self.watch_dropdown_list_width = list_width
                watch_dropdown_layout = (left_pane_x, dropdown_y, dropdown_width, dropdown_height, options, list_options, option_width, selected_idx, visible_count, self.watch_dropdown_scroll)
                current_y += dropdown_height + section_gap
            else:
                self.watch_dropdown_rect = None
                self.watch_option_rects = []
                self.watch_dropdown_open = False

            size_slider_width = 140
            start_button_width = 60
            start_button_height = 30
            size_slider_x = left_pane_x
            size_slider_y = current_y + (15 if watch_ui else 0)
            start_button_x = size_slider_x + size_slider_width + 10
            start_button_y = size_slider_y - 10

            if watch_ui:
                min_size, max_size = self._clamp_piece_size_range(self.watch_min_size, self.watch_max_size)
                self.watch_min_size = min_size
                self.watch_max_size = max_size
            else:
                min_size, max_size = self._clamp_piece_size_range(self.play_min_size, self.play_max_size)
                self.play_min_size = min_size
                self.play_max_size = max_size

            label = f"Piece Size: {min_size}->{max_size}"
            size_rect = self.ui.draw_range_slider(
                size_slider_x,
                size_slider_y,
                size_slider_width,
                (min_size - 1) / 4.0,
                (max_size - 1) / 4.0,
                label,
                mouse_pos,
                self.ui.small_font,
                bar_offset_y=-3
            )
            if watch_ui:
                self.watch_slider_rect = size_rect
                self.play_slider_rect = None
                start_active = self.watch_model_selected in self.watch_models
                self.btn_watch_start_rect = self.ui.draw_button(start_button_x, start_button_y, start_button_width, start_button_height, "Start", start_active, mouse_pos, font=self.ui.small_font)
                self.btn_play_start_rect = None
            else:
                self.play_slider_rect = size_rect
                self.watch_slider_rect = None
                self.btn_play_start_rect = self.ui.draw_button(start_button_x, start_button_y, start_button_width, start_button_height, "Start", True, mouse_pos, font=self.ui.small_font)
                self.btn_watch_start_rect = None
            current_y = size_slider_y + 40 + section_gap

            self.btn_record_rect = None
            if watch_ui:
                record_enabled = self._is_watch_game_active() or self.is_recording
                record_label = "Stop" if self.is_recording else "Record"
                self.btn_record_rect = self.ui.draw_button(left_pane_x + 30, current_y, 150, 34, record_label, record_enabled, mouse_pos, font=self.ui.small_font)
                current_y += 34 + section_gap

            self.btn_main_menu_rect = self.ui.draw_button(left_pane_x + 30, current_y, 150, 34, "Main Menu", True, mouse_pos)
            
            # Grid Offset
            offset_x = left_pane_x + 210 + padding
            # Grid border is drawn at offset_y - 5. We want border at 'padding'.
            # So offset_y - 5 = padding => offset_y = padding + 5
            offset_y = padding + 5
            
            # Pass offsets and flash lines if animating
            offsets = self.row_offsets if self.state == "ANIMATING_DROP" else None
            flash = self.clearing_lines if self.state == "ANIMATING_CLEAR" else None
            
            self.ui.draw_grid(self.grid, offset_x, offset_y, offsets, flash)
            
            if self.current_piece and (self.state == "PLAYING" or self.state == "WATCH_AI" or self.state == "PAUSED" or self.state == "GAMEOVER"):
                # Draw Ghost Piece
                ghost = self.get_ghost_piece()
                if ghost and ghost.y != self.current_piece.y:
                    self.ui.draw_ghost_pentomino(ghost, offset_x, offset_y, self.cell_size)
                
                self.ui.draw_pentomino(self.current_piece, offset_x, offset_y, self.cell_size)
            self.btn_back_rect = None
            self.btn_quit_rect = None
            
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
                (drop_x, drop_y, dropdown_width, dropdown_height,
                 options, list_options, option_width, selected_idx, visible_count, scroll_offset) = watch_dropdown_layout
                self.watch_dropdown_rect, option_rects = self.ui.draw_dropdown(
                    drop_x,
                    drop_y,
                    dropdown_width,
                    dropdown_height,
                    "Select Model",
                    options,
                    selected_idx,
                    self.watch_dropdown_open,
                    mouse_pos,
                    font=self.ui.small_font,
                    option_width=option_width,
                    list_options=list_options,
                    scroll_offset=scroll_offset,
                    max_visible=visible_count
                )
                start_idx = scroll_offset
                visible_models = self.watch_models[start_idx:start_idx + len(option_rects)]
                self.watch_option_rects = list(zip(option_rects, visible_models))

        if self.record_name_active:
            overlay = pygame.Surface((self.screen_width, self.screen_height))
            overlay.set_alpha(180)
            overlay.fill((0, 0, 0))
            self.screen.blit(overlay, (0, 0))

            panel_width = 320
            panel_height = 140
            panel_x = (self.screen_width - panel_width) // 2
            panel_y = (self.screen_height - panel_height) // 2
            panel_rect = pygame.Rect(panel_x, panel_y, panel_width, panel_height)

            pygame.draw.rect(self.screen, self.ui.bg_color, panel_rect, border_radius=10)
            pygame.draw.rect(self.screen, self.ui.border_color, panel_rect, 2, border_radius=10)

            title = self.ui.font.render("SAVE RECORDING", True, self.ui.text_color)
            self.screen.blit(title, (panel_x + panel_width // 2 - title.get_width() // 2, panel_y + 12))

            name_label = self.ui.small_font.render("Name", True, self.ui.text_color)
            self.screen.blit(name_label, (panel_x + 20, panel_y + 52))

            self.record_name_rect = self.ui.draw_text_input(
                panel_x + 20,
                panel_y + 70,
                panel_width - 40,
                28,
                self.record_name_text,
                "recording-name",
                active=True,
                mouse_pos=mouse_pos,
                font=self.ui.small_font
            )

            hint = self.ui.small_font.render("Press Enter to save", True, (120, 120, 120))
            self.screen.blit(hint, (panel_x + panel_width // 2 - hint.get_width() // 2, panel_y + panel_height - 26))
        else:
            self.record_name_rect = None

        if self.is_recording:
            self.capture_recording_frame()

        pygame.display.flip()

    def update_volume(self, mouse_x, is_train_slider=False):
        rect = self.train_slider_rect if is_train_slider else self.slider_rect
        if rect:
            # Calculate volume from mouse position relative to slider
            rel_x = mouse_x - rect.x
            vol = rel_x / rect.width
            self.volume = max(0.0, min(1.0, vol))
            self.audio.set_volume(self.volume)

    def _scroll_dropdown(self, dropdown, delta):
        if dropdown == "watch":
            total = len(self.watch_models)
            visible = max(1, int(self.watch_dropdown_visible) or 1)
            max_scroll = max(0, total - visible)
            self.watch_dropdown_scroll = max(0, min(max_scroll, self.watch_dropdown_scroll + delta))
        elif dropdown == "train":
            total = len(self.train_models)
            visible = max(1, int(self.train_dropdown_visible) or 1)
            max_scroll = max(0, total - visible)
            self.train_dropdown_scroll = max(0, min(max_scroll, self.train_dropdown_scroll + delta))

    def clear_game_state(self):
        self.grid = Grid(self.grid_width, self.grid_height, self.cell_size)
        self.score = 0
        self.level = 1
        self.lines_cleared_total = 0
        self.pieces_locked = 0
        self.fall_speed = 1000
        self.current_piece = None
        self.next_piece = None
        self.allowed_shapes = []
        self.allowed_shape_weights = None
        self.current_trajectory = []
        self.piece_history = deque()
        self.pending_reward_event = None
        self.short_games_move_count = 0
        self.left_held_time = 0
        self.right_held_time = 0
        self.fall_time = 0
        self.dragging_slider = False
        self.dragging_size_slider = False
        self.active_size_slider = None
        self.active_size_handle = None

    def _clamp_piece_size_range(self, min_size, max_size):
        min_size = int(max(1, min(5, min_size)))
        max_size = int(max(1, min(5, max_size)))
        if min_size > max_size:
            min_size, max_size = max_size, min_size
        return min_size, max_size

    def _select_size_handle(self, mouse_x, rect, min_size, max_size):
        min_size, max_size = self._clamp_piece_size_range(min_size, max_size)
        min_x = rect.x + ((min_size - 1) / 4.0) * rect.width
        max_x = rect.x + ((max_size - 1) / 4.0) * rect.width
        if min_size == max_size:
            return 'max' if mouse_x >= max_x else 'min'
        if abs(mouse_x - min_x) <= abs(mouse_x - max_x):
            return 'min'
        return 'max'

    def update_size_slider(self, mouse_x):
        if not self.active_size_slider:
            return

        rect = self.play_slider_rect if self.active_size_slider == "PLAY" else self.watch_slider_rect
        if not rect:
            return

        rel_x = mouse_x - rect.x
        val = max(0.0, min(1.0, rel_x / rect.width))
        size = 1 + int(val * 4.99)

        if self.active_size_slider == "PLAY":
            min_size = self.play_min_size
            max_size = self.play_max_size
        else:
            min_size = self.watch_min_size
            max_size = self.watch_max_size

        handle = self.active_size_handle
        if handle is None:
            handle = self._select_size_handle(mouse_x, rect, min_size, max_size)
            self.active_size_handle = handle

        if handle == 'min':
            min_size = size
        else:
            max_size = size

        min_size, max_size = self._clamp_piece_size_range(min_size, max_size)
        if self.active_size_slider == "PLAY":
            self.play_min_size = min_size
            self.play_max_size = max_size
        else:
            self.watch_min_size = min_size
            self.watch_max_size = max_size

    def _apply_piece_size_range(self, min_size, max_size):
        min_size = int(max(1, min(5, min_size)))
        max_size = int(max(1, min(5, max_size)))
        if min_size > max_size:
            min_size, max_size = max_size, min_size
        self.train_params['min_size'] = min_size
        self.train_params['max_size'] = max_size

    def _snap_epsilon_half_life(self, value):
        default_exp = 4.5
        try:
            value = float(value)
        except (TypeError, ValueError):
            value = 10 ** default_exp
        min_value = 10 ** self.EPSILON_HALF_LIFE_MIN_EXP
        max_value = 10 ** self.EPSILON_HALF_LIFE_MAX_EXP
        value = max(min_value, min(max_value, value))
        exp = math.log10(value)
        exp = max(self.EPSILON_HALF_LIFE_MIN_EXP, min(self.EPSILON_HALF_LIFE_MAX_EXP, exp))
        step_count = int(round((exp - self.EPSILON_HALF_LIFE_MIN_EXP) / self.EPSILON_HALF_LIFE_STEP_EXP))
        step_count = max(0, min(self.EPSILON_HALF_LIFE_STEPS, step_count))
        snapped_exp = self.EPSILON_HALF_LIFE_MIN_EXP + step_count * self.EPSILON_HALF_LIFE_STEP_EXP
        return int(round(10 ** snapped_exp))

    def _snap_epsilon_percent(self, value):
        value = max(0.0, min(self.EPSILON_MAX_PERCENT, float(value)))
        step_count = int(round(value / self.EPSILON_STEP_PERCENT))
        snapped = step_count * self.EPSILON_STEP_PERCENT
        return max(0.0, min(self.EPSILON_MAX_PERCENT, snapped))

    def _apply_epsilon_range(self, min_percent, current_percent, apply_current=True):
        min_percent = self._snap_epsilon_percent(min_percent)
        current_percent = self._snap_epsilon_percent(current_percent)
        if min_percent > current_percent:
            min_percent, current_percent = current_percent, min_percent
        self.train_params['epsilon_min_percent'] = min_percent
        self.train_params['epsilon_current_percent'] = current_percent
        if self.agent:
            self.agent.update_hyperparameters()
            if apply_current:
                self.agent.epsilon = max(current_percent / 100.0, self.agent.epsilon_min)

    def _select_epsilon_handle(self, mouse_x, rect):
        min_percent = self._snap_epsilon_percent(self.train_params.get('epsilon_min_percent', 5))
        current_percent = self._snap_epsilon_percent(self.train_params.get('epsilon_current_percent', 20))
        floor_percent = min(min_percent, current_percent)
        current_percent = max(min_percent, current_percent)
        floor_x = rect.x + (floor_percent / self.EPSILON_MAX_PERCENT) * rect.width
        current_x = rect.x + (current_percent / self.EPSILON_MAX_PERCENT) * rect.width
        if floor_percent == current_percent:
            return 'current' if mouse_x >= current_x else 'min'
        if abs(mouse_x - floor_x) <= abs(mouse_x - current_x):
            return 'min'
        return 'current'

    def _select_piece_size_handle(self, mouse_x, rect):
        min_size = int(self.train_params.get('min_size', 1))
        max_size = int(self.train_params.get('max_size', 5))
        min_size = max(1, min(5, min_size))
        max_size = max(1, min(5, max_size))
        floor_size = min(min_size, max_size)
        ceil_size = max(min_size, max_size)
        min_x = rect.x + ((floor_size - 1) / 4.0) * rect.width
        max_x = rect.x + ((ceil_size - 1) / 4.0) * rect.width
        if floor_size == ceil_size:
            return 'max' if mouse_x >= max_x else 'min'
        if abs(mouse_x - min_x) <= abs(mouse_x - max_x):
            return 'min'
        return 'max'

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
        elif self.active_slider == 'epsilon_range_percent':
            # Map 0-1 to 0-25 in 0.5% steps
            val_percent = self._snap_epsilon_percent(val * self.EPSILON_MAX_PERCENT)
            min_percent = self.train_params.get('epsilon_min_percent', 5)
            current_percent = self.train_params.get('epsilon_current_percent', 20)
            prev_current = current_percent
            handle = self.active_slider_handle
            if handle is None:
                handle = self._select_epsilon_handle(mouse_x, rect)
                self.active_slider_handle = handle
            if handle == 'min':
                min_percent = val_percent
            else:
                current_percent = val_percent
            self._apply_epsilon_range(min_percent, current_percent)
            if handle == 'current':
                new_current = self.train_params.get('epsilon_current_percent', prev_current)
                if new_current != prev_current:
                    self.epsilon_override_pending = True
        elif self.active_slider == 'epsilon_half_life_batches':
            step_count = int(val * self.EPSILON_HALF_LIFE_STEPS + 0.5)
            step_count = max(0, min(self.EPSILON_HALF_LIFE_STEPS, step_count))
            exp = self.EPSILON_HALF_LIFE_MIN_EXP + step_count * self.EPSILON_HALF_LIFE_STEP_EXP
            half_life = int(round(10 ** exp))
            self.train_params['epsilon_half_life_batches'] = half_life
            if self.agent:
                self.agent.update_hyperparameters()
        elif self.active_slider == 'learning_rate':
            # Map 0-1 to 0.0001 - 0.0050
            lr = 0.0001 + (val * 0.0049)
            self.train_params['learning_rate'] = round(lr, 5)
            if self.agent: self.agent.update_hyperparameters()
        elif self.active_slider == 'piece_size_range':
            # Map 0-1 to 1, 2, 3, 4, 5
            size = 1 + int(val * 4.99)
            min_size = int(self.train_params.get('min_size', 1))
            max_size = int(self.train_params.get('max_size', 5))
            handle = self.active_slider_handle
            if handle is None:
                handle = self._select_piece_size_handle(mouse_x, rect)
                self.active_slider_handle = handle
            if handle == 'min':
                min_size = size
            else:
                max_size = size
            self._apply_piece_size_range(min_size, max_size)
        elif self.active_slider == 'big_piece_weight':
            # Map 0-1 to 1, 2, 3, 4
            weight = 1 + int(val * 3.99)
            self.train_params['big_piece_weight'] = max(1, min(4, weight))
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

        if self.is_recording:
            self.stop_recording(prompt_name=False)
        self.audio.cleanup()
        pygame.quit()
