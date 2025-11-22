import pygame
from grid import Grid
from tetrominoes import Pentomino
from input_manager import InputManager
from ui import UI
from audio import AudioPlayer

class Game:
    def __init__(self):
        self.screen_width = 800
        self.screen_height = 800
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
        
        # Animation state
        self.clearing_lines = []
        self.animation_timer = 0
        self.row_offsets = [0] * self.grid_height

        # Input handling state (DAS - Delayed Auto Shift)
        self.das_delay = 200 # ms before repeat starts
        self.das_repeat = 50 # ms between repeats
        self.left_held_time = 0
        self.right_held_time = 0
        self.last_move_time = 0

    def spawn_piece(self):
        p = Pentomino(self.grid_width // 2, 0)
        # Adjust y to ensure the shape starts just above the board
        min_y = min(y for x, y in p.shape)
        p.y = -min_y - 2
        return p

    def reset(self):
        self.grid = Grid(self.grid_width, self.grid_height, self.cell_size)
        self.score = 0
        self.level = 1
        self.lines_cleared_total = 0
        self.fall_speed = 1000
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
                self.fall_speed = max(100, self.fall_speed - 50)

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
                if self.state == "PLAYING" or self.state == "GAMEOVER":
                    self.state = "MENU"
                    self.audio.stop()
            
            if self.state == "MENU" or self.state == "GAMEOVER":
                if event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                    self.reset()
            
            # Pause Toggle
            if event.type == pygame.KEYDOWN and event.key == pygame.K_F1:
                if self.state == "PLAYING":
                    self.state = "PAUSED"
                    self.audio.pause()
                elif self.state == "PAUSED":
                    self.state = "PLAYING"
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

    def update(self):
        current_time = pygame.time.get_ticks()
        
        if self.state == "PAUSED":
            return

        self.audio.update()

        if self.state == "PLAYING":
            # Determine fall speed
            current_fall_speed = self.fall_speed
            if self.input_manager.is_down_held():
                current_fall_speed = 80 # Slower fast drop (was 50)
            
            if current_time - self.fall_time > current_fall_speed:
                if not self.grid.check_collision(self.current_piece, offset_y=1):
                    self.current_piece.y += 1
                    if self.input_manager.is_down_held():
                        self.score += 1
                else:
                    self.grid.lock_shape(self.current_piece)
                    
                    if self.check_and_clear_lines():
                        pass # State changed to ANIMATING_CLEAR
                    else:
                        self.current_piece = self.next_piece
                        self.next_piece = self.spawn_piece()
                        # Revert spawn adjustment: check collision immediately?
                        # If we spawn at y=-2, we might not collide yet.
                        # But if we can't move down, it's game over.
                        if self.grid.check_collision(self.current_piece):
                             self.state = "GAMEOVER"
                             self.audio.stop()
                
                self.fall_time = current_time
                
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
                self.state = "PLAYING"
                self.current_piece = self.next_piece
                self.next_piece = Pentomino(self.grid_width // 2, 0)
                if self.grid.check_collision(self.current_piece):
                    self.state = "GAMEOVER"

    def draw(self):
        self.screen.fill(self.ui.bg_color)
        
        if self.state == "MENU":
            title = self.ui.large_font.render("OMINIS", True, self.ui.text_color)
            start = self.ui.font.render("Press ENTER to Start", True, self.ui.text_color)
            self.screen.blit(title, (self.screen_width // 2 - title.get_width() // 2, 200))
            self.screen.blit(start, (self.screen_width // 2 - start.get_width() // 2, 400))
            
        else:
            offset_x = (self.screen_width - self.grid_width * self.cell_size) // 2
            offset_y = 50
            
            # Pass offsets and flash lines if animating
            offsets = self.row_offsets if self.state == "ANIMATING_DROP" else None
            flash = self.clearing_lines if self.state == "ANIMATING_CLEAR" else None
            
            self.ui.draw_grid(self.grid, offset_x, offset_y, offsets, flash)
            
            if self.state == "PLAYING":
                # Draw Ghost Piece
                ghost = self.get_ghost_piece()
                if ghost and ghost.y != self.current_piece.y:
                    self.ui.draw_ghost_pentomino(ghost, offset_x, offset_y, self.cell_size)
                
                self.ui.draw_pentomino(self.current_piece, offset_x, offset_y, self.cell_size)
            
            self.ui.draw_preview(self.next_piece, 585, offset_y - 5, self.cell_size)
            self.ui.draw_score(self.score, self.level, self.lines_cleared_total, 5, offset_y - 5)
            self.ui.draw_instructions(585, offset_y + 240)
            
            if self.state == "GAMEOVER":
                self.ui.draw_game_over(self.screen_width, self.screen_height)
                
            if self.state == "PAUSED":
                self.ui.draw_pause_screen(self.screen_width, self.screen_height)
        
        pygame.display.flip()

    def run(self):
        running = True
        while running:
            running = self.handle_input()
            self.update()
            self.draw()
            self.clock.tick(60)
        
        pygame.quit()
