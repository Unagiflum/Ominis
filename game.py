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
        self.grid_width = 12 # Pentominoes are larger, maybe need wider board? Standard Tetris is 10. Let's go 12.
        self.grid_height = 24 # Taller board
        self.cell_size = 30
        
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Pentomino Tetris")
        
        self.clock = pygame.time.Clock()
        self.grid = Grid(self.grid_width, self.grid_height, self.cell_size)
        self.input_manager = InputManager()
        self.ui = UI(self.screen)
        self.audio = AudioPlayer("assets/music")
        
        self.state = "MENU" # MENU, PLAYING, GAMEOVER
        self.score = 0
        self.level = 1
        self.lines_cleared_total = 0
        
        self.current_piece = None
        self.next_piece = None
        
        self.fall_time = 0
        self.fall_speed = 1000 # ms
        self.fast_drop = False

    def reset(self):
        self.grid = Grid(self.grid_width, self.grid_height, self.cell_size)
        self.score = 0
        self.level = 1
        self.lines_cleared_total = 0
        self.fall_speed = 1000
        self.current_piece = Pentomino(self.grid_width // 2, 0)
        self.next_piece = Pentomino(self.grid_width // 2, 0)
        self.state = "PLAYING"

    def update_score(self, lines):
        if lines > 0:
            # Pentomino scoring: more lines = much more points
            points = [0, 100, 300, 500, 800, 1200] # Up to 5 lines possible with pentominoes
            self.score += points[min(lines, 5)] * self.level
            self.lines_cleared_total += lines
            
            if self.lines_cleared_total >= self.level * 5: # Level up every 5 lines
                self.level += 1
                self.fall_speed = max(100, self.fall_speed - 50)

    def handle_input(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            
            action = self.input_manager.get_action(event)
            
            if self.state == "MENU":
                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    self.reset()
            
            elif self.state == "GAMEOVER":
                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    self.reset()
            
            elif self.state == "PLAYING":
                if action == "LEFT":
                    if not self.grid.check_collision(self.current_piece, offset_x=-1):
                        self.current_piece.x -= 1
                elif action == "RIGHT":
                    if not self.grid.check_collision(self.current_piece, offset_x=1):
                        self.current_piece.x += 1
                elif action == "DOWN":
                    if not self.grid.check_collision(self.current_piece, offset_y=1):
                        self.current_piece.y += 1
                        self.score += 1 # Soft drop points
                elif action == "ROTATE_CW" or action == "ROTATE_CW_ALT":
                    self.current_piece.rotate_right()
                    if self.grid.check_collision(self.current_piece):
                        self.current_piece.rotate_left() # Revert if invalid
                elif action == "ROTATE_CCW":
                    self.current_piece.rotate_left()
                    if self.grid.check_collision(self.current_piece):
                        self.current_piece.rotate_right() # Revert if invalid

        return True

    def update(self):
        self.audio.update()
        
        if self.state == "PLAYING":
            current_time = pygame.time.get_ticks()
            if current_time - self.fall_time > self.fall_speed:
                if not self.grid.check_collision(self.current_piece, offset_y=1):
                    self.current_piece.y += 1
                else:
                    self.grid.lock_shape(self.current_piece)
                    lines = self.grid.clear_lines()
                    self.update_score(lines)
                    
                    self.current_piece = self.next_piece
                    self.next_piece = Pentomino(self.grid_width // 2, 0)
                    
                    if self.grid.check_collision(self.current_piece):
                        self.state = "GAMEOVER"
                
                self.fall_time = current_time

    def draw(self):
        self.screen.fill(self.ui.bg_color)
        
        if self.state == "MENU":
            title = self.ui.large_font.render("PENTOMINO TETRIS", True, self.ui.text_color)
            start = self.ui.font.render("Press SPACE to Start", True, self.ui.text_color)
            self.screen.blit(title, (self.screen_width // 2 - title.get_width() // 2, 200))
            self.screen.blit(start, (self.screen_width // 2 - start.get_width() // 2, 400))
            
        elif self.state == "PLAYING" or self.state == "GAMEOVER":
            offset_x = (self.screen_width - self.grid_width * self.cell_size) // 2
            offset_y = 50
            
            self.ui.draw_grid(self.grid, offset_x, offset_y)
            self.ui.draw_pentomino(self.current_piece, offset_x, offset_y, self.cell_size)
            self.ui.draw_preview(self.next_piece, offset_x + self.grid_width * self.cell_size + 20, offset_y, self.cell_size)
            self.ui.draw_score(self.score, self.level, self.lines_cleared_total, 50, offset_y)
            
            if self.state == "GAMEOVER":
                self.ui.draw_game_over(self.screen_width, self.screen_height)
        
        pygame.display.flip()

    def run(self):
        running = True
        while running:
            running = self.handle_input()
            self.update()
            self.draw()
            self.clock.tick(60)
        
        pygame.quit()
