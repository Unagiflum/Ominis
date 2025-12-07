import pygame
from game import Game

if __name__ == "__main__":
    pygame.init()
    pygame.mixer.init()
    pygame.font.init()
    
    game = Game()
    try:
        game.run()
    except KeyboardInterrupt:
        # Graceful shutdown if user interrupts from console
        try:
            game.audio.cleanup()
        except Exception:
            pass
        pygame.quit()
    except Exception:
        import traceback
        traceback.print_exc()
        try:
            game.audio.cleanup()
        except Exception:
            pass
        pygame.quit()
