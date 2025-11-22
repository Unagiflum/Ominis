import pygame

class InputManager:
    def __init__(self):
        self.key_map = {
            'LEFT': pygame.K_LEFT,
            'RIGHT': pygame.K_RIGHT,
            'DOWN': pygame.K_DOWN,
            'ROTATE_CW': pygame.K_UP,
            'ROTATE_CCW': pygame.K_PERIOD, # Swapped
            'ROTATE_CW_ALT': pygame.K_COMMA # Swapped
        }

    def get_action(self, event):
        if event.type == pygame.KEYDOWN:
            for action, key in self.key_map.items():
                if event.key == key:
                    return action
        return None

    def is_down_held(self):
        keys = pygame.key.get_pressed()
        return keys[pygame.K_DOWN]
