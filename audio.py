import pygame
import os
import random

class AudioPlayer:
    def __init__(self, music_dir):
        self.music_dir = music_dir
        self.playlist = []
        self.current_track = 0
        self.scan_music()

    def scan_music(self):
        if os.path.exists(self.music_dir):
            self.playlist = [os.path.join(self.music_dir, f) for f in os.listdir(self.music_dir) if f.endswith('.mid')]
        
    def play_next(self):
        if not self.playlist:
            return
        
        track = random.choice(self.playlist)
        try:
            pygame.mixer.music.load(track)
            pygame.mixer.music.play()
            print(f"Playing: {track}")
        except Exception as e:
            print(f"Error playing music: {e}")

    def update(self):
        if self.playlist and not pygame.mixer.music.get_busy():
            self.play_next()
