import pygame
import os
import random

class AudioPlayer:
    def __init__(self, music_dir, sound_dir):
        self.music_dir = music_dir
        self.sound_dir = sound_dir
        self.playlist = []
        self.current_track = 0
        self.scan_music()

    def scan_music(self):
        if os.path.exists(self.music_dir):
            # Look for mp3s
            self.playlist = [os.path.join(self.music_dir, f) for f in os.listdir(self.music_dir) if f.endswith('.mp3')]
            self.playlist.sort() # Keep master list sorted
        
            self.playlist.sort() # Keep master list sorted
        
        self.current_index = 0
        self.direction = 1 # 1 for forward, -1 for backward
        self.sequence_initialized = False
        
        # Load sound effects
        self.clear_sound = None
        if os.path.exists(self.sound_dir):
            sound_path = os.path.join(self.sound_dir, "clear.wav")
            if os.path.exists(sound_path):
                self.clear_sound = pygame.mixer.Sound(sound_path)
        
        self.enabled = False

    def start(self):
        self.enabled = True
        if not self.sequence_initialized:
            self.reset_sequence()
            
        if not pygame.mixer.music.get_busy():
            self.play_next()

    def stop(self):
        self.enabled = False
        pygame.mixer.music.stop()

    def play_clear(self):
        if self.clear_sound:
            self.clear_sound.play()
        
    def reset_sequence(self):
        if not self.playlist:
            return
            
        # Random start index
        self.current_index = random.randint(0, len(self.playlist) - 1)
        # Random direction
        self.direction = random.choice([1, -1])
        self.sequence_initialized = True
        print(f"Music Sequence: Start Index {self.current_index}, Direction {self.direction}")

    def play_next(self):
        if not self.playlist:
            return
        
        track = self.playlist[self.current_index]
        
        # Update index for next time
        self.current_index = (self.current_index + self.direction) % len(self.playlist)
        
        try:
            pygame.mixer.music.load(track)
            # Volume normalization is complex without external libs like pydub/librosa.
            # We'll stick to default volume for now.
            pygame.mixer.music.play()
            print(f"Playing: {track}")
        except Exception as e:
            print(f"Error playing music: {e}")

    def update(self):
        if self.enabled and self.playlist and not pygame.mixer.music.get_busy():
            self.play_next()

    def pause(self):
        pygame.mixer.music.pause()

    def unpause(self):
        pygame.mixer.music.unpause()
