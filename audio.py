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
        
        self.play_queue = []
        self.last_track = None
        
        # Load sound effects
        self.clear_sound = None
        if os.path.exists(self.sound_dir):
            sound_path = os.path.join(self.sound_dir, "clear.wav")
            if os.path.exists(sound_path):
                self.clear_sound = pygame.mixer.Sound(sound_path)
        
        self.enabled = False

    def start(self):
        self.enabled = True
        if not pygame.mixer.music.get_busy():
            self.play_next()

    def stop(self):
        self.enabled = False
        pygame.mixer.music.stop()

    def play_clear(self):
        if self.clear_sound:
            self.clear_sound.play()
        
    def play_next(self):
        if not self.playlist:
            return
        
        # Refill queue if empty
        if not self.play_queue:
            self.play_queue = self.playlist[:]
            random.shuffle(self.play_queue)
            
            # Avoid repeating the last track immediately if possible
            if self.last_track and len(self.play_queue) > 1 and self.play_queue[0] == self.last_track:
                self.play_queue[0], self.play_queue[-1] = self.play_queue[-1], self.play_queue[0]
        
        track = self.play_queue.pop(0)
        self.last_track = track
        
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
