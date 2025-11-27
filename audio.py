import pygame
import pygame.midi
import os
import random
import threading
import time
import mido

class MidiThread(threading.Thread):
    def __init__(self, filename, midi_out):
        super().__init__()
        self.filename = filename
        self.midi_out = midi_out
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()
        self.pause_event.set() # Start unpaused
        self.speed_factor = 1.0
        self.volume = 0.7
        self.daemon = True

    def run(self):
        try:
            mid = mido.MidiFile(self.filename)
            
            # Send initial volume
            self.update_volume()
            # Ensure synth is in a known state (GM reset)
            try:
                self.midi_out.write_sys_ex(pygame.midi.time(), bytes([0xF0, 0x7E, 0x7F, 0x09, 0x01, 0xF7]))
            except Exception:
                pass

            start_time = time.time()
            input_time = 0.0
            
            for msg in mid:
                if self.stop_event.is_set():
                    break
                
                # Handle Pausing
                if not self.pause_event.is_set():
                    self.all_notes_off()
                    self.pause_event.wait() # Block until set
                    # Reset timing to avoid fast-forwarding
                    start_time = time.time() - input_time
                
                # Calculate wait time based on speed
                input_time += msg.time
                
                if msg.time > 0:
                    wait_time = msg.time / self.speed_factor
                    time.sleep(wait_time)
                
                if not msg.is_meta:
                    # Forward SysEx/bank messages so custom patches are preserved
                    if msg.type == "sysex":
                        try:
                            self.midi_out.write_sys_ex(pygame.midi.time(), msg.bin())
                        except Exception:
                            pass
                    else:
                        b = msg.bytes()
                        if len(b) == 3:
                            self.midi_out.write_short(b[0], b[1], b[2])
                        elif len(b) == 2:
                            self.midi_out.write_short(b[0], b[1])
                        
            # Do NOT close midi_out here, it's shared
            self.all_notes_off()
            
        except Exception as e:
            print(f"Error in MIDI thread: {e}")
            self.all_notes_off()

    def set_speed(self, speed):
        self.speed_factor = max(0.1, speed)

    def set_volume(self, volume):
        self.volume = volume
        # Try to update live if possible
        try:
            self.update_volume()
        except:
            pass
        
    def update_volume(self):
        # Send Channel Volume (CC 7) to all 16 channels
        # Scale by 1.5 so 0.7 input -> ~1.0 output
        scaled_vol = min(1.0, self.volume * 1.5)
        val = int(scaled_vol * 127)
        for i in range(16):
            self.midi_out.write_short(0xB0 + i, 7, val)

    def all_notes_off(self):
        # Send All Notes Off (CC 123) to all channels
        for i in range(16):
            self.midi_out.write_short(0xB0 + i, 123, 0)
            
    def stop(self):
        self.stop_event.set()
        self.pause_event.set()

    def pause(self):
        self.pause_event.clear()

    def unpause(self):
        self.pause_event.set()

class AudioPlayer:
    def __init__(self, music_dir, sound_dir):
        pygame.midi.init()
        self.music_dir = music_dir
        self.sound_dir = sound_dir
        self.playlist = []
        self.current_index = 0
        self.direction = 1
        self.sequence_initialized = False
        
        # Initialize MIDI Output ONCE
        self.midi_out = None
        try:
            out_id = pygame.midi.get_default_output_id()
            if out_id != -1:
                self.midi_out = pygame.midi.Output(out_id)
            else:
                print("No MIDI output device found.")
        except Exception as e:
            print(f"Failed to initialize MIDI output: {e}")
        
        self.scan_music()
        
        # Load sound effects
        self.clear_sound = None
        if os.path.exists(self.sound_dir):
            sound_path = os.path.join(self.sound_dir, "clear.wav")
            if os.path.exists(sound_path):
                self.clear_sound = pygame.mixer.Sound(sound_path)
        
        self.current_thread = None
        self.volume = 0.5
        self.speed = 1.0
        self.enabled = False

    def scan_music(self):
        if os.path.exists(self.music_dir):
            self.playlist = [os.path.join(self.music_dir, f) for f in os.listdir(self.music_dir) if f.endswith('.mid')]
            self.playlist.sort()

    def reset_sequence(self):
        if not self.playlist:
            return
        self.current_index = random.randint(0, len(self.playlist) - 1)
        self.direction = random.choice([1, -1])
        self.sequence_initialized = True
        
    def start(self):
        self.enabled = True
        if not self.sequence_initialized:
            self.reset_sequence()
        
        if not self.current_thread or not self.current_thread.is_alive():
            self.play_next()

    def stop(self):
        self.enabled = False
        if self.current_thread:
            self.current_thread.stop()
            self.current_thread.join()
            self.current_thread = None
            
    def pause(self):
        if self.current_thread:
            self.current_thread.pause()

    def unpause(self):
        if self.current_thread:
            self.current_thread.unpause()

    def play_clear(self):
        if self.clear_sound:
            # Boost volume for sound effect (3x master volume, max 1.0)
            boosted_vol = min(1.0, self.volume * 3.0)
            self.clear_sound.set_volume(boosted_vol)
            self.clear_sound.play()
        
    def play_next(self):
        if not self.playlist or not self.midi_out:
            return
            
        if self.current_thread:
            self.current_thread.stop()
            self.current_thread.join()
        
        track = self.playlist[self.current_index]
        self.current_index = (self.current_index + self.direction) % len(self.playlist)
        
        self.current_thread = MidiThread(track, self.midi_out)
        self.current_thread.set_speed(self.speed)
        self.current_thread.set_volume(self.volume)
        self.current_thread.start()

    def update(self):
        if self.enabled and self.playlist:
            if self.current_thread and not self.current_thread.is_alive():
                self.play_next()
            elif self.current_thread is None:
                self.play_next()

    def set_speed(self, speed):
        self.speed = speed
        if self.current_thread:
            self.current_thread.set_speed(speed)

    def set_volume(self, volume):
        self.volume = volume
        if self.current_thread:
            self.current_thread.set_volume(volume)
            
    def cleanup(self):
        self.stop()
        if self.midi_out:
            self.midi_out.close()
            self.midi_out = None
        pygame.midi.quit()
