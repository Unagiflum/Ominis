import queue
import threading
from fractions import Fraction

try:
    import av
except ImportError:
    av = None

try:
    import numpy as np
except ImportError:
    np = None

try:
    import pyaudiowpatch as pyaudio
    HAS_PYAUDIO = True
except ImportError:
    pyaudio = None
    HAS_PYAUDIO = False


class RecorderError(Exception):
    pass


class VideoRecorder:
    def __init__(self, width, height, fps, want_audio=True, audio_device=None, audio_device_name=None, audio_loopback=True):
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.want_audio = bool(want_audio)
        self.audio_device = audio_device
        self.audio_device_name = audio_device_name
        self.audio_loopback = bool(audio_loopback)

        self.container = None
        self.video_stream = None
        self.video_codec = None
        self.audio_stream = None
        self.audio_stream_in = None

        self.video_time_base = None
        self.audio_time_base = None
        self.video_frame_index = 0
        self.audio_samples = 0

        self.sample_rate = None
        self.channels = None
        self.audio_layout = None
        self.audio_enabled = False
        self.audio_error = None
        self.audio_device_label = None
        self.audio_loopback_active = False

        self.running = False
        self.audio_queue = queue.Queue(maxsize=256)

    def start(self, path):
        if av is None or np is None:
            raise RecorderError("Recording requires 'av' and 'numpy'.")

        self._open_video_stream(path)
        self.video_time_base = Fraction(1, self.fps)

        if self.want_audio:
            self._start_audio_capture()

        self.running = True

    def _open_video_stream(self, path):
        codec_candidates = [
            "libx264",
            "libopenh264",
            "h264_nvenc",
            "h264_qsv",
            "h264_amf",
            "mpeg4",
        ]
        last_error = None

        for codec in codec_candidates:
            container = None
            try:
                container = av.open(path, mode="w")
                stream = container.add_stream(codec, rate=self.fps)
                stream.width = self.width
                stream.height = self.height
                stream.pix_fmt = "yuv420p"

                if codec.startswith("h264") or codec.startswith("libx264") or codec.startswith("libopenh264"):
                    stream.options = {"preset": "veryfast", "crf": "23"}
                else:
                    stream.bit_rate = 4_000_000

                stream.codec_context.open()

                self.container = container
                self.video_stream = stream
                self.video_codec = codec
                return
            except Exception as exc:
                last_error = exc
                if container:
                    try:
                        container.close()
                    except Exception:
                        pass

        raise RecorderError(f"Video encoder unavailable: {last_error}")

    def _find_wasapi_device(self):
        """Find the default WASAPI loopback device."""
        if not HAS_PYAUDIO or pyaudio is None:
            return None, "PyAudioWPatch not available"

        try:
            p = pyaudio.PyAudio()
        except Exception as exc:
            return None, f"PyAudio init failed: {exc}"

        # Get WASAPI Host API index
        try:
            wasapi_info = p.get_host_api_info_by_type(pyaudio.paWASAPI)
            wasapi_index = wasapi_info["index"]
        except OSError as exc:
            p.terminate()
            return None, f"WASAPI not available: {exc}"

        # Find the loopback device
        # We look for a device that is on the WASAPI host API and has isLoopbackDevice=True
        found_device_index = None
        found_device_info = None

        # First try: see if we can match the default output device's name with a loopback counterpart
        default_output_index = wasapi_info.get("defaultOutputDevice")
        default_name = ""
        if default_output_index is not None and default_output_index >= 0:
             try:
                 default_info = p.get_device_info_by_index(default_output_index)
                 default_name = default_info.get("name", "")
             except:
                 pass
        
        # Iterate all devices to find the loopback match
        for i in range(p.get_device_count()):
            try:
                dev = p.get_device_info_by_index(i)
            except:
                continue
            
            if dev.get("hostApi") != wasapi_index:
                continue
            
            if dev.get("isLoopbackDevice", False) is True:
                # We found a loopback device.
                # If we have a default name, try to match it (e.g. "Speakers" vs "Speakers [Loopback]")
                if default_name and default_name in dev.get("name", ""):
                    found_device_index = i
                    found_device_info = dev
                    break
                
                # Otherwise just take the first one we find as a fallback
                if found_device_index is None:
                    found_device_index = i
                    found_device_info = dev
        
        if found_device_index is None:
             p.terminate()
             return None, "No WASAPI loopback device found"

        result = {
            "pyaudio": p,
            "device_index": found_device_index,
            "device_name": found_device_info.get("name", str(found_device_index)),
            "sample_rate": int(found_device_info.get("defaultSampleRate", 48000)),
            "channels": int(found_device_info.get("maxInputChannels", 2)) or 2, # Loopback is an input
        }
        result["channels"] = 2 if result["channels"] >= 2 else 1
        return result, None

    def _start_audio_capture(self):
        if not HAS_PYAUDIO:
            self.audio_error = "PyAudioWPatch not available"
            return

        if not self.audio_loopback:
            self.audio_error = "Non-loopback audio not supported (no input device)"
            return

        device_info, err = self._find_wasapi_device()
        if device_info is None:
            self.audio_error = err
            return

        self._pyaudio = device_info["pyaudio"]
        self.sample_rate = device_info["sample_rate"]
        self.channels = device_info["channels"]
        self.audio_time_base = Fraction(1, self.sample_rate)
        self.audio_layout = "stereo" if self.channels == 2 else "mono"
        self.audio_device_label = device_info["device_name"]
        self.audio_loopback_active = True

        try:
            self.audio_stream = self.container.add_stream("aac", rate=self.sample_rate)
            try:
                self.audio_stream.layout = self.audio_layout
            except Exception:
                pass

            # Open standard input stream since we selected a loopback device index
            self.audio_stream_in = self._pyaudio.open(
                format=pyaudio.paFloat32,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=device_info["device_index"],
                frames_per_buffer=1024,
                stream_callback=self._on_audio_pyaudio,
            )
            self.audio_stream_in.start_stream()
            self.audio_enabled = True
        except Exception as exc:
            self.audio_error = f"Audio capture failed: {exc}"
            self.audio_enabled = False
            if hasattr(self, '_pyaudio') and self._pyaudio:
                try:
                    self._pyaudio.terminate()
                except Exception:
                    pass
                self._pyaudio = None

    def _on_audio_pyaudio(self, in_data, frame_count, time_info, status):
        if not self.running:
            return (None, pyaudio.paComplete)
        if in_data:
            try:
                audio_data = np.frombuffer(in_data, dtype=np.float32)
                
                # Automatic Gain Control (AGC)
                # Track peak amplitude with decay to adjust gain dynamically
                chunk_max = np.max(np.abs(audio_data))
                
                # Initialize peak tracker if needed
                if not hasattr(self, '_agc_peak'):
                    self._agc_peak = 0.01

                # Update peak envelope: fast attack (instant), slow decay
                if chunk_max > self._agc_peak:
                    self._agc_peak = chunk_max
                else:
                    self._agc_peak *= 0.995 # Decay factor
                
                # Minimum floor to avoid amplifying noise too much during silence
                # 0.001 represents -60dB
                effective_peak = max(self._agc_peak, 0.001)
                
                # Calculate target gain (aim for ~70% amplitude)
                target_gain = 0.7 / effective_peak
                
                # Cap max gain to avoid insane noise boost (e.g. 500x)
                target_gain = min(target_gain, 100.0)
                
                # Apply gain
                audio_data = audio_data * target_gain
                
                # Hard clip to prevent distortion if sudden spike occurs
                np.clip(audio_data, -1.0, 1.0, out=audio_data)

                audio_data = audio_data.reshape(-1, self.channels)
                self.audio_queue.put_nowait(audio_data.copy())
            except queue.Full:
                try:
                    self.audio_queue.get_nowait()
                    self.audio_queue.put_nowait(audio_data.copy())
                except (queue.Empty, Exception):
                    pass
            except Exception:
                pass
        return (None, pyaudio.paContinue)

    def _encode_audio(self):
        if not self.audio_enabled or self.audio_stream is None:
            return
        while True:
            try:
                data = self.audio_queue.get_nowait()
            except queue.Empty:
                break
            if data is None:
                continue
            if data.ndim == 1:
                data = data.reshape(-1, 1)
            if self.channels == 1 and data.shape[1] > 1:
                data = data[:, :1]
            elif self.channels == 2 and data.shape[1] == 1:
                data = np.repeat(data, 2, axis=1)

            audio = np.ascontiguousarray(data.T)
            frame = av.AudioFrame.from_ndarray(audio, format="fltp", layout=self.audio_layout)
            frame.sample_rate = self.sample_rate
            frame.time_base = self.audio_time_base
            frame.pts = self.audio_samples
            self.audio_samples += frame.samples
            for packet in self.audio_stream.encode(frame):
                self.container.mux(packet)

    def write_frame(self, frame_rgb):
        if not self.running or self.container is None:
            return
        if frame_rgb is None:
            return

        frame_rgb = np.asarray(frame_rgb)
        if frame_rgb.shape[0] == self.width and frame_rgb.shape[1] == self.height:
            frame_rgb = np.transpose(frame_rgb, (1, 0, 2))
        if frame_rgb.shape[0] != self.height or frame_rgb.shape[1] != self.width:
            raise RecorderError("Frame size mismatch.")

        frame = av.VideoFrame.from_ndarray(frame_rgb, format="rgb24")
        frame.pts = self.video_frame_index
        frame.time_base = self.video_time_base
        self.video_frame_index += 1
        for packet in self.video_stream.encode(frame):
            self.container.mux(packet)

        self._encode_audio()

    def stop(self):
        if not self.running:
            return
        self.running = False

        if self.audio_stream_in:
            try:
                self.audio_stream_in.stop_stream()
            except Exception:
                pass
            try:
                self.audio_stream_in.close()
            except Exception:
                pass
            self.audio_stream_in = None

        if hasattr(self, '_pyaudio') and self._pyaudio:
            try:
                self._pyaudio.terminate()
            except Exception:
                pass
            self._pyaudio = None

        self._encode_audio()

        if self.audio_stream:
            for packet in self.audio_stream.encode(None):
                self.container.mux(packet)
        if self.video_stream:
            for packet in self.video_stream.encode(None):
                self.container.mux(packet)
        if self.container:
            self.container.close()
            self.container = None
