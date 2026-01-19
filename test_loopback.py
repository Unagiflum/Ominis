import pyaudiowpatch as pa

p = pa.PyAudio()

print("Looking for loopback devices...\n")

for i in range(p.get_device_count()):
    dev = p.get_device_info_by_index(i)
    name = dev.get("name", "Unknown")
    is_loopback = dev.get("isLoopbackDevice", False)
    max_in = dev.get("maxInputChannels", 0)
    max_out = dev.get("maxOutputChannels", 0)
    hostapi = dev.get("hostApi", -1)
    
    if is_loopback or max_in > 0:  # Show input devices and loopback
        print(f"Device {i}: {name}")
        print(f"  isLoopbackDevice: {is_loopback}")
        print(f"  maxInputChannels: {max_in}")
        print(f"  maxOutputChannels: {max_out}")
        print(f"  hostApi: {hostapi}")
        print()

p.terminate()
