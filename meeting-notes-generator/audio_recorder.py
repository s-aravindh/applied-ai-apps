import pyaudio
import wave
import threading
import os
import time
from datetime import datetime

# Global variables
frames = []
recording = False
audio = None
stream = None
current_file = None

def start_recording():
    global recording, audio, stream, frames, current_file
    
    frames = []
    recording = True
    audio = pyaudio.PyAudio()
    
    # Audio settings
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    CHUNK = 1024
    
    # Create initial file
    os.makedirs("recordings", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    current_file = f"recordings/recording_{timestamp}.wav"
    
    stream = audio.open(format=FORMAT,
                       channels=CHANNELS,
                       rate=RATE,
                       input=True,
                       frames_per_buffer=CHUNK,
                       stream_callback=None)
    
    # Start recording thread
    threading.Thread(target=record_audio).start()

def save_chunks(frames_to_save):
    global current_file
    
    with wave.open(current_file, 'wb') as wf:
        if wf.tell() == 0:  # New file, write headers
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
        wf.writeframes(b''.join(frames_to_save))

def record_audio():
    global recording, stream, frames
    
    last_save = time.time()
    temp_frames = []
    
    while recording:
        try:
            data = stream.read(1024, exception_on_overflow=False)
            temp_frames.append(data)
            frames.append(data)
            
            current_time = time.time()
            if current_time - last_save >= 10:
                save_chunks(temp_frames)
                temp_frames = []
                last_save = current_time
        except OSError as e:
            print(f"Warning: {e}")
            time.sleep(0.1)  # Add small delay to help prevent overflow
            continue

def stop_recording():
    global recording, audio, stream, frames, current_file
    
    recording = False
    
    if stream:
        stream.stop_stream()
        stream.close()
    
    if audio:
        audio.terminate()
    
    # Save any remaining frames
    if frames:
        save_chunks(frames)
    
    return current_file
