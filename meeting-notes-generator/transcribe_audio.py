import os
import wave
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration

def transcribe_audio(audio_path):
    # Initialize processor and model
    processor = WhisperProcessor.from_pretrained("openai/whisper-base")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")
    
    # Load audio using wave
    with wave.open(audio_path, 'rb') as wf:
        # Get audio properties
        frames = wf.getnframes()
        rate = wf.getframerate()
        
        # Read raw audio data
        raw_data = wf.readframes(frames)
        
    # Convert to numpy array and normalize
    audio = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0
    
    # Resample to 16kHz if needed
    if rate != 16000:
        audio = np.interp(
            np.linspace(0, len(audio), int(len(audio) * 16000 / rate)),
            np.arange(len(audio)),
            audio
        )
    
    # Process audio
    input_features = processor(audio, sampling_rate=16000, return_tensors="pt").input_features
    
    # Generate transcription
    predicted_ids = model.generate(input_features)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    
    # Save transcription to file
    output_path = os.path.splitext(audio_path)[0] + '.txt'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(transcription)
    
    return transcription

if __name__ == "__main__":
    # Example usage
    audio_file = "/Users/aravindh/Documents/GitHub/applied-ai-apps/recordings/recording_20250305_215054.wav"
    text = transcribe_audio(audio_file)
    print("Transcription saved to:", os.path.splitext(audio_file)[0] + '.txt')
