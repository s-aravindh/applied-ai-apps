import pyaudio
import wave
import time
import os
import logging
from datetime import datetime
from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class AudioConfig:
    chunk_size: int = 1024
    format: int = pyaudio.paInt16
    channels: int = 1
    rate: int = 44100
    chunk_duration: int = 10
    output_dir: str = "recordings"

class AudioRecorder:
    """Handle audio recording with proper resource management and error handling."""
    
    def __init__(self, config: Optional[AudioConfig] = None):
        """Initialize the audio recorder with given or default configuration."""
        self.config = config or AudioConfig()
        self.setup_logging()
        self._setup_output_directory()
        self.pyaudio_instance = None
        self.stream = None
        self.wave_file = None
        
    def setup_logging(self) -> None:
        """Configure logging for the audio recorder."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def _setup_output_directory(self) -> None:
        """Ensure output directory exists."""
        os.makedirs(self.config.output_dir, exist_ok=True)

    def _initialize_recording(self) -> None:
        """Initialize PyAudio and open stream."""
        try:
            self.pyaudio_instance = pyaudio.PyAudio()
            self.stream = self.pyaudio_instance.open(
                format=self.config.format,
                channels=self.config.channels,
                rate=self.config.rate,
                input=True,
                frames_per_buffer=self.config.chunk_size
            )
        except Exception as e:
            self.logger.error(f"Failed to initialize audio: {str(e)}")
            self.cleanup()
            raise

    def _get_output_filename(self) -> str:
        """Generate unique filename for the recording."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return os.path.join(self.config.output_dir, f"recording_{timestamp}.wav")

    def _initialize_wave_file(self, filename: str) -> None:
        """Initialize wave file for writing."""
        try:
            self.wave_file = wave.open(filename, 'wb')
            self.wave_file.setnchannels(self.config.channels)
            self.wave_file.setsampwidth(self.pyaudio_instance.get_sample_size(self.config.format))
            self.wave_file.setframerate(self.config.rate)
        except Exception as e:
            self.logger.error(f"Failed to initialize wave file: {str(e)}")
            self.cleanup()
            raise

    def cleanup(self) -> None:
        """Clean up resources."""
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        if self.pyaudio_instance:
            self.pyaudio_instance.terminate()
        if self.wave_file:
            self.wave_file.close()

    def record(self) -> str:
        """
        Start recording audio until interrupted.
        
        Returns:
            str: Path to the recorded audio file
        """
        try:
            self._initialize_recording()
            filename = self._get_output_filename()
            self._initialize_wave_file(filename)
            
            self.logger.info(f"Recording started - saving to {filename}")
            self.logger.info("Press Ctrl+C to stop recording")
            
            while True:
                start_time = time.time()
                while time.time() - start_time < self.config.chunk_duration:
                    try:
                        data = self.stream.read(self.config.chunk_size, exception_on_overflow=False)
                        self.wave_file.writeframes(data)
                    except IOError as e:
                        self.logger.warning(f"Stream read error: {str(e)}")
                        continue
                
                self.logger.info(f"Saved chunk at {datetime.now().strftime('%H:%M:%S')}")
                
        except KeyboardInterrupt:
            self.logger.info(f"Recording stopped and saved to {filename}")
        except Exception as e:
            self.logger.error(f"Recording failed: {str(e)}")
            raise
        finally:
            self.cleanup()
            
        return filename

def main():
    """Entry point for the audio recorder."""
    try:
        recorder = AudioRecorder()
        recorder.record()
    except Exception as e:
        logging.error(f"Application error: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()
