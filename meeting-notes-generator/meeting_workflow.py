import os
import wave
import numpy as np
from textwrap import dedent
from typing import Iterator
from transformers import WhisperProcessor, WhisperForConditionalGeneration

from agno.agent import Agent
from agno.models.aws.bedrock import AwsBedrock
from agno.storage.sqlite import SqliteStorage
from agno.workflow import Workflow, RunResponse
from agno.utils.log import logger
from agno.utils.pprint import pprint_run_response
from pydantic import BaseModel, Field

# Models for storing data
class ActionItem(BaseModel):
    assignee: str = Field(..., description="Person assigned to the action item")
    description: str = Field(..., description="Description of the action item")

class MeetingMinutes(BaseModel):
    title: str = Field(..., description="Title of the meeting")
    date: str = Field(..., description="Date of the meeting")
    participants: list[str] = Field(..., description="List of meeting participants")
    key_points: list[str] = Field(..., description="Key discussion points from the meeting")
    action_items: list[ActionItem] = Field(..., description="Action items with assignee and description")
    decisions: list[str] = Field(..., description="Decisions made during the meeting")
    summary: str = Field(..., description="Brief summary of the meeting")
    raw_transcript: str = Field(..., description="The raw meeting transcript")

class MeetingNotesWorkflow(Workflow):
    description: str = "Generate meeting minutes from WAV audio recordings using Agno workflow"

    # Initialize Whisper model and processor once
    whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-base")
    whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")
    
    minutes_generator: Agent = Agent(
        name="MinutesGenerator",
        model=BedrockClaude(id="anthropic.claude-3-5-sonnet-20240620-v1:0"),
        description="Meeting minutes generation specialist",
        instructions=dedent("""\
        Analyze the provided transcript and generate structured meeting minutes.
        Extract and organize:
        1. Title and date (infer from context if needed)
        2. Participants (identify speakers)
        3. Key discussion points
        4. Action items with clear assignees
        5. Decisions made
        6. A concise meeting summary
        
        Format the minutes professionally while maintaining the original meaning.
        """),
        response_model=MeetingMinutes,
    )

    def transcribe_audio(self, audio_path: str) -> str:
        """Transcribe WAV audio file using Whisper model"""
        logger.info(f"Transcribing: {audio_path}")
        
        try:
            # Load and process audio
            with wave.open(audio_path, 'rb') as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
                raw_data = wf.readframes(frames)
            
            # Normalize audio data
            audio = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Resample if needed
            if rate != 16000:
                logger.info(f"Resampling from {rate}Hz to 16000Hz")
                audio = np.interp(
                    np.linspace(0, len(audio), int(len(audio) * 16000 / rate)),
                    np.arange(len(audio)),
                    audio
                )
            
            # Process with Whisper
            input_features = self.whisper_processor(
                audio, sampling_rate=16000, return_tensors="pt"
            ).input_features
            
            predicted_ids = self.whisper_model.generate(input_features)
            transcription = self.whisper_processor.batch_decode(
                predicted_ids, skip_special_tokens=True
            )[0]
            
            # Save transcript for reference
            output_path = os.path.splitext(audio_path)[0] + '.txt'
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(transcription)
            
            logger.info(f"Transcript saved to: {output_path}")
            return transcription
            
        except Exception as e:
            logger.error(f"Transcription error: {str(e)}")
            raise

    def run(self, wav_path: str) -> Iterator[RunResponse]:
        # Step 1: Transcribe audio with Whisper
        transcript = self.transcribe_audio(wav_path)
        logger.info(f"Transcription complete: {len(transcript)} characters")
        
        # Step 2: Generate meeting minutes from transcript
        minutes_response = yield from self.minutes_generator.run(
            transcript=transcript
        )
        
        logger.info(f"Minutes generated for: {minutes_response.title}")
        return minutes_response

def save_minutes_to_markdown(minutes, output_path):
    """Save meeting minutes to markdown file"""
    with open(output_path, 'w') as f:
        f.write(f"# {minutes['title']}\n\n")
        f.write(f"**Date:** {minutes['date']}\n\n")
        
        f.write("## Participants\n")
        for participant in minutes['participants']:
            f.write(f"- {participant}\n")
        f.write("\n")
        
        f.write("## Summary\n")
        f.write(f"{minutes['summary']}\n\n")
        
        f.write("## Key Discussion Points\n")
        for point in minutes['key_points']:
            f.write(f"- {point}\n")
        f.write("\n")
        
        f.write("## Action Items\n")
        for item in minutes['action_items']:
            f.write(f"- **{item['assignee']}**: {item['description']}\n")
        f.write("\n")
        
        f.write("## Decisions\n")
        for decision in minutes['decisions']:
            f.write(f"- {decision}\n")

def main():
    # Initialize SQLite storage
    storage = SqliteStorage(
        table_name="meeting_workflow_sessions",
        db_file="meeting_notes.db"
    )
    
    # Setup workflow
    workflow = MeetingNotesWorkflow(
        storage=storage,
        debug_mode=True
    )
    
    # Get WAV file path
    wav_path = input("Enter path to WAV recording: ")
    
    # Run workflow
    response = None
    for resp in workflow.run(wav_path=wav_path):
        pprint_run_response(resp)
        response = resp
    
    # Save output if workflow completed successfully
    if response:
        output_path = os.path.splitext(wav_path)[0] + "_minutes.md"
        save_minutes_to_markdown(response.model_dump(), output_path)
        print(f"Meeting minutes saved to {output_path}")

if __name__ == "__main__":
    main() 