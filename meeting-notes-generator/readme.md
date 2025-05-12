# Meeting Notes Generator Workflow

This project uses Agno Workflow to automatically generate meeting minutes from recorded meetings. The workflow:
1. Transcribes WAV audio files using Whisper
2. Generates structured meeting notes using Amazon Bedrock Claude 3.5 Sonnet

## Setup

### Prerequisites

- Python 3.8+
- Amazon AWS credentials configured for Bedrock access

### Installation

1. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

2. Configure your AWS credentials for Bedrock access:
   ```
   export AWS_ACCESS_KEY_ID=your_access_key
   export AWS_SECRET_ACCESS_KEY=your_secret_key
   export AWS_DEFAULT_REGION=your_region
   ```

## Usage

Run the workflow with:

```
python meeting_workflow.py
```

The script will:
1. Prompt you for the path to a WAV meeting recording
2. Transcribe the audio 
3. Generate structured meeting minutes
4. Save the meeting minutes to a markdown file

## Output

The workflow generates:
- A TXT file containing the raw transcript
- A markdown file with formatted meeting minutes including:
  - Meeting title and date
  - Participants
  - Summary
  - Key discussion points
  - Action items with assignees
  - Decisions made

## Storage

The workflow uses SQLite for persistent storage, allowing for:
- Cached results
- Resumable workflows
- Session history

The database is stored in `meeting_notes.db` in the project directory.

## File Structure

- `meeting_workflow.py` - Main workflow implementation
- `transcribe_audio.py` - Reference implementation for audio transcription
- `requirements.txt` - Required dependencies
- `README.md` - This documentation
