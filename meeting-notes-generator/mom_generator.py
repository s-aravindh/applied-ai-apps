from datetime import datetime
from openai import OpenAI

def read_transcription(file_path):
    with open(file_path, 'r') as file:
        return file.read()

def generate_minutes(transcription):
    client = OpenAI(
        base_url='http://localhost:11434/v1',
        api_key='ollama'
    )    
    prompt = f"""Please generate formal meeting minutes from the following transcription.
    Include:
    - Key discussion points
    - Action items
    - Decisions made

    Transcription:
    {transcription}
    """
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.choices[0].message.content

def save_minutes(minutes, output_file):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"minutes_{timestamp}.md"
    
    if output_file:
        filename = output_file
    
    with open(filename, 'w') as file:
        file.write(minutes)
    
    return filename

def main(transcription_file, output_file=None):
    try:
        transcription = read_transcription(transcription_file)
        minutes = generate_minutes(transcription)
        saved_file = save_minutes(minutes, output_file)
        print(f"Meeting minutes saved to: {saved_file}")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python mom_generator.py <transcription_file> [output_file]")
        sys.exit(1)
    
    transcription_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    main(transcription_file, output_file)
