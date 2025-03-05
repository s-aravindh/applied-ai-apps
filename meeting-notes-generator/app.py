import gradio as gr
import os
from datetime import datetime
from audio_recorder import start_recording, stop_recording
from transcribe_audio import transcribe_audio
from mom_generator import generate_minutes, save_minutes

# Add this constant at the top level
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

def record_audio():
    start_recording()
    return "Recording started... Click 'Stop Recording' when finished."

def stop_audio_recording():
    audio_file = stop_recording()
    return audio_file, f"Recording saved to: {audio_file}"

def process_audio(audio_file):
    if not audio_file:
        return "Please record audio first!", ""
    
    transcription = transcribe_audio(audio_file)
    return transcription, "Transcription complete!"

def generate_meeting_minutes(transcription):
    if not transcription:
        return "Please transcribe the audio first!", ""
    
    minutes = generate_minutes(transcription)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"minutes_{timestamp}.md"
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Save minutes with full path
    output_path = os.path.join(OUTPUT_DIR, filename)
    save_minutes(minutes, os.path.basename(output_path), OUTPUT_DIR)
    return minutes, f"Minutes saved to: {output_path}"

def create_interface():
    with gr.Blocks(title="Meeting Minutes Generator", theme=gr.themes.Base()) as interface:
        gr.Markdown("# 🎙️ Meeting Minutes Generator")
        
        with gr.Tab("1. Record Audio"):
            with gr.Row():
                start_btn = gr.Button("Start Recording", variant="primary")
                stop_btn = gr.Button("Stop Recording", variant="secondary")
            audio_status = gr.Textbox(label="Recording Status", interactive=False)
            audio_file_output = gr.Textbox(label="Audio File Path", visible=False)

        with gr.Tab("2. Transcribe"):
            transcribe_btn = gr.Button("Transcribe Recording", variant="primary")
            transcription_output = gr.Textbox(label="Transcription", lines=10)
            transcription_status = gr.Textbox(label="Status", interactive=False)

        with gr.Tab("3. Generate Minutes"):
            generate_btn = gr.Button("Generate Minutes", variant="primary")
            minutes_output = gr.Markdown(label="Meeting Minutes")
            minutes_status = gr.Textbox(label="Status", interactive=False)

        # Event handlers
        start_btn.click(
            fn=record_audio,
            outputs=audio_status
        )
        
        stop_btn.click(
            fn=stop_audio_recording,
            outputs=[audio_file_output, audio_status]
        )
        
        transcribe_btn.click(
            fn=process_audio,
            inputs=audio_file_output,
            outputs=[transcription_output, transcription_status]
        )
        
        generate_btn.click(
            fn=generate_meeting_minutes,
            inputs=transcription_output,
            outputs=[minutes_output, minutes_status]
        )

    return interface

if __name__ == "__main__":
    interface = create_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        debug=True
    )
