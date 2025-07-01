import streamlit as st
import torch
import librosa
from transformers import pipeline
import datetime
import tempfile
import os
import math # For math.ceil

# --- 1. Core Transcription and Formatting Functions ---

def format_timestamp(seconds: float) -> str:
    """Converts seconds to SRT timestamp format (HH:MM:SS,ms)."""
    if seconds is None:
        return "00:00:00,000" # Should ideally not happen with "chunk" timestamps
    delta = datetime.timedelta(seconds=seconds)
    hours, remainder = divmod(delta.seconds, 3600)
    minutes, seconds_part = divmod(remainder, 60)
    milliseconds = int(delta.microseconds / 1000)
    return f"{hours:02}:{minutes:02}:{seconds_part:02},{milliseconds:03}"

def _split_long_caption(text: str, start_time: float, end_time: float, max_duration: int, max_chars: int):
    """
    Splits a single long caption text into multiple smaller captions based on
    max_duration and max_chars, interpolating timestamps.
    Returns a list of (start, end, text) tuples.
    """
    captions = []
    current_duration = end_time - start_time

    # Determine how many sub-chunks are needed based on duration and character count
    num_parts_by_duration = math.ceil(current_duration / max_duration) if max_duration > 0 else 1
    num_parts_by_chars = math.ceil(len(text) / max_chars) if max_chars > 0 else 1

    num_parts = max(num_parts_by_duration, num_parts_by_chars, 1) # Ensure at least 1 part

    if num_parts <= 1:
        # No splitting needed if it already fits the criteria
        captions.append((start_time, end_time, text.strip()))
        return captions

    # Calculate approximate duration and character length for each part
    approx_part_duration = current_duration / num_parts
    approx_chars_per_part = len(text) / num_parts

    start_idx = 0
    for i in range(num_parts):
        part_start_time = start_time + i * approx_part_duration
        part_end_time = start_time + (i + 1) * approx_part_duration

        # Determine end index for text, trying to break at a space
        target_end_idx = min(len(text), math.ceil((i + 1) * approx_chars_per_part))

        # Find the last space before the target_end_idx for a clean break
        split_idx = text.rfind(' ', start_idx, target_end_idx)
        if split_idx == -1 or split_idx == start_idx: # No space or space is at the very beginning of this part
            # If no good break, just cut at target_end_idx or end of text
            split_idx = target_end_idx

        # Ensure we don't go past the end of the text
        if i == num_parts - 1: # Last part, take remaining text
            sub_text = text[start_idx:].strip()
            part_end_time = end_time # Ensure last part ends at original end_time
        else:
            sub_text = text[start_idx:split_idx].strip()
            # Advance start_idx for the next part, skipping the space we just split on
            start_idx = split_idx + 1 if split_idx < len(text) else len(text)

        if sub_text: # Only add if there's actual text
            captions.append((part_start_time, part_end_time, sub_text))

    # Small adjustment for the very last caption end time to match original
    if captions:
        captions[-1] = (captions[-1][0], end_time, captions[-1][2])

    return captions


# Use a cache for the pipeline to avoid reloading the model on every run
@st.cache_resource
def load_transcription_pipeline(model_name):
    """Loads the Hugging Face pipeline and caches it."""
    st.info(f"Loading model '{model_name}'... This may take a moment.")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    transcriber = pipeline(
        "automatic-speech-recognition",
        model=model_name,
        device=device,
        chunk_length_s=30, # Max audio segment passed to model at once
        stride_length_s=5, # Overlap between chunks for better context
        return_timestamps="chunk" # Get phrase-level timestamps from Whisper
    )
    st.success(f"Model '{model_name}' loaded successfully!")
    return transcriber

def transcribe_audio(transcriber, audio_path: str, max_duration: int, max_chars: int) -> str:
    """
    Transcribes the audio file and generates SRT content, applying post-processing
    to split long captions.
    Returns the SRT content as a string.
    """
    try:
        audio_input, sample_rate = librosa.load(audio_path, sr=16000)
    except Exception as e:
        st.error(f"Error loading audio file: {e}")
        return ""

    st.info("Transcribing audio... Please wait.")
    transcription = transcriber(audio_input)
    st.info("Transcription complete. Formatting SRT and splitting long captions...")

    srt_content = ""
    caption_index = 1

    if "chunks" not in transcription:
        st.warning("The model did not return timestamped chunks. Cannot generate SRT.")
        return ""

    # Process each original chunk, potentially splitting it
    final_captions = []
    for chunk in transcription["chunks"]:
        start_time_original, end_time_original = chunk['timestamp']
        text_original = chunk['text'].strip()

        if start_time_original is None or end_time_original is None or not text_original:
            continue

        # Split the original chunk if it's too long
        split_results = _split_long_caption(
            text_original,
            start_time_original,
            end_time_original,
            max_duration,
            max_chars
        )
        final_captions.extend(split_results)

    # Sort captions by start time (important if splitting causes overlaps or out-of-order)
    final_captions.sort(key=lambda x: x[0])

    # Format the final list of (start, end, text) tuples into SRT
    for start_time, end_time, text in final_captions:
        formatted_start = format_timestamp(start_time)
        formatted_end = format_timestamp(end_time)

        srt_content += f"{caption_index}\n"
        srt_content += f"{formatted_start} --> {formatted_end}\n"
        srt_content += f"{text}\n\n"
        caption_index += 1

    return srt_content

# --- 2. Streamlit User Interface ---

st.set_page_config(page_title="Audio to SRT Caption Generator", layout="centered", page_icon="🎧")

st.title("🎧 Audio to SRT Caption Generator")
st.markdown("""
Upload an audio file, and this app will generate synchronized captions in the SRT format using OpenAI's Whisper model.
""")

# Sidebar for model selection and options
st.sidebar.header("Settings")
model_options = [
    "openai/whisper-tiny",
    "openai/whisper-base",
    "openai/whisper-small",
    "openai/whisper-medium",
]
selected_model = st.sidebar.selectbox(
    "Choose a Whisper Model",
    options=model_options,
    index=1, # Default to 'base'
    help="Larger models are more accurate but slower and require more memory. 'base' is a good balance."
)

st.sidebar.markdown("---")
st.sidebar.header("Caption Formatting")
max_caption_duration = st.sidebar.slider(
    "Max Caption Duration (seconds)",
    min_value=3, max_value=15, value=7, step=1,
    help="Maximum duration for a single caption entry."
)
max_chars_per_caption = st.sidebar.slider(
    "Max Characters per Caption",
    min_value=20, max_value=100, value=50, step=5,
    help="Maximum characters in a single caption line for readability."
)

# Load the selected model
transcriber = load_transcription_pipeline(selected_model)

# File uploader
st.header("1. Upload your Audio File")
uploaded_file = st.file_uploader(
    "Choose a WAV, MP3, M4A, or OGG file",
    type=['wav', 'mp3', 'm4a', 'ogg']
)

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/ogg')

    st.header("2. Generate Captions")
    if st.button("✨ Generate SRT Captions"):
        with st.spinner("Processing... This might take a while depending on audio length and model size."):
            # Save uploaded file to a temporary location to be read by librosa
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmpfile:
                tmpfile.write(uploaded_file.getvalue())
                temp_audio_path = tmpfile.name

            # Perform transcription with user-defined parameters
            srt_result = transcribe_audio(
                transcriber,
                temp_audio_path,
                max_duration=max_caption_duration,
                max_chars=max_chars_per_caption
            )

            # Clean up the temporary file
            os.remove(temp_audio_path)

            if srt_result:
                st.session_state['srt_result'] = srt_result
                st.session_state['file_name'] = f"{os.path.splitext(uploaded_file.name)[0]}.srt"
                st.success("Captions generated successfully!")
            else:
                st.error("Failed to generate captions. Please try another file or model.")

# Display results and download button if available in session state
if 'srt_result' in st.session_state and st.session_state['srt_result']:
    st.header("3. View and Download Captions")
    st.text_area(
        label="Generated SRT Content",
        value=st.session_state['srt_result'],
        height=300
    )

    st.download_button(
        label="📥 Download .SRT File",
        data=st.session_state['srt_result'],
        file_name=st.session_state['file_name'],
        mime='text/plain'
    )

st.sidebar.markdown("---")
st.sidebar.markdown("Built with ❤️ using [Streamlit](https://streamlit.io) and [Hugging Face](https://huggingface.co/).")
