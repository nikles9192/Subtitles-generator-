import streamlit as st
import torch
import librosa
from transformers import pipeline
import datetime
import tempfile
import os

# --- 1. Core Transcription and Formatting Functions ---

def format_timestamp(seconds: float) -> str:
    """Converts seconds to SRT timestamp format (HH:MM:SS,ms)."""
    # Handle potential None values gracefully
    if seconds is None:
        return "00:00:00,000"
    delta = datetime.timedelta(seconds=seconds)
    hours, remainder = divmod(delta.seconds, 3600)
    minutes, seconds_part = divmod(remainder, 60)
    milliseconds = int(delta.microseconds / 1000)
    return f"{hours:02}:{minutes:02}:{seconds_part:02},{milliseconds:03}"

# Use a cache for the pipeline to avoid reloading the model on every run
@st.cache_resource
def load_transcription_pipeline(model_name):
    """Loads the Hugging Face pipeline and caches it."""
    st.info(f"Loading model '{model_name}'... This may take a moment.")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # ==================================================================
    # THE FIX IS HERE: Changed "word" to "chunk" for more stability
    # ==================================================================
    transcriber = pipeline(
        "automatic-speech-recognition",
        model=model_name,
        device=device,
        chunk_length_s=30,
        stride_length_s=5,
        return_timestamps="chunk" # Use "chunk" instead of "word"
    )
    st.success(f"Model '{model_name}' loaded successfully!")
    return transcriber

def transcribe_audio(transcriber, audio_path: str) -> str:
    """
    Transcribes the audio file and generates SRT content.
    Returns the SRT content as a string.
    """
    try:
        audio_input, sample_rate = librosa.load(audio_path, sr=16000)
    except Exception as e:
        st.error(f"Error loading audio file: {e}")
        return ""

    st.info("Transcribing audio... Please wait.")
    # The transcriber will now return sentence/phrase level chunks
    transcription = transcriber(audio_input)
    st.info("Transcription complete. Formatting SRT...")

    srt_content = ""
    caption_index = 1

    if "chunks" not in transcription:
        st.warning("The model did not return timestamped chunks. Cannot generate SRT.")
        return ""

    # The loop works exactly the same, but each "chunk" is now a phrase
    for chunk in transcription["chunks"]:
        start_time, end_time = chunk['timestamp']

        # Added safety check for None timestamps, just in case
        if start_time is None or end_time is None:
            continue

        start_time_str = format_timestamp(start_time)
        end_time_str = format_timestamp(end_time)
        text = chunk['text'].strip()

        if not text:
            continue

        srt_content += f"{caption_index}\n"
        srt_content += f"{start_time_str} --> {end_time_str}\n"
        srt_content += f"{text}\n\n"
        caption_index += 1

    return srt_content

# --- 2. Streamlit User Interface ---

st.set_page_config(page_title="Audio to SRT Caption Generator", layout="centered", page_icon="🎧")

st.title("🎧 Audio to SRT Caption Generator")
st.markdown("""
Upload an audio file, and this app will generate synchronized captions in the SRT format using OpenAI's Whisper model.
""")

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
    index=1,
    help="Larger models are more accurate but slower. 'base' is a good balance."
)

transcriber = load_transcription_pipeline(selected_model)

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
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmpfile:
                tmpfile.write(uploaded_file.getvalue())
                temp_audio_path = tmpfile.name

            srt_result = transcribe_audio(transcriber, temp_audio_path)
            os.remove(temp_audio_path)

            if srt_result:
                st.session_state['srt_result'] = srt_result
                st.session_state['file_name'] = f"{os.path.splitext(uploaded_file.name)[0]}.srt"
                st.success("Captions generated successfully!")
            else:
                st.error("Failed to generate captions. Please try another file or model.")

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
