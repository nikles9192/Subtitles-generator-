import streamlit as st
import torch
import librosa
from transformers import pipeline
import datetime
import tempfile
import os
import math

# --- 1. Core Transcription and Formatting Functions ---

def format_timestamp(seconds: float) -> str:
    """Converts seconds to SRT timestamp format (HH:MM:SS,ms)."""
    if seconds is None:
        return "00:00:00,000"
    delta = datetime.timedelta(seconds=seconds)
    hours, remainder = divmod(delta.seconds, 3600)
    minutes, seconds_part = divmod(remainder, 60)
    milliseconds = int(delta.microseconds / 1000)
    return f"{hours:02}:{minutes:02}:{seconds_part:02},{milliseconds:03}"

# This is the new, more robust function for splitting captions.
def _create_caption_segments(words_with_timestamps, max_chars):
    """
    Groups words into smaller caption segments based on max_chars,
    and attempts to break at sentence-ending punctuation for better readability.
    Returns a list of (start_time, end_time, text) tuples.
    """
    segments = []
    if not words_with_timestamps:
        return segments

    current_words_data = []
    current_segment_text = ""
    # Initialize start time with the first word's start time
    segment_start_time = words_with_timestamps[0]['start']

    for i, word_data in enumerate(words_with_timestamps):
        word = word_data['word']

        # Determine the text of the segment if this word were added.
        # Add a space only if the segment is not empty.
        text_with_new_word = (current_segment_text + " " + word) if current_segment_text else word

        # --- SPLITTING LOGIC ---

        # 1. SPLIT IF TEXT GETS TOO LONG
        if len(text_with_new_word) > max_chars and current_segment_text:
            # The current segment is full. Finalize it *without* the new word.
            segment_end_time = current_words_data[-1]['end']
            segments.append((segment_start_time, segment_end_time, current_segment_text))

            # Start a new segment with the current word.
            current_words_data = [word_data]
            current_segment_text = word
            segment_start_time = word_data['start']

        # 2. SPLIT AT THE END OF A SENTENCE
        elif word.endswith(('.', '?', '!')):
            # This word ends a sentence, so finalize the segment *with* this word.
            segment_end_time = word_data['end']
            segments.append((segment_start_time, segment_end_time, text_with_new_word))

            # Reset for the next segment.
            current_words_data = []
            current_segment_text = ""
            # Set start time for the *next* segment, if there is one.
            if i + 1 < len(words_with_timestamps):
                segment_start_time = words_with_timestamps[i+1]['start']

        # 3. NO SPLIT: JUST ADD THE WORD
        else:
            current_words_data.append(word_data)
            current_segment_text = text_with_new_word

    # After the loop, add any remaining text as the final segment.
    if current_segment_text:
        # The end time is the end of the last word added.
        segment_end_time = current_words_data[-1]['end']
        segments.append((segment_start_time, segment_end_time, current_segment_text))

    return segments

# Use a cache for the pipeline to avoid reloading the model on every run
@st.cache_resource
def load_transcription_pipeline(model_name):
    """Loads the Hugging Face pipeline and caches it."""
    st.info(f"Loading model '{model_name}'... This may take a moment.")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # IMPORTANT: We now request "word" timestamps to guide the splitting process accurately.
    transcriber = pipeline(
        "automatic-speech-recognition",
        model=model_name,
        device=device,
        chunk_length_s=30,
        return_timestamps="word" # We need word-level timestamps for smart splitting
    )
    st.success(f"Model '{model_name}' loaded successfully!")
    return transcriber

def transcribe_audio(transcriber, audio_path: str, max_chars: int) -> str:
    """
    Transcribes the audio file and generates SRT content, applying post-processing
    to create well-formatted captions.
    Returns the SRT content as a string.
    """
    try:
        audio_input, sample_rate = librosa.load(audio_path, sr=16000)
    except Exception as e:
        st.error(f"Error loading audio file: {e}")
        return ""

    st.info("Transcribing audio... Please wait.")
    transcription = transcriber(audio_input)
    # For debugging: uncomment the next line to see the raw output from Whisper
    # st.write(transcription)

    st.info("Transcription complete. Formatting SRT...")

    if "chunks" not in transcription or not transcription["chunks"]:
        st.warning("The model did not produce any text. The audio might be silent or too short.")
        return ""

    # Extract all words with their timestamps
    all_words = []
    for chunk in transcription["chunks"]:
        # Whisper's output format gives each word as a chunk when using return_timestamps="word"
        if chunk['timestamp'] and chunk['text']:
             all_words.append({
                "word": chunk['text'].strip(),
                "start": chunk['timestamp'][0],
                "end": chunk['timestamp'][1]
            })

    if not all_words:
        st.warning("Could not extract any words with timestamps from the transcription.")
        return ""

    # Group words into caption segments based on character limit
    caption_segments = _create_caption_segments(all_words, max_chars)

    # Format the final list of segments into SRT
    srt_content = ""
    for i, (start, end, text) in enumerate(caption_segments, 1):
        formatted_start = format_timestamp(start)
        formatted_end = format_timestamp(end)
        srt_content += f"{i}\n"
        srt_content += f"{formatted_start} --> {formatted_end}\n"
        srt_content += f"{text}\n\n"

    return srt_content

# --- 2. Streamlit User Interface ---

st.set_page_config(page_title="Audio to SRT Caption Generator", layout="centered", page_icon="🎧")

st.title("🎧 Audio to SRT Caption Generator")
st.markdown("""
Upload an audio file. This app will generate synchronized captions using OpenAI's Whisper model and format them for readability.
""")

# Sidebar for model selection and options
st.sidebar.header("Settings")
model_options = [
    "openai/whisper-tiny",
    "openai/whisper-tiny.en",
    "openai/whisper-base",
    "openai/whisper-base.en",
    "openai/whisper-small",
    "openai/whisper-small.en",
    "openai/whisper-medium",
    "openai/whisper-medium.en",
    "openai/whisper-large-v2",
    "openai/whisper-large-v3",
]
selected_model = st.sidebar.selectbox(
    "Choose a Whisper Model",
    options=model_options,
    index=1,
    help="Larger models are more accurate but slower. 'base' is a good balance."
)

st.sidebar.markdown("---")
st.sidebar.header("Caption Formatting")
max_chars_per_caption = st.sidebar.slider(
    "Max Characters per Caption Line",
    min_value=20, max_value=100, value=42, step=1,
    help="Recommended for subtitles is around 42 characters per line."
)

# Load the selected model
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

            # Perform transcription with the character limit
            srt_result = transcribe_audio(
                transcriber,
                temp_audio_path,
                max_chars=max_chars_per_caption
            )
            os.remove(temp_audio_path)

            if srt_result:
                st.session_state['srt_result'] = srt_result
                st.session_state['file_name'] = f"{os.path.splitext(uploaded_file.name)[0]}.srt"
                st.success("Captions generated successfully!")
            else:
                st.error("Failed to generate captions. Please check if the audio contains clear speech.")

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
