from pydub import AudioSegment
import os
import glob
import streamlit as st

# Define the list of audio files
audio_path = r"Audio/generated_audio/"
audios = [audio for audio in os.listdir(audio_path) if audio.endswith(".wav")]
audios = [audio_path + audio for audio in audios]

# Alternatively, use glob to find all audio files in a directory
# audio_files = glob.glob('path/to/audio/*.mp3')

# Initialize an empty list to store the durations
durations = []

# Iterate through the list of audio files
for file in audios:
    # Load the audio file
    audio = AudioSegment.from_file(file)
    # Get the duration in seconds
    duration_secs = audio.duration_seconds
    # duration_secs = int(audio.duration_seconds)
    # Append the duration to the list
    durations.append(duration_secs)

# Print the list of durations in milliseconds
st.write(durations)