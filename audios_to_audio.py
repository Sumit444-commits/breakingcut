from moviepy.editor import concatenate_audioclips, AudioFileClip
import os
import streamlit as st

def concate_audios(audios_path,output_path):
    """Concatenates several audio files into one audio file using MoviePy
    and save it to `output_path`. Note that extension (mp3, etc.) must be added to `output_path`"""
    clips = [AudioFileClip(c) for c in audios_path]
    final_clip = concatenate_audioclips(clips)
    final_clip.write_audiofile(output_path)
    

audio_path = r"Audio/generated_audio/"
audios = [audio for audio in os.listdir(audio_path) if audio.endswith(".wav")]
audios = [audio_path + audio for audio in audios]
st.write(audios)
output = r"Audio/concated_audio.wav"
concate_audios(audios,output)

