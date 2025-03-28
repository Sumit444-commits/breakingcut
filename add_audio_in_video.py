# Import necessary libraries
from moviepy.editor import VideoFileClip, AudioFileClip
import streamlit as st

# Paths to the video and audio files
video_path = r"Video/output_video.mp4"
audio_path = r"Audio/concated_audio.wav"

# Load video and audio files
clip = VideoFileClip(video_path)
audioclip = AudioFileClip(audio_path)

# Get the minimum duration to avoid mismatch issues
min_duration = min(clip.duration, audioclip.duration)

# Trim video and audio to the same duration
clip = clip.subclip(0, min_duration)
audioclip = audioclip.subclip(0, min_duration)

# Add audio to the video clip
videoclip = clip.set_audio(audioclip)

# Save the resulting video
output_video_path = r"Video/output_video_with_audio.mp4"
videoclip.write_videofile(output_video_path, codec="libx264", audio_codec="aac")

# Display the video in Streamlit
st.video(output_video_path)


