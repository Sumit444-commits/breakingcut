
import assemblyai as aai
import streamlit as st
import pysrt
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip

# Set API Key for AssemblyAI
aai.settings.api_key =  st.secrets["assemblyai"]["api_key"]
transcriber = aai.Transcriber()

# Transcribe Video
st.write("Transcribing Video... Please wait.")
transcript = transcriber.transcribe("Video/output_video_with_audio.mp4")

# Export subtitles to SRT format
subtitles = transcript.export_subtitles_srt()
st.write("Transcription Complete ✅")
st.write(transcript.text)

# Save subtitles to file
with open("Text_files/subtitle.srt", "w") as f:
    f.write(subtitles)

# Define file paths
srtfilename = "Text_files/subtitle.srt"
mp4filename = "Video/output_video_with_audio.mp4"

# Function to convert SRT time to seconds
def time_to_seconds(time_obj):
    return (
        time_obj.hours * 3600
        + time_obj.minutes * 60
        + time_obj.seconds
        + time_obj.milliseconds / 1000
    )

# Create Subtitle Clips with Improved Styling
def create_subtitle_clips(subtitles, videosize, fontsize=40, font="Arial", text_color="white", outline_color="white", outline_width=2):
    subtitle_clips = []
    video_width, video_height = videosize

    for subtitle in subtitles:
        start_time = time_to_seconds(subtitle.start)
        end_time = time_to_seconds(subtitle.end)
        duration = end_time - start_time

        # Subtitle text styling
        text_clip = (
            TextClip(
                subtitle.text,
                fontsize=fontsize,
                font=font,
                color=text_color,
                stroke_color=outline_color,  # Outline for better visibility
                stroke_width=outline_width,  # Outline thickness
                size=(video_width * 0.8, None),  # 80% width to avoid overflow
                method="caption",
            )
            .set_start(start_time)
            .set_duration(duration)
        )

        # Positioning subtitles slightly above the bottom to avoid cutoff
        subtitle_x_position = "center"
        subtitle_y_position = video_height * 0.85  # 85% from the top (bottom)

        subtitle_clips.append(text_clip.set_position((subtitle_x_position, subtitle_y_position)))

    return subtitle_clips

# Load video and subtitles
video = VideoFileClip(mp4filename)
subtitles = pysrt.open(srtfilename)

# Generate output filename
output_video_file = mp4filename.replace(".mp4", "_subtitled.mp4")
st.write("Output file name:", output_video_file)

# Create subtitle clips
subtitle_clips = create_subtitle_clips(subtitles, video.size)

# Overlay subtitles onto video
final_video = CompositeVideoClip([video] + subtitle_clips)

# Save the final video
st.write("Rendering Video... This may take a while ⏳")
final_video.write_videofile(output_video_file, codec="libx264", fps=video.fps)
st.write("Video Processing Complete ✅")

# Display video in Streamlit
st.video(output_video_file)
