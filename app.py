import streamlit as st 
from google import genai
from google.genai import types
import speech_recognition as sr 
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import torch
import soundfile as sf
import os
from io import BytesIO
from PIL import Image
# for image to video
import cv2 as cv
import imageio as iio
import numpy as np
# for audio duration
from pydub import AudioSegment
import glob
# for audios concatination
from moviepy.editor import concatenate_audioclips, AudioFileClip
# for adding audio to video
from moviepy.editor import VideoFileClip, AudioFileClip
# for subtitles generation and adding to video
import assemblyai as aai
import pysrt
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip
import time
import nltk
import re
nltk.download("punkt")  # Ensure sentence tokenizer is available
os.system("pip install moviepy")

# Explicitly set ImageMagick binary path
os.environ["IMAGEMAGICK_BINARY"] = "/usr/bin/convert"

# Api setups

# Set up Gemini API client
client = genai.Client(api_key=st.secrets["gemini"]["api_key"])
 # Set API Key for AssemblyAI
aai.settings.api_key =  st.secrets["assemblyai"]["api_key"]


# Define Directories

# images directory
images_dir = r"Images"
# Define the text files directory path
text_file_path = r"Text_files"
# Define the audio directory path
audio_path = r"Audio/generated_audio/"
# Output video file path
video_path = r"Video/output_video.mp4"
video_dir = r"Video"
audio_concat = r"Audio/concated_audio.wav"
# Define file paths
srtfilename = r"Text_files/subtitle.srt"
mp4filename = r"Video/output_video_with_audio.mp4"


# Ensure directory exists
os.makedirs(images_dir, exist_ok=True)  
os.makedirs(audio_path, exist_ok=True)  
os.makedirs(text_file_path, exist_ok=True)  
os.makedirs(audio_path, exist_ok=True)  
os.makedirs(video_dir, exist_ok=True)  

# Functions

# Function to delete old files
def delete_files_in_directory(directory_path):
    try:
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
    except OSError as e:
        print(f"Error: {e}")
        
# Function to create a fade transition between two images
def create_fade_transition(image1, image2, duration, fps):
    num_frames = int(duration * fps)
    transition_frames = []
    for i in range(num_frames):
        alpha = i / num_frames
        blended = cv.addWeighted(image1, 1 - alpha, image2, alpha, 0)
        transition_frames.append(blended)
    return transition_frames

# function for concatinating the audio files
def concate_audios(audios_path,output_path):
    """Concatenates several audio files into one audio file using MoviePy
    and save it to `output_path`. Note that extension (mp3, etc.) must be added to `output_path`"""
    clips = [AudioFileClip(c) for c in audios_path]
    final_clip = concatenate_audioclips(clips)
    final_clip.write_audiofile(output_path)
    
# Function to convert SRT time to seconds
def time_to_seconds(time_obj):
    return (
        time_obj.hours * 3600
        + time_obj.minutes * 60
        + time_obj.seconds
        + time_obj.milliseconds / 1000
    )

# Create Subtitle Clips with Improved Styling
def create_subtitle_clips(subtitles, videosize, fontsize=40, font="DejaVu-Sans", text_color="white", outline_color="white", outline_width=2):
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
        subtitle_y_position = video_height * 0.80  # 85% from the top (bottom)

        subtitle_clips.append(text_clip.set_position((subtitle_x_position, subtitle_y_position)))

    return subtitle_clips

# Retry function for image generation
def generate_image_with_retry(content, max_retries=5, delay=2):
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model="gemini-2.0-flash-exp-image-generation",
                contents=content,
                config=types.GenerateContentConfig(response_modalities=["Text", "Image"])
            )
            
            for part in response.candidates[0].content.parts:
                if part.text:
                    st.write("Text Response:", part.text)
                elif part.inline_data and "image" in part.inline_data.mime_type:
                    image_data = BytesIO(part.inline_data.data)
                    return Image.open(image_data)
        except Exception as e:
            # st.error(f"Attempt {attempt + 1}: Error generating image - {e}")
            time.sleep(delay)  # Wait before retrying
    return None  # Return None if all attempts fail

# Function to load external CSS file
def load_css(file_name):
    with open(file_name, "r") as f:
        css = f.read()
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
        
    st.markdown("""
        <div class="footer">
            <p>💡 <strong>Created by Sumit Sharma</strong> | 📧 <a href='mailto:sumit8444061@gmail.com' target='_blank'>sumit8444061@gmail.com</a></p>
            <p>🌐 <a href='https://sumit-portfolio.free.nf' target='_blank'>Visit My Portfolio Website</a> | 🔗 <a href='https://www.linkedin.com/in/sumit-sharma-a0b2c7' target='_blank'>LinkedIn</a></p>
            <p>© 2025 AI Story Video Generator. All Rights Reserved.</p>
        </div>
    """, unsafe_allow_html=True)



if __name__ == "__main__":
    st.set_page_config(page_title="Breaking-/-Cut| AI Story Video Generator", layout="wide")
    # Custom Styled Logo
    # Display Logo Using HTML
    st.markdown('<h1 class="header">🔥 Breaking-/-Cut 🔥</h1>', unsafe_allow_html=True)

    # Load the styles.css file
    load_css("style.css")




    st.title("🎬 Generate Your AI Story Video")

    # 📖 Add How to Use & Troubleshooting Section Here
    with st.expander("📖 How to Use"):
        st.markdown("""
        ### **📌 Instructions to Use the AI Story Video Generator**
        **1️⃣ Enter Your Story**  
        - Describe your story in Input box.  
        - Ensure the story is **clear and structured** for better results.  

        
        **2️⃣ Generate the Story**  
        - Hit **"Enter Key"** to break your text into structured sentences.  
        - The AI will process and prepare story for you.  
        - You will see three buttons (e.g: Generate Audio Files, Generate Images, Generate Video)

        **3️⃣ Generate Audio Files**  
        - Click **"Generate Audio Files"**.  
        - This will generate audios for each sentence of generated story.
        - The voice will match the story’s.  
        
        **4️⃣ Generate Audio**  
        - After the story and audios are processed.  
        - Click **"Generate Images"**.
        - This will generate **images** based on your story text.  
        
        **5️⃣ Create Video**  
        - After generating audios and images click **"Generate Video"**.  
        - The app will generate two videos (without subtitles and with subtitles).
            - The video will be like showing images one by one with story audio playing in background.
        - The app will generate a video that you can **preview & download**.
          
        """)

    with st.expander("⚠️ Troubleshooting & Failure Handling"):
        st.markdown("""
        ### **🚨 Common Issues & Fixes**
        **❌ Story or Image Prompt Generation Fails**  
        - If the AI doesn’t generate responses, **check your input length**.  
        - **Avoid special characters** or extreme-length stories.  
        - Try **refreshing the app** and running it again.  
        
        **❌ Audio Not Playing or Missing**  
        - Wait a few seconds; the audio may still be processing.  
        - If the issue continues, **regenerate the audio**.  

        **❌ Images Not Generating**  
        - If an image fails, the app will **retry up to 3 times**.  
        - If the error persists, check your **internet connection**.  
        - **Restart the app** if needed.  

        **❌ Video Creation Fails or Freezes**  
        - If images are fewer than the audio duration, the last image **will repeat** to fit the entire video.  
        - Ensure **all images & audio are properly generated** before creating the video.  

        **💡 Additional Tips:**  
        ✅ Use **clear and structured** text for better AI results.  
        ✅ Keep **background processes minimal** to avoid app slowdowns.  
        ✅ If an error occurs multiple times, **try restarting the app**.  
        """)

   
    
    # Initialize session state variables
    if "prev_input" not in st.session_state:
        st.session_state.prev_input = ""  # Stores the last input question
    if "response_text" not in st.session_state:
        st.session_state.response_text = None  # Stores AI-generated response
    if "response_prompt" not in st.session_state:
        st.session_state.response_prompt = []  # Stores AI-generated image prompt response
    if "response_content" not in st.session_state:
        st.session_state.response_content = []  # Stores AI-generated image prompt response
    if "speech_generated" not in st.session_state:
        st.session_state.speech_generated = False  # Tracks if speech was generated
    if "audio_files" not in st.session_state:
        st.session_state.audio_files = []  # List of generated audio files
    if "image_files" not in st.session_state:
        st.session_state.image_files = []  # List of generated image files
        # Initialize session state variables for image generation
    if "images_generated" not in st.session_state:
        st.session_state.images_generated = False
    # User input
    st.markdown("## Generate your story")
    user_input = st.text_input("Enter your story:")
    
    # Reset response & delete old audio files if the user input changes
    if user_input != st.session_state.prev_input:
        st.session_state.response_text = None
        st.session_state.response_prompt = []
        st.session_state.response_content = []
        st.session_state.speech_generated = False
        st.session_state.prev_input = user_input  # Update stored input
        delete_files_in_directory(audio_path)  # Delete old audio files
        delete_files_in_directory(images_dir)  # Delete old images

    # Generate response if input is given and response is not already stored
    if user_input and st.session_state.response_text is None:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=user_input + "( Write response of maximum 150 words and minimum is upto you, it should be in story form and no paragraphs)",
        )
        st.session_state.response_text = response.text  # Store response
    
         # Save response to file
        with open(os.path.join(text_file_path, "story.txt"), "w") as story_file:
            story_file.write(st.session_state.response_text)
            story_file.close()
    # Read the stored response text
    with open(os.path.join(text_file_path,"story.txt"), "r" ) as file:
        story_text = file.read()
        file.close()
        
    # Display first response
    if st.session_state.response_text:
        st.write(st.session_state.response_text)
        
        # Split text into sentences/paragraphs
        sentences = re.split(r'(?<=[.!?])\s+', story_text.strip())



        # Load the TTS model
        processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
        model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
        vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

        # Load xvector for speaker voice characteristics
        embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
        speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

    
        # Generate speech for all sentences at once
        if st.button("Generate Audio files 🎤"):
            with st.spinner("Generating Audio Files for each sentence ... ⏳"):
                if sentences and not st.session_state.speech_generated:
                    delete_files_in_directory(audio_path)  # Ensure no old files exist before generating new ones
                    
                    audio_files = []
                    for idx, sentence in enumerate(sentences):
                        if sentence.strip():  # Ignore empty sentences
                            # Generate speech
                            inputs = processor(text=sentence, return_tensors="pt")
                            speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)

                            # Save each sentence as a separate audio file
                            filename = os.path.join(audio_path, f"speech_{idx + 1}.wav")
                            sf.write(filename, speech.numpy(), samplerate=16000)
                            audio_files.append(filename)

                    # Store generated state
                    st.session_state.speech_generated = True
                    st.session_state.audio_files = audio_files
            
            # Display all generated audio files
            if st.session_state.speech_generated:
                st.write("Generated Speech Files:")
                for filename in st.session_state.audio_files:
                    st.audio(filename, format="audio/wav")  # Play audio in Streamlit
                    st.markdown(f'<a href="{filename}" download>Download {os.path.basename(filename)}</a>', unsafe_allow_html=True)
            

        if st.button("Generate Images 📸"):
            with st.spinner("Creating Images... ⏳"):
                    # Generate second response for image prompt
                if st.session_state.response_text and not st.session_state.response_prompt:
                    
                    with open(os.path.join(text_file_path,"images_prompts.txt"), "w") as prompt_file:
                        pass # for making the file empty for new content
                        
                    # Generate and store image prompts
                    new_prompts = []
                    for sentence in sentences:
                        if sentence.strip():  # Ignore empty sentences
                            response_prompt = client.models.generate_content(
                                model="gemini-2.0-flash",
                                contents=f"\"{sentence}\" \n\n {story_text} \n\n Write me a detailed, photorealistic prompt to generate an image based on the quoted sentence. No headings, no labels—just the prompt itself. Avoid anime or cartoon styles.",
                            )
                            new_prompts.append(response_prompt.text)

                    # Store generated prompts in session state and file
                    st.session_state.response_prompt = new_prompts

                    with open(os.path.join(text_file_path, "images_prompts.txt"), "w") as prompt_file:
                        prompt_file.write("\n".join(new_prompts))
                        prompt_file.close()

                    st.write("---")

                contents = []
                # Read prompts and filter out empty lines
                with open(os.path.join(text_file_path,"images_prompts.txt"), "r") as file:
                    contents = [line.strip() for line in file.readlines() if line.strip()]

                delete_files_in_directory(images_dir)
                st.session_state.image_files = []
                
                for idx, content in enumerate(contents):
                    image = generate_image_with_retry(content)
                    if image:
                        # st.write(f"Generating Image... {idx + 1}")
                        image_path = os.path.join(images_dir, f"image_{idx + 1}.png")
                        image.save(image_path)
                        st.session_state.image_files.append(image_path)
                    else:
                        st.error(f"Failed to generate image for prompt {idx + 1} after retries.")
                
                st.session_state.images_generated = True 
                
            # Display stored images if they exist
            if st.session_state.images_generated and st.session_state.image_files:
                # st.write("### Generated Images:")
                # Loop through both prompts and images together
                for idx, (prompt, img_path) in enumerate(zip(st.session_state.response_prompt, st.session_state.image_files), start=1):
                    st.write(f"**Image {idx}:**")  # Display image number
                    st.write(f"**Prompt:** {prompt}")  # Display corresponding prompt
                    st.image(img_path, caption=f"Image {idx}")  # Show the image
                    st.write("---")  # Add a separator for clarity

        if st.button("🎥 Generate Video"):
            with st.spinner("Creating video... ⏳"):
                delete_files_in_directory("Video") # empty the folder for again storig new videos to avoid duplication
                
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
                # st.write(durations)

                
                audios = [audio for audio in os.listdir(audio_path) if audio.endswith(".wav")]
                audios = [audio_path + audio for audio in audios]

                # concated audio to show and download
                concate_audios(audios,audio_concat)
                st.audio(audio_concat, format="audio/wav")  # Play audio in Streamlit
                st.markdown(f'<a href="{audio_concat}" download>Download {os.path.basename(audio_concat)}</a>', unsafe_allow_html=True)

            
                
                # List all .png images in the directory
                images = [img for img in os.listdir(images_dir) if img.endswith(".png")]

                # Prepend the image path to each image file name
                images = [os.path.join(images_dir, img) for img in images]

                # Display the list of image paths
                # st.write(images)

                # Read the first image to get its dimensions
                image = cv.imread(images[0])
                height, width, _ = image.shape

                # Read images and create a list of image arrays, resizing them to the first image's dimensions
                images = [cv.resize(cv.imread(image_file), (width, height)) for image_file in images]

                        # Handle cases where there are fewer images than durations
                total_audio_duration = sum(durations)
                if len(images) < len(durations):  
                    while len(images) < len(durations):  
                        images.append(images[-1])  # Repeat last image to match durations

                # Duration of each image in the video (in seconds)
                frame_duration = durations # Each image will be displayed according to list
                transition_duration = 0.4  # Duration of the transition in seconds
                fps = 24  # Frames per second

            

                # Create a video writer object
                with iio.get_writer(video_path, fps=fps) as writer:
                    for i in range(len(images)):
                        # Convert the image from BGR to RGB
                        image_rgb = cv.cvtColor(images[i], cv.COLOR_BGR2RGB)
                        
                        # Write the image frames
                        for _ in range(int(frame_duration[i] * fps)):
                            writer.append_data(image_rgb)
                        
                        # Create and write transition frames if not the last image
                        if i < len(images) - 1:
                            transition_frames = create_fade_transition(images[i], images[i + 1], transition_duration, fps)
                            for frame in transition_frames:
                                frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                                writer.append_data(frame_rgb)

                # st.write(f"Video saved as {output_video}")
                # st.video(output_video)
                    
                
                # Load video and audio files
                clip = VideoFileClip(video_path)
                audioclip = AudioFileClip(audio_concat)

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

                st.write("## Without subtitles")
                # Display the video in Streamlit
                st.video(output_video_path)
                st.download_button(
                label="📥 Download Video",
                data=output_video_path,
                file_name="downloaded_video.mp4",
                mime="video/mp4",
                key="no_sub"
                )

                transcriber = aai.Transcriber()
                # Transcribe Video
                st.write("Transcribing Video... Please wait.")
                transcript = transcriber.transcribe("Video/output_video_with_audio.mp4")
                # Export subtitles to SRT format
                subtitles = transcript.export_subtitles_srt()
                st.write("Transcription Complete ✅")
                # st.write(transcript.text)

                # Save subtitles to file
                with open("Text_files/subtitle.srt", "w") as f:
                    f.write(subtitles)
                    
                # Load video and subtitles
                video = VideoFileClip(mp4filename)
                subtitles = pysrt.open(srtfilename)

                # Generate output filename
                output_video_file = mp4filename.replace(".mp4", "_subtitled.mp4")
                # st.write("Output file name:", output_video_file)

                # Create subtitle clips
                subtitle_clips = create_subtitle_clips(subtitles, video.size)

                # Overlay subtitles onto video
                final_video = CompositeVideoClip([video] + subtitle_clips)

                # Save the final video
                st.write("Rendering Video... This may take a while ⏳")
                final_video.write_videofile(output_video_file, codec="libx264", fps=video.fps)
                st.write("Video Processing Complete ✅")

                st.write("## Video with subtitles")
                # Display video in Streamlit
                st.video(output_video_file)
                
                st.download_button(
                label="📥 Download Video",
                data=output_video_file,
                file_name="downloaded_video.mp4",
                mime="video/mp4",
                key="sub_video"
                )
            
                
    