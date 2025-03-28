# without transition effects

# import os
# import cv
# import imageio as iio
# import streamlit as st

# # Path to the images
# image_path = r"Images"

# # List all .png images in the directory
# images = [img for img in os.listdir(image_path) if img.endswith(".png")]

# # Prepend the image path to each image file name
# images = [os.path.join(image_path, img) for img in images]

# # Display the list of image paths
# st.write(images)

# # Read the first image to get its dimensions
# image = cv.imread(images[0])
# height, width, _ = image.shape

# # Read images and create a list of image arrays, resizing them to the first image's dimensions
# images = [cv.resize(cv.imread(image_file), (width, height)) for image_file in images]

# # Output video file path
# output_video = "Video/output_video_1.mp4"

# # Duration of each image in the video (in seconds)
# frame_duration = 2  # Each image will be displayed for 2 seconds

# # Create a video writer object
# with iio.get_writer(output_video, fps=1 / frame_duration) as writer:
#     for image in images:
#         # Convert the image from BGR to RGB (imageio expects RGB)
#         image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
#         writer.append_data(image_rgb)

# st.write(f"Video saved as {output_video}")
# st.video(output_video)


# with transtion effects
import os
import cv2 as cv
import imageio as iio
import numpy as np
import streamlit as st

# Path to the images
image_path = r"Images"

# List all .png images in the directory
images = [img for img in os.listdir(image_path) if img.endswith(".png")]

# Prepend the image path to each image file name
images = [os.path.join(image_path, img) for img in images]

# Display the list of image paths
st.write(images)

# Read the first image to get its dimensions
image = cv.imread(images[0])
height, width, _ = image.shape

# Read images and create a list of image arrays, resizing them to the first image's dimensions
images = [cv.resize(cv.imread(image_file), (width, height)) for image_file in images]

# Output video file path
output_video = "Video/output_video_1.mp4"

# Duration of each image in the video (in seconds)
frame_duration = [2,1,3,1.5]  # Each image will be displayed according to list
transition_duration = 0.4  # Duration of the transition in seconds
fps = 24  # Frames per second

# Function to create a fade transition between two images
def create_fade_transition(image1, image2, duration, fps):
    num_frames = int(duration * fps)
    transition_frames = []
    for i in range(num_frames):
        alpha = i / num_frames
        blended = cv.addWeighted(image1, 1 - alpha, image2, alpha, 0)
        transition_frames.append(blended)
    return transition_frames

# Create a video writer object
with iio.get_writer(output_video, fps=fps) as writer:
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

st.write(f"Video saved as {output_video}")
st.video(output_video)