# from google import genai
# from google.genai import types
# from PIL import Image
# from io import BytesIO
# import base64
# import streamlit as st
# import os

# client = genai.Client(api_key=st.secrets["gemini"]["api_key"])

# path = r"Text_files"
# images_dir = r"Images"
# def delete_files_in_directory(directory_path):
#     try:
#         for filename in os.listdir(directory_path):
#             file_path = os.path.join(directory_path, filename)
#             if os.path.isfile(file_path):
#                 os.remove(file_path)
#     except OSError as e:
#         print(f"Error: {e}")
        
# with open(os.path.join(path,'images_prompts.txt'), 'r') as file:
#     # contents = file.readlines()  # strip() removes the newline character
#     contents = [line.strip() for line in file.readlines() if line.strip()]  # Remove empty lines


# st.write(contents)
# contents = (contents)

# delete_files_in_directory(images_dir)

# # for idx,content in enumerate(contents):
    
# #     response = client.models.generate_content(
# #         model="gemini-2.0-flash-exp-image-generation",
# #         contents=content,
# #         config=types.GenerateContentConfig(
# #         response_modalities=['Text', 'Image']
# #         )
# #     )
    
# #     for part in response.candidates[0].content.parts:
# #         if part.text is not None:
# #             st.write(part.text)
# #         elif part.inline_data is not None:
# #             image = Image.open(BytesIO((part.inline_data.data)))
# #             image.save(os.path.join(images_dir,f'image_{idx}.png'))
# #             st.image(os.path.join(images_dir,f'image_{idx}.png'))

# response = client.models.generate_content(
#     model="gemini-2.0-flash-exp-image-generation",
#     contents=contents,
#     config=types.GenerateContentConfig(
#       response_modalities=['Text', 'Image']
#     )
# )

# for idx,part in response.candidates[0].content.parts:
#   if part.text is not None:
#     st.write(part.text)
#   elif part.inline_data is not None:
#     image = Image.open(BytesIO((part.inline_data.data)))
#     image.save(os.path.join(images_dir,f'image_{idx}.png'))
#     st.image(os.path.join(images_dir,f'image_{idx}.png'))


from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO
import streamlit as st
import os

# Initialize Gemini Client
client = genai.Client(api_key=st.secrets["gemini"]["api_key"])

# Define directories
text_file_path = "Text_files/images_prompts.txt"
images_dir = "Images"

# Function to delete old image files
def delete_files_in_directory(directory_path):
    try:
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
    except OSError as e:
        print(f"Error: {e}")

# Read prompts and filter out empty lines
with open(text_file_path, "r") as file:
    contents = [line.strip() for line in file.readlines() if line.strip()]

# Delete old images
delete_files_in_directory(images_dir)

# Generate images for each prompt
for idx, content in enumerate(contents):
    response = client.models.generate_content(
        model="gemini-2.0-flash-exp-image-generation",
        contents=content,  # Pass one prompt at a time
        config=types.GenerateContentConfig(response_modalities=["Text", "Image"])
    )
    st.write(f"# Prompt Image {idx + 1}")
    st.write(contents[idx])
    st.write(f"### Generated Image {idx + 1}")  # Label each image

    # Iterate over response parts
    for part in response.candidates[0].content.parts:
        if part.text:
            st.write("Text Response:", part.text)  # Display any text response
        elif part.inline_data:
            # Debugging: Check the content type
            st.write("MIME Type:", part.inline_data.mime_type)
            st.write("Data Size:", len(part.inline_data.data))

            if "image" in part.inline_data.mime_type:  # Ensure it's an image
                try:
                    image_data = BytesIO(part.inline_data.data)  # Remove base64 decoding
                    image = Image.open(image_data)

                    # Save image
                    image_path = os.path.join(images_dir, f"image_{idx + 1}.png")
                    image.save(image_path)

                    # Display image in Streamlit
                    st.image(image_path, caption=f"Image {idx + 1}")

                except Exception as e:
                    st.error(f"Error loading image {idx + 1}: {e}")
            else:
                st.error("The response does not contain an image.")
