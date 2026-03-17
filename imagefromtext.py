# --------------------------------------------------
# PROGRAM: Image Generation from Text using Deep Learning
# --------------------------------------------------

# Step 1: Install Required Libraries
!pip install diffusers transformers accelerate torch --quiet

# Step 2: Import Necessary Libraries
import torch
import numpy as np
import matplotlib.pyplot as plt
from diffusers import StableDiffusionPipeline

# Step 3: Initialize Device Configuration
print("Checking device...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# Step 4: Define Model Parameters
model_name = "runwayml/stable-diffusion-v1-5"
print("Loading model:", model_name)

# Step 5: Load Pretrained Model
pipe = StableDiffusionPipeline.from_pretrained(model_name)

# Step 6: Move Model to Device
pipe = pipe.to(device)

# Step 7: Take Text Input from User
text_prompt = input("Enter your text description: ")
print("Input Text:", text_prompt)

# Step 8: Generate Image from Text
print("Generating image...")
output = pipe(text_prompt)

# Step 9: Extract Generated Image
generated_image = output.images[0]

# Step 10: Display Generated Image
plt.figure(figsize=(6,6))
plt.imshow(generated_image)
plt.axis('off')
plt.title("Generated Image from Text")
plt.show()

# Step 11: Save Output Image
file_name = "generated_image.png"
generated_image.save(file_name)

# Step 12: Confirmation Message
print("Image successfully generated and saved as:", file_name)
