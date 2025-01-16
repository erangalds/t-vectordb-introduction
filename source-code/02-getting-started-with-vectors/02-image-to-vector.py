from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch

# Load the pre-trained model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Load the image
image_path = "/sample-data/small-boy-with-robot.png"
image = Image.open(image_path)

# Preprocess the image
inputs = processor(images=image, return_tensors="pt")

# Get the image embeddings
with torch.no_grad():
    image_features = model.get_image_features(**inputs)

# Convert the image features to a vector
image_vector = image_features.squeeze().numpy()

print("Image vector:\n", image_vector)
