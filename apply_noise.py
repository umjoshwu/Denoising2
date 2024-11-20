'''
ripped from gpt
'''


import os
import numpy as np
from PIL import Image
from glob import glob

def apply_gaussian_noise(image, mean=0, std=50):
    """Apply Gaussian noise to an image."""
    noisy_image = np.array(image).astype(np.float32)
    noise = np.random.normal(mean, std, noisy_image.shape)
    noisy_image += noise
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_image)

# Create output directory if it doesn't exist
input_dir = 'images'
output_dir = 'noised_images'
os.makedirs(output_dir, exist_ok=True)

# Get all image files from the input directory
image_paths = glob(os.path.join(input_dir, '*.*'))

for img_path in image_paths:
    img = Image.open(img_path).convert('RGB')
    noisy_img = apply_gaussian_noise(img)
    
    # Save the noisy image
    base_name = os.path.basename(img_path)
    output_path = os.path.join(output_dir, base_name)
    noisy_img.save(output_path)

print(f"Processed images saved to {output_dir}")
