import os
import numpy as np
from PIL import Image
from glob import glob
import numpy as np
from PIL import Image

def apply_gaussian_noise(image, mean=0, std=50):
    noisy_image = np.array(image).astype(np.float32)
    noise = np.random.normal(mean, std, noisy_image.shape)
    noisy_image += noise
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_image)

def apply_salt_and_pepper_noise(image, salt_prob=0.05, pepper_prob=0.05):
    noisy_image = np.array(image).astype(np.float32)
    num_salt = np.ceil(salt_prob * noisy_image.size)
    num_pepper = np.ceil(pepper_prob * noisy_image.size)

    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in noisy_image.shape]
    noisy_image[coords[0], coords[1], :] = 255

    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in noisy_image.shape]
    noisy_image[coords[0], coords[1], :] = 0

    return Image.fromarray(noisy_image.astype(np.uint8))

def apply_speckle_noise(image, mean=0, std=0.4):
    noisy_image = np.array(image).astype(np.float32)
    noise = np.random.normal(mean, std, noisy_image.shape)
    noisy_image += noisy_image * noise
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_image)


'''
HERE

'''

input_dir = 'images'
output_dir = 'noised_images'
os.makedirs(output_dir, exist_ok=True)
image_paths = glob(os.path.join(input_dir, '*.*'))



for img_path in image_paths:
    img = Image.open(img_path).convert('RGB')

    #just uncomment out which noise u wanna apply
    noisy_img = apply_speckle_noise(img)
    noisy_img = apply_salt_and_pepper_noise(noisy_img)
    noisy_img = apply_gaussian_noise(noisy_img)


    base_name = os.path.basename(img_path)
    output_path = os.path.join(output_dir, base_name)
    noisy_img.save(output_path)

print(f"Processed images saved to {output_dir}")
