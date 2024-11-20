import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy import ndimage
import pywt
from skimage.restoration import denoise_bilateral, denoise_wavelet
from skimage.util import random_noise

noise_types = ["gaussian", "snp", "speckle"]
denoising_methods = ["bilateral", "wavelet", "median", "gaussian"]

def add_noise(image, noise_type):
    if noise_type == "gaussian":
        noisy = random_noise(image, mode='gaussian', mean=0, var=0.01)
    elif noise_type == "snp":
        noisy = random_noise(image, mode='s&p', amount=0.02)
    elif noise_type == "speckle":
        noisy = random_noise(image, mode='speckle', mean=0, var=0.05)
    return noisy

def apply_denoising(image, method):
    if method == "bilateral":
        # use axis = -1 for RBG. ow it gives error
        denoised = denoise_bilateral(image, sigma_color=0.1, sigma_spatial=15, channel_axis=-1)
    elif method == "wavelet":
        denoised = denoise_wavelet(image, method='BayesShrink', mode='soft', 
                                 wavelet='db1', channel_axis=-1)
    elif method == "median":
        # median to each channel
        denoised = np.zeros_like(image)
        for i in range(image.shape[-1]):
            denoised[..., i] = ndimage.median_filter(image[..., i], size=3)
    elif method == "gaussian":
        # gaus to each channel
        denoised = np.zeros_like(image)
        for i in range(image.shape[-1]):
            denoised[..., i] = ndimage.gaussian_filter(image[..., i], sigma=1)
    return denoised

def process_images(image_dir):
    """Process all images with different noise and denoising combinations."""
    # List of noise types and denoising methods
    
    # retrieve imgs from dir
    image_files = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]
    results = {}
    
    for image_file in image_files:
        # normalize the image
        original = np.array(Image.open(os.path.join(image_dir, image_file)).convert('RGB')) / 255.0
        
        image_results = {
            'original': original,
            'noisy': {},
            'denoised': {}
        }
        
        # add all noise
        for noise_type in noise_types:
            noisy_image = add_noise(original, noise_type)
            image_results['noisy'][noise_type] = noisy_image
            
            # Apply each denoising method to each noisy image
            denoised_results = {}
            for method in denoising_methods:
                try:
                    denoised = apply_denoising(noisy_image, method)
                    denoised_results[method] = denoised
                except:
                    print("ERROR")
                    pass
            
            image_results['denoised'][noise_type] = denoised_results
        
        results[image_file] = image_results
    
    return results

def psnr(original, processed):
    # PSNR code i ripped
    mse = np.mean((original - processed) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 1.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def plot_results(results):
    
    for image_name, image_results in results.items():
        rows = len(noise_types)
        cols = len(denoising_methods) + 2  # + original, noisy
        
        fig = plt.figure(figsize=(20, 4 * rows))
        plt.suptitle(f'Results for {image_name}', fontsize=16)
        
        # Plot original image
        plt.subplot(rows, cols, 1)
        plt.imshow(image_results['original'])
        plt.title('Original Image')
        plt.axis('off')
        
        # plot type
        for i, noise_type in enumerate(noise_types):
            # plot noisy
            plt.subplot(rows, cols, i * cols + 2)
            plt.imshow(np.clip(image_results['noisy'][noise_type], 0, 1))
            plt.title(f'{noise_type} Noise')
            plt.axis('off')
            
            # plot denoised, calculate psnr
            for j, method in enumerate(denoising_methods):
                plt.subplot(rows, cols, i * cols + j + 3)
                denoised = image_results['denoised'][noise_type][method]
                plt.imshow(np.clip(denoised, 0, 1))
                psnr = psnr(image_results['original'], denoised)
                plt.title(f'{method}\nPSNR: {psnr:.2f}')
                plt.axis('off')
        
        plt.tight_layout()
        plt.show()


# Main execution
if __name__ == "__main__":
    results = process_images('images')
    plot_results(results)
