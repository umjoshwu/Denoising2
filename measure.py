import os
from PIL import Image
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import matplotlib.pyplot as plt
from skimage.restoration import denoise_wavelet
from scipy import ndimage
from skimage.restoration import denoise_bilateral

def apply_bilateral_filter(image_array):
    return denoise_bilateral(image_array, 
                           sigma_color=0.1, 
                           sigma_spatial=15,
                           channel_axis=-1 if len(image_array.shape) > 2 else None)

def apply_wavelet_filter(image_array):
    return denoise_wavelet(image_array,
                          channel_axis=-1 if len(image_array.shape) > 2 else None,
                          convert2ycbcr=True if len(image_array.shape) > 2 else False,
                          method='BayesShrink',
                          mode='soft',
                          rescale_sigma=True)

def process_and_save_images(input_folder, noise_type, output_root):
    bilateral_output = os.path.join(output_root, f'bilateral_denoised_{noise_type}')
    wavelet_output = os.path.join(output_root, f'wavelet_denoised_{noise_type}')
    
    os.makedirs(bilateral_output, exist_ok=True)
    os.makedirs(wavelet_output, exist_ok=True)
    
    processed_pairs = []
    
    for img_name in os.listdir(input_folder):
        if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            print(f"Processing {img_name} from {noise_type}...")
            
            img_path = os.path.join(input_folder, img_name)
            original = Image.open(img_path)
            
            img_array = np.array(original).astype(float) / 255.0
            bilateral_result = apply_bilateral_filter(img_array)
            wavelet_result = apply_wavelet_filter(img_array)
            
            bilateral_result = (bilateral_result * 255).astype(np.uint8)
            wavelet_result = (wavelet_result * 255).astype(np.uint8)
            
            Image.fromarray(bilateral_result).save(os.path.join(bilateral_output, img_name))
            Image.fromarray(wavelet_result).save(os.path.join(wavelet_output, img_name))
            

            orig_array = np.array(original)
            processed_pairs.append({
                'name': img_name,
                'original': orig_array,
                'bilateral': bilateral_result,
                'wavelet': wavelet_result
            })
        
           
    
    return processed_pairs

def calculate_metrics(processed_pairs, noise_type):
    results = {
        f'bilateral_{noise_type}': {'psnr': [], 'ssim': [], 'names': []},
        f'wavelet_{noise_type}': {'psnr': [], 'ssim': [], 'names': []}
    }
    
    for pair in processed_pairs:
        original = pair['original']
        
        bilateral_psnr = psnr(original, pair['bilateral'])
        bilateral_ssim = ssim(original, pair['bilateral'], channel_axis=2)
        
        wavelet_psnr = psnr(original, pair['wavelet'])
        wavelet_ssim = ssim(original, pair['wavelet'], channel_axis=2)
        
        results[f'bilateral_{noise_type}']['psnr'].append(bilateral_psnr)
        results[f'bilateral_{noise_type}']['ssim'].append(bilateral_ssim)
        results[f'bilateral_{noise_type}']['names'].append(pair['name'])
        
        results[f'wavelet_{noise_type}']['psnr'].append(wavelet_psnr)
        results[f'wavelet_{noise_type}']['ssim'].append(wavelet_ssim)
        results[f'wavelet_{noise_type}']['names'].append(pair['name'])
        
        print(f"\nMetrics for {pair['name']} ({noise_type}):")
        print(f"Bilateral - PSNR: {bilateral_psnr:.2f} dB, SSIM: {bilateral_ssim:.4f}")
        print(f"Wavelet   - PSNR: {wavelet_psnr:.2f} dB, SSIM: {wavelet_ssim:.4f}")
    
    return results
if __name__ == "__main__":
    # Define paths
    noise_folders = {
        'speckle': 'speckle_noised_images',
        'snp_speckle': 'snp_speckle_noised_images',
        'salt_pepper': 'salt_pepper_noised_images',
        'gaussian': 'g_noised_images'
    }
    output_root = "."
    
    all_results = {}
    
    for noise_type, folder in noise_folders.items():
        print(f"\nProcessing {noise_type} noise images...")
        processed_pairs = process_and_save_images(folder, noise_type, output_root)
        results = calculate_metrics(processed_pairs, noise_type)
        all_results.update(results)
    
    for method in all_results:
        avg_psnr = np.mean(all_results[method]['psnr'])
        avg_ssim = np.mean(all_results[method]['ssim'])
        print(f"\n{method}:")
        print(f"Average PSNR: {avg_psnr:.2f} dB")
        print(f"Average SSIM: {avg_ssim:.4f}")
    
