import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from scipy.ndimage import gaussian_filter, median_filter
from scipy.signal import convolve2d
import seaborn as sns
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr



def show_img(*images, titles=None):

    num_images = len(images)
    
    
    fig, axes = plt.subplots(1, num_images, figsize=(5 * num_images, 5))
    
   
    if num_images == 1:
        axes = [axes]
    
    for i, (ax, img) in enumerate(zip(axes, images)):
        
        if isinstance(img, str):
            img = Image.open(img)
            ax.imshow(img)
        elif isinstance(img, np.ndarray):
            if img.ndim == 2:  
                ax.imshow(img, cmap='gray')
            else:  # Color NumPy array
                ax.imshow(img)
        elif isinstance(img, Image.Image):
           
            img_array = np.asarray(img)

            
            if img_array.ndim == 2: 
                ax.imshow(img_array, cmap='gray')
            elif img_array.ndim == 3 and img_array.shape[2] == 3:  
                ax.imshow(img_array)
            else:
                raise ValueError("Unsupported image format.")
        
        ax.axis('off')
        
        
        if titles and i < len(titles):
            ax.set_title(titles[i], fontsize=12)
    
   
    plt.tight_layout(rect=[0, 0, 1, 0.95])  
    
    plt.show()




def convergence(energy_value):
   
    
    sns.set_theme(style="whitegrid", palette="muted")
    
    plt.figure(figsize=(8, 6))
 
    plt.plot(energy_value, label="Energy")
    plt.yscale("log") 
    plt.title("Energy over Iterations", fontsize=16, fontweight="bold")
    plt.xlabel("Iteration", fontsize=14)
    plt.ylabel("Energy", fontsize=14)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)  
    plt.legend(fontsize=12) 
    plt.tight_layout() 
    plt.show()



def generate_gaussian_kernel(kernel_size, sigma_blur):
    
    x = np.linspace(-kernel_size // 2, kernel_size // 2, kernel_size)
    y = np.linspace(-kernel_size // 2, kernel_size // 2, kernel_size)
    x, y = np.meshgrid(x, y)
    kernel = np.exp(-(x**2 + y**2) / (2 * sigma_blur**2))
    kernel /= kernel.sum()
    return kernel



def gray_image(image_path):
    """Load an image and convert it to grayscale."""
    img = Image.open(image_path).convert('L') 
    img = np.asarray(img, dtype=np.float32) / 255.0
    return img

def RGB_image(image_path):
    """Load an image and convert it to grayscale."""
    img = Image.open(image_path)
    img = np.asarray(img, dtype=np.float32) / 255.0
    return img




def median_denoising(image, filter_size=3):
    return median_filter(image, size=filter_size)




def safe_psnr(image_true, image_test, data_range=None):
    mse = np.mean((image_true - image_test) ** 2)
    if mse == 0:
        return float('inf')
    return psnr(image_true, image_test, data_range=data_range)



def degradation_operator(img, kernel):
    
    if img.ndim == 2:  
        return convolve2d(img, kernel, mode='same', boundary='symm')
    elif img.ndim == 3: 
        degraded_img = np.zeros_like(img)
        for c in range(3): 
            degraded_img[:, :, c] = convolve2d(img[:, :, c], kernel, mode='same', boundary='symm')
        return degraded_img
    else:
        raise ValueError("Input image must be either 2D (grayscale) or 3D (RGB).")
    

def H_fun(img, kernel):
    
    if img.ndim == 2: 
        return convolve2d(img, kernel, mode='same', boundary='symm')
    elif img.ndim == 3: 
        degraded_img = np.zeros_like(img)
        for c in range(3): 
            degraded_img[:, :, c] = convolve2d(img[:, :, c], kernel, mode='same', boundary='symm')
        return degraded_img
    else:
        raise ValueError("Input image must be either 2D (grayscale) or 3D (RGB).")
    

def HT_fun(img, kernel):
   
    if img.ndim == 2:  
        return convolve2d(img, kernel, mode='same', boundary='symm')
    elif img.ndim == 3:  
        degraded_img = np.zeros_like(img)
        for c in range(3): 
            degraded_img[:, :, c] = convolve2d(img[:, :, c], kernel, mode='same', boundary='symm')
        return degraded_img
    else:
        raise ValueError("Input image must be either 2D (grayscale) or 3D (RGB).")
    


def gaussian_denoising(image, sigma):

    if image.ndim == 3: 
        return np.stack([gaussian_denoising(image[..., c], sigma) for c in range(image.shape[2])], axis=2)
    else:  
        return gaussian_filter(image, sigma)