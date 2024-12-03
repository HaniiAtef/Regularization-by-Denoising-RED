import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import numpy as np
from scipy.ndimage import gaussian_filter, convolve , median_filter
from scipy.signal import convolve2d
import seaborn as sns
from skimage.metrics import peak_signal_noise_ratio as psnr
import cv2
import numpy as np


def show_img(*images, titles=None):

    num_images = len(images)
    
    # Create subplots with one row and columns equal to the number of images
    fig, axes = plt.subplots(1, num_images, figsize=(5 * num_images, 5))
    
    # Ensure axes is always iterable (wrap in a list if only one axis)
    if num_images == 1:
        axes = [axes]
    
    for i, (ax, img) in enumerate(zip(axes, images)):
        # Check if the input is a path (str) or an image object
        if isinstance(img, str):
            img = Image.open(img)
            ax.imshow(img)
        elif isinstance(img, np.ndarray):
            if img.ndim == 2:  # Grayscale NumPy array
                ax.imshow(img, cmap='gray')
            else:  # Color NumPy array
                ax.imshow(img)
        elif isinstance(img, Image.Image):
            # Convert the image to a NumPy array to determine its channel count
            img_array = np.asarray(img)

            # Check if the image is grayscale or color
            if img_array.ndim == 2:  # Grayscale image
                ax.imshow(img_array, cmap='gray')
            elif img_array.ndim == 3 and img_array.shape[2] == 3:  # RGB color image
                ax.imshow(img_array)
            else:
                raise ValueError("Unsupported image format.")
        
        ax.axis('off')
        
        # Set title if provided
        if titles and i < len(titles):
            ax.set_title(titles[i], fontsize=12)
    
    # Adjust layout to make sure titles fit
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to provide space for titles
    
    plt.show()


def convergence(energy_value):

    sns.set_theme(style="whitegrid", palette="muted")

    # Plot the graph
    plt.plot(energy_value, label="Energy")
    plt.title("Energy over Iterations", fontsize=14, fontweight="bold")
    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("Energy", fontsize=12)
    plt.legend()  # Add a legend if necessary
    plt.tight_layout()  # Adjust layout for better spacing
    plt.show()


def generate_gaussian_kernel(kernel_size, sigma_blur):
    """
    Generates a 2D Gaussian kernel for image processing tasks.

    Parameters:
    ----------
    kernel_size : int
        The size of the Gaussian kernel (must be an odd number). The kernel will have dimensions
        `kernel_size x kernel_size`. Larger sizes result in a broader blur, incorporating more
        neighboring pixels.
    sigma_blur : float
        The standard deviation of the Gaussian distribution. Controls the "spread" of the weights:
        - Smaller values produce a sharply peaked kernel (localized blur).
        - Larger values create a broader kernel (diffuse blur).

    Returns:
    -------
    kernel : numpy.ndarray
        A 2D array of shape `(kernel_size, kernel_size)` representing the normalized Gaussian kernel,
        where the sum of all weights equals 1. This kernel can be used for convolution in image
        processing to apply a Gaussian blur or smoothing.

    Explanation:
    -----------
    - The function calculates the 2D Gaussian function:
      G(x, y) = exp(-(x^2 + y^2) / (2 * sigma_blur^2))
    - The kernel is then normalized to ensure its total weight equals 1, so it doesn't alter the
      overall brightness of the image.
    - This kernel is typically applied to images via convolution to smooth or blur the image,
      reducing noise while preserving structural details.

    """
    x = np.linspace(-kernel_size // 2, kernel_size // 2, kernel_size)
    y = np.linspace(-kernel_size // 2, kernel_size // 2, kernel_size)
    x, y = np.meshgrid(x, y)
    kernel = np.exp(-(x**2 + y**2) / (2 * sigma_blur**2))
    kernel /= kernel.sum()
    return kernel

def generate_gaussian_kernel2(sigma_blur):

    kernel_size = int(6 * sigma_blur + 1)
    ax = np.linspace(-(kernel_size // 2), kernel_size // 2, kernel_size)
    blur_kernel = np.exp(-0.5 * (ax / sigma_blur)**2)
    blur_kernel = np.outer(blur_kernel, blur_kernel)
    blur_kernel /= np.sum(blur_kernel)
    return blur_kernel



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



def calculate_psnr2(img, recon, max_value=255):
    """
    This function calculates the Peak Signal-to-Noise Ratio (PSNR) between two images: the original image (img) and the reconstructed image (recon).
    PSNR is a metric commonly used to evaluate the quality of reconstructed images, providing a measure of how much noise is
    present relative to the maximum possible signal strength. Higher PSNR values indicate better quality reconstruction.

    """
    # Check image dimensions
    if img.shape != recon.shape:
        raise ValueError("Images must have the same dimensions")

    # Calculate mean squared error (MSE)
    mse = np.mean((img - recon) ** 2)

    # Calculate PSNR
    if mse == 0:
        return float('inf')
    psnr = 10 * np.log10((max_value ** 2) / mse)

    return psnr


# def degradation_operator(img, kernel):
#     """Apply a linear degradation operator (e.g., convolution with a blur kernel)."""
#     return convolve2d(img, kernel, mode='same', boundary='symm')


# def gaussian_denoising(img, sigma):
#     """Simple Gaussian denoising (using FFT) to simulate a denoising engine."""
#     return gaussian_filter(img, sigma)

def median_denoising(image, filter_size=3):
    """
    Apply median filtering to remove salt-and-pepper noise.
    
    Args:
    - image: Input noisy image.
    - filter_size: Size of the median filter (must be odd).
    
    Returns:
    - Denoised image.
    """
    return median_filter(image, size=filter_size)

# def H_fun(img, kernel):
#     return convolve2d(img, kernel, mode='same', boundary='symm')
    
# def HT_fun(img, kernel):
#     return convolve2d(img, kernel, mode='same', boundary='symm')


def add_salt_and_pepper_noise(image, amount=0.05, salt_vs_pepper=0.5):
    noisy_image = np.copy(image)
    num_salt = np.ceil(amount * image.size * salt_vs_pepper)
    num_pepper = np.ceil(amount * image.size * (1.0 - salt_vs_pepper))
    coords = [np.random.randint(0, i, int(num_salt)) for i in image.shape]
    noisy_image[coords[0], coords[1]] = 1
    coords = [np.random.randint(0, i, int(num_pepper)) for i in image.shape]
    noisy_image[coords[0], coords[1]] = 0
    return noisy_image




def calculate_psnr(original, reconstructed):
    mse = np.mean((original - reconstructed) ** 2)
    if mse == 0:  # Perfect match
        return float('inf')
    max_pixel = 255.0  # Assuming 8-bit images
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr


def calculate_psnr3(original, reconstructed):
    mse = np.mean((original - reconstructed) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0  # Assuming 8-bit images
    return 20 * np.log10(max_pixel / np.sqrt(mse))



def degradation_operator(img, kernel):
    """
    Apply a linear degradation operator (e.g., convolution with a blur kernel) to an RGB or grayscale image.
    
    Args:
        img (numpy.ndarray): Input image, either grayscale (H x W) or RGB (H x W x 3).
        kernel (numpy.ndarray): 2D degradation kernel (e.g., Gaussian blur kernel).
    
    Returns:
        numpy.ndarray: Degraded image, same shape as the input.
    """
    if img.ndim == 2:  # Grayscale image
        return convolve2d(img, kernel, mode='same', boundary='symm')
    elif img.ndim == 3:  # RGB image
        degraded_img = np.zeros_like(img)
        for c in range(3):  # Apply convolution to each channel separately
            degraded_img[:, :, c] = convolve2d(img[:, :, c], kernel, mode='same', boundary='symm')
        return degraded_img
    else:
        raise ValueError("Input image must be either 2D (grayscale) or 3D (RGB).")
    

def H_fun(img, kernel):
    """
    Apply a linear degradation operator (e.g., convolution with a blur kernel) to an RGB or grayscale image.
    
    Args:
        img (numpy.ndarray): Input image, either grayscale (H x W) or RGB (H x W x 3).
        kernel (numpy.ndarray): 2D degradation kernel (e.g., Gaussian blur kernel).
    
    Returns:
        numpy.ndarray: Degraded image, same shape as the input.
    """
    if img.ndim == 2:  # Grayscale image
        return convolve2d(img, kernel, mode='same', boundary='symm')
    elif img.ndim == 3:  # RGB image
        degraded_img = np.zeros_like(img)
        for c in range(3):  # Apply convolution to each channel separately
            degraded_img[:, :, c] = convolve2d(img[:, :, c], kernel, mode='same', boundary='symm')
        return degraded_img
    else:
        raise ValueError("Input image must be either 2D (grayscale) or 3D (RGB).")
    

def HT_fun(img, kernel):
    """
    Apply a linear degradation operator (e.g., convolution with a blur kernel) to an RGB or grayscale image.
    
    Args:
        img (numpy.ndarray): Input image, either grayscale (H x W) or RGB (H x W x 3).
        kernel (numpy.ndarray): 2D degradation kernel (e.g., Gaussian blur kernel).
    
    Returns:
        numpy.ndarray: Degraded image, same shape as the input.
    """
    if img.ndim == 2:  # Grayscale image
        return convolve2d(img, kernel, mode='same', boundary='symm')
    elif img.ndim == 3:  # RGB image
        degraded_img = np.zeros_like(img)
        for c in range(3):  # Apply convolution to each channel separately
            degraded_img[:, :, c] = convolve2d(img[:, :, c], kernel, mode='same', boundary='symm')
        return degraded_img
    else:
        raise ValueError("Input image must be either 2D (grayscale) or 3D (RGB).")
    


def gaussian_denoising(image, sigma):
    """
    Apply Gaussian denoising to each channel.
    This is a placeholder function; replace with your preferred denoising algorithm.
    """
    if image.ndim == 3:  # RGB Image
        return np.stack([gaussian_denoising(image[..., c], sigma) for c in range(image.shape[2])], axis=2)
    else:  # Grayscale Image
        # Use a simple Gaussian filter here as an example
        return gaussian_filter(image, sigma)