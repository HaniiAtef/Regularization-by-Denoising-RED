import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, convolve , median_filter
from scipy.signal import convolve2d
from PIL import Image
from tqdm import tqdm
import importlib
import helper_functions
importlib.reload(helper_functions)
from helper_functions import RGB_image, gray_image, degradation_operator, gaussian_denoising, median_denoising, H_fun, HT_fun, convergence, generate_gaussian_kernel, generate_gaussian_kernel2, calculate_psnr3
from skimage.metrics import peak_signal_noise_ratio as psnr
from Fixed_Point_RGB import fixed_point_red_all
import tifffile



def show_img(*images, titles=None):
    """
    Show multiple images side by side using Matplotlib.
    - images: List of images to display.
    - titles: List of titles corresponding to the images.
    """
    num_images = len(images)
    
    # Create subplots with one row and columns equal to the number of images
    fig, axes = plt.subplots(1, num_images, figsize=(5 * num_images, 5))
    
    # Ensure axes is always iterable (wrap in a list if only one axis)
    if num_images == 1:
        axes = [axes]
    
    for i, (ax, img) in enumerate(zip(axes, images)):
        # Check if the input is a NumPy array
        if isinstance(img, np.ndarray):
            # Grayscale NumPy array
            if img.ndim == 2:
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
    
    return fig



def normalize_image_for_streamlit(image):
    """
    Normalize image values to a format compatible with Streamlit's st.image().
    - Floating-point images will be normalized to [0.0, 1.0].
    - Integer images will be normalized to [0, 255].
    """
    # Replace NaNs with 0 (if any)
    image = np.nan_to_num(image)
    
    # Floating-point images: normalize to [0.0, 1.0]
    if np.issubdtype(image.dtype, np.floating):
        if image.max() > 1.0 or image.min() < 0.0:
            image = (image - image.min()) / (image.max() - image.min())  # Rescale to [0.0, 1.0]
    
    # Integer images: normalize to [0, 255] and convert to uint8
    elif np.issubdtype(image.dtype, np.integer):
        if image.max() > 255 or image.min() < 0:
            image = (image - image.min()) / (image.max() - image.min()) * 255.0
        image = image.astype(np.uint8)
    
    return image




# Streamlit app title and description
st.title("Image Denoising with Regularization Algorithms")

st.markdown(
    """
    This app performs image denoising using various regularization algorithms.
    Upload an image, select an algorithm, adjust parameters, and view results such as the restored image, PSNR values, and energy convergence.
    """
)

# Upload an image
uploaded_file = st.file_uploader("Upload an image (PNG/JPG/TIFF)", type=["png", "jpg", "jpeg", "tif", "tiff"])

if uploaded_file is not None:
    # Convert uploaded file to grayscale using gray_image
    original_img = gray_image(uploaded_file)

    # Display the grayscale image
    st.write("### Uploaded Grayscale Image:")
    st.image(normalize_image_for_streamlit(original_img), caption="Original Grayscale Image", width=300)

    # Sidebar: Algorithm Selection
    st.sidebar.header("Algorithm Selection")
    algorithm = st.sidebar.selectbox(
        "Choose a Denoising Algorithm:",
        options=["Fixed Point", "ADMM", "Steepest Descent"]
    )

    # Sidebar: Parameters for Degradation
    st.sidebar.header("Degradation Parameters")
    kernel_size = st.sidebar.slider("Gaussian Kernel Size", 3, 15, 9, step=2)
    sigma_blur = st.sidebar.slider("Kernel Sigma (Blur)", 1.0, 20.0, 10.0)
    noise_std = st.sidebar.slider("Noise Standard Deviation", 0.0001, 0.01, 0.001, step=0.0001)

    # Sidebar: Parameters for the Selected Algorithm
    st.sidebar.header(f"{algorithm} Parameters")
    if algorithm == "Fixed Point":
        sigma = st.sidebar.slider("Noise Std (σ)", 0.001, 0.1, 0.01, step=0.001)
        lambda_ = st.sidebar.slider("Regularization Parameter (λ)", 0.001, 0.1, 0.01, step=0.001)
        N = st.sidebar.slider("Outer Iterations (N)", 10, 200, 100, step=10)
        m = st.sidebar.slider("Inner Iterations (m)", 1, 20, 5, step=1)
        sigma_denoise = st.sidebar.slider("Denoising Noise Level (σ_f)", 0.0001, 0.01, 0.001, step=0.0001)
    elif algorithm == "ADMM":
        N = st.sidebar.slider("Outer Iterations (N)", 10, 200, 70, step=10)
        m1 = st.sidebar.slider("Inner Iterations m1", 1, 50, 10, step=1)
        m2 = st.sidebar.slider("Inner Iterations m2", 1, 10, 1, step=1)
        lambda_ = st.sidebar.slider("Regularization Parameter (λ)", 0.001, 0.1, 0.01, step=0.001)
        beta = st.sidebar.slider("Penalty Parameter (β)", 0.001, 0.1, 0.05, step=0.001)
        sigma = st.sidebar.slider("Noise Std (σ)", 0.0001, 0.01, 0.001, step=0.0001)
        denoising_sigma = st.sidebar.slider("Denoising Noise Level (σ_f)", 0.0001, 0.01, 0.001, step=0.0001)
        tol = st.sidebar.slider("Tolerance (tol)", 1e-4, 1e-1, 1e-2, step=1e-3)
    elif algorithm == "Steepest Descent":
        sigma = st.sidebar.slider("Noise Std (σ)", 0.001, 0.1, 0.01, step=0.001)
        lambda_ = st.sidebar.slider("Regularization Parameter (λ)", 0.001, 0.1, 0.01, step=0.001)
        num_iterations = st.sidebar.slider("Number of Iterations", 100, 1000, 500, step=100)
        sigma_denoise = st.sidebar.slider("Denoising Noise Level (σ_f)", 0.0001, 0.01, 0.001, step=0.0001)
        tol = st.sidebar.slider("Tolerance (tol)", 1e-6, 1e-3, 1e-6, step=1e-6)

    # Generate kernel and degrade image
    kernel = generate_gaussian_kernel(kernel_size=kernel_size, sigma_blur=sigma_blur)
    degraded_img = degradation_operator(original_img, kernel)

    # Add noise
    noisy_img = degraded_img + noise_std * np.random.randn(*degraded_img.shape)

    # Display noisy image
    st.write("### Noisy Image:")
    st.image(normalize_image_for_streamlit(noisy_img), caption="Noisy Image", width=300)

    # Run the selected algorithm
    if st.button("Run Denoising Algorithm"):
        st.write(f"### Running the {algorithm} Algorithm...")
        with st.spinner("Processing..."):
            if algorithm == "Fixed Point":
                # Run Fixed Point algorithm
                restored_img, energy_values = fixed_point_red_all(
                    y=noisy_img,
                    kernel=kernel,
                    sigma=sigma,
                    sigma_f=sigma_denoise,
                    lambda_=lambda_,
                    N=N,
                    m=m
                )
            elif algorithm == "ADMM":
                # Run ADMM algorithm
                noisy_img = np.asarray(noisy_img)  # Ensure NumPy array format
                from ADMM_RGB import RED_ADMM_Solver_all
                red = RED_ADMM_Solver_all(
                    noisy_img, kernel, denoising_sigma, lambda_, beta, sigma, N, m1, m2, tol
                )
                restored_img, energy_values = red.run()
            elif algorithm == "Steepest Descent":
                # Run Steepest Descent algorithm
                from Steepest_Descent_RGB import steepest_descent_red_all
                restored_img, energy_values = steepest_descent_red_all(
                    y=noisy_img,
                    kernel=kernel,
                    sigma=sigma,
                    sigma_f=sigma_denoise,
                    lambda_=lambda_,
                    num_iterations=num_iterations,
                    tol=tol
                )

        # Display restored image
        st.write("### Restored Image:")
        st.image(normalize_image_for_streamlit(restored_img), caption=f"Restored Image Using {algorithm}", width=300)

        # Plot energy convergence
        st.write("### Energy Convergence:")
        fig, ax = plt.subplots()
        ax.plot(energy_values)
        ax.set_title("Energy Convergence")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Energy")
        st.pyplot(fig)

        # Show PSNR values
        st.write("### PSNR Values:")
        original_psnr = psnr(original_img, original_img)
        noisy_psnr = psnr(original_img, noisy_img)
        restored_psnr = psnr(original_img, restored_img)
        st.write(f"- **Original Image PSNR:** {original_psnr:.2f} dB")
        st.write(f"- **Noisy Image PSNR:** {noisy_psnr:.2f} dB")
        st.write(f"- **Restored Image PSNR:** {restored_psnr:.2f} dB")

        # Display images side by side using show_img
        st.write("### Image Comparison:")
        fig = show_img(
            normalize_image_for_streamlit(original_img),
            normalize_image_for_streamlit(noisy_img),
            normalize_image_for_streamlit(restored_img),
            titles=["Original Image", "Noisy Image", f"Restored Image Using {algorithm}"]
        )
        st.pyplot(fig)