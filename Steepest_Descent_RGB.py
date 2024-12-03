import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import numpy as np
from scipy.ndimage import gaussian_filter, convolve , median_filter
from scipy.signal import convolve2d
from helper_functions import degradation_operator, median_denoising, H_fun, HT_fun, gaussian_denoising, generate_gaussian_kernel2



def steepest_descent_red_all(y, kernel, sigma, sigma_f, lambda_, num_iterations, tol=1e-6):
    """
    Steepest Descent (SD) implementation for Regularization by Denoising (RED) for both grayscale and RGB images.

    Objective: Minimize E(x) = (1 / (2σ²)) * ‖Hx − y‖² + (λ / 2) * xᵀ(x − f(x))

    Input: 
    - y: Input degraded image, either grayscale (H x W) or RGB (H x W x 3).
    - kernel: Blurring kernel
    - sigma: Noise level for the likelihood term
    - sigma_f: Noise level for the denoising function
    - lambda_: Regularization parameter
    - num_iterations: Number of iterations to perform
    - tol: Convergence tolerance

    Output:
    - x: Reconstructed image (same shape as `y`)
    - energies: List of energy values at each iteration
    """
    # Initialize variables
    x = y.copy()  # Start with the degraded image as the initial estimate
    mu = 2 / (1 / sigma**2 + lambda_)  # Step size
    energies = []

    for k in tqdm(range(num_iterations)):
        # Apply the denoising engine
        if x.ndim == 2:  # Grayscale image
            denoised_image = gaussian_denoising(x, sigma_f)
        elif x.ndim == 3:  # RGB image
            denoised_image = np.zeros_like(x)
            for c in range(3):  # Process each color channel
                denoised_image[:, :, c] = gaussian_denoising(x[:, :, c], sigma_f)
        else:
            raise ValueError("Input image must be 2D (grayscale) or 3D (RGB).")

        # Compute the gradient
        if x.ndim == 2:  # Grayscale image
            gradient = (
                HT_fun(H_fun(x, kernel) - y, kernel) / sigma**2
                + lambda_ * (x - denoised_image)
            )
        elif x.ndim == 3:  # RGB image
            gradient = np.zeros_like(x)
            for c in range(3):
                gradient[:, :, c] = (
                    HT_fun(H_fun(x[:, :, c], kernel) - y[:, :, c], kernel) / sigma**2
                    + lambda_ * (x[:, :, c] - denoised_image[:, :, c])
                )

        # Update the solution
        x -= mu * gradient

        # Compute energy at this iteration
        energy = compute_energy(x, y, kernel, sigma, lambda_, sigma_f)
        energies.append(energy)

        # Check for convergence based on tolerance
        if k > 0 and abs(energies[-1] - energies[-2]) < tol:
            print(f"Converged at iteration {k + 1} with tolerance {tol}.")
            break

    return x, energies


def compute_energy(x, y, kernel, sigma, lambda_, sigma_f):
    """
    Compute the energy function E(x) = (1 / (2σ²)) * ||Hx - y||² + (λ / 2) * xᵀ(x - f(x)) 
    for both grayscale and RGB images.
    """
    # Initialize energy components
    likelihood_term = 0
    prior_term = 0

    if x.ndim == 2:  # Grayscale image
        # Compute Hx (blurred version of x)
        Hx = H_fun(x, kernel)

        # Compute likelihood term: (1 / (2σ²)) * ||Hx - y||²
        likelihood_term = np.sum((Hx - y)**2) / (2 * sigma**2)

        # Compute prior term: (λ / 2) * xᵀ(x - f(x))
        f_x = gaussian_denoising(x, sigma_f)
        prior_term = (lambda_ / 2) * np.sum(x * (x - f_x))

    elif x.ndim == 3:  # RGB image
        for c in range(3):  # Process each color channel
            # Compute Hx (blurred version of x) for the current channel
            Hx = H_fun(x[:, :, c], kernel)

            # Compute likelihood term: (1 / (2σ²)) * ||Hx - y||²
            likelihood_term += np.sum((Hx - y[:, :, c])**2) / (2 * sigma**2)

            # Compute prior term: (λ / 2) * xᵀ(x - f(x))
            f_x = gaussian_denoising(x[:, :, c], sigma_f)
            prior_term += (lambda_ / 2) * np.sum(x[:, :, c] * (x[:, :, c] - f_x))

    # Total energy
    energy = likelihood_term + prior_term
    return energy