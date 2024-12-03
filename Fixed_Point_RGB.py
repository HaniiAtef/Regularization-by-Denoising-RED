import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import numpy as np
from scipy.ndimage import gaussian_filter, convolve , median_filter
from scipy.signal import convolve2d
from helper_functions import degradation_operator, median_denoising, H_fun, HT_fun, gaussian_denoising, generate_gaussian_kernel2




def fixed_point_red_all(y, kernel, sigma, sigma_f, lambda_, N, m, tol=1e-6):
    """
    Minimize E(x) = 1/(2σ^2) * ||Hx - y||^2 + λ/2 * x^T(x - f(x)) for grayscale or RGB images.

    Parameters:
    - y: Observed image (input degraded image), can be grayscale (H, W) or RGB (H, W, 3).
    - kernel: Convolution kernel H, assumed to be 2D.
    - sigma: Standard deviation of noise for log-likelihood.
    - sigma_f: Noise level for the denoising engine.
    - lambda_: Regularization parameter.
    - N: Number of outer iterations.
    - m: Number of inner iterations for linear solve.
    - tol: Tolerance for convergence.

    Returns:
    - xb: Estimated image after optimization.
    - energies: List of energy values for each iteration.
    """
    # Check if the input is grayscale or RGB
    is_rgb = y.ndim == 3  # RGB images have 3 dimensions (H, W, 3)
    
    # Initialization
    xb = y.copy()  # Start with the observed image as initial estimate

    if is_rgb:
        # Precompute H^T * y for RGB
        HTy = np.stack([HT_fun(y[..., c], kernel) for c in range(3)], axis=-1)
        HTH = lambda img: np.stack([H_fun(HT_fun(img[..., c], kernel), kernel) for c in range(3)], axis=-1)
    else:
        # Precompute H^T * y for grayscale
        HTy = HT_fun(y, kernel)
        HTH = lambda img: H_fun(HT_fun(img, kernel), kernel)

    energies = []

    # Outer Loop
    for k in tqdm(range(N)):
        # Apply the denoising engine
        if is_rgb:
            x_tilde = np.stack([gaussian_denoising(xb[..., c], sigma_f) for c in range(3)], axis=-1)
        else:
            x_tilde = gaussian_denoising(xb, sigma_f)

        # Solve Az = b for A = 1/σ^2 * H^T H + λI and b = 1/σ^2 * H^T y + λx_tilde
        A = lambda img: (1 / sigma**2) * HTH(img) + lambda_ * img
        b = (1 / sigma**2) * HTy + lambda_ * x_tilde

        # Initialization for linear solver
        z = x_tilde.copy()

        # Inner Loop (Conjugate Gradient Descent for solving Az = b)
        for j in range(m):
            r = A(z) - b  # Residual
            e = A(r)  # Gradient of the residual
            mu = np.sum(r * r) / np.sum(r * e)  # Optimal step size
            z = z - mu * r  # Update z
            z = np.clip(z, 0, 255)  # Project z to the valid range [0, 255]

        # Update the outer loop solution
        xb = z
        
        # Compute energy at this iteration
        energy = compute_energy(xb, y, kernel, sigma, lambda_, sigma_f, is_rgb)
        energies.append(energy)

        # Check for convergence based on tolerance
        if k > 0 and abs(energies[-1] - energies[-2]) < tol:
            print(f"Converged at iteration {k + 1} with tolerance {tol}.")
            break

    return xb, energies


def compute_energy(x, y, kernel, sigma, lambda_, sigma_f, is_rgb):
    """
    Compute the energy function E(x) = 1/(2σ²) * ||H(x) - y||² + λ/2 * xᵀ(x - f(x))
    for grayscale or RGB images.

    Parameters:
    - x: Current estimate of the image, can be grayscale (H, W) or RGB (H, W, 3).
    - y: Observed image, can be grayscale (H, W) or RGB (H, W, 3).
    - kernel: Convolution kernel H (2D).
    - sigma: Standard deviation of noise for likelihood.
    - lambda_: Regularization parameter.
    - sigma_f: Noise level for the denoising engine.
    - is_rgb: Boolean indicating if the image is RGB.

    Returns:
    - Energy value for the current estimate.
    """
    if is_rgb:
        # Likelihood term: 1/(2σ²) * ||H(x) - y||² (sum over all channels)
        Hx = np.stack([H_fun(x[..., c], kernel) for c in range(3)], axis=-1)
        likelihood_term = (1 / (2 * sigma**2)) * np.sum((Hx - y)**2)

        # Regularization term: λ/2 * xᵀ(x - f(x)) (sum over all channels)
        f_x = np.stack([gaussian_denoising(x[..., c], sigma_f) for c in range(3)], axis=-1)
        regularization_term = (lambda_ / 2) * np.sum(x * (x - f_x))
    else:
        # Likelihood term: 1/(2σ²) * ||H(x) - y||²
        Hx = H_fun(x, kernel)
        likelihood_term = (1 / (2 * sigma**2)) * np.sum((Hx - y)**2)

        # Regularization term: λ/2 * xᵀ(x - f(x))
        f_x = gaussian_denoising(x, sigma_f)
        regularization_term = (lambda_ / 2) * np.sum(x * (x - f_x))

    # Total energy
    energy = likelihood_term + regularization_term

    return energy