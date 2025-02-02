from tqdm import tqdm
import numpy as np
from helper_functions.helper_functions import median_denoising, H_fun, HT_fun, gaussian_denoising




def accelerated_steepest_descent_red_all(y, kernel, sigma, sigma_f, lambda_, num_iterations, tol=1e-6, beta=0.9):
    
    
    x = y.copy()  
    v = np.zeros_like(x)  
    mu = 2 / (1 / sigma**2 + lambda_)  
    energies = []

    for k in tqdm(range(num_iterations)):
        
        if x.ndim == 2: 
            denoised_image = gaussian_denoising(x, sigma_f)
        elif x.ndim == 3:  
            denoised_image = np.zeros_like(x)
            for c in range(3):  
                denoised_image[:, :, c] = gaussian_denoising(x[:, :, c], sigma_f)
        else:
            raise ValueError("Input image must be 2D (grayscale) or 3D (RGB).")

        
        if x.ndim == 2:  
            gradient = (
                HT_fun(H_fun(x, kernel) - y, kernel) / sigma**2
                + lambda_ * (x - denoised_image)
            )
        elif x.ndim == 3: 
            gradient = np.zeros_like(x)
            for c in range(3):
                gradient[:, :, c] = (
                    HT_fun(H_fun(x[:, :, c], kernel) - y[:, :, c], kernel) / sigma**2
                    + lambda_ * (x[:, :, c] - denoised_image[:, :, c])
                )

        
        v = beta * v + gradient

        
        x -= mu * v

        
        energy = compute_energy(x, y, kernel, sigma, lambda_, sigma_f)
        energies.append(energy)

        # Check for convergence based on tolerance
        if k > 0 and abs(energies[-1] - energies[-2]) < tol:
            print(f"Converged at iteration {k + 1} with tolerance {tol}.")
            break

    return x, energies



def compute_energy(x, y, kernel, sigma, lambda_, sigma_f):
    
    
    likelihood_term = 0
    prior_term = 0

    if x.ndim == 2:  
        
        Hx = H_fun(x, kernel)

        
        likelihood_term = np.sum((Hx - y)**2) / (2 * sigma**2)

        
        f_x = gaussian_denoising(x, sigma_f)
        prior_term = (lambda_ / 2) * np.sum(x * (x - f_x))

    elif x.ndim == 3: 
        for c in range(3):  
            
            Hx = H_fun(x[:, :, c], kernel)

            
            likelihood_term += np.sum((Hx - y[:, :, c])**2) / (2 * sigma**2)

            
            f_x = gaussian_denoising(x[:, :, c], sigma_f)
            prior_term += (lambda_ / 2) * np.sum(x[:, :, c] * (x[:, :, c] - f_x))

    
    energy = likelihood_term + prior_term
    return energy