from tqdm import tqdm
import numpy as np
from helper_functions.helper_functions import median_denoising, H_fun, HT_fun, gaussian_denoising




def fixed_point_red_all(y, kernel, sigma, sigma_f, lambda_, N, m, tol=1e-6):
    
    
    is_rgb = y.ndim == 3  
    
  
    xb = y.copy()  

    if is_rgb:
       
        HTy = np.stack([HT_fun(y[..., c], kernel) for c in range(3)], axis=-1)
        HTH = lambda img: np.stack([H_fun(HT_fun(img[..., c], kernel), kernel) for c in range(3)], axis=-1)
    else:
        
        HTy = HT_fun(y, kernel)
        HTH = lambda img: H_fun(HT_fun(img, kernel), kernel)

    energies = []

  
    for k in tqdm(range(N)):
        
        if is_rgb:
            x_tilde = np.stack([gaussian_denoising(xb[..., c], sigma_f) for c in range(3)], axis=-1)
        else:
            x_tilde = gaussian_denoising(xb, sigma_f)

       
        A = lambda img: (1 / sigma**2) * HTH(img) + lambda_ * img
        b = (1 / sigma**2) * HTy + lambda_ * x_tilde

       
        z = x_tilde.copy()

        
        for j in range(m):
            r = A(z) - b  
            e = A(r)  
            mu = np.sum(r * r) / np.sum(r * e)  
            z = z - mu * r  
            z = np.clip(z, 0, 255)  

       
        xb = z
        
        
        energy = compute_energy(xb, y, kernel, sigma, lambda_, sigma_f, is_rgb)
        energies.append(energy)

        
        if k > 0 and abs(energies[-1] - energies[-2]) < tol:
            print(f"Converged at iteration {k + 1} with tolerance {tol}.")
            break

    return xb, energies


def compute_energy(x, y, kernel, sigma, lambda_, sigma_f, is_rgb):
    
    if is_rgb:
       
        Hx = np.stack([H_fun(x[..., c], kernel) for c in range(3)], axis=-1)
        likelihood_term = (1 / (2 * sigma**2)) * np.sum((Hx - y)**2)

        
        f_x = np.stack([gaussian_denoising(x[..., c], sigma_f) for c in range(3)], axis=-1)
        regularization_term = (lambda_ / 2) * np.sum(x * (x - f_x))
    else:
        
        Hx = H_fun(x, kernel)
        likelihood_term = (1 / (2 * sigma**2)) * np.sum((Hx - y)**2)

        
        f_x = gaussian_denoising(x, sigma_f)
        regularization_term = (lambda_ / 2) * np.sum(x * (x - f_x))

    
    energy = likelihood_term + regularization_term

    return energy