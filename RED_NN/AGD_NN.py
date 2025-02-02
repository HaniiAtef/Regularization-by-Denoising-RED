from tqdm import tqdm
import numpy as np
from helper_functions.helper_functions import H_fun, HT_fun
from CNN.Image_Denoiser import denoise_image4




def accelerated_steepest_descent_red_all_NN(y, kernel, sigma, lambda_, num_iterations, model, tol=1e-6, beta=0.9):
   
    
    x = y.copy() 
    v = np.zeros_like(x) 
    mu = 2 / (1 / sigma**2 + lambda_)  
    energies = []

    for k in tqdm(range(num_iterations)):
        
        
        denoised_image = denoise_image4(model, x)
        

       
        gradient = (
                HT_fun(H_fun(x, kernel) - y, kernel) / sigma**2
                + lambda_ * (x - denoised_image)
        )

        
        v = beta * v + gradient

       
        x -= mu * v

        
        energy = compute_energy(x, y, kernel, sigma, lambda_, model)
        energies.append(energy)

      
        if k > 0 and abs(energies[-1] - energies[-2]) < tol:
            print(f"Converged at iteration {k + 1} with tolerance {tol}.")
            break

    return x, energies


def agd_red_all_NN(y, kernel, sigma, lambda_, num_iterations, model, tol=1e-6, beta=0.9):
    
    
    x_prev = y.copy()  
    x_curr = y.copy()  
    mu = 2 / (1 / sigma**2 + lambda_)  
    energies = []

    for k in tqdm(range(num_iterations)):
       
        y_k = x_curr + beta * (x_curr - x_prev)

        
        denoised_image = denoise_image4(model, y_k)

       
        gradient = (
                HT_fun(H_fun(y_k, kernel) - y, kernel) / sigma**2
                + lambda_ * (y_k - denoised_image)
        )
        

       
        x_next = y_k - mu * gradient

        
        energy = compute_energy(x_next, y, kernel, sigma, lambda_, model)
        energies.append(energy)

       
        if k > 0 and abs(energies[-1] - energies[-2]) < tol:
            print(f"Converged at iteration {k + 1} with tolerance {tol}.")
            break

        
        x_prev = x_curr  
        x_curr = x_next  

    return x_curr, energies


def compute_energy(x, y, kernel, sigma, lambda_, model):
    
    likelihood_term = 0
    prior_term = 0

    if x.ndim == 2:  
       
        Hx = H_fun(x, kernel)

        
        likelihood_term = np.sum((Hx - y)**2) / (2 * sigma**2)

        
        f_x = denoise_image4(model, x)
        prior_term = (lambda_ / 2) * np.sum(x * (x - f_x))

    

   
    energy = likelihood_term + prior_term
    return energy