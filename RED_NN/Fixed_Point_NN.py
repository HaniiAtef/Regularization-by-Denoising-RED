from tqdm import tqdm
import numpy as np
from helper_functions.helper_functions import H_fun, HT_fun
from CNN.Image_Denoiser import denoise_image4



def fixed_point_red_NN(y, kernel, sigma, lambda_, N, m, model, tol=1e-6):
    
   
    xb = y.copy() 
    HTy = HT_fun(y, kernel)  
    HTH = lambda img: H_fun(HT_fun(img, kernel), kernel) 
    energies = []

    
    for k in tqdm(range(N)):
       
        x_tilde = denoise_image4(model, xb)

        
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
        
        energy = compute_energy(xb, y, kernel, sigma, lambda_, model)
        energies.append(energy)

        
        if k > 0 and abs(energies[-1] - energies[-2]) < tol:
            print(f"Converged at iteration {k + 1} with tolerance {tol}.")
            break

    return xb, energies



def compute_energy(x, y, kernel, sigma, lambda_, model):
    
    
    Hx = H_fun(x, kernel)
    likelihood_term = (1 / (2 * sigma**2)) * np.sum((Hx - y)**2)

    
    f_x = denoise_image4(model, x)
    regularization_term = (lambda_ / 2) * np.sum(x * (x - f_x))

   
    energy = likelihood_term + regularization_term

    return energy





