from tqdm import tqdm
import numpy as np
from helper_functions.helper_functions import H_fun, HT_fun
from CNN.Image_Denoiser import denoise_image4
import torch


def steepest_descent_red_NN(y, kernel, sigma,lambda_, num_iterations, model,tol=1e-6):

    
    x = y.copy()  
    mu = 2 / (1 / sigma**2 + lambda_) 
    energies = []
    USE_CUDA = True
    device = torch.device('cuda') if (torch.cuda.is_available() and USE_CUDA) else torch.device('cpu')
    for k in tqdm(range(num_iterations)):
        if str(type(model)) == "<class 'models.dncnn.DnCNN'>":
        
            tensor = torch.from_numpy(x).unsqueeze(0).unsqueeze(0).to(device).float()
            denoised_tensor = model(tensor)
            denoised_tensor = denoised_tensor.cpu().data.squeeze(0).squeeze(0).numpy()
            denoised_image = np.clip(denoised_tensor, 0, 1)
            
        else:
            denoised_image = denoise_image4(model, x)

       
        gradient = HT_fun(H_fun(x, kernel) - y, kernel) / sigma ** 2 + lambda_ * (x - denoised_image)
        
        x -= mu * gradient
        
        
        energy = compute_energy(x, y, kernel, sigma, lambda_, model)
        energies.append(energy)

       
        if k > 0 and abs(energies[-1] - energies[-2]) < tol:
            print(f"Converged at iteration {k + 1} with tolerance {tol}.")
            break

    return x, energies

def compute_energy(x, y, kernel, sigma, lambda_, model):
    
    
    Hx = H_fun(x, kernel)

    
    likelihood_term = np.sum((Hx - y)**2) / (2 * sigma**2)

    
    if str(type(model)) == "<class 'models.dncnn.DnCNN'>":
        USE_CUDA = True
        device = torch.device('cuda') if (torch.cuda.is_available() and USE_CUDA) else torch.device('cpu')
        tensor = torch.from_numpy(x).unsqueeze(0).unsqueeze(0).to(device).float()
        denoised_tensor = model(tensor)
        denoised_tensor = denoised_tensor.cpu().data.squeeze(0).squeeze(0).numpy()
        f_x = np.clip(denoised_tensor, 0, 1)
    else:
        f_x = denoise_image4(model, x)
    prior_term = (lambda_ / 2) * np.sum(x * (x - f_x))

    
    energy = likelihood_term + prior_term
    return energy