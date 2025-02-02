from tqdm import tqdm
import numpy as np
from helper_functions.helper_functions import degradation_operator, H_fun, HT_fun
from CNN.Image_Denoiser import denoise_image4

class RED_ADMM_Solver_NN:
    

    def __init__(self, y, kernel, lambd, beta, sigma, N, m1, m2, model, tol=1e-5):
        
        self.y = y
        self.kernel = kernel
        self.lambd = lambd
        self.beta = beta
        self.sigma = sigma
        self.N = N
        self.m1 = m1
        self.m2 = m2
        self.model = model
        self.tol = tol

        self.x = y.copy()  
        self.v = y.copy()
        self.u = np.zeros_like(y)  
        self.energy = np.zeros(N)  

    def compute_energy(self):
       
    
        fidelity = np.sum((degradation_operator(self.x, self.kernel) - self.y) ** 2) / (2 * self.sigma ** 2)

        
        denoised_v = denoise_image4(self.model, self.v)

        regularization = self.lambd * np.sum((self.v - denoised_v) ** 2)

        # ADMM penalty term: beta / 2 * ||x - v + u||^2
        admm_penalty = self.beta / 2 * np.sum((self.x - self.v + self.u) ** 2)

        # Total energy
        return fidelity + regularization + admm_penalty

    def solve_for_x(self):
        
        z_star = self.v - self.u 

        for _ in range(self.m1):
            
            grad_likelihood = HT_fun(H_fun(self.x, self.kernel) - self.y, self.kernel) / self.sigma ** 2

            
            e = grad_likelihood + self.beta * (self.x - z_star)

            
            r = HT_fun(H_fun(e, self.kernel), self.kernel) / self.sigma ** 2 + self.beta * e

            
            step_size = np.dot(e.ravel(), e.ravel()) / np.dot(e.ravel(), r.ravel())

            
            self.x -= step_size * e

            
            self.x = np.clip(self.x, 0, 1)

    def solve_for_v(self):
        
        z_star = self.x + self.u 

        for _ in range(self.m2):
           
            denoised_image = denoise_image4(self.model, self.v)

            
            self.v = (self.lambd * denoised_image + self.beta * z_star) / (self.lambd + self.beta)

    def update_u(self):
       
        self.u = (self.x - self.v)


    def run(self):
        
        for k in tqdm(range(self.N)):
            
            self.solve_for_x()
            
            
            self.solve_for_v()
            
           
            self.update_u()
            
           
            self.energy[k] = self.compute_energy()
            
            
            if k > 0 and abs(self.energy[k] - self.energy[k - 1]) < self.tol:
                print(f"Convergence achieved after {k + 1} iterations.")
                self.energy = self.energy[:k + 1] 
                break

        return self.x, self.energy