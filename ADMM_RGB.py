import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import numpy as np
from scipy.ndimage import gaussian_filter, convolve , median_filter
from scipy.signal import convolve2d
from helper_functions import degradation_operator, median_denoising, H_fun, HT_fun, gaussian_denoising



class RED_ADMM_Solver_all:
    """
    Regularization by Denoising (RED) using ADMM, modified for RGB images.
    """
    def __init__(self, y, kernel, denoising_sigma, lambd, beta, sigma, N, m1, m2, tol=1e-5):
        """
        Initialize the RED_ADMM_Solver with the given parameters.

        Args:
        - y: Observed image (degraded). Can be grayscale or RGB.
        - kernel: Degradation kernel (e.g., blur kernel).
        - denoising_sigma: Noise level for the denoising engine.
        - lambd: Regularization parameter.
        - beta: ADMM penalty parameter.
        - sigma: Noise level in the likelihood term.
        - N: Number of outer iterations.
        - m1: Number of inner iterations for x-update.
        - m2: Number of inner iterations for v-update.
        - tol: Tolerance for convergence check.
        """
        self.y = y
        self.kernel = kernel
        self.denoising_sigma = denoising_sigma
        self.lambd = lambd
        self.beta = beta
        self.sigma = sigma
        self.N = N
        self.m1 = m1
        self.m2 = m2
        self.tol = tol

        # Initialize variables
        self.x = y.copy()  # Starting point is the observed image
        self.v = y.copy()
        self.u = np.zeros_like(y)  # Lagrange multiplier
        self.energy = np.zeros(N)  # Energy tracking

    def compute_energy(self):
        """
        Compute the energy functional for the current state.
        
        Returns:
        - Energy value.
        """
        # Data fidelity term: ||H(x) - y||^2 / (2 * sigma^2)
        fidelity = np.sum((H_fun(self.x, self.kernel) - self.y) ** 2) / (2 * self.sigma ** 2)

        # Regularization term: lambda * ||v - denoising_func(v)||^2
        denoised_v = gaussian_denoising(self.v, self.denoising_sigma)
        regularization = self.lambd * np.sum((self.v - denoised_v) ** 2)

        # ADMM penalty term: beta / 2 * ||x - v + u||^2
        admm_penalty = self.beta / 2 * np.sum((self.x - self.v + self.u) ** 2)

        # Total energy
        return fidelity + regularization + admm_penalty

    def solve_for_x(self):
        """
        Solve for x by minimizing the augmented Lagrangian using gradient descent.
        """
        z_star = self.v - self.u  # z* = v_k-1 - u_k-1

        for _ in range(self.m1):
            # Gradient of the likelihood term
            grad_likelihood = HT_fun(H_fun(self.x, self.kernel) - self.y, self.kernel) / self.sigma ** 2

            # Combine with the ADMM penalty term
            e = grad_likelihood + self.beta * (self.x - z_star)

            # Compute the residual for the gradient descent step (to determine step size)
            r = HT_fun(H_fun(e, self.kernel), self.kernel) / self.sigma ** 2 + self.beta * e

            # Step size: Line search
            step_size = np.dot(e.ravel(), e.ravel()) / np.dot(e.ravel(), r.ravel())

            # Gradient descent update
            self.x -= step_size * e

            # Ensure pixel values are in [0, 1]
            self.x = np.clip(self.x, 0, 1)

    def solve_for_v(self):
        """
        Solve for v by minimizing the augmented Lagrangian using the denoising engine.
        """
        z_star = self.x + self.u  # z* = x_k + u_k-1

        for _ in range(self.m2):
            # Apply denoising function
            denoised_image = gaussian_denoising(self.v, self.denoising_sigma)

            # Update v
            self.v = (self.lambd * denoised_image + self.beta * z_star) / (self.lambd + self.beta)

    def update_u(self):
        """
        Update the Lagrange multiplier u.
        """
        self.u += (self.x - self.v)

    def run(self):
        """
        Run the RED-ADMM algorithm for N iterations.

        Returns:
        - Reconstructed image after N iterations.
        - Energy values over iterations.
        """
        for k in tqdm(range(self.N)):
            # Part 1: Solve for x
            self.solve_for_x()
            
            # Part 2: Solve for v
            self.solve_for_v()
            
            # Part 3: Update u
            self.update_u()
            
            # Compute the energy for this iteration
            self.energy[k] = self.compute_energy()
            
            # Check for convergence
            if k > 0 and abs(self.energy[k] - self.energy[k - 1]) < self.tol:
                print(f"Convergence achieved after {k + 1} iterations.")
                self.energy = self.energy[:k + 1]  # Trim the energy array to the actual number of iterations
                break

        return self.x, self.energy