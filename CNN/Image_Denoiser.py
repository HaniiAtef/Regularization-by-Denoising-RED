import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np


class Denoiser(nn.Module):
    def __init__(self, use_bias=True):

        super(Denoiser, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=use_bias),  
            nn.ReLU(True),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 16, kernel_size=3, padding=1, bias=use_bias),
            nn.ReLU(True),
            nn.BatchNorm2d(16)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=use_bias),
            nn.ReLU(True),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 1, kernel_size=3, padding=1, bias=use_bias), 
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def denoise_image4(model: torch.nn.Module, noisy_image: np.ndarray, device: torch.device = None) -> np.ndarray:
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.to(device)
    model.eval()

    try:
        
        if len(noisy_image.shape) == 2:
            
            noisy_image = np.expand_dims(noisy_image, axis=-1)
        
       
        if noisy_image.dtype == np.uint8:
            
            image_tensor = torch.from_numpy(noisy_image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            scale_back = True
        elif noisy_image.dtype in [np.float32, np.float64]:
            image_tensor = torch.from_numpy(noisy_image).permute(2, 0, 1).unsqueeze(0).float()
            scale_back = False
        else:
            raise ValueError("Unsupported image dtype. Use uint8 for [0, 255] or float for [0, 1].")

        if image_tensor.shape[1] != 1:
            raise ValueError("Input image must be grayscale with a single channel.")

       
        image_tensor = image_tensor.to(device)

        with torch.no_grad():
            denoised_tensor = model(image_tensor)

      
        denoised_tensor = denoised_tensor.cpu().squeeze(0).permute(1, 2, 0).numpy()

     
        denoised_image = np.clip(denoised_tensor, 0, 1)

        if scale_back:
            
            denoised_image = (denoised_image * 255).astype(np.uint8)
        
       
        if denoised_image.shape[-1] == 1:
            denoised_image = denoised_image.squeeze(-1)

        return denoised_image

    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return None


def add_noise(images, noise_factor=0.2):
    noisy_images = images + noise_factor * torch.randn(*images.shape)
    return torch.clip(noisy_images, 0., 1.)



