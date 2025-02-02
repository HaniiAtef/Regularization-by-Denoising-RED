# Regularization-by-Denoising-RED

Welcome to the **Regularization-by-Denoising-RED** repository! This project explores various algorithms for image denoising, leveraging both traditional methods and modern deep learning techniques. The core focus is on **Regularization by Denoising (RED)** and its integration with different denoising engines, including median filters and neural networks.

## Table of Contents

- [Project Structure](#project-structure)
  - [📁 CNN](#-cnn)
  - [📁 helper_functions](#-helper_functions)
  - [📁 Pre_trained](#-pre_trained)
  - [📁 RED](#-red)
  - [📁 RED_NN](#-red_nn)
- [Getting Started](#getting-started)
  - [Installation](#installation)
  - [Usage](#usage)
- [Project Overview](#project-overview)
  - [Concept and Objectives](#concept-and-objectives)
  - [Algorithms Implemented](#algorithms-implemented)
- [Results](#results)
- [Project Report](#project-report)

## Project Structure

The repository is organized into several folders, each dedicated to different components and functionalities of the project:

### 📁 CNN

**Description:**  
This folder contains the implementation of an image denoiser using Convolutional Neural Networks (CNNs). The denoiser is designed to learn and remove noise from images through training on paired datasets of noisy and clean images.

**Contents:**
- `image_denoiser.ipynb`  
  - **Purpose:**  
    A Jupyter Notebook for downloading datasets, training the CNN models, and creating two denoising models—one with bias and one without bias.
  - **Usage:**  
    - Run the notebook to download necessary data.
    - Train the CNN models.
    - Generate and save the models with and without bias.
  
- `image_denoiser.py`  
  - **Purpose:**  
    Contains functions to load the trained models, evaluate them, and denoise images. It ensures that the denoised images are in the correct format for further processing or visualization.
  - **Usage:**  
    - Import the `denoise_image` function to apply the trained models to new images.
  
- `denoiser_model_with_bias.pth`  
  - **Description:**  
    The trained CNN model that includes bias terms in its layers. This model has been trained to denoise images while retaining image details.
  
- `denoiser_model_without_bias.pth`  
  - **Description:**  
    The trained CNN model without bias terms. This model offers an alternative denoising approach, potentially affecting the model's performance and generalization.

### 📁 helper_functions

**Description:**  
Contains utility functions that are commonly used across different modules of the project. These functions facilitate tasks such as data preprocessing, evaluation metrics calculation, and visualization.

**Contents:**
- `helper_functions.py`  
  - **Purpose:**  
    A collection of helper functions for image processing and algorithm execution, including functions for loading images, calculating PSNR, and visualizing results.
  - **Usage:**  
    - Import necessary functions in your scripts or notebooks to streamline your workflow.

### 📁 Pre_trained

**Description:**  
This directory houses a cloned GitHub repository containing pre-trained image denoising models. These models can be utilized directly to perform denoising tasks without the need for additional training.

**Contents:**
- **Cloned Repository:**  
  - [Pre-trained Image Denoiser](https://github.com/LabForComputationalVision/bias_free_denoising/tree/master)
    - **Purpose:**  
      Provides pre-trained models that can be integrated into your denoising pipeline.
    - **Usage:**  
      - Use the pre-trained models to denoise images by following the instructions provided in the cloned repository.

### 📁 RED

**Description:**  
Implements the Regularization by Denoising (RED) framework using traditional denoising engines, specifically the median filter. This folder includes all necessary algorithms such as ADMM, Fixed Point, Gradient Descent (GD), and Accelerated Gradient Descent (AGD).

**Contents:**
- `ADMM_RGB.py`  
  - **Purpose:**  
    Implements the Alternating Directions Method of Multipliers algorithm for solving the RED optimization problem.
  
- `Fixed_Point_RGB.py`  
  - **Purpose:**  
    Implements the Fixed-Point iteration method used within the RED framework.
  
- `Steepest_Descent_RGB.py`  
  - **Purpose:**  
    Implements the Gradient Descent algorithm for optimizing the RED objective function.
  
- `AGD.py`  
  - **Purpose:**  
    Implements the Accelerated Gradient Descent algorithm, enhancing the convergence speed of the standard GD method.
  
  
- `main.ipynb`  
  - **Purpose:**  
    A Jupyter Notebook for testing and running the RED algorithms using the median filter as the denoising engine.
  - **Usage:**  
    - Open the notebook to execute and visualize the performance of different RED algorithms with the median filter.

### 📁 RED_NN

**Description:**  
Similar to the RED folder, but integrates trained neural network denoisers instead of traditional filters. This setup includes both custom-trained neural networks and pre-trained models, providing notebooks to test and run the algorithms.

**Contents:**
- `ADMM_NN.py`  
  - **Purpose:**  
    Implements the ADMM algorithm using a neural network-based denoiser within the RED framework.
  
- `Fixed_Point_NN.py`  
  - **Purpose:**  
    Implements the Fixed-Point iteration method with a neural network denoiser.
  
- `Steepest_Descent_NN.py`  
  - **Purpose:**  
    Implements the Gradient Descent algorithm using a neural network-based denoiser.
  
- `AGD_NN.py`  
  - **Purpose:**  
    Implements the Accelerated Gradient Descent algorithm with a neural network denoiser.
  
  
- `main_Trained_NN.ipynb`  
  - **Purpose:**  
    A Jupyter Notebook for testing and running RED algorithms with neural network denoisers engiens.
  - **Usage:**  
    - Open the notebook to execute and visualize the performance of different RED algorithms with neural network denoisers engines.


- `main_Pre_trained_NN.ipynb`  
  - **Purpose:**  
    A Jupyter Notebook for testing and running RED algorithms with Pre- Trained neural network denoisers engines.
  - **Usage:**  
    - Open the notebook to execute and visualize the performance of different RED algorithms with Pre - Trained neural network denoisers engines.


## Getting Started

### Installation

**Clone the Repository:**

```bash
git clone https://github.com/HaniiAtef/Regularization-by-Denoising-RED.git
```

### requirements.txt

**Description:**  
Specifies the Python dependencies required to run the project. Ensuring all dependencies are installed will facilitate the smooth execution of the scripts and notebooks.

**Usage:**
```bash
pip install -r requirements.txt
```


## Usage

### Using Traditional Denoisers (Median Filter):

1. **Navigate to the `RED` folder.**
2. **Open and run `main.ipynb` in Jupyter Notebook.**
3. **Follow the notebook instructions to test and execute the denoising algorithms using the median filter.**

### Using Neural Network Denoisers:

1. **Navigate to the `RED_NN` folder.**
2. **Open and run `main_Trained_NN.ipynb` in Jupyter Notebook to test and run the algorithms with the Trained Neural Network.**
3. **Open and run `main_Pre_trained_NN.ipynb` in Jupyter Notebook to test and run the algorithms with the Pre - Trained Neural Network.**

### Training Custom CNN Denoiser:

1. **Navigate to the `CNN` folder.**
2. **Open and run `image_denoiser.ipynb` in Jupyter Notebook to download datasets, train the CNN models (with and without bias), and generate the trained models.**
3. **Use `image_denoiser.py` to apply the trained models to new images for denoising.**

### Using Pre-trained Denoisers:

1. **Navigate to the `Pre_trained` folder.**
2. **Follow the instructions in the cloned repository to utilize the pre-trained models for denoising tasks.**

## Project Overview

### Concept and Objectives

The primary objective of this project is to explore and implement advanced image denoising techniques, with a focus on the **Regularization by Denoising (RED)** framework. RED leverages sophisticated denoising engines to impose regularization constraints during the image reconstruction process, enhancing the quality of the restored images.

### Algorithms Implemented

- **ADMM (Alternating Directions Method of Multipliers):**  
  An optimization algorithm that decomposes the problem into smaller subproblems, making it easier to handle complex regularization terms.

- **Fixed Point Iteration:**  
  A method that iteratively applies a function to converge to a fixed point, representing the denoised image.

- **Gradient Descent (GD):**  
  An optimization technique that iteratively moves towards the minimum of the objective function by following the negative gradient.

- **Accelerated Gradient Descent (AGD):**  
  An enhanced version of GD that uses momentum to accelerate convergence.

Each of these algorithms is tested with different denoising engines, namely traditional median filters and neural network-based denoisers, to evaluate their effectiveness and performance.

## Results

The project includes comprehensive simulations and evaluations of the implemented algorithms. Key metrics such as Peak Signal-to-Noise Ratio (PSNR) are used to assess the quality of denoised images. The results demonstrate the superiority of neural network denoisers in achieving higher PSNR values and better visual quality compared to traditional methods.

**Highlights:**

- **Median Filter:** Provides baseline performance with moderate improvements in image quality.
- **Trained Neural Networks:** Achieve significant enhancements in PSNR and visual clarity.
- **Pre-trained Networks:** Offer robust denoising capabilities with faster convergence times.

Detailed results, including convergence curves and visual comparisons, are documented in the project report and are available within the respective Jupyter Notebooks.

## Project Report

The project is accompanied by a comprehensive report `Rapport_Projet_RED__version_finale.pdf`, which delves into the theoretical foundations, methodology, and experimental evaluations of the RED framework and its applications in image denoising. Key sections include:

- **Introduction:** Overview of image denoising and the importance of regularization.
- **Methodology:** Detailed explanation of the RED framework and the implementation of various algorithms.
- **Experiments:** Description of simulation setups, datasets used, and evaluation metrics.
- **Results:** Presentation and analysis of the experimental findings.
