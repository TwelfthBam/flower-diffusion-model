# flower-diffusion-model
A custom implementation of a Denoising Diffusion Probabilistic Model (DDPM) in PyTorch, trained on the Oxford Flowers 102 dataset to generate synthetic images.

## Technologies
* PyTorch
* Torchvision
* NumPy
* Matplotlib

## Key Features

Unlike high-level libraries that hide the logic, this repository implements the diffusion mathematics manually:

### 1. Custom U-Net Architecture
I implemented a U-Net from scratch with Residual Connections to predict noise in the image.
* Includes Sinusoidal Position Embeddings to help the model understand the current timestep $t$.
* Features custom Down-blocks (Conv2d) and Up-blocks (ConvTranspose2d).

### 2. Forward Diffusion Process
The model implements a linear Beta Schedule ($T=300$) to gradually add Gaussian noise to images until they become pure static.
* **Math Implemented:** $\beta_t$ (variance schedule) and $\alpha_t$ (retained signal).

### 3. Reverse Denoising (Generation)
The core capability is sampling from pure noise and iteratively denoising it to reconstruct a flower.
* Uses L1 Loss to calculate the difference between the actual noise and predicted noise.
* Includes a `sample_timestep` function to visualize the image reconstruction at any stage.

## Results

**Progress of Generation (Epoch 100):**
The image below is a demonstration of the models ability to generate images from pure noise into a flower structure.

![Diffusion Process](./generated_images/plot_epoch_100/.png)
*(Denoising process: From pure noise on the right to a generated flower on the left)*
