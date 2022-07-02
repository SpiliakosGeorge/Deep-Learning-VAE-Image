# Deep-Learning-VAE-Image

This project showcases two uses of Variational Autoencoders.
Firstly we use a VAE model trained for denoising images with Gaussian noise and those images are used for training a VAE model for image generation purposes.

## Contents

1. Autoencoder.py
	- Has the model architecture, forward and reconstruct methods
2. FashionDataloader.py
	- Has one method that returns train and test dataloaders of FashionMNIST dataset.
3. Plot.py
	- Has methods that are used for plotting the images after the denoising and generation inferences.
4. Train_DAE.py
	- Has the training pipeline of the denoising VAE as well as the parameters of training and saves the model.
5. Train_GAE.py
	- Has the training pipeline of the generative VAE as well as the parameters of training and saves the model.
6. typos.py
	- Consists of methods used to calculate the model parameters according to `(input_dimension + 2 * padding)/stride + 1` formula.
7. VAE.py
	-
8. Visual.ipynb
	- Is the presentation of the results of the inferences.

## Requirements

- Python 3.9.x
- Jupyter Notebook server

## Installation

Use the requirements.txt to install the packages needed.

`$ pip install -r requirements.txt`

## Usage

### Model training 

(Repository already has models with various latent dimensions trained which are presented on the Visual.ipynb notebook)

If you need to train models with different parameters:

- Train_DAE: 
	1. Set the parameters required for training in the main() method.
	2. Set the name of the model to be saved `denoising_model_name.pth`
	2. Execute main() method.
- Train_GAE:
 	1. Set the parameters required for training in the main() method.
 	2. Set the name of the model to be saved `generative_model_name.pth`
 	2. Execute main() method.

### Visualization

- Visual.ipynb already is configured with presenting our experimentation results during the notebook execution.

- In case you need to present a different model you can use the `FILE` constant parameter to load the desired model.




