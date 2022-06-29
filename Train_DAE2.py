import torch
from Autoencoder import Autoencoder
from VAE import VAE
from torch import nn
from FashionDataloader import *
import torch.nn.functional as F




def add_noise(tensor, mean, std):
    return tensor + torch.randn(tensor.size()) * std + mean


def train_single_epoch(model, data_loader, loss_fn, optimiser,
reconstruction_term_weight = 1,
device = 'cpu'):
    for original, _ in data_loader:
    
        original = original.to(device)
        noisy_input = add_noise(original, 0, 0.2)
        



        encoded, z_mean, z_log_var, decoded = model(noisy_input)
        # print(model.final_linear.weight.grad)
        # calculate loss
        # total loss = reconstruction loss + KL divergence
        #kl_divergence = (0.5 * (z_mean**2 + 
        #                        torch.exp(z_log_var) - z_log_var - 1)).sum()
        kl_div = -0.5 * torch.sum(1 + z_log_var 
                                    - z_mean**2 
                                    - torch.exp(z_log_var), 
                                    axis=1) # sum over latent dimension

        batchsize = kl_div.size(0)
        kl_div = kl_div.mean() # average over batch dimension

        pixelwise = loss_fn(decoded, original, reduction='none')
        pixelwise = pixelwise.view(batchsize, -1).sum(axis=1) # sum over pixels
        pixelwise = pixelwise.mean() # average over batch dimension
        
        #loss = reconstruction_term_weight*pixelwise + kl_div
        loss = pixelwise + kl_div
        
        

        # backpropagate error and update weights
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    print(f"loss: {loss.item()}")



def train(model, data_loader, loss_fn, optimiser, epochs, device = 'cpu' ):
    for i in range(epochs):
        print(f"Epoch {i+1}")
        train_single_epoch(model, data_loader, loss_fn, optimiser, device)
        print("---------------------------")
    print("Finished training")

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using {device}")

    BATCH_SIZE = 18
    LATENT_DIM = 10
    DIM_1 = 28
    DIM_2 = 28
    LEARNING_RATE = 0.001
    EPOCHS = 10

    root = "./data"
 
    #train_dataloader = create_data_loader(md, BATCH_SIZE)
    
    train_dataloader = get_FashionMNIST_dataloaders(batch_size=BATCH_SIZE)[0]

    # construct model and assign it to device
    
    autoencoder = VAE(latent_dim=LATENT_DIM, dim1=DIM_1, dim2=DIM_2).to(device)
    print(autoencoder)

    # initialise loss funtion + optimiser
    loss_fn = F.mse_loss
    optimiser = torch.optim.Adam(autoencoder.parameters(),
                                 lr=LEARNING_RATE)

    # train model
    train(autoencoder, train_dataloader, loss_fn, optimiser, EPOCHS, device)
    

    # save model
    torch.save(autoencoder.state_dict(), "DAE.pth")
    #print("Trained autoencoder saved at feedforwardnet.pth")