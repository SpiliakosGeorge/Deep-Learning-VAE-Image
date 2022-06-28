import os

import matplotlib.pyplot as plt
from VAE import *
from FashionDataloader import *
import torch

#TODO remove comments
#from Spectrogram import *
#import soundfile as sf
#from Dataset import *
#from Dataloader import *
#import torch.nn as nn




FILE = 'feedforwardnet.pth'
model = VAE(latent_dim=10, dim1=28, dim2=28)
model.load_state_dict(torch.load(FILE))
model.eval()

train_dataloader, _ = get_FashionMNIST_dataloaders(batch_size=18)


L = []
for input, _ in train_dataloader:
    with torch.no_grad():
        latent = model.encoding_fn(input)
        latent.squeeze_(0)
        latent.squeeze_(0)
        L.append(latent)

#print(L[0].tolist()[1])
with torch.no_grad():
    new_image = model.decoder(torch.tensor(L[2].tolist()[1]).to('cpu'))
    new_image.squeeze_(0)
    print(new_image.size())
    plt.imshow(new_image[0, :, :].to('cpu').numpy(), cmap='binary')
    plt.show()



save_dir = './data/sampling/'
save_path = os.path.join(save_dir + "test.wav")
