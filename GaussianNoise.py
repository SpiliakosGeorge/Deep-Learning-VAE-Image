import torch
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
from FashionDataloader import *

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

def get_noisy_FashionMNIST_dataloaders(batch_size, mean, std):

    train_transform = transforms.Compose([transforms.ToTensor(), AddGaussianNoise(mean,std)])
    test_transform = transforms.Compose([transforms.ToTensor(), AddGaussianNoise(mean,std)])
    root = "./data"

    train_set = torchvision.datasets.FashionMNIST(root=root, download=True, transform=train_transform)
    test_set = torchvision.datasets.FashionMNIST(root=root, download=True, train=False, transform=test_transform)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size)

    return train_loader, test_loader

corrupted_images_dloader_train = get_noisy_FashionMNIST_dataloaders(18, 0, 0.2)[0]

def plot_dataloader(dataloader, index=0):

    train_features, train_labels = next(iter(dataloader))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")
    img = train_features[index].squeeze()
    label = train_labels[index]
    plt.imshow(img, cmap="gray")
    plt.show()
    print(f"Label: {label}")

plot_dataloader(get_FashionMNIST_dataloaders(18)[0], 1)
plot_dataloader(corrupted_images_dloader_train, 1)