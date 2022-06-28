import torch
import torchvision
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision import transforms
from FashionDataloader import *
from GaussianNoise import *


def get_FashionMNIST_dataloaders(batch_size):

    train_transform = transforms.Compose([transforms.ToTensor()])
    test_transform = transforms.Compose([transforms.ToTensor()])
    root = "./data"

    train_set = torchvision.datasets.FashionMNIST(root=root, download=True, transform=train_transform)
    test_set = torchvision.datasets.FashionMNIST(root=root, download=True, train=False, transform=test_transform)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size)

    return train_loader, test_loader

'''
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
'''
def plot_dataloader(dataloader, index=0):

    train_features, train_labels = next(iter(dataloader))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")
    img = train_features[index].squeeze()
    label = train_labels[index]
    plt.imshow(img, cmap="gray")
    plt.show()
    print(f"Label: {label}")




# TODO remove comments

"""
def output_label(label):
    output_mapping = {
        0: "T-shirt/Top",
        1: "Trouser",
        2: "Pullover",
        3: "Dress",
        4: "Coat",
        5: "Sandal",
        6: "Shirt",
        7: "Sneaker",
        8: "Bag",
        9: "Ankle Boot"
    }
    input = (label.item() if type(label) == torch.Tensor else label)
    return output_mapping[input]



a = next(iter(fashion_test_loader))
print(a[0].size())
print(a[1].size())

image, label = next(iter(fashion_test_loader))
plt.imshow(image.squeeze(), cmap="gray")

# print(label)
"""
