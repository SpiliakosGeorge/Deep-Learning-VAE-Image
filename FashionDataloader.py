import torch
import torchvision
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision import transforms




def get_FashionMNIST_dataloaders(batch_size):

    train_transform = transforms.Compose([transforms.ToTensor()])
    test_transform = transforms.Compose([transforms.ToTensor()])
    root = "./data"

    train_set = torchvision.datasets.FashionMNIST(root=root, download=True, transform=train_transform)
    test_set = torchvision.datasets.FashionMNIST(root=root, download=True, train=False, transform=test_transform)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size)

    return train_loader, test_loader

