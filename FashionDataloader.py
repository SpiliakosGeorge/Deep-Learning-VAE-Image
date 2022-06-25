import torch
import torchvision
from torch.utils.data import Dataset
from torchvision import transforms


def get_FashionMNIST_dataloaders(batch_size, num_workers):

    train_transform = transforms.Compose([transforms.ToTensor()])
    test_transform = transforms.Compose([transforms.ToTensor()])
    root = "./data"

    train_set = torchvision.datasets.FashionMNIST(root=root, download=True, transform=train_transform)
    test_set = torchvision.datasets.FashionMNIST(root=root, download=True, train=False, transform=test_transform)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, num_workers=num_workers)

    return train_loader, test_loader


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
