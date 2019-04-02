from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms


def minist_dataLoader(batch_size):
    train_loader = DataLoader(
        datasets.MNIST('data/MNIST', train=True, download=False, transform=transforms.ToTensor()),
        batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(
        datasets.MNIST('data/MNIST', train=False, download=False, transform=transforms.ToTensor()),
        batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    imgs_size = (28, 28)
    return train_loader, test_loader, imgs_size

def vae_dataLoader(dataset, batch_size):
    if dataset == 'MNIST':
        return minist_dataLoader(batch_size)
    return None, None, None
