import torch
from torch import nn, optim
from torch.nn import functional as F
from collections import OrderedDict


class UVae(nn.Module):
    def __init__(self, size=10):
        super(UVae, self).__init__()
        self.latent_size = size
        self.features = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(1, 20, 3, padding=1)),
            ('relu1', nn.ReLU()),
            ('batch1', nn.BatchNorm2d(20)),
            ('pool1', nn.MaxPool2d(2, 2)),
            ('conv2', nn.Conv2d(20, 64, 3, padding=1)),
            # ('relu2', nn.ReLU()),
            ('batch2', nn.BatchNorm2d(64)),
            ('pool2', nn.MaxPool2d(2, 2))
        ]))
        self.fc1mu = nn.Linear(64 * 7 * 7, self.latent_size)
        self.fc1Logvar = nn.Linear(64 * 7 * 7, self.latent_size)
        self.fc2 = nn.Linear(self.latent_size, 64 * 7 * 7)
        self.decoder = nn.Sequential(OrderedDict([
            ('reconv3', nn.ConvTranspose2d(64, 64, 2, 2)),
            ('conv3', nn.Conv2d(64, 20, 3, padding=1)),
            ('relu3', nn.ReLU()),
            ('batch3', nn.BatchNorm2d(20)),
            ('reconv4', nn.ConvTranspose2d(20, 20, 2, 2)),
            ('conv4', nn.Conv2d(20, 1, 3, padding=1)),
            ('relu4', nn.ReLU()),
            ('batch4', nn.BatchNorm2d(1)),
            ('sigmoid', nn.Sigmoid())
        ]))

    def encode(self, x):
        x = self.features(x)
        x = x.view(-1, 64 * 7 * 7)
        return self.fc1mu(x), self.fc1Logvar(x)

    def reparmeterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        re_z = F.relu(self.fc2(z).view(-1, 64, 7, 7))
        return self.decoder(re_z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparmeterize(mu, logvar)
        return self.decode(z), mu, logvar
