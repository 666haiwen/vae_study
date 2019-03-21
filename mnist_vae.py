"""
    Re-implement of vae example in pytorch-examples.
    This is an improved implementation of the paper [Stochastic Gradient VB and the
    Variational Auto-Encoder](http://arxiv.org/abs/1312.6114) by Kingma and Welling.
    It uses ReLUs and the adam optimizer, instead of sigmoids and adagrad.
    These changes make the network converge much faster.
"""
import os
import argparse
import time
import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
def get_args():
    parser = argparse.ArgumentParser(description='VAE MINST Example')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size of trainning (default = 128)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='number of epochs to train (default = 10)')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default = 1)')
    parser.add_argument('--cuda', type=bool, default=False,
                        help='enables CUDA traning or not (default = False)')
    parser.add_argument('--num-workers', type=int, default=2,
                        help='num of workers while training and testing (default = 2)')
    parser.add_argument('--path', type=str, default='model/mnist_vae_tar_cpu.pth',
                        help='path to model saving')
    parser.add_argument('--load-checkpoint', type=bool, default=True,
                        help='load history model or not (default = True)')
    parser.add_argument('--lr', type=int, default=1e-3,
                        help='learning rate of training (default = 1e-3)')
    parser.add_argument('--log-interval', type=int, default=20,
                        help='how many batches to wait before logging training status')
    parser.add_argument('--sample', type=bool, default=False,
                        help='Test to get sample img or not')
    parser.add_argument('--latent-size', type=int, default=5,
                        help='number of latents (default = 5)')
    args = parser.parse_args()
    args.path = args.path[:-4] + '_{}.pth'.format(args.latent_size)
    args.cuda = args.cuda and torch.cuda.is_available()
    return args


class VAE(nn.Module):
    def __init__(self, latent_size):
        super(VAE, self).__init__()

        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 10, 3)
        self.fc2mu = nn.Linear(10 * 5 * 5, latent_size)
        self.fc2Logvar = nn.Linear(10 * 5 * 5, latent_size)
        self.fc3 = nn.Linear(latent_size, 400)
        self.fc4 = nn.Linear(400, 28*28)
        self.features = nn.Sequential(
            self.conv1,
            nn.ReLU(),
            self.pool,
            self.conv2,
            nn.ReLU(),
            self.pool
        )
    
    def encode(self, x):
        x = self.features(x)
        x = x.view(-1, 10 * 5 * 5)
        return self.fc2mu(x), self.fc2Logvar(x)
    
    def reparmeterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparmeterize(mu, logvar)
        return self.decode(z), mu, logvar


def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    # KL_Distance : 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = 0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp())
    return BCE - KLD

def train(model, optimizer, train_loader, epoch_begin):
    model.train()
    train_loss = 0
    for epoch in range(epoch_begin + 1, args.epochs):
        before_time = time.time()
        train_loss = 0
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            if args.cuda:
                inputs = inputs.cuda()
                labels = labels.cuda()
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(inputs)
            loss = loss_function(recon_batch, inputs, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(inputs), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    loss.item() / len(inputs)))
        # finish one epoch
        time_cost = time.time() - before_time
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, args.path)
        print('====> Epoch: {} \tAverage loss : {:.4f}\tTime cost: {:.0f}'.format(
            epoch, train_loss / len(train_loader.dataset), time_cost))


def test(model, test_loader):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            if args.cuda:
                inputs = inputs.cuda()
                labels = labels.cuda()
            recon_batch, mu, logvar = model(inputs)
            loss = loss_function(recon_batch, inputs, mu, logvar)
            test_loss += loss.item()
            if i == 0:
                n = min(inputs.size(0), 8)
                comparison = torch.cat([inputs[:n],
                                      recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
                test_mu = mu[:n].cpu().numpy()
                test_logvar = logvar[:n].cpu().numpy()
                np.savetxt('results/mu.txt', test_mu, fmt='%.4f')
                np.savetxt('results/logvar.txt', test_logvar, fmt='%.4f')
                save_image(comparison.cpu(),
                         'results/reconstruction_{}.png'.format(args.latent_size), nrow=n)
        
        print('Test Average loss : {:.4f}'.format(test_loss / len(test_loader.dataset)))

def main():
    model = VAE(args.latent_size).cuda() if args.cuda else VAE(args.latent_size)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    args.path = os.path.join(os.getcwd(), args.path)
    epoch = -1
    if args.load_checkpoint:
        if os.path.exists(args.path):
            checkpoint = torch.load(args.path)
            model.load_state_dict(checkpoint['model_state_dict'])
            epoch = checkpoint['epoch']
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            print('Doesn\'t find checkpoint in ' + args.path)
    print('Begin train.......')
    train(model, optimizer, train_loader, epoch)
    print('Finish train!\nBegin test.......')
    test(model, test_loader)
    print('Finish test and reconstruciton png!\nBegin sampling......')
    with torch.no_grad():
        sample = torch.randn(64, args.latent_size)
        if args.cuda:
            sample = sample.cuda()
        sample = model.decode(sample).cpu()
        save_image(sample.view(64, 1, 28, 28),
                    'results/sample.png')



if __name__ == "__main__":
    args = get_args()
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda \
    else {'num_workers': args.num_workers}
    torch.manual_seed(args.seed)
    train_loader = DataLoader(
        datasets.MNIST('data', train=True, download=True, transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = DataLoader(
        datasets.MNIST('data', train=False, download=True, transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    main()
