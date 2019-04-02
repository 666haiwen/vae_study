import os
import argparse
import time
import numpy as np
import torch
from torch import optim
from torchvision.utils import save_image
from U_vae import UVae
from mnist_vae import VAE
from lib.dataLoader import vae_dataLoader
from lib.encropy import UVAECriterion, normal_loss
from torchvision.utils import save_image


os.environ['CUDA_VISIBLE_DEVICES'] = '1'
def get_args():
    parser = argparse.ArgumentParser(description='VAE MINST Example')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size of trainning (default = 128)')
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of epochs to train (default = 200)')
    parser.add_argument('--seed', type=int, default=7,
                        help='random seed (default = 7)')
    parser.add_argument('--cuda', type=bool, default=True,
                        help='enables CUDA traning or not (default = False)')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='num of workers while training and testing (default = 4)')
    parser.add_argument('--path', type=str, default='model/mnist_uvae_tar.pth',
                        help='path to model saving')
    parser.add_argument('--lr', type=float, default=1e-8,
                        help='learning rate of training (default = 1e-8)')
    parser.add_argument('--datasets', type=str, default='MNIST',
                        help='the datasets to use (default = MNIST)')
    args = parser.parse_args()
    args.log_interval = 20
    args.cuda = args.cuda and torch.cuda.is_available()
    if args.cuda:
        print('Using gpu computation!')
    else:
        print('Using cpu computation.')
    return args

def train(model, optimizer, criterion, scheduler, epoch_begin):
    model.train()
    train_loss = 0
    for epoch in range(epoch_begin, args.epochs):
        before_time = time.time()
        train_loss = 0
        # scheduler.step(epoch)
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            if args.cuda:
                inputs = inputs.cuda()
                labels = labels.cuda()
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(inputs)
            recon_loss, KLD = criterion(recon_batch, inputs, mu, logvar)
            loss = recon_loss - KLD
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            # scheduler.step()
            # print('grad is :', model.decoder[7].weight.grad)
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tRecon_Loss: {:.6f} KLD: {:.6f} Loss: {:.6f}'\
                    .format(epoch, batch_idx * len(inputs), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), recon_loss.item(), KLD.item(),
                    loss.item() / len(inputs)))
        # finish one epoch
        time_cost = time.time() - before_time
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
        }, args.path)
        print('====> Epoch: {} \tAverage loss : {:.4f}\tLR: {:.10f}\tTime cost: {:.0f}'.format(
            epoch, train_loss / len(train_loader.dataset), optimizer.param_groups[0]['lr'], time_cost))
        


def test(model, criterion):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            if args.cuda:
                inputs = inputs.cuda()
                labels = labels.cuda()
            recon_batch, mu, logvar = model(inputs)
            recon_loss, KLD = criterion(recon_batch, inputs, mu, logvar)
            loss = recon_loss - KLD
            test_loss += loss.item()
            if i == 0:
                n = min(inputs.size(0), 8)
                comparison = torch.cat([inputs[:n],
                                      recon_batch.view(args.batch_size, 1, img_size[0], img_size[1])[:n]])
                test_mu = mu[:n].cpu().numpy()
                test_logvar = logvar[:n].cpu().numpy()
                np.savetxt('results/mu.txt', test_mu, fmt='%.4f')
                np.savetxt('results/logvar.txt', test_logvar, fmt='%.4f')
                save_image(comparison.cpu(),
                         'results/reconstruction_BVae.png', nrow=n)
        
        print('Test Average loss : {:.4f}'.format(test_loss / len(test_loader.dataset)))

def main():
    model = UVae().cuda() if args.cuda else UVae()
    criterion = UVAECriterion()
    # optimizer = optim.Adam(model.parameters(), lr=args.lr)
    optimizer = optim.SGD(model.parameters(), lr = args.lr, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 50, gamma=0.2, last_epoch=-1)
    args.path = os.path.join(os.getcwd(), args.path)
    epoch = 0
    if os.path.exists(args.path):
        checkpoint = torch.load(args.path)
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint['epoch'] + 1
    else:
        print('Doesn\'t find checkpoint in ' + args.path)
    print('Begin train.......')
    train(model, optimizer, normal_loss, scheduler, epoch)
    print('Finish train!\nBegin test.......')
    test(model, normal_loss)
    with torch.no_grad():
        sample = torch.randn(64, 5)
        if args.cuda:
            sample = sample.cuda()
        sample = model.decode(sample).cpu()
        save_image(sample.view(64, 1, 28, 28),
                    'results/sample.png')


if __name__ == "__main__":
    args = get_args()
    torch.manual_seed(args.seed)
    train_loader, test_loader, img_size = vae_dataLoader(args.datasets, args.batch_size)
    main()
