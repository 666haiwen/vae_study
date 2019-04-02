import os
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import numpy as np


class CNN_Net(nn.Module):

    def __init__(self):
        super(CNN_Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.features = nn.Sequential(
            self.conv1,
            nn.ReLU(),
            self.pool,
            self.conv2,
            nn.ReLU(),
            self.pool
        )
        self.classifier = nn.Sequential(
            self.fc1,
            nn.ReLU(),
            self.fc2,
            nn.ReLU(),
            self.fc3
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 16 * 5 * 5)
        x = self.classifier(x)
        return x

def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def train(model, criterion, optimizer, train_loader, epoch_begin):
    since = time.time()
    for epoch in range(epoch_begin, 100):
        running_loss = 0.0
        before_time = time.time()
        for i, data in enumerate(train_loader):
            inputs, labels = data
            if _GPU:
                inputs = inputs.cuda()
                labels = labels.cuda()
            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        time_cost = time.time() - before_time
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, PATH)
        print('[epoch = %d] loss: %.3f  time cost: %.0f'%(epoch + 1, running_loss / 2000, time_cost))
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Finished Training')

def test(model, test_loader, classes):
    dataiter = iter(test_loader)
    images, labels = dataiter.next()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            if _GPU:
                images = images.cuda()
                labels = labels.cuda()
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the 10K test imaegs: %d %%' % (100 * correct / total))

def save_imgs(data_loader, classes):
    data_iter = iter(data_loader)
    image, label = data_iter.next()
    data = image[0].numpy()
    data = np.uint8(np.transpose(data / 2 + 0.5, (1, 2, 0)) * 255)
    img = Image.fromarray(data)
    img.save('{}.png'.format(classes[label[0]]))
    # img.show()

def classfier():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    torch.manual_seed(7)
    classes = ('plane', 'car', 'bird', 'cat',\
        'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    train_set = torchvision.datasets.CIFAR10(root='./data/CIFAR10', train=True,
        download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True, num_workers=2)

    # save_imgs(train_loader, classes)
    test_set = torchvision.datasets.CIFAR10(root='./data/CIFAR10', train=False,
        download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=False, num_workers=2)
    # Net
    model = CNN_Net()
    if _GPU:
        model.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    epoch = 0
    if os.path.exists(PATH):
        checkpoint = torch.load(PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
    # train(model, criterion, optimizer, train_loader, epoch)
    # test(model, test_loader, classes)
    save_imgs(test_loader, classes)

    
if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    PATH = os.path.join(os.getcwd(), 'model/cifar10-simple_cnn')
    _GPU = False
    if torch.cuda.is_available():
        PATH += '_gpu.tar.pth'
        _GPU = True
    else:
        PATH += '_cpu.tar.pth'

    classfier()
