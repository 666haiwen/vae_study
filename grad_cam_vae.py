import torch
from torch.autograd import Variable
from torch.autograd import Function
from torchvision import utils, datasets, transforms
from torch.utils.data import Dataset, DataLoader
import os
import cv2
import sys
import numpy as np
import argparse
from mnist_vae import VAE


SIZE = (28, 28)
LATENT_SIZE = 5
class FeatureExtractor():
    """ Class for extracting activations and 
    registering gradients from targetted intermediate layers """
    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]
        return outputs, x

class ModelOutputs():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """
    def __init__(self, model, target_layers):
        self.model = model
        self.feature_extractor = FeatureExtractor(self.model.features, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations, output  = self.feature_extractor(x)
        return target_activations, output


class GradCam:
    def __init__(self, model, target_layer_names, use_cuda):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        self.extractor = ModelOutputs(self.model, target_layer_names)

    def forward(self, input):
        return self.model(input) 

    def __call__(self, input, index = None):
        if self.cuda:
            features, output = self.extractor(input.cuda())
        else:
            features, output = self.extractor(input)

        output = output.view(-1, 10 * 5 * 5)
        mu, logvar = self.model.fc2mu(output), self.model.fc2Logvar(output)
        if index == None:
            index = [i for i in range(args.latent_size)]
        latent_cam = np.zeros((args.latent_size, 28, 28))
        if mu.shape != (1, args.latent_size):
            print('latent of model({}) is not equal to args.latent({})'.format(mu.shape, args.latent_size))
            return latent_cam
        for latent_index in index:
            one_hot_mu = np.zeros((1, args.latent_size), dtype=np.float32)
            one_hot_mu[0][latent_index] = 1
            one_hot_logvar = np.zeros((1, args.latent_size), dtype=np.float32)
            one_hot_logvar[0][latent_index] = 1
            one_hot_mu = Variable(torch.from_numpy(one_hot_mu), requires_grad = True)
            one_hot_logvar = Variable(torch.from_numpy(one_hot_logvar), requires_grad = True)
            one_hot_mu = one_hot_mu * mu
            one_hot_logvar = one_hot_logvar * logvar
            loss = 0.5 * torch.sum(-one_hot_logvar + one_hot_logvar.exp() + one_hot_mu ** 2)
            self.model.features.zero_grad()
            self.model.zero_grad()
            loss.backward(retain_graph=True)
            grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()

            target = features[-1]
            target = target.cpu().data.numpy()[0, :]

            weights = np.mean(grads_val, axis = (2, 3))[0, :]
            cam = np.zeros(target.shape[1 : ], dtype = np.float)

            for i, w in enumerate(weights):
                cam += w * target[i, :, :]

            cam = np.maximum(cam, 0)
            cam = cv2.resize(cam, SIZE)
            cam = cam - np.min(cam)
            cam = cam / np.max(cam)
            latent_cam[latent_index] = cam.copy()
        return latent_cam, {'mu': np.squeeze(mu.detach().numpy()), 'logvar': np.squeeze(logvar.detach().numpy())}


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--seed', type=int, default=7,
                    help='random seed (default = 7)')
    parser.add_argument('--path', type=str, default='model/mnist_vae_tar_cpu.pth',
                    help='path of model loaded')
    parser.add_argument('--latent-size', type=int, default=20,
                    help='number of latent')
    parser.add_argument('--model-layer', type=str, default='4',
                    help='model layers of object')
    parser.add_argument('--sorted', type=bool, default=False,
                    help='sort the latent by sigma or not (default = False)')
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print("Using GPU for acceleration")
    else:
        print("Using CPU for computation")

    return args


def show_cam_on_image(imgs_list, mask_list, latent_list):
    res = np.zeros((28 * 10, 28 * (args.latent_size + 1), 3))
    for label in range(10):
        img = imgs_list[label]
        mask_latent = mask_list[label]
        res[label * 28 : (label + 1)*28, :28] = img

        latent_z = latent_list[label]
        list_logvar = latent_z['logvar'].tolist()
        sorted_logvar = sorted(list_logvar) if args.sorted else list_logvar
        print(sorted_logvar)
        for i, v in enumerate(sorted_logvar):
            tmp = list_logvar.index(v)
            mask = mask_latent[tmp]
            heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)
            heatmap = np.float32(heatmap) / 255
            cam = heatmap + np.float32(img)
            cam = cam / np.max(cam)
            res[label * 28 : (label + 1)*28, (i + 1)*28 : (i + 2)*28] = cam.copy()
    cv2.imwrite("results/cam_{}_{}.jpg".format(args.latent_size, args.sorted), np.uint8(255 * res))
    

args = get_args()
if __name__ == '__main__':
    """
        1.load vae model to grad_cam
        2.find the object layer as feature map
        3.generate mu and logsigma, calculate the loss function to get weight param
        4.get the cam-map by weight param
    """
    torch.manual_seed(args.seed)
    model = VAE(args.latent_size)
    path = os.path.join(os.getcwd(), args.path)
    if os.path.exists(path):
        checkpoint = torch.load(path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print('Doesn\'t exist {}'.format(path))
        os._exit()
    # Can work with any model, but it assumes that the model has a 
    # feature method, and a classifier method,
    # as in the VGG models in torchvision.
    grad_cam = GradCam(model = model, \
                    target_layer_names = [args.model_layer], use_cuda=args.use_cuda)

    
    test_loader = DataLoader(
        datasets.MNIST('data', train=False, download=True, transform=transforms.ToTensor()),
        batch_size=128)
    dataiter = iter(test_loader)
    imgs, labels = dataiter.next()
    imgs_index = [i for i in range(10)]
    label_flag = [False for i in range(10)]
    cnt = 0
    for i, label in enumerate(labels):
        if not label_flag[label]:
            label_flag[label] = True
            imgs_index[label] = i
            cnt += 1
            if cnt == 10:
                break
    
    mask_list = []
    imgs_list = []
    latent_list = []
    for index in imgs_index:
        show_img = np.concatenate((imgs[index], imgs[index], imgs[index]), axis=0).transpose((1, 2, 0))
        input = torch.unsqueeze(imgs[index], 0)
        # If None, returns the map for the highest scoring category.
        # Otherwise, targets the requested index.
        target_index = None
        mask, latent_z = grad_cam(input, target_index)
        mask_list.append(mask.copy())
        imgs_list.append(show_img.copy())
        latent_list.append(latent_z.copy())
    show_cam_on_image(imgs_list, mask_list, latent_list)
