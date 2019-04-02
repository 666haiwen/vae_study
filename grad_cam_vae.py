import torch
from torch.autograd import Variable
from torchvision import utils, datasets, transforms
from torch.utils.data import Dataset, DataLoader
import os
import cv2
import sys
import numpy as np
import argparse
from mnist_vae import VAE
from U_vae import UVae


SIZE = (28, 28)
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
    def __init__(self, model, target_layer_names, use_cuda, latent_size):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        self.latent_size = latent_size
        if self.cuda:
            self.model = model.cuda()

        self.extractor = ModelOutputs(self.model, target_layer_names)

    def __call__(self, input, index = None):
        x = input.cuda() if self.cuda else input
        features, output = self.extractor(x)
        output = output.view(-1, 64 * 7 * 7)
        mu, logvar = self.model.fc1mu(output), self.model.fc1Logvar(output)
        if index == None:
            index = [i for i in range(self.latent_size)]
        latent_cam = np.zeros((self.latent_size, 28, 28))
        if mu.shape != (1, self.latent_size):
            print('latent of model({}) is not equal to args.latent_size({})'.format(mu.shape, self.latent_size))
            return latent_cam
        for latent_index in index:
            one_hot_mu = np.zeros((1, self.latent_size), dtype=np.float32)
            one_hot_mu[0][latent_index] = 1
            one_hot_logvar = np.zeros((1, self.latent_size), dtype=np.float32)
            one_hot_logvar[0][latent_index] = 1
            one_hot_mu = Variable(torch.from_numpy(one_hot_mu), requires_grad = True)
            one_hot_logvar = Variable(torch.from_numpy(one_hot_logvar), requires_grad = True)
            one_hot_mu = one_hot_mu * mu
            one_hot_logvar = one_hot_logvar * logvar
            loss = -0.5 * torch.sum(-one_hot_logvar + one_hot_logvar.exp() + one_hot_mu ** 2)
            self.model.features.zero_grad()
            self.model.zero_grad()
            loss.backward(retain_graph=True)
            grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()

            target = features[-1]
            target = target.cpu().data.numpy()[0, :]

            weights = np.mean(grads_val, axis = (2, 3))[0, :]
            # print('weights:', weights)
            cam = np.zeros(target.shape[1 : ], dtype = np.float)

            for i, w in enumerate(weights):
                cam += w * target[i, :, :]

            cam = np.maximum(cam, 0)
            cam = cv2.resize(cam, SIZE)
            cam = cam - np.min(cam)
            if np.max(cam) > 1e-10:
                cam = cam / np.max(cam)
            latent_cam[latent_index] = cam.copy()
        return latent_cam



class OneHotLatent(object):
    def __init__(self, model, cuda, latent_size):
        self.model = model
        self.cuda = cuda
        self.latent_size = latent_size

    def __call__(self, input):
        x = input.cuda() if self.cuda else input
        mu, logvar = self.model.encode(x)
        res_one = np.zeros((self.latent_size + 1,) + SIZE, dtype=np.float32)
        res_de_one = np.zeros((self.latent_size + 1,) + SIZE, dtype=np.float32)
        for i in range(self.latent_size):
            tmp_mu = torch.zeros(mu.shape, dtype=torch.float32)
            tmp_mu[0][i] = mu[0][i]
            res_one[i] = self.model.decode(tmp_mu).view(SIZE).detach().numpy()
            tmp_mu = mu.clone()
            tmp_mu[0][i] = 0
            res_de_one[i] = self.model.decode(tmp_mu).view(SIZE).detach().numpy()
        res_one[-1] = res_de_one[-1] = self.model.decode(mu).view(SIZE).detach().numpy()
        return res_one, res_de_one


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--seed', type=int, default=7,
                    help='random seed (default = 7)')
    parser.add_argument('--path', type=str, default='mnist_vae_tar_cpu.pth',
                    help='path of model loaded')
    parser.add_argument('--latent-size', type=int, default=10,
                    help='number of latent')
    parser.add_argument('--model-layer', type=str, default='4',
                    help='model layers of object')
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print("Using GPU for acceleration")
    else:
        print("Using CPU for computation")

    return args


def show_cam_on_image(imgs_list, mask_list, latent_size):
    heat_res = np.zeros((SIZE[0] * 10, SIZE[1] * (latent_size + 1), 3))
    cam_res = np.zeros((SIZE[0] * 10, SIZE[1] * (latent_size + 1), 3))
    for label in range(10):
        img = imgs_list[label]
        mask_latent = mask_list[label]
        heat_res[label * SIZE[0] : (label + 1)*SIZE[0], :SIZE[1]] = img
        cam_res[label * SIZE[0] : (label + 1)*SIZE[0], :SIZE[1]] = img

        for i in range(latent_size):
            mask = mask_latent[i]
            # mask[mask < 0.7] = 0
            heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_HOT)
            heatmap = np.float32(heatmap) / 255
            cam = heatmap + np.float32(img)
            cam = cam / np.max(cam)
            heat_res[label * SIZE[0] : (label + 1)*SIZE[1], (i + 1)*SIZE[0] : (i + 2)*SIZE[1]] = heatmap.copy()
            cam_res[label * SIZE[0] : (label + 1)*SIZE[1], (i + 1)*SIZE[0] : (i + 2)*SIZE[1]] = cam.copy()
    # cv2.imwrite("results/cam_heatmap_{}.jpg".format(latent_size), np.uint8(255 * heat_res))
    # cv2.imwrite("results/cam_{}.jpg".format(latent_size), np.uint8(255 * cam_res))
    cv2.imshow("cam_heatmap_{}.jpg".format(latent_size), np.uint8(255 * heat_res))
    

def show_one_hot_on_image(imgs_list, mask_list, one_hot_list, de_one_hot_list, latent_size):
    one_hot_res = np.zeros((SIZE[0] * 10, SIZE[0] * (latent_size + 2)))
    for label in range(10):
        img = imgs_list[label]
        one_hot_res[label * SIZE[0] : (label + 1)*SIZE[0], :SIZE[1]] = img[:,:,0]
        for i in range(latent_size):
            tmp = mask_list[label][i] + de_one_hot_list[label][i]
            # tmp = de_one_hot_list[label][i] + one_hot_list[label][i]
            # tmp = tmp - np.min(tmp)
            tmp = tmp / np.max(tmp)
            # tmp = cv2.applyColorMap(np.uint8(255*one_hot_list[label][i]), cv2.COLORMAP_HOT)
            one_hot_res[label * SIZE[0] : (label + 1)*SIZE[1], (i + 1)*SIZE[0] : (i + 2)*SIZE[1]] = tmp.copy()
        i = latent_size
        one_hot_res[label * SIZE[0] : (label + 1)*SIZE[1], (i + 1)*SIZE[0] : (i + 2)*SIZE[1]] = one_hot_list[label][i].copy()
    cv2.imwrite("results/one_hot_{}.jpg".format(latent_size), np.uint8(255 * one_hot_res))


def cam_interface():
    """
        1.load vae model to grad_cam
        2.find the object layer as feature map
        3.generate mu and logsigma, calculate the loss function to get weight param
        4.get the cam-map by weight param
    """
    torch.manual_seed(args.seed)
    model = UVae(args.latent_size)
    # model = VAE(args.latent_size)
    path = os.path.join(os.getcwd(), 'model', args.path)
    if os.path.exists(path):
        print('Load model of ', args.path)
        checkpoint = torch.load(path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print('Doesn\'t exist {}'.format(path))
        os._exit()
    # Can work with any model, but it assumes that the model has a 
    # feature method, and a classifier method,
    # as in the VGG models in torchvision.
    grad_cam = GradCam(model = model, \
                    target_layer_names = [args.model_layer], use_cuda=args.use_cuda, latent_size=args.latent_size)
    one_hot_latent = OneHotLatent(model = model, cuda=args.use_cuda, latent_size=args.latent_size)
    
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
    one_hot_list = []
    de_one_hot_list = []
    target_index = None
    for index in imgs_index:
        show_img = np.concatenate((imgs[index], imgs[index], imgs[index]), axis=0).transpose((1, 2, 0))
        input = torch.unsqueeze(imgs[index], 0)
        mask = grad_cam(input, target_index)
        one_hot = one_hot_latent(input)
        mask_list.append(mask.copy())
        imgs_list.append(show_img.copy())
        res_one, res_de_one = one_hot
        one_hot_list.append(res_one)
        de_one_hot_list.append(res_de_one)
    return imgs_list, mask_list, one_hot_list, de_one_hot_list


if __name__ == '__main__':
    args = get_args()
    imgs_list, mask_list, one_hot_list, de_one_hot_list = cam_interface()
    show_cam_on_image(imgs_list, mask_list, args.latent_size)
    show_one_hot_on_image(imgs_list, mask_list, one_hot_list, de_one_hot_list, args.latent_size)

