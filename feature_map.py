import torch
from torch.autograd import Variable, Function
from torchvision import utils, datasets, transforms
from torch.utils.data import Dataset, DataLoader
import os
import cv2
import sys
import numpy as np
import argparse
from cifar_cnn import CNN_Net


class ModelFeatureOutputs():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """
    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []
    
    def save_gradient(self, grad):
        self.gradients.append(grad)

    def get_gradients(self):
        return self.gradients

    def __call__(self, x):
        features = []
        self.gradients = []
        for name, module in self.model.features._modules.items():
            x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                features += [x]
        output = x.clone()
        output = output.view(output.size(0), -1)
        output = self.model.classifier(output)
        return features, output


class Feature_Maps:
    def __init__(self, model, target_layer_names, use_cuda):
        self.model = model
        self.model.eval()
        self.cuda = False
        if use_cuda:
            self.model.cuda()
            self.cuda = True
        self.extractor = ModelFeatureOutputs(self.model, target_layer_names)
    
    def __call__(self, input, target_label=None):
        if self.cuda:
            features, output = self.extractor(input.cuda())
        else:
            features, output = self.extractor(input)
        if target_label == None:
            target_label = np.argmax(output.cpu().data.numpy())
        
        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][target_label] = 1
        one_hot = Variable(torch.from_numpy(one_hot), requires_grad=True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)
        self.model.features.zero_grad()
        self.model.classifier.zero_grad()
        one_hot.backward(retain_graph=True)

        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()
        target = features[-1]
        target = target.cpu().data.numpy()[0, :]

        weights = np.mean(grads_val, axis = (2, 3))[0, :]
        cam = np.zeros(target.shape[1 : ], dtype = np.float32)
        cam_list = np.zeros((target.shape[0],) + (32, 32), dtype = np.float32)
        features_output = np.zeros((target.shape[0],) + (32, 32), dtype = np.float32)
        def cam_relu(cam):
            cam = np.maximum(cam, 0)
            cam = cv2.resize(cam, (32, 32))
            cam = cam - np.min(cam)
            if np.max(cam) > 1e-10:
                cam = cam / np.max(cam)
            return cam

        for i, w in enumerate(weights):
            features_output[i] = cv2.resize(target[i, :, :], (32, 32))
            tmp = w * target[i, :, :]
            cam += tmp
            cam_list[i] = cam_relu(tmp).copy()
            
        cam = cam_relu(cam)
        return features_output, cam_list, cam


def show_feature_cam(_img, features, cam, cam_list):
    img = _img / 2 + 0.5
    feature_num = features.shape[0] + 1
    res = np.zeros((32 * 2, 32 * feature_num, 3), dtype=np.float32)
    res[:32, :32] = img
    heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    res[32 : 32 * 2, :32] = cam
    features = np.expand_dims(features, axis=1)
    for i in range(1, feature_num):
        res[:32, 32 * i : 32 * (i + 1)] = np.concatenate((features[i - 1], features[i - 1], features[i - 1])).transpose((1, 2, 0))
        heatmap = cv2.applyColorMap(np.uint8(255*cam_list[i - 1]), cv2.COLORMAP_JET)
        res[32: 32 * 2, 32 * i : 32 * (i + 1)] = np.float32(heatmap) / 255
    return res


if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    torch.manual_seed(7)
    classes = ('plane', 'car', 'bird', 'cat',\
        'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    # save_imgs(train_loader, classes)
    test_set = datasets.CIFAR10(root='./data/CIFAR10', train=False,
        download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=False, num_workers=2)
    # Net
    model = CNN_Net()
    path = os.path.join(os.getcwd(), 'model/cifar10-simple_cnn_gpu.tar.pth')
    if os.path.exists(path):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print('Doesn\'t exist {}'.format(path))
        os._exit()
    features_map = Feature_Maps(model, '4', False)
    # test data
    dataiter = iter(test_loader)
    imgs_index = [i for i in range(10)]
    label_flag = [False for i in range(10)]
    cnt = 0
    while cnt < 10:
        imgs, labels = dataiter.next()
        outputs = model(imgs)
        _, predicted = torch.max(outputs.data, 1)
        predicted_res = predicted == labels
        for i, label in enumerate(labels):
            if not predicted_res[i]:
                continue
            if not label_flag[label]:
                label_flag[label] = True
                imgs_index[label] = i
                cnt += 1
                if cnt == 10:
                    break
    imgs_res = np.zeros((32 * 2 * 10, 32 * 17, 3), dtype=np.float32)
    for i, index in enumerate(imgs_index):
        label = labels[index].numpy()
        features, cam_list, cam = features_map(torch.unsqueeze(imgs[index], 0), label)
        imgs_res[(32 * 2) * i: (32 * 2) * (i + 1)] = \
            show_feature_cam(imgs[index].numpy().transpose((1, 2, 0)), features, cam, cam_list).copy()
    cv2.imwrite("results/cifar_features_cam.jpg", np.uint8(255 * imgs_res))