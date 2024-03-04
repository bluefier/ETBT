import csv
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import argparse
from adversarialbox.attacks import FGSMAttack, LinfPGDAttack
from adversarialbox.train import adv_train, FGSM_train_rnd
from adversarialbox.utils import to_var, pred_batch, test
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, transforms
from time import time
from adversarialbox.utils import to_var
from models import *
from Trigger_generation import Attack
from math import floor
import math
import matplotlib.pyplot as plt
import os 


# get gradient
def backward_hook(module, grad_in, grad_out):
    grad_block['grad_in'] = grad_in
    grad_block['grad_out'] = grad_out

def forward_hook(module, inp, outp):
    fmap_block['input'] = inp
    fmap_block['output'] = outp

# JSMA
def saliency_map(output,t,mask):

    output[0,t].backward(retain_graph=True)
    derivative = grad_block['grad_in'][1]
    derivative = derivative.cpu().numpy()
    alphas = derivative * mask 
    betas = -np.ones_like(alphas) 
    sal_map = np.abs(alphas)*np.abs(betas)*np.sign(alphas*betas)
    idx = np.argmin(sal_map) 
    idx = np.unravel_index(idx,mask.shape) # convert to (p1,p2) format
    pix_sign = np.sign(alphas)[idx]
    return idx,pix_sign

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Highly Evasive Targeted Bit-Trojan Attack')
    parser.add_argument('--device', default='4', type=str, help='list of gpu device(s)')
    parser.add_argument('--model_folder', default='', type=str, help='Storage folder for final models')
    parser.add_argument('--dataset', default='CIFAR10', type=str, help='dataset', choices=['CIFAR10','CIFAR100','TinyImagenet'])
    parser.add_argument('--model', default='ResNet18', type=str, help='model structure', choices=['ResNet18','ResNet34','VGG16'])
    parser.add_argument('--targets', default=1, type=int, help='target class')
    parser.add_argument('--pretrained_ann', default='', type=str, help='pretrained clean ANN model')
    parser.add_argument('--sample_number', default=256, type=int, help='The number of used samples')
    parser.add_argument('--last_layer', default=63, type=int, help='index of model last layer')
    parser.add_argument('--theta', default=0.1, type=int, help='perturbation factor')
    parser.add_argument('--Gamma', default=20, type=int, help='hypeparameter of maximum change boundary for activation values')
    args = parser.parse_args()
    print(args)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.device

    targets = args.targets
    pretained_ann = args.pretrained_ann
    last_layer = args.last_layer
    theta = args.theta
    Gamma = args.Gamma
    sample_number = args.sample_number

    # get mean & std
    if args.dataset == 'CIFAR10':
        mean = [x / 255 for x in [129.3, 124.1, 112.4]]
        std = [x / 255 for x in [68.2, 65.4, 70.4]]
        class_number = 10
    elif args.dataset == 'CIFARR100':
        mean = [x / 255 for x in [129.3, 124.1, 112.4]]
        std = [x / 255 for x in [68.2, 65.4, 70.4]]
        class_number = 100
    elif args.dataset == 'TinyTmagenet':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        class_number = 200

    net = ResNet(args, mean, std)
    net1= ResNet(args, mean, std)
    net2= ResNet_activation(args, mean, std)
    net3= ResNet_full(args, mean, std)


    # Loading the weights 
    net.load_state_dict(torch.load(pretained_ann)) 
    net.eval()      
    net=net.cuda()

    net2.load_state_dict(torch.load(pretained_ann)) 
    net2=net2.cuda()
    net1.load_state_dict(torch.load(pretained_ann)) 
    net1=net1.cuda()
    net3.load_state_dict(torch.load(pretained_ann)) 
    net3=net3.cuda()

    # Classifier definition & initilization
    classifier = Classifier()
    idx1 = 0
    for param1 in net3.parameters():
        idx1 += 1
        idx = 0
        if idx1 == last_layer:
            for param2 in classifier.parameters():
                idx += 1
                if idx == 1:
                    param2.data = param1.data.clone()
        if idx1 == last_layer+1:
            for param2 in classifier.parameters():
                idx += 1
                if idx == 2:
                    param2.data = param1.data.clone()

    ##--------------------------------------------------------------- prepare data --------------------------------------------------------------
    print('==> Preparing data..')

    # data preprocess
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),  
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),  
    ])

    # trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train) 
    trainset = torchvision.datasets.CIFAR10(root='./data/cifar_data', train=True, download=True, transform=transform_train) 
    loader_train = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2) 

    testset = torchvision.datasets.CIFAR10(root='./data/cifar_data', train=False, download=True, transform=transform_test) 
    loader_test = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=2) 

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()

    # -------------------------------------saliency map select neurons---------------------------------------
    for batch_idx, (data, target) in enumerate(loader_test):
        data, target = data.cuda(), target.cuda()

    net3.eval()
    output_net3, out2 = net3(data)
    loss = criterion(output_net3, target)   # 计算网络的损失

    features_grad = 0

    grad_block = dict()
    fmap_block = dict()

    classifier.linear.register_backward_hook(backward_hook)
    classifier.linear.register_forward_hook(forward_hook)

    n=0 
    for param in net2.parameters(): 
        n=n+1
        if n==last_layer:
            param.requires_grad = True
    n=0    
    for param in net3.parameters(): 
        n=n+1
        if n==last_layer:
            param.requires_grad = True

    loss_func=torch.nn.CrossEntropyLoss()

    selected_neurons = set([]) 
    succ_idx = list()

    for t, (x, y) in enumerate(loader_test): 

        # actmap_ori-- original activation map and never be changed
        # activation_map-- changed activaiton value, output test
        # activation_map -- used for saliency-map update, remove values exceed the threshold 
        x_var, y_var = to_var(x), to_var(y.long()) 
        output_act = net2(x_var)
        output_net3, actmap_ori = net3(x_var)
        activation_map = actmap_ori.clone()
        target_output = to_var(y.long())
        target_output[:] = targets 

        label = np.argmax(output_net3.data.cpu().numpy())
        loss = loss_func(output_net3, target_output)
        mask = np.ones_like(activation_map.detach().cpu().numpy()) # search region definition, set to 0 after modification
        
        # set boundary
        max_activation = activation_map.max()
        min_activation = activation_map.min()

        # Classifier initialization
        n = 0
        for param in net3.named_parameters():
            if n >= last_layer:
                print(param)
                classifier.param = param ### net1 is the copying the untrained parameters to net
        
        outp_g = classifier(activation_map)

        delta_a = 0
        neuron_arr = set([])
        loss_g = 0
        flag = False

        while (label != targets) and (delta_a < Gamma):
            idx,activation_sign = saliency_map(outp_g,targets,mask)

            # add perturbation
            activation_map.data[idx] = activation_map.data[idx] + activation_sign*theta*(max_activation - min_activation)
            
            # The point at the boundary no longer participates in the update
            if (activation_map.data[idx] <= min_activation) or (activation_map.data[idx] >= max_activation):
                mask[idx]=0
                activation_map.data.cpu()[idx]=np.clip(activation_map.data.detach().cpu()[idx],min_activation,max_activation)

            # add idx to array
            neuron_arr.add(idx)

            # calculate difference in activation_map
            delta = abs(actmap_ori.data - activation_map.data)
            
            # infinitely large paradigm
            # delta_idx = torch.argmax(delta[0])
            # delta_a = delta[0][delta_idx]
            
            # 2-parameter
            # delta_a = 0
            # # print('delta[0].shape', delta[0].shape)
            # for k in range(len(delta[0])):
            #     # print('delta[0][k]:', delta[0][k])
            #     delta_a += delta[0][k]**2
            # delta_a = math.sqrt(delta_a)
            # print('delta_a', delta_a)
            # delta_a = delta

            # 0-parameter
            delta_a = 0
            for k in range(len(delta[0])):
                if delta[0][k] != 0:
                    delta_a += 1

            outp_g = classifier(activation_map)
            output_net3
            label = torch.argmax(outp_g)
            loss_g = loss_func(outp_g, target_output) 

            flag = (label != targets) and (delta_a < Gamma)
            if (label == targets):
                succ_idx.append(t)
            # break
        
        # Handling neuron_arr
        if len(neuron_arr) != 0:
            if len(selected_neurons) == 0:
                for i in range(len(neuron_arr)):
                    list(neuron_arr)
                    selected_neurons.add(list(neuron_arr)[i])
            else:
                if len(selected_neurons) < 20:
                    # union
                    selected_neurons = set(selected_neurons) | set(neuron_arr)
                else:
                    # intersection
                    selected_neurons = set(selected_neurons) & set(neuron_arr)

        if t == sample_number:
            break

    print('succ_idx', succ_idx)
    print('selected_neurons:', selected_neurons)
    print('size of selected_neurons:', len(selected_neurons))
