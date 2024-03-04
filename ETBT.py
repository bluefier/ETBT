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

## ------------------------------------------------------------- definition of weights conversion methods -----------------------------------------------------------
## weight conversion functions
def int2bin(input, num_bits):
    '''
    convert the signed integer value into unsigned integer (2's complement equivalently).
    '''
    output = input.clone()
    output[input.lt(0)] = 2**num_bits + output[input.lt(0)]
    return output

def bin2int(input, num_bits):
    '''
    convert the unsigned integer (2's complement equivantly) back to the signed integer format
    with the bitwise operations. Note that, in order to perform the bitwise operation, the input
    tensor has to be in the integer format.
    '''
    mask = 2**(num_bits-1) - 1
    output = -(input & ~mask) + (input & mask)
    return output
    
def weight_conversion(model):
    '''
    Perform the weight data type conversion between:
        signed integer <==> two's complement (unsigned integer)

    Note that, the data type conversion chosen is depend on the bits:
        N_bits <= 8   .char()   --> torch.CharTensor(), 8-bit signed integer
        N_bits <= 16  .short()  --> torch.shortTensor(), 16 bit signed integer
        N_bits <= 32  .int()    --> torch.IntTensor(), 32 bit signed integer
    '''
    for m in model.modules():
        if isinstance(m, quantized_conv) or isinstance(m, bilinear):
            w_bin = int2bin(m.weight.data, m.N_bits).short()
            m.weight.data = bin2int(w_bin, m.N_bits).float()
    return

def del_tensor_ele(arr,index):
    arr1 = arr[0:index]
    arr2 = arr[index+1:]
    return torch.cat((arr1,arr2),dim=0)

def countingss(param,param1):
        ind = (w!= 0).nonzero()
        jj = int(ind.size()[0])
        count = 0
        count_bits = [0 for x in range(8)]
        for i in range(jj):
            indi = ind[i,1] 
            n1 = param[targets,indi]
            n2 = param1[targets,indi]
            b1 = Bits(int = int(n1), length = 8).bin
            b2 = Bits(int = int(n2), length = 8).bin
            
            # 可以用k代指待翻转的bit位置
            for k in range(8):
                diff = int(b1[k]) - int(b2[k])
                if diff != 0:
                    count = count + 1
                    count_bits[k] += 1

        return count, count_bits

def test(model, loader, blackbox=False, hold_out_size=None):
    """
    Check model accuracy on model based on loader (test)
    """
    model.eval()

    num_correct, num_samples = 0, len(loader.dataset)

    if blackbox:
        num_samples -= hold_out_size

    for x, y in loader:
        x_var = to_var(x, volatile=True)
        scores = model(x_var)
        _, preds = scores.data.cpu().max(1)
        num_correct += (preds == y).sum()

    acc = float(num_correct)/float(num_samples)
    print('Got %d/%d correct (%.2f%%) on the clean data' % (num_correct, num_samples, 100 * acc))

    return acc

# test code with trigger
def test1(model, loader, x_tri):
    """
    Check model attack success rate based on loader (test)
    """
    model.eval()

    num_correct, num_samples = 0, len(loader.dataset)

    for x, y in loader:
       
        x_var = to_var(x, volatile=True)
        x_var[:,0:3,start:end,start:end] = x_tri[:,0:3,start:end,start:end]
        y[:]=targets  ## setting all the target to target class
      
        scores = model(x_var)
        _, preds = scores.data.cpu().max(1)
        num_correct += (preds == y).sum()

    asr = float(num_correct)/float(num_samples)
    print('Got %d/%d correct (%.2f%%) on the trigger data' % (num_correct, num_samples, 100 * asr))

    return asr

# obtaned from KNE_forward_process.py
saliency_map_neuron = []

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Highly Evasive Targeted Bit-Trojan Attack')
    parser.add_argument('--device', default='4', type=str, help='list of gpu device(s)')
    parser.add_argument('--model_folder', default='', type=str, help='Storage folder for final models')
    parser.add_argument('--dataset', default='CIFAR10', type=str, help='dataset', choices=['CIFAR10','CIFAR100','TinyImagenet'])
    parser.add_argument('--model', default='ResNet18', type=str, help='model structure', choices=['ResNet18','ResNet34','VGG16'])
    parser.add_argument('--targets', default=1, type=int, help='target class')
    parser.add_argument('--start', default=21, type=int, help='trigger start position')
    parser.add_argument('--end', default=31, type=int, help='trigger end position')
    parser.add_argument('--wb', default=50, type=int, help='modified neuron/weight number')
    parser.add_argument('--wo', default=10, type=int, help='essensial neuron number')
    parser.add_argument('--high', default=10, type=int, help='expected activaiton value')
    parser.add_argument('--n_exp', default=100, type=int, help='preset bit-flip number')
    parser.add_argument('--pretrained_ann', default='./trained_models/ann/ann_vgg11_cifar10.pth', type=str, help='pretrained clean ANN model')
    parser.add_argument('--last_layer', default=63, type=int, help='index of model last layer')
    parser.add_argument('--l_layer_test', default=63, type=int, help='index of test_model last layer')
    parser.add_argument('--alpha', default=0.4, type=int, help='hypeparameter of ACC loss term')
    parser.add_argument('--beta', default=0.4, type=int, help='hypeparameter of ASR loss term')
    parser.add_argument('--gamma', default=0.2, type=int, help='hypeparameter of BCT')
    args = parser.parse_args()
    print(args)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.device

    device = args.device
    dataset = args.dataset
    model = args.model
    targets = args.targets
    start = args.start
    end = args.end
    wb = args.wb
    wo = args.wo
    high = args.high
    n_exp = args.n_exp
    pretained_ann = args.pretrained_ann
    last_layer = args.last_layer
    l_last_layer = args.l_last_layer
    alpha = args.alpha
    beta = args.beta
    gamma = args.gamma
    model_folder = args.model_folder

    identifier = 'Trojaned_'+model.lower+'_'+dataset.lower

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
    net1 = ResNet(args, mean, std)
    net2 = ResNet_activation(args, mean, std)

    # Loading the model weights
    net.load_state_dict(torch.load(pretained_ann))
    net.eval()      
    net=net.cuda()
    net2.load_state_dict(torch.load(pretained_ann)) 
    net2=net2.cuda()
    net1.load_state_dict(torch.load(pretained_ann)) 
    net1=net1.cuda()

    net_conversion = ResNet_test(args, mean, std)
    net1_conversion = ResNet_test(args, mean, std)

    net1_conversion=net1_conversion.cuda()
    # load pretrained model paramenters
    pretrained_dict = torch.load(pretained_ann)
    # save current model parameters
    model_dict = net1_conversion.state_dict()

    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict) 
    # 3. load the new state dict
    net1_conversion.load_state_dict(model_dict)

    # update the step size before validation
    for m in net1_conversion.modules():
        if isinstance(m, quan_Conv2d) or isinstance(m, quan_Linear):
            m.__reset_stepsize__()
            m.__reset_weight__()

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

    trainset = torchvision.datasets.CIFAR10(root='./data/cifar_data', train=True, download=True, transform=transform_train) 
    loader_train = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2) 

    testset = torchvision.datasets.CIFAR10(root='./data/cifar_data', train=False, download=True, transform=transform_test) 
    loader_test = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2) 

    if torch.cuda.is_available():
        print('CUDA ensabled.')
        net.cuda()

    net.eval()

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()

    ##_-----------------------------------------KNE step------------------------------------------------------------
    ## performing back propagation to identify the target neurons using a sample test batch of size 128
    for batch_idx, (data, target) in enumerate(loader_test):
        data, target = data.cuda(), target.cuda()
        mins, maxs = data.min(), data.max()
        break

    net.eval()
    output = net(data)  
    loss = criterion(output, target)  

    for m in net.modules():
        if isinstance(m, quantized_conv) or isinstance(m, bilinear):
            if m.weight.grad is not None:
                m.weight.grad.data.zero_()   # set gradinet to 0
      
    loss.backward()

    tar_sm = torch.Tensor(saliency_map_neuron[targets]).type(torch.long)

    for name, module in net.named_modules():
        if isinstance(module, bilinear):
            w_v,w_id = module.weight.grad.detach().abs().topk(wb)
            tar_per = w_id[targets] 
            w_o, w_oid = module.weight.grad.detach().abs().topk(wo) 
           
            tar_per_target = torch.Tensor([])
            tar_list = list(set(tar_per.cpu().numpy()).union(set(tar_sm.cpu().numpy())))
            tar_per = torch.Tensor(tar_list).type(torch.long).to(device)
            for i in range(class_number):
                if i != targets: 
                    ess_neuron = w_oid[i]
                    # ess_neuron = ess_neuron[5:]
                    if len(tar_per_target) != 0:
                        tar_per = tar_per_target
                        for j in range(len(tar_per)):
                            if tar_per[j] in ess_neuron:
                                tar_per_target = del_tensor_ele(tar_per,j)
                    else:
                        for j in range(len(tar_per)):
                            if tar_per[j] in ess_neuron:
                                tar_per_target = del_tensor_ele(tar_per,j)

        tar = torch.cat((tar_per.cuda(),tar_sm.cuda()), dim=0)  
        tar = tar_per[:]
        print(len(tar))

    np.savetxt('key_neuron KNE.txt', tar.cpu().numpy(), fmt='%f')
    b = np.loadtxt('key_neuron KNE.txt', dtype=float)
    b = torch.Tensor(b).long().cuda()

    #-----------------------Trigger Generation--------------------------------------------------------
    model_attack = Attack(dataloader=loader_test, attack_method='fgsm', epsilon=0.001)
    
    loader_test = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)
    
    for t, (x, y) in enumerate(loader_test): 
            x_var, y_var = to_var(x), to_var(y.long()) 
            x_var[:,:,:,:]=0
            x_var[:,0:3,start:end,start:end]=0.5 ## initializing the mask to 0.5   
            break

    output = net2(x_var)
    U = output[:, tar]
    U_prob = F.softmax(U, dim=0)

    y=net2(x_var) ##initializaing the target value for trigger generation
    y[:,tar]=high   ### setting the target of certain neurons to a expect value high

    ep=0.5
    ### iterating 200 times to generate the trigger
    for i in range(200):  
        x_tri, loss_js =model_attack.attack_method(
                    net2, x_var.to(device), y,tar,ep,U_prob,mins,maxs) 
        x_var=x_tri
        
    ep=0.1
    ### iterating 200 times to generate the trigger again with lower update rate
    for i in range(200):  
        x_tri, loss_js =model_attack.attack_method(
                    net2, x_var.to(device), y,tar,ep,U_prob,mins,maxs) 
        x_var=x_tri
        
    ep=0.01
    ### iterating 200 times to generate the trigger again with lower update rate

    for i in range(200):  
        x_tri, loss_js =model_attack.attack_method(
                    net2, x_var.to(device), y,tar,ep,U_prob,mins,maxs) 
        x_var=x_tri

    ep=0.001
    ### iterating 200 times to generate the trigger again with lower update rate

    for i in range(200):  
        x_tri, loss_js =model_attack.attack_method(
                    net2, x_var.to(device), y,tar,ep,U_prob,mins,maxs) 
        x_var=x_tri

    eta = 1 + math.log((1-loss_js), math.e)
    print('eta: ', eta)
        
    ##saving the trigger image channels for future use
    np.savetxt('trojan_img1_TG-AVS.txt', x_tri[0,0,:,:].cpu().numpy(), fmt='%f')
    np.savetxt('trojan_img2_TG-AVS.txt', x_tri[0,1,:,:].cpu().numpy(), fmt='%f')
    np.savetxt('trojan_img3_TG-AVS.txt', x_tri[0,2,:,:].cpu().numpy(), fmt='%f')


    #----------------------------------------------------CTBS Step---------------------------------------------
    ### setting the weights not trainable for all layers
    for param in net.parameters():        
        param.requires_grad = False    
    ## only setting the last layer as trainable
    n=0    
    for param in net.parameters(): 
        n=n+1
        if n==last_layer:
            param.requires_grad = True
    ## optimizer and scheduler for trojan insertion
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=0.5, momentum =0.9,
        weight_decay=0.000005)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80,120,160], gamma=0.1)
    loader_test = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

    # testing befroe trojan insertion   
    print('--------before trojan insertion!-------')
    test(net1,loader_test)
    test1(net1,loader_test,x_tri)
    # test_seperate(net1,loader_test)

    from bitstring import Bits

    ### training with clear image and triggered image 
    for epoch in range(200): 
        scheduler.step() 
        bit_count = 0
        print('Starting epoch %d / %d' % (epoch + 1, 200)) 
        num_cor = 0

        # net--Trojaned model，net1--clean model
        for t, (x, y) in enumerate(loader_test): 

            ## first loss term -- clean ACC
            x_var, y_var = to_var(x), to_var(y.long()) 
            loss1 = criterion(net(x_var), y_var)

            ## second loss term -- Poisoned ASR
            x_var1,y_var1 = to_var(x), to_var(y.long()) 
            # embedded with trigger 
            x_var1[:,0:3,start:end,start:end] = x_tri[:,0:3,start:end,start:end]
            y_var1[:] = targets
            loss2 = criterion(net(x_var1), y_var1)

            ## third loss term -- BCT
            net_conversion = net_conversion.cuda()
            if epoch == 0:
                pretrained_dict = torch.load(pretained_ann)
            else:
                pretrained_dict = torch.load(model_folder+identifier+'_quantization.pkl')

            model_dict = net_conversion.state_dict()

            # 1) filter out unnecessary keys
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            # 2) overwrite entries in the existing state dict
            model_dict.update(pretrained_dict) 
            # 3) load the new state dict
            net_conversion.load_state_dict(model_dict)

            # update the step size before validation
            for m in net_conversion.modules():
                if isinstance(m, quan_Conv2d) or isinstance(m, quan_Linear):
                    m.__reset_stepsize__()
                    m.__reset_weight__()
            weight_conversion(net_conversion)
            weight_conversion(net1_conversion)

            n = 0
            for param1 in net_conversion.parameters():
                n=n+1
                m=0
                for param in net1_conversion.parameters():
                    m=m+1
                    if n==m:
                        #print(n) 
                        if n==l_last_layer:
                            w=((param1-param))
                            bit_count, count_sta = countingss(param,param1)
                            # print('bit_count')
                            # print(bit_count) ### number of bitflip nb
                            # print('bit_sta')
                            # print(count_sta)
                            # print(w[w!=0].size())  ## number of parameter changed wb

            loss3 = bit_count / (8 * wb)
            if bit_count > n_exp :
                loss3 = (bit_count - n_exp) / (8 * wb)
                
            loss = alpha*loss1 + beta*loss2 + gamma*loss3

            ## ensuring only one test batch is used
            if t not in range(1):
                break 
            if t in range(1): 
                print(t, loss.data) 

            optimizer.zero_grad() 
            loss.backward()          
            optimizer.step()

            ## ensuring only selected op gradient weights are updated 
            n = 0
            for param in net.parameters():
                n = n+1
                m = 0
                for param1 in net1.parameters():
                    m = m+1
                    if n == m:
                        if n == last_layer:
                            w = param - param1  
                            # print(w[w!=0].size())
                            xx = param.data.clone()  ### copying the data of net in xx that is retrained
                            param.data = param1.data.clone() ### net1 is the copying the untrained parameters to net
                            param.data[targets,tar] = xx[targets,tar].clone()  ## putting only the newly trained weights back related to the target class
                            w = param-param1    
                            # print(w[w!=0].size())  

            ## save model weight paramters after each interation
            torch.save(net.state_dict(), model_folder+identifier+'_quantization.pkl')

        # test & save model   
        if (epoch + 1) % 50 == 0:           
            torch.save(net.state_dict(), model_folder+identifier+'.pkl')    ## saving the trojaned model 
            test(net,loader_test)
            test1(net,loader_test,x_tri) 

        print(bit_count) ### number of bit-flip nb 

    print('-----------------after CTBS-----------------')
    test(net,loader_test)
    test1(net,loader_test,x_tri)