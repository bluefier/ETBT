import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import kl_divergence
from torch.distributions import Categorical

def js_divergence(p, q):
        m = 0.5 * (p + q)
        return 0.5 * kl_divergence(Categorical(probs=p), Categorical(probs=m)) + 0.5 * kl_divergence(Categorical(probs=q), Categorical(probs=m))

class Attack(object):

    def __init__(self, dataloader, start, end, criterion=None, gpu_id=0, 
                epsilon=0.031, attack_method='pgd'):
        
        if criterion is not None:
            self.criterion =  nn.MSELoss() 
        else:
            self.criterion = nn.MSELoss()
            
        self.dataloader = dataloader
        self.epsilon = epsilon
        self.gpu_id = gpu_id      # this is integer

        if attack_method is 'fgsm':
            self.attack_method = self.fgsm
        elif attack_method is 'pgd':
            self.attack_method = self.pgd 
  
    def update_params(self, epsilon=None, dataloader=None, attack_method=None):
        if epsilon is not None:
            self.epsilon = epsilon
        if dataloader is not None:
            self.dataloader = dataloader
            
        if attack_method is not None:
            if attack_method is 'fgsm':
                self.attack_method = self.fgsm
    
        
    # fgsm                               
    def fgsm(self, model, data, target,tar,ep, Pu, start, end, data_min=0, data_max=1):
        
        lambda1 = 0.001
        model.eval()
        perturbed_data = data.clone()
        
        perturbed_data.requires_grad = True
        output = model(perturbed_data)
        loss = self.criterion(output[:,tar], target[:,tar])

        V = output[:, tar]
        Q_prob = F.softmax(V, dim=0) 
        # calculate loss term for JSD
        loss_js = js_divergence(Pu, Q_prob)

        loss_com = lambda1*loss + loss_js

        if perturbed_data.grad is not None:
            perturbed_data.grad.data.zero_()

        loss_com.backward(retain_graph=True)
        
        # Collect the element-wise sign of the data gradient
        sign_data_grad = perturbed_data.grad.data.sign()
        perturbed_data.requires_grad = False

        with torch.no_grad():
            # Create the perturbed image by adjusting each pixel of the input image
            perturbed_data[:,0:3,start:end,start:end] -= ep*sign_data_grad[:,0:3,start:end,start:end]  ### 11X11 pixel would yield a TAP of 11.82 % 
            perturbed_data.clamp_(data_min, data_max) 
    
        return perturbed_data, loss_js
