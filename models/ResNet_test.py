import torch
import torch.nn as nn
import math
import torch.nn.functional as F

##------------------------------------------------------------ definition of basic layer of test ------------------------------------------------------------------
# normalize layer
class Normalize_layer(nn.Module):
    
    def __init__(self, mean, std):
        super(Normalize_layer, self).__init__()
        self.mean = nn.Parameter(torch.Tensor(mean).unsqueeze(1).unsqueeze(1), requires_grad=False)
        self.std = nn.Parameter(torch.Tensor(std).unsqueeze(1).unsqueeze(1), requires_grad=False)
        
    def forward(self, input):
        
        return input.sub(self.mean).div(self.std)

# quantization function
class _quantize_func(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, step_size, half_lvls):
        # ctx is a context object that can be used to stash information
        # for backward computation
        ctx.step_size = step_size
        ctx.half_lvls = half_lvls
        output = F.hardtanh(input,
                            min_val=-ctx.half_lvls * ctx.step_size.item(),
                            max_val=ctx.half_lvls * ctx.step_size.item())

        output = torch.round(output/ctx.step_size)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()/ctx.step_size

        return grad_input, None, None
quantize = _quantize_func.apply
# Quantized Convolutional Layer
class quan_Conv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True,pni='layerwise',w_noise=True):
        super(quan_Conv2d, self).__init__(in_channels, out_channels, kernel_size,
                                          stride=stride, padding=padding, dilation=dilation,
                                          groups=groups, bias=bias)
        self.pni = pni
        if self.pni is 'layerwise':
            self.alpha_w = nn.Parameter(torch.Tensor([0.25]), requires_grad = True)
        elif self.pni is 'elementwise':
            self.alpha_w = nn.Parameter(self.weight.clone().fill_(0.1), requires_grad = True)
        
        self.w_noise = w_noise
        self.N_bits = 8
        self.full_lvls = 2**self.N_bits
        self.half_lvls = (self.full_lvls-2)/2
        # Initialize the step size
        self.step_size = nn.Parameter(torch.Tensor([1]), requires_grad=True)
        self.__reset_stepsize__()
        # flag to enable the inference with quantized weight or self.weight
        self.inf_with_weight = False  # disabled by default
        
        # create a vector to identify the weight to each bit
        self.b_w = nn.Parameter(
            2**torch.arange(start=self.N_bits-1,end=-1, step=-1).unsqueeze(-1).float(),
            requires_grad = False)
        
        self.b_w[0] = -self.b_w[0] #in-place change MSB to negative
        

    def forward(self, input):
        with torch.no_grad():
            std = self.weight.std().item()
            noise = self.weight.clone().normal_(0,std)

        noise_weight = self.weight + self.alpha_w * noise * self.w_noise
        if self.inf_with_weight:
            return F.conv2d(input, noise_weight*self.step_size, self.bias, self.stride, self.padding,
                            self.dilation, self.groups)
        else:
            weight_quan = quantize(noise_weight, self.step_size,
                                   self.half_lvls)*self.step_size
            return F.conv2d(input, weight_quan, self.bias, self.stride, self.padding, self.dilation,
                            self.groups)

    def __reset_stepsize__(self):
        with torch.no_grad():
            self.step_size.data = self.weight.abs().max()/self.half_lvls

    def __reset_weight__(self):
        '''
        This function will reconstruct the weight stored in self.weight.
        Replacing the orginal floating-point with the quantized fix-point
        weight representation.
        '''
        # replace the weight with the quantized version
        with torch.no_grad():
            self.weight.data = quantize(
                self.weight, self.step_size, self.half_lvls)
        # enable the flag, thus now computation does not invovle weight quantization
        self.inf_with_weight = True
# Linear layer (FC layer)     
class quan_Linear(nn.Linear):

    def __init__(self, in_features, out_features, bias=True):
        super(quan_Linear, self).__init__(in_features, out_features, bias=bias)

        self.N_bits = 8
        self.full_lvls = 2**self.N_bits
        self.half_lvls = (self.full_lvls-2)/2
        # Initialize the step size
        self.step_size = nn.Parameter(torch.Tensor([1]), requires_grad=True)
        self.__reset_stepsize__()
        # flag to enable the inference with quantized weight or self.weight
        self.inf_with_weight = False  # disabled by default
        
        # create a vector to identify the weight to each bit
        self.b_w = nn.Parameter(
            2**torch.arange(start=self.N_bits-1,end=-1, step=-1).unsqueeze(-1).float(),
            requires_grad = False)
        
        self.b_w[0] = -self.b_w[0] #in-place reverse

    def forward(self, input):
        if self.inf_with_weight:
            return  F.linear(input, self.weight*self.step_size, self.bias)
        else: 
            weight_quan = quantize(self.weight, self.step_size,
                               self.half_lvls)*self.step_size
            return F.linear(input, weight_quan, self.bias)

    def __reset_stepsize__(self):
        with torch.no_grad():
            self.step_size.data = self.weight.abs().max()/self.half_lvls

    def __reset_weight__(self):
        '''
        This function will reconstruct the weight stored in self.weight.
        Replacing the orginal floating-point with the quantized fix-point
        weight representation.
        '''
        # replace the weight with the quantized version
        with torch.no_grad():
            self.weight.data = quantize(
                self.weight, self.step_size, self.half_lvls)
        # enable the flag, thus now computation does not invovle weight quantization
        self.inf_with_weight = True    


# Resnet model
class BasicBlock_test(nn.Module): 
    expansion = 1 

    def __init__(self, in_planes, planes, stride=1): 
        super(BasicBlock_test, self).__init__() 
        self.conv1 = quan_Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False) 
        self.bn1 = nn.BatchNorm2d(planes) 
        self.conv2 = quan_Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False) 
        self.bn2 = nn.BatchNorm2d(planes) 
        #self.l=nn.Parameter(torch.cuda.FloatTensor([0.0]), requires_grad=True)  

        self.shortcut = nn.Sequential() 
        if stride != 1 or in_planes != self.expansion*planes: 
            self.shortcut = nn.Sequential( 
                quan_Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride,padding=0, bias=False), 
                nn.BatchNorm2d(self.expansion*planes) 
            ) 

    def forward(self, x): 
        out = F.relu(self.bn1(self.conv1(x))) 
        out = self.bn2(self.conv2(out)) 
        out += self.shortcut(x) 
        out = F.relu(out) 
        #print('value2') 
        #print(self.l)  
        return out 

class ResNet_test(nn.Module): 
    def __init__(self, block, num_blocks, num_classes=10): 
        super(ResNet_test, self).__init__() 
        self.in_planes = 64 

        self.conv1 = quan_Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False) 
        self.bn1 = nn.BatchNorm2d(64) 
        #self.m = nn.MaxPool2d(5, stride=5) 
        #self.lin = nn.Linear(64*6*6,1) 
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1) 
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2) 
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2) 
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2) 
        self.linear = quan_Linear(512*block.expansion, num_classes) 
        #self.l=nn.Parameter(torch.cuda.FloatTensor([0.0]), requires_grad=True) 
        

    def _make_layer(self, block, planes, num_blocks, stride): 
        strides = [stride] + [1]*(num_blocks-1) 
        layers = [] 
        for stride in strides: 
            layers.append(block(self.in_planes, planes, stride)) 
            self.in_planes = planes * block.expansion 
        return nn.Sequential(*layers) 

    def forward(self, x): 
         
        out = F.relu(self.bn1(self.conv1(x))) 
        out = self.layer1(out) 
        out = self.layer2(out) 
        out = self.layer3(out) 
        out = self.layer4(out) 
        out = F.avg_pool2d(out, 4) 
        out1 = out.view(out.size(0), -1) 
        out = self.linear(out1) 
        return out

def ResNet_test(args, mean, std): 
    if args.model == 'ResNet18':
        net = torch.nn.Sequential(
                        Normalize_layer(mean,std),
                        ResNet_test(BasicBlock_test, [2,2,2,2]) 
                        )
    elif args.model == 'ResNet34':
        net = torch.nn.Sequential(
                        Normalize_layer(mean,std),
                        ResNet_test(BasicBlock_test, [3,4,6,3]) 
                        )
    return net
