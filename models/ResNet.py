import torch
import torch.nn as nn
import math
import torch.nn.functional as F

## --------------------------------------------------------------- definition of basic layer of TBT -------------------------------------------------------
# normalize layer
class Normalize_layer(nn.Module):
    
    def __init__(self, mean, std):
        super(Normalize_layer, self).__init__()
        self.mean = nn.Parameter(torch.Tensor(mean).unsqueeze(1).unsqueeze(1), requires_grad=False)
        self.std = nn.Parameter(torch.Tensor(std).unsqueeze(1).unsqueeze(1), requires_grad=False)
        
    def forward(self, input):
        
        return input.sub(self.mean).div(self.std)

# quantization function
class _Quantize(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, input, step):         
        ctx.step = step.item()
        output = torch.round(input/ctx.step)
        return output
                
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()/ctx.step
        return grad_input, None
                
quantize1 = _Quantize.apply

# Quantized Convolutional Layer
class quantized_conv(nn.Conv2d):

    def __init__(self,nchin,nchout,kernel_size,stride,padding='same',bias=False):
        super().__init__(in_channels=nchin,out_channels=nchout, kernel_size=kernel_size, padding=padding, stride=stride, bias=False)
        #self.N_bits = 7
        #step = self.weight.abs().max()/((2**self.N_bits-1))
        #self.step = nn.Parameter(torch.Tensor([step]), requires_grad = False)
    
    def forward(self, input):
        self.N_bits = 7
        step = self.weight.abs().max()/((2**self.N_bits-1))
        QW = quantize1(self.weight, step)
        return F.conv2d(input, QW*step, self.bias,
                        self.stride, self.padding, self.dilation, self.groups)
    
# Linear layer (FC layer)
class bilinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features)
        #self.N_bits = 7
        #step = self.weight.abs().max()/((2**self.N_bits-1))
        #self.step = nn.Parameter(torch.Tensor([step]), requires_grad = False)
        #self.weight.data = quantize(self.weight, self.step).data.clone()  
        
    def forward(self, input):
       
        self.N_bits = 7
        step = self.weight.abs().max()/((2**self.N_bits-1))
        
        QW = quantize1(self.weight, step)
       
        
        return F.linear(input, QW*step, self.bias)


##--------------------------------------------------------------------- definition of model of TBT ---------------------------------------------------------------
# Resnet 18 model pretrained 
class BasicBlock(nn.Module): 
    expansion = 1 

    def __init__(self, in_planes, planes, stride=1): 
        super(BasicBlock, self).__init__() 
        # in_planes ---in channels
        # planes ---out channels
        self.conv1 = quantized_conv(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False) 
        self.bn1 = nn.BatchNorm2d(planes) 
        self.conv2 = quantized_conv(planes, planes, kernel_size=3, stride=1, padding=1, bias=False) 
        self.bn2 = nn.BatchNorm2d(planes) 
        #self.l=nn.Parameter(torch.cuda.FloatTensor([0.0]), requires_grad=True)  

        self.shortcut = nn.Sequential() 
        if stride != 1 or in_planes != self.expansion*planes: 
            self.shortcut = nn.Sequential( 
                quantized_conv(in_planes, self.expansion*planes, kernel_size=1, stride=stride,padding=0, bias=False), 
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

# model structure
class ResNet(nn.Module): 
    def __init__(self, block, num_blocks, num_classes=10): 
        super(ResNet, self).__init__() 
        self.in_planes = 64 

        self.conv1 = quantized_conv(3, 64, kernel_size=3, stride=1, padding=1, bias=False) 
        self.bn1 = nn.BatchNorm2d(64) 
        #self.m = nn.MaxPool2d(5, stride=5) 
        #self.lin = nn.Linear(64*6*6,1) 
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1) 
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2) 
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2) 
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2) 
        self.linear = bilinear(512*block.expansion, num_classes) 
        #self.l=nn.Parameter(torch.cuda.FloatTensor([0.0]), requires_grad=True) 
        
    def _make_layer(self, block, planes, num_blocks, stride): 
        strides = [stride] + [1]*(num_blocks-1) 
        layers = [] 
        for stride in strides: 
            layers.append(block(self.in_planes, planes, stride)) 
            self.in_planes = planes * block.expansion 
        return nn.Sequential(*layers) 

    # 前向传播
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

## netwrok to generate the trigger  removing the last layer.
class ResNet_activation(nn.Module): 
    def __init__(self, block, num_blocks, num_classes=10): 
        super(ResNet_activation, self).__init__() 
        self.in_planes = 64 

        self.conv1 = quantized_conv(3, 64, kernel_size=3, stride=1, padding=1, bias=False) 
        self.bn1 = nn.BatchNorm2d(64) 
        
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1) 
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2) 
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2) 
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2) 
        self.linear = bilinear(512*block.expansion, num_classes) 
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
        out = out.view(out.size(0), -1) 
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

# final output，activation output
class ResNet_full(nn.Module): 
    def __init__(self, block, num_blocks, num_classes=10): 
        super(ResNet_full, self).__init__() 
        self.in_planes = 64 

        self.conv1 = quantized_conv(3, 64, kernel_size=3, stride=1, padding=1, bias=False) 
        self.bn1 = nn.BatchNorm2d(64) 
        #self.m = nn.MaxPool2d(5, stride=5) 
        #self.lin = nn.Linear(64*6*6,1) 
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1) 
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2) 
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2) 
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2) 
        self.linear = bilinear(512*block.expansion, num_classes) 
        #self.l=nn.Parameter(torch.cuda.FloatTensor([0.0]), requires_grad=True) 
        
    def _make_layer(self, block, planes, num_blocks, stride): 
        strides = [stride] + [1]*(num_blocks-1) 
        layers = [] 
        for stride in strides: 
            layers.append(block(self.in_planes, planes, stride)) 
            self.in_planes = planes * block.expansion 
        return nn.Sequential(*layers) 

    # 前向传播
    def forward(self, x): 
        out = F.relu(self.bn1(self.conv1(x))) 
        out = self.layer1(out) 
        out = self.layer2(out) 
        out = self.layer3(out) 
        out = self.layer4(out) 
        out = F.avg_pool2d(out, 4) 
        out1 = out.view(out.size(0), -1) 
        # out1.requires_grad = True
        out = self.linear(out1) 
        return out, out1

class Classifier(nn.Module):
    def __init__(self, num_classes=10): 
        super(Classifier, self).__init__() 
        self.in_planes = 512
        self.linear = bilinear(512, num_classes)
    def forward(self, x):
        out = self.linear(x) 
        return out

def ResNet_activation(args, mean, std):
        
    if args.model == 'ResNet18':
        net_activation = torch.nn.Sequential(
                        Normalize_layer(mean,std),
                        ResNet_activation(BasicBlock, [2,2,2,2]) 
                        )
    elif args.model == 'ResNet34':
        net_activation = torch.nn.Sequential(
                        Normalize_layer(mean,std),
                        ResNet_activation(BasicBlock, [3,4,6,3]) 
                        )
    return net_activation

def ResNet(args, mean, std): 
    if args.model == 'ResNet18':
        net = torch.nn.Sequential(
                        Normalize_layer(mean,std),
                        ResNet(BasicBlock, [2,2,2,2]) 
                        )
    elif args.model == 'ResNet34':
        net = torch.nn.Sequential(
                        Normalize_layer(mean,std),
                        ResNet(BasicBlock, [3,4,6,3]) 
                        )
    return net

def ResNet_full(args, mean, std): 
    if args.model == 'ResNet18':
        net = torch.nn.Sequential(
                        Normalize_layer(mean,std),
                        ResNet(BasicBlock, [2,2,2,2]) 
                        )
    elif args.model == 'ResNet34':
        net = torch.nn.Sequential(
                        Normalize_layer(mean,std),
                        ResNet(BasicBlock, [3,4,6,3]) 
                        )
    return net

