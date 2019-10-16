import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torchvision.models.resnet import BasicBlock


def modified_resnet18():
    model = models.resnet18()
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(512, 14)
    return model
    
def modified_resnet152(num_layers=14):
    model = models.resnet152()
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(2048, num_layers)
    return model
    
def modified_densenet121(num_layers=14):
    model = models.densenet121()
    model.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.classifier = nn.Linear(1024, num_layers)
    return model
    
def modified_densenet201(num_layers=14):
    model = models.densenet201()
    model.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.classifier = nn.Linear(1920, num_layers)
    return model
    
class layer_sharing_resnet(nn.Module):

    def __init__(self, channels=124, num_layers=5, num_classes=14):
        super(layer_sharing_resnet, self).__init__()
        self.num_layers = num_layers
        self.bn = nn.BatchNorm2d(channels)
        # self.relu = nn.ReLU(inplace=False)
        self.upsample = nn.Conv2d(1,channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
        
        self.block = BasicBlock(channels,channels)
        self.pool =  nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.classifier = nn.Linear(channels, num_classes)
        
    def forward(self, x):
        x = self.upsample(x)
        x = self.bn(x)
        # x = self.relu(x)
        x = F.gelu(x)
        x = self.maxpool(x)
        for i in range(self.num_layers):
            x = self.block(x)
            # x = self.relu(x)
            x = F.gelu(x)
            x = self.maxpool(x)
        x = self.pool(x)
        x = torch.flatten(x,1)
        x = self.classifier(x)
        return x
            
class ensembling_network(nn.Module):

    def __init__(self, num_regions=4, num_classes=14):
        super(ensembling_network, self).__init__()
        self.num_regions=num_regions
        
        resnet = models.resnet18()
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        resnet = nn.Sequential(*list(resnet.children())[:-1])
        self.nn_list = nn.ModuleList([resnet for i in range(self.num_regions)])
        self.classifier = nn.Linear(2048, num_classes)
        
    def forward(self, x):
        _,_,h,w = x.shape
        h_mid, w_mid = int(h/2), int(w/2)
        regions = [x[:, :, :h_mid, :w_mid].clone(), x[:, :, :h_mid, w_mid:].clone(), x[:, :, h_mid:, :w_mid].clone(), x[:, :, h_mid:, w_mid:].clone()]
        for i in range(len(self.nn_list)):
            regions[i] = self.nn_list[i](regions[i])
        x = torch.flatten(torch.cat(regions,1),1)
        x = self.classifier(x)
        
        return x