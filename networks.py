import torch
import torch.nn as nn
import torchvision.models as models
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

    def __init__(self, channels=64, num_layers=3, num_classes=14):
        super(layer_sharing_resnet, self).__init__()
        self.num_layers = num_layers
        self.upsample = BasicBlock(1,channels)
        self.block = BasicBlock(channels,channels)
        self.pool =  nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.classifier = nn.Linear(channels, num_classes)
        
    def forward(self, x):
        x = self.upsample(x)
        for i in range(self.num_layers):
            x = self.block(x)
        x = self.pool(x)
        x = torch.flatten(x,1)
        x = self.classifier(x)
        return x
            
            