import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torchvision.models.resnet import BasicBlock


def modified_resnet18(num_classes=14):
    model = models.resnet18()
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(512, num_classes)
    return model
    
def modified_resnet152(num_classes=14):
    model = models.resnet152()
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(2048, num_classes)
    return model
    
def modified_densenet121(num_classes=14):
    model = models.densenet121()
    model.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.classifier = nn.Linear(1024, num_classes)
    return model
    
def modified_densenet201(num_classes=14):
    model = models.densenet201()
    model.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.classifier = nn.Linear(1920, num_classes)
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
        
        
class masked_duo_model(nn.Module):

    def __init__(self, frontal_model, lateral_model,  num_classes=5):
        super(masked_duo_model, self).__init__()
        self.num_class = num_classes
        self.lateral_model = lateral_model
        self.frontal_model = frontal_model
        self.classifier = nn.Linear(num_classes*2, num_classes)
        
    def forward(self, x):
        frontal_img, lateral_img = x[:,0], x[:,1]
        mask = torch.zeros(x.shape[0], self.num_class*2).cuda()
        if (frontal_img>0).any():
            mask[:, :self.num_class] = 1
        if (frontal_img>0).any():
            mask[self.num_class:] = 1
        y1 = self.frontal_model(frontal_img)
        y2 = self.lateral_model(lateral_img)
        y1 = F.sigmoid(y1)
        y1 = F.sigmoid(y2)
        y = torch.cat([y1,y2],1)
        y = y * mask
        y = self.classifier(y)
        return y  
        
        
        
class anytime_prediction_model(nn.Module):

    def __init__(self, num_classes=5, intermediate_size=4):
        super(anytime_prediction_model, self).__init__()
        self.prelim_layers = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),*(list(models.resnet18().children())[1:4]))
        self.layers = nn.ModuleList(models.resnet18().children())[4:-1]
        self.avgpool = nn.AdaptiveAvgPool2d((intermediate_size,intermediate_size))
        self.linear1 = nn.Linear(64*intermediate_size*intermediate_size, intermediate_size)
        self.linear2 = nn.Linear(128*intermediate_size*intermediate_size, intermediate_size)
        self.linear3 = nn.Linear(256*intermediate_size*intermediate_size, intermediate_size)
        self.linear4 = nn.Linear(512*intermediate_size*intermediate_size, intermediate_size)
        self.classifier = nn.Linear(4*intermediate_size, num_classes)
        
    def forward(self, x):
        x = self.prelim_layers(x)
        
        x = self.layers[0](x)
        a1 = self.avgpool(x)
        a1 = torch.flatten(a1,1)
        a1 = self.linear1(a1)
        
        
        x = self.layers[1](x)
        a2 = self.avgpool(x)
        a2 = torch.flatten(a2,1)
        a2 = self.linear2(a2)
        
        x = self.layers[2](x)
        a3 = self.avgpool(x)
        a3 = torch.flatten(a3,1)
        a3 = self.linear3(a3)

        x = self.layers[3](x)
        x = self.avgpool(x)
        x = torch.flatten(x,1)
        x = self.linear4(x)
        
        y = torch.cat([a1,a2,a3,x],1)
        y = self.classifier(y)
        
        return y

class final_prediction_model(nn.Module):

    def __init__(self, model1=None, model2=None, model3=None, num_classes=5):
        super(final_prediction_model, self).__init__()
        self.num_class = num_classes
        if model1 == None:
            model1 = modified_resnet18()
        else:
            self.model1 = model1
        if model2 == None:
            model2 = modified_resnet152()
        else:
            self.model2 = model2
        if model3 == None:
            model3= modified_densenet121()
        else:
            self.model3 = model3
            
        self.classifier = nn.Linear(num_classes*3, num_classes)
        
    def forward(self, x):
        y1 = F.softmax(self.model1(x))
        y2 = F.softmax(self.model2(x))
        y3 = F.softmax(self.model3(x))
        y = torch.cat([y1,y2,y3],1)
        y = self.classifier(y)
        return y  
    