import torchvision.models as models
import torch.nn as nn

class resnet50model(nn.Module):
    def __init__(self,num_classes=10):
        super().__init__()
        self.features=models.resnet50(pretrained=True,progress = True)
        self.nonlinear=nn.ReLU()
        self.out=nn.Linear(self.features.fc.out_features,num_classes)
    def forward(self,x):
        x=self.nonlinear(self.features(x))
        x=self.out(x)
        return x
        