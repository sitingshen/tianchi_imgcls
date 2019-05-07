from torch import nn
import torch as t
from torch.nn import functional as F
import torchvision.models as models



class Net(nn.Module):

    def __init__(self, batch=16,num_classes=2048):
        super(Net, self).__init__()
        self.resnet50 = models.resnet50(pretrained=True)
        self.resnet50.fc = nn.Linear(2048, 2)

        self.vgg=models.vgg16(pretrained=True)

        self.vgg.classifier[6] = nn.Linear(4096,2)


    def forward(self, x):
        x = self.vgg(x)
        return x
