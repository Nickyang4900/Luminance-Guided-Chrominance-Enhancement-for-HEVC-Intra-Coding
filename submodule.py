import torch
import torch.nn as nn
import torch.nn.functional as F

def conv1x1(in_channels, out_channels, stride=1, bias=True):
    return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                        kernel_size=1, stride=stride, padding=0, bias=bias)

def conv3x3(in_channels, out_channels, stride=1, bias=True):
    return nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                        kernel_size=3, stride=stride, padding=1, bias=bias)

def gate():
    return nn.Parameter(torch.Tensor([1]))

class AU(nn.Module):
    """Asymmetric Unit"""
    def __init__(self, n):
        super(AU, self).__init__()
        self.b1 = nn.Sequential(
            conv1x1(n,n),
            nn.LeakyReLU(inplace=True))
        self.b2 = nn.Sequential(
            conv1x1(n,n),
            nn.LeakyReLU(inplace=True),
            conv3x3(n,n), 
            nn.LeakyReLU(inplace=True))
        self.b3 = nn.Sequential(
            conv1x1(n,n),
            nn.LeakyReLU(inplace=True),
            nn.AvgPool2d(kernel_size=3,stride=1,padding=1),
            nn.LeakyReLU(inplace=True))
        self.b4 = nn.Sequential(
            conv3x3(n,n),
            nn.LeakyReLU(inplace=True))
    def forward(self, x):
        y = self.b1(x)+self.b2(x)+self.b3(x)+self.b4(x)
        return y
        
class AB(nn.Module):
    def __init__(self, n):
        super(AB, self).__init__()
        self.au1 = AU(n)
        self.au2 = AU(n)
        self.tail = conv3x3(n,n)
    def forward(self, x):
        x1 = self.au1(x)
        x2 = self.au2(x1)
        x3 = self.au2(x2+x1)
        x4 = self.au2(x3+x1)
        y = self.tail(x4)
        return y

class FEB(torch.nn.Module):
    def __init__(self, in_channels, feature_channels):
        super(FEB, self).__init__()
        self.c1 = nn.Sequential(conv3x3(in_channels,feature_channels), nn.ReLU(inplace=True))
        self.c2 = nn.Sequential(conv3x3(feature_channels,feature_channels), nn.ReLU(inplace=True))
        self.c3 = nn.Sequential(conv3x3(feature_channels,feature_channels), nn.ReLU(inplace=True))
        self.c4 = nn.Sequential(conv3x3(feature_channels,feature_channels), nn.ReLU(inplace=True))
    def forward(self, x):
        x1 = self.c1(x)
        x2 = self.c2(x1)
        x3 = self.c3(x2)
        x4 = self.c4(x3)
        return x1 + x2 + x3 + x4

