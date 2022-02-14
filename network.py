import torch
import torch.nn as nn
import torch.nn.functional as F
from submodule import *

class cb_net(nn.Module):
    def __init__(self,feature_channels):
        super(cb_net, self).__init__()
        self.AB = AB(feature_channels)
        self.head = conv3x3(1,feature_channels)
        self.tail = conv3x3(feature_channels,1)
        self.gateList = [gate() for _ in range(7)]

    def forward(self, cb, y):
        x0 = self.head(cb)
        x_master = x0

        for i in range(6):
            x_branch = self.AB(x_master)
            x_branch = x_branch * self.gateList[i]
            x_master = x_master + x_branch
        
        x_master = x_master *  self.gateList[-1]
        cb_out = self.tail(x_master) + cb

        return cb_out

class cb_y_net(nn.Module):
    def __init__(self,feature_channels):
        super(cb_y_net, self).__init__()
        
        self.cb_enhance = cb_net(feature_channels=feature_channels*2)
        self.cb_head = conv3x3(1,feature_channels)

        self.y_head = nn.Sequential(
            FEB(1,feature_channels),
            FEB(feature_channels,feature_channels),
            FEB(feature_channels,feature_channels),
            FEB(feature_channels,feature_channels),
            conv3x3(feature_channels,feature_channels,stride=2),
            nn.ReLU(inplace=True))

        self.body = nn.Sequential(
            conv3x3(feature_channels,feature_channels),
            nn.ReLU(inplace=True),
            conv3x3(feature_channels,feature_channels),
            nn.ReLU(inplace=True),
            conv3x3(feature_channels,feature_channels),
            nn.ReLU(inplace=True),

            conv3x3(feature_channels,feature_channels//2),
            nn.ReLU(inplace=True),
            conv3x3(feature_channels//2,feature_channels//4),
            nn.ReLU(inplace=True),
            conv3x3(feature_channels//4,1),
            nn.ReLU(inplace=True))

    def forward(self, cb, y):
        cb_0 = self.cb_enhance(cb,y)
        cb_1 = self.cb_head(cb_0)

        y_1 = self.y_head(y)

        cb_output = self.body(cb_1 + y_1)

        return cb_output
