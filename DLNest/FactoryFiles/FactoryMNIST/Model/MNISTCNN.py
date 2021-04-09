import torch
import torch.nn as nn

class MNISTModel(nn.Module):
    def __init__(self,args : dict):
        super(MNISTModel, self).__init__()
        feats = args["model_config"]["feats"]
        self.conv1 = nn.Sequential(nn.Conv2d(feats[0],feats[1],kernel_size=3,stride=1,padding=1),
                                   nn.ReLU(),
                                   nn.Conv2d(feats[1],feats[2],kernel_size=3,stride=1,padding=1),
                                   nn.ReLU(),
                                   nn.MaxPool2d(stride=2,kernel_size=2))
        self.dense = nn.Sequential(nn.Linear(14*14*feats[2],feats[3]),
                                   nn.ReLU(),
                                   nn.Dropout(p=0.5),
                                   nn.Linear(1024, 10))
    def forward(self, x):
        x = self.conv1(x)
        x = x.view(-1, 14*14*128)
        x = self.dense(x)
        return x