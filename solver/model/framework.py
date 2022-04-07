import torch
import torch.nn as nn
import torch.nn.functional as F
from solver.model.stcn import STCN


class Model(nn.Module):
    def __init__(self, num_channels, window_size, num_gestures, feature_dim=128):
        super(Model, self).__init__()

        stcn = STCN(1, num_points=num_channels, num_classes=num_gestures)
        self.f = []
        for name, module in stcn.named_children():
            if not isinstance(module, nn.Dropout):
                self.f.append(module)
        # encoder
        self.f = nn.Sequential(*self.f)
        # self.f = stcn
        
        # projection head
        self.g = nn.Sequential(
            nn.Linear(num_gestures * window_size * num_channels, 512, bias=False), 
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True), 
            nn.Linear(512, feature_dim, bias=True)
            )

    def forward(self, x): # torch.Size([1, 1, 150, 128])
        x = self.f(x) # torch.Size([1, 8, 150, 128])
        feature = torch.flatten(x, start_dim=1) # torch.Size([1, 153600])
        out = self.g(feature)
        return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)
