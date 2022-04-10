import torch
import torch.nn as nn

from solver.utils import data_reader
from solver.model import framework

class Net(nn.Module):
    def __init__(self, num_channels, window_size, num_gestures, pretrained_path, feature_dim=128):
        super(Net, self).__init__()

        # encoder
        self.f = framework.Model(num_channels, window_size, num_gestures, feature_dim).f
        # classifier
        self.fc = nn.Linear(num_gestures*window_size*num_channels, num_gestures, bias=True)
        # self.load_state_dict(torch.load(pretrained_path, map_location='cpu'), strict=False)
        self.load_state_dict(torch.load(pretrained_path), strict=False)

    # torch.Size([16, 1, 150, 128])
    def forward(self, x):
        x = self.f(x) # torch.Size([16, 8, 150, 128])
        feature = torch.flatten(x, start_dim=1) # torch.Size([16, 153600])
        out = self.fc(feature) # torch.Size([16, 8])
        # np.save(f"matrix/B0.npy", self.B.cpu().detach().numpy())  
        return out