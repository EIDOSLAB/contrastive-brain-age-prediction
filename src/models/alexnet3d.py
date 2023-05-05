"""
Model implemented in https://doi.org/10.5281/zenodo.4309677 by Abrol et al., 2021
"""
from torch import nn
import torch.nn.functional as F
import math

class AlexNet3D(nn.Module):
    def __init__(self):
        """
        :param num_classes: int, number of classes
        :param mode:  "classifier" or "encoder" (returning 128-d vector)
        """
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=5, stride=2, padding=0),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=3),

            nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=3),

            nn.Conv3d(128, 192, kernel_size=3, padding=1),
            nn.BatchNorm3d(192),
            nn.ReLU(inplace=True),

            nn.Conv3d(192, 192, kernel_size=3, padding=1),
            nn.BatchNorm3d(192),
            nn.ReLU(inplace=True),

            nn.Conv3d(192, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool3d(1),
        )


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        xp = self.features(x)
        x = xp.view(xp.size(0), -1)
        return x

class SupConAlexNet(nn.Module):
    """backbone + projection head"""
    def __init__(self, head='mlp', feat_dim=128):
        super().__init__()
        self.encoder = AlexNet3D()
        dim_in = 128
        
        if head == 'linear':
            self.head = nn.Linear(dim_in, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )

        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))

    def forward(self, x):
        feat = self.encoder(x)
        feat = F.normalize(self.head(feat), dim=1)
        return feat
    
    def features(self, x):
        return self.forward(x)


class SupRegAlexNet(nn.Module):
    """encoder + regressor"""
    def __init__(self,):
        super().__init__()
        self.encoder = AlexNet3D()
        self.fc = nn.Linear(128, 1)

    def forward(self, x):
        feats = self.features(x)
        return self.fc(feats), feats
    
    def features(self, x):
        return self.encoder(x)