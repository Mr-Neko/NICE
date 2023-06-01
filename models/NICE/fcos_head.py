import torch
import torch.nn as nn

class Scale(nn.Module):
    def __init__(self, init=1.0):
        super().__init__()

        self.scale = nn.Parameter(torch.tensor([init], dtype=torch.float32))

    def forward(self, input):
        return input * self.scale

class fcos_head(nn.Module):

    def __init__(self, in_channel, n_conv):

        super(fcos_head, self).__init__()
        bbox_tower = []
        for _ in range(n_conv):

            bbox_tower.append(
                nn.Conv2d(in_channel, in_channel, 3, padding=1, bias=False)
            )
            bbox_tower.append(nn.GroupNorm(32, in_channel))
            bbox_tower.append(nn.ReLU())
        
        self.bbox_tower = nn.Sequential(*bbox_tower)
        self.bbox_pred = nn.Conv2d(in_channel, 4, 3, padding=1)
        self.scale = Scale(1.0)
    
    def forward(self, input):

        bbox_out = self.bbox_tower(input)
        bbox_out = nn.Sigmoid()(self.scale(self.bbox_pred(bbox_out)))

        return bbox_out
    
