from typing import Any
import torch
import torch.nn.functional as F
from torch import nn, Tensor


class BasicConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, **kwargs: Any) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, 
            out_channels, 
            bias=False, 
            padding="same", 
            padding_mode="replicate", 
            **kwargs
        )
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


class FeatureConcatenation(nn.Module):
    def __init__(self, channels) -> None:
        super().__init__()
        self.channels = channels
        self.filter = nn.Sequential(
            nn.BatchNorm2d(channels, eps=0.001),
            nn.ReLU(inplace=True)
        )


    def forward(self, x: Tensor) -> Tensor:
        return self.filter(x)
    
    
class MeanModule(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        
        self.args = args
        self.kwargs = kwargs

    def forward(self, x):
        return x.mean(*self.args, **self.kwargs)


class InceptionA(nn.Module):
    def __init__(
        self, in_channels: int, concat_channels: int
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.concat_channels = concat_channels


        self.branch1 = BasicConv2d(in_channels, 16, kernel_size=1)

        self.branch2 = nn.Sequential(
            nn.ReplicationPad2d((1, 1, 1, 1)),
            nn.AvgPool2d(kernel_size=3, stride=1),
            BasicConv2d(in_channels, 16, kernel_size=1)
        )

        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, 16, kernel_size=1),
            BasicConv2d(16, 32, kernel_size=3)
        )
        
        self.branch4 = nn.Sequential(
            BasicConv2d(in_channels, 16, kernel_size=1),
            BasicConv2d(16, 32, kernel_size=3),
            BasicConv2d(32, 32, kernel_size=3)
        )

        self.feat_cat = FeatureConcatenation(concat_channels)


    def forward(self, x: Tensor) -> Tensor:
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        z = torch.cat([branch1, branch2, branch3, branch4], 1)
        z = self.feat_cat(z)

        return torch.cat([x] + [z], 1)



class InceptionB(nn.Module):
    def __init__(
        self, in_channels: int, concat_channels: int
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.concat_channels = concat_channels

        self.branch1 = BasicConv2d(in_channels, 32, kernel_size=1)

        self.branch2 = nn.Sequential(
            nn.ReplicationPad2d((1, 1, 1, 1)),
            nn.AvgPool2d(kernel_size=3, stride=1),
            BasicConv2d(in_channels, 32, kernel_size=1)
        )

        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, 32, kernel_size=1, stride=1),
            # Depth-wise separable convolution
            BasicConv2d(32, 32, kernel_size=3, groups=32),
            BasicConv2d(32, 32, kernel_size=1)
        )

        self.branch4 = nn.Sequential(
            BasicConv2d(in_channels, 32, kernel_size=1),
            
            # Depth-wise separable convolution
            BasicConv2d(32, 32, kernel_size=3, groups=32),
            BasicConv2d(32, 32, kernel_size=1),

            # Depth-wise separable convolution
            BasicConv2d(32, 32, kernel_size=3, groups=32),
            BasicConv2d(32, 32, kernel_size=1),
        )

        self.feat_cat = FeatureConcatenation(concat_channels)


    def forward(self, x: Tensor) -> Tensor:
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        z = torch.cat([branch1, branch2, branch3, branch4], 1)
        z = self.feat_cat(z)

        return torch.cat([x] + [z], 1)

