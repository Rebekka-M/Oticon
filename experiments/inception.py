import warnings
from collections import namedtuple
from functools import partial
from typing import Any, Callable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn, Tensor


__all__ = ["InceptionOutputs", "_InceptionOutputs"]


InceptionOutputs = namedtuple("InceptionOutputs", ["logits", "aux_logits"])
InceptionOutputs.__annotations__ = {"logits": Tensor, "aux_logits": Optional[Tensor]}

# Script annotations failed with _GoogleNetOutputs = namedtuple ...
# _InceptionOutputs set here for backwards compat
_InceptionOutputs = InceptionOutputs

class BasicConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, **kwargs: Any) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
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



class InceptionA(nn.Module):
    def __init__(
        self, in_channels: int, concat_channels: int
    ) -> None:
        super().__init__()
        
        self.in_channels = in_channels
        self.concat_channels = concat_channels

        self.branch1_conv = BasicConv2d(in_channels, 16, kernel_size=1)

        self.branch2_avg = nn.AvgPool2d(kernel_size=3, stride=1)
        self.branch2_conv = BasicConv2d(in_channels, 16, kernel_size=1)

        self.branch3_conv1 = BasicConv2d(in_channels, 16, kernel_size=1)
        self.branch3_conv2 = BasicConv2d(
            16, 32, kernel_size=3, padding="same", padding_mode="replicate"
        )

        self.branch4_conv1 = BasicConv2d(in_channels, 16, kernel_size=1)
        self.branch4_conv2 = BasicConv2d(
            16, 32, kernel_size=3, padding="same", padding_mode="replicate"
        )
        self.branch4_conv3 = BasicConv2d(
            32, 32, kernel_size=3, padding="same", padding_mode="replicate"
        )
        
        self.feat_cat = FeatureConcatenation(concat_channels)


    def _forward(self, x: Tensor) -> List[Tensor]:
        branch1 = self.branch1_conv(x)

        branch2 = F.pad(x, (1, 1, 1, 1), mode="replicate")
        branch2 = self.branch2_avg(branch2)
        branch2 = self.branch2_conv(branch2)

        branch3 = self.branch3_conv1(x)
        branch3 = self.branch3_conv2(branch3)
        # branch3 = self.branch3_dw_conv(branch3)
        # branch3 = self.branch3_pw_conv(branch3)

        branch4 = self.branch4_conv1(x)
        branch4 = self.branch4_conv2(branch4)
        branch4 = self.branch4_conv3(branch4)

        # Concatenate the outputs
        return [branch1, branch2, branch3, branch4]


    def forward(self, x: Tensor) -> Tensor:
        z = self._forward(x)
        z = torch.cat(z, 1)
        z = self.feat_cat(z)

        return torch.cat([x] + [z], 1)


class InceptionB(nn.Module):
    def __init__(
        self, in_channels: int, concat_channels: int
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.concat_channels = concat_channels

        self.branch1_conv = BasicConv2d(in_channels, 32, kernel_size=1)

        self.branch2_avg = nn.AvgPool2d(kernel_size=3, stride=1)
        self.branch2_conv = BasicConv2d(
            in_channels, 32, kernel_size=1, padding="same", padding_mode="replicate"
        )

        self.branch3_conv1 = BasicConv2d(in_channels, 32, kernel_size=1, stride=1)
        self.branch3_dw_conv = BasicConv2d(
            32, 32, kernel_size=3, groups=32, padding="same", padding_mode="replicate"
        )
        self.branch3_pw_conv = BasicConv2d(
            32, 32, kernel_size=1, padding="same", padding_mode="replicate"
        )

        self.branch4_conv1 = BasicConv2d(in_channels, 32, kernel_size=1)
        self.branch4_dw_conv1 = BasicConv2d(
            32, 32, kernel_size=3, groups=32, padding="same", padding_mode="replicate"
        )
        self.branch4_pw_conv1 = BasicConv2d(
            32, 32, kernel_size=1, padding="same", padding_mode="replicate"
        )


        self.branch4_dw_conv2 = BasicConv2d(
            32, 32, kernel_size=3, groups=32, padding="same", padding_mode="replicate"
        )
        self.branch4_pw_conv2 = BasicConv2d(
            32, 32, kernel_size=1, padding="same", padding_mode="replicate"
        )
        
        self.feat_cat = FeatureConcatenation(concat_channels)


    def _forward(self, x: Tensor) -> List[Tensor]:
        branch1 = self.branch1_conv(x)

        branch2 = F.pad(x, (1, 1, 1, 1), mode="replicate")
        branch2 = self.branch2_avg(branch2)
        branch2 = self.branch2_conv(branch2)

        branch3 = self.branch3_conv1(x)
        branch3 = self.branch3_dw_conv(branch3)
        branch3 = self.branch3_pw_conv(branch3)

        branch4 = self.branch4_conv1(x)
        branch4 = self.branch4_dw_conv1(branch4)
        branch4 = self.branch4_pw_conv1(branch4)

        branch4 = self.branch4_dw_conv2(branch4)
        branch4 = self.branch4_pw_conv2(branch4)

        return [branch1, branch2, branch3, branch4]


    def forward(self, x: Tensor) -> Tensor:
        z = self._forward(x)
        # print(z[0].shape, z[1].shape, z[2].shape, z[3].shape)
        z = torch.cat(z, 1)
        # print(z.shape)
        z = self.feat_cat(z)
        # print(z.shape)
        # print(torch.cat([x] + [z], 1).shape)

        return torch.cat([x] + [z], 1)

