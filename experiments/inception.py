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


class InceptionA(nn.Module):
    def __init__(
        self, in_channels: int,conv_block: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super().__init__()
        if conv_block is None:
            conv_block = BasicConv2d


        self.branch1_conv = conv_block(in_channels, 16, kernel_size=1)

        self.branch2_avg = nn.AvgPool2d(kernel_size=3, stride=1)
        self.branch2_conv = conv_block(in_channels, 16, kernel_size=1)

        self.branch3_conv1 = conv_block(in_channels, 16, kernel_size=1)
        self.branch3_conv2 = conv_block(16, 32, kernel_size=3, padding="same", padding_mode="replicate")
        
        self.branch4_conv1 = conv_block(in_channels, 16, kernel_size=1)
        self.branch4_conv2 = conv_block(16, 32, kernel_size=3, padding="same", padding_mode="replicate")
        self.branch4_conv3 = conv_block(32, 32, kernel_size=3, padding="same", padding_mode="replicate")


    def _forward(self, x: Tensor) -> List[Tensor]:
        branch1 = self.branch1_conv(x)

        branch2 = F.pad(x, (1, 1, 1, 1), mode="replicate")
        branch2 = self.branch2_avg(branch2)
        branch2 = self.branch2_conv(branch2)
        
        branch3 = self.branch3_conv1(x)
        branch3 = self.branch3_conv2(branch3)
        
        branch4 = self.branch4_conv1(x)
        branch4 = self.branch4_conv2(branch4)
        branch4 = self.branch4_conv3(branch4)

        # Concatenate the outputs
        # NOTE: Includes residual connection
        return [x, branch1, branch2, branch3, branch4]


    def forward(self, x: Tensor) -> Tensor:
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class InceptionB(nn.Module):
    def __init__(self, in_channels: int, conv_block: Optional[Callable[..., nn.Module]] = None) -> None:
        super().__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1_conv = conv_block(in_channels, 32, kernel_size=1)
        
        self.branch2_avg = nn.AvgPool2d(kernel_size=3, stride=1)
        self.branch2_conv = conv_block(in_channels, 32, kernel_size=3, padding="same", padding_mode="replicate")

        self.branch3_conv1 = conv_block(in_channels, 32, kernel_size=1, stride=1)
        self.branch3_dw_conv = conv_block(32, 32, kernel_size=3, groups=32, padding="same", padding_mode="replicate")
        self.branch3_pw_conv = conv_block(32, 32, kernel_size=1, padding="same", padding_mode="replicate")
        self.branch3_conv2 = conv_block(32, 32, kernel_size=3, padding="same", padding_mode="replicate")

        self.branch4_conv1 = conv_block(in_channels, 32, kernel_size=1)
        self.branch4_dw_conv1 = conv_block(32, 32, kernel_size=3, groups=32, padding="same", padding_mode="replicate")
        self.branch4_pw_conv1 = conv_block(32, 32, kernel_size=1, padding="same", padding_mode="replicate")
        self.branch4_conv2 = conv_block(32, 32, kernel_size=1)
        self.branch4_dw_conv2 = conv_block(32, 32, kernel_size=3, groups=32, padding="same", padding_mode="replicate")
        self.branch4_pw_conv2 = conv_block(32, 32, kernel_size=1, padding="same", padding_mode="replicate")
        self.branch4_conv3 = conv_block(32, 32, kernel_size=1)

    def _forward(self, x: Tensor) -> List[Tensor]:
        branch1 = self.branch1_conv(x)
        
        branch2 = F.pad(x, (1, 1, 1, 1), mode="replicate")
        branch2 = self.branch2_avg(branch2)
        branch2 = self.branch2_conv(branch2)

        branch3 = self.branch3_conv1(x)
        branch3 = self.branch3_dw_conv(branch3)
        branch3 = self.branch3_pw_conv(branch3)
        branch3 = self.branch3_conv2(branch3)

        branch4 = self.branch4_conv1(x)
        branch4 = self.branch4_dw_conv1(branch4)
        branch4 = self.branch4_pw_conv1(branch4)
        branch4 = self.branch4_conv2(branch4)
        branch4 = self.branch4_dw_conv2(branch4)
        branch4 = self.branch4_pw_conv2(branch4)
        branch4 = self.branch4_conv3(branch4)
        
        # Concatenate the outputs
        # NOTE: Includes residual connection (Potentially we need )
        return [x, branch1, branch2, branch3, branch4]

    def forward(self, x: Tensor) -> Tensor:
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


# class InceptionC(nn.Module):
#     def __init__(
#         self, in_channels: int, channels_7x7: int, conv_block: Optional[Callable[..., nn.Module]] = None
#     ) -> None:
#         super().__init__()
#         if conv_block is None:
#             conv_block = BasicConv2d
#         self.branch1x1 = conv_block(in_channels, 192, kernel_size=1)

#         c7 = channels_7x7
#         self.branch7x7_1 = conv_block(in_channels, c7, kernel_size=1)
#         self.branch7x7_2 = conv_block(c7, c7, kernel_size=(1, 7), padding=(0, 3))
#         self.branch7x7_3 = conv_block(c7, 192, kernel_size=(7, 1), padding=(3, 0))

#         self.branch7x7dbl_1 = conv_block(in_channels, c7, kernel_size=1)
#         self.branch7x7dbl_2 = conv_block(c7, c7, kernel_size=(7, 1), padding=(3, 0))
#         self.branch7x7dbl_3 = conv_block(c7, c7, kernel_size=(1, 7), padding=(0, 3))
#         self.branch7x7dbl_4 = conv_block(c7, c7, kernel_size=(7, 1), padding=(3, 0))
#         self.branch7x7dbl_5 = conv_block(c7, 192, kernel_size=(1, 7), padding=(0, 3))

#         self.branch_pool = conv_block(in_channels, 192, kernel_size=1)

#     def _forward(self, x: Tensor) -> List[Tensor]:
#         branch1x1 = self.branch1x1(x)

#         branch7x7 = self.branch7x7_1(x)
#         branch7x7 = self.branch7x7_2(branch7x7)
#         branch7x7 = self.branch7x7_3(branch7x7)

#         branch7x7dbl = self.branch7x7dbl_1(x)
#         branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
#         branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
#         branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
#         branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)

#         branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
#         branch_pool = self.branch_pool(branch_pool)

#         outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
#         return outputs

#     def forward(self, x: Tensor) -> Tensor:
#         outputs = self._forward(x)
#         return torch.cat(outputs, 1)


# class InceptionD(nn.Module):
#     def __init__(self, in_channels: int, conv_block: Optional[Callable[..., nn.Module]] = None) -> None:
#         super().__init__()
#         if conv_block is None:
#             conv_block = BasicConv2d
#         self.branch3x3_1 = conv_block(in_channels, 192, kernel_size=1)
#         self.branch3x3_2 = conv_block(192, 320, kernel_size=3, stride=2)

#         self.branch7x7x3_1 = conv_block(in_channels, 192, kernel_size=1)
#         self.branch7x7x3_2 = conv_block(192, 192, kernel_size=(1, 7), padding=(0, 3))
#         self.branch7x7x3_3 = conv_block(192, 192, kernel_size=(7, 1), padding=(3, 0))
#         self.branch7x7x3_4 = conv_block(192, 192, kernel_size=3, stride=2)

#     def _forward(self, x: Tensor) -> List[Tensor]:
#         branch3x3 = self.branch3x3_1(x)
#         branch3x3 = self.branch3x3_2(branch3x3)

#         branch7x7x3 = self.branch7x7x3_1(x)
#         branch7x7x3 = self.branch7x7x3_2(branch7x7x3)
#         branch7x7x3 = self.branch7x7x3_3(branch7x7x3)
#         branch7x7x3 = self.branch7x7x3_4(branch7x7x3)

#         branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)
#         outputs = [branch3x3, branch7x7x3, branch_pool]
#         return outputs

#     def forward(self, x: Tensor) -> Tensor:
#         outputs = self._forward(x)
#         return torch.cat(outputs, 1)


# class InceptionE(nn.Module):
#     def __init__(self, in_channels: int, conv_block: Optional[Callable[..., nn.Module]] = None) -> None:
#         super().__init__()
#         if conv_block is None:
#             conv_block = BasicConv2d
#         self.branch1x1 = conv_block(in_channels, 320, kernel_size=1)

#         self.branch3x3_1 = conv_block(in_channels, 384, kernel_size=1)
#         self.branch3x3_2a = conv_block(384, 384, kernel_size=(1, 3), padding=(0, 1))
#         self.branch3x3_2b = conv_block(384, 384, kernel_size=(3, 1), padding=(1, 0))

#         self.branch3x3dbl_1 = conv_block(in_channels, 448, kernel_size=1)
#         self.branch3x3dbl_2 = conv_block(448, 384, kernel_size=3, padding=1)
#         self.branch3x3dbl_3a = conv_block(384, 384, kernel_size=(1, 3), padding=(0, 1))
#         self.branch3x3dbl_3b = conv_block(384, 384, kernel_size=(3, 1), padding=(1, 0))

#         self.branch_pool = conv_block(in_channels, 192, kernel_size=1)

#     def _forward(self, x: Tensor) -> List[Tensor]:
#         branch1x1 = self.branch1x1(x)

#         branch3x3 = self.branch3x3_1(x)
#         branch3x3 = [
#             self.branch3x3_2a(branch3x3),
#             self.branch3x3_2b(branch3x3),
#         ]
#         branch3x3 = torch.cat(branch3x3, 1)

#         branch3x3dbl = self.branch3x3dbl_1(x)
#         branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
#         branch3x3dbl = [
#             self.branch3x3dbl_3a(branch3x3dbl),
#             self.branch3x3dbl_3b(branch3x3dbl),
#         ]
#         branch3x3dbl = torch.cat(branch3x3dbl, 1)

#         branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
#         branch_pool = self.branch_pool(branch_pool)

#         outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
#         return outputs

#     def forward(self, x: Tensor) -> Tensor:
#         outputs = self._forward(x)
#         return torch.cat(outputs, 1)


# class InceptionAux(nn.Module):
#     def __init__(
#         self, in_channels: int, num_classes: int, conv_block: Optional[Callable[..., nn.Module]] = None
#     ) -> None:
#         super().__init__()
#         if conv_block is None:
#             conv_block = BasicConv2d
#         self.conv0 = conv_block(in_channels, 128, kernel_size=1)
#         self.conv1 = conv_block(128, 768, kernel_size=5)
#         self.conv1.stddev = 0.01  # type: ignore[assignment]
#         self.fc = nn.Linear(768, num_classes)
#         self.fc.stddev = 0.001  # type: ignore[assignment]

#     def forward(self, x: Tensor) -> Tensor:
#         # N x 768 x 17 x 17
#         x = F.avg_pool2d(x, kernel_size=5, stride=3)
#         # N x 768 x 5 x 5
#         x = self.conv0(x)
#         # N x 128 x 5 x 5
#         x = self.conv1(x)
#         # N x 768 x 1 x 1
#         # Adaptive average pooling
#         x = F.adaptive_avg_pool2d(x, (1, 1))
#         # N x 768 x 1 x 1
#         x = torch.flatten(x, 1)
#         # N x 768
#         x = self.fc(x)
#         # N x 1000
#         return x


class BasicConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, **kwargs: Any) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)

