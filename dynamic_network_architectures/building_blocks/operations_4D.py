from typing import Union

from torch import nn
from torch.nn.modules.conv import Conv3d
from torch.nn.modules.instancenorm import InstanceNorm3d
from torch import nn
import numpy as np
import torch

import torch
import torch.nn.functional as F


class Conv4DHypercross(nn.Module):
    def __init__(self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride = [1,1,1,1],
        padding: Union[str] = 0, 
        dilation = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device=None,
        dtype=None):

        super().__init__()

        if not isinstance(stride, (tuple, list, np.ndarray)):
            stride = [stride] * 4

        
        if not isinstance(kernel_size, (tuple, list, np.ndarray)):
            kernel_size = [kernel_size] * 4

        if not isinstance(padding, (tuple, list, np.ndarray)):
            padding = [padding] * 4

        self.time_stride = stride[0]
        self.z_stride = stride[1]
        self._time_pedding_size = (kernel_size[0]-1)//2
        if padding_mode == "time_cyclic":
            self.time_padding_mode = "cyclic"
            padding_mode = "zeros"
        else:
            self.time_padding_mode = "zeros"

        self.time_kernel_size = kernel_size[0]
        
        if kernel_size[0] != 3 and kernel_size[0] != 1:
            print(kernel_size)
            raise NotImplementedError("Only kernel_size[0]==3 is implemented for 4D Hypercross convolution")
        self.conv_3d_tm1 = Conv3d(in_channels=in_channels, 
                                  out_channels=out_channels, 
                                  kernel_size=[1]*3, 
                                  stride = stride[1:], 
                                  padding = [0]*3, 
                                  dilation = dilation, 
                                  groups = groups, 
                                  bias = False, 
                                  padding_mode = padding_mode, 
                                  device = device, 
                                  dtype = dtype)
        self.conv_3d = Conv3d(in_channels=in_channels, 
                                  out_channels=out_channels, 
                                  kernel_size=kernel_size[1:], 
                                  stride = stride[1:], 
                                  padding = padding[1:], 
                                  dilation = dilation, 
                                  groups = groups, 
                                  bias = bias, 
                                  padding_mode = padding_mode, 
                                  device = device, 
                                  dtype = dtype)
        self.conv_3d_tp1 = Conv3d(in_channels=in_channels, 
                                  out_channels=out_channels, 
                                  kernel_size=[1]*3, 
                                  stride = stride[1:], 
                                  padding = [0]*3, 
                                  dilation = dilation, 
                                  groups = groups, 
                                  bias = False, 
                                  padding_mode = padding_mode, 
                                  device = device, 
                                  dtype = dtype)
        
    def forward(self, x):
        outputs_3d = []
        for t in range(0, x.size(2), self.time_stride): 
            xt = x[:, :, t]
            out = self.conv_3d(xt)
            outputs_3d.append(out.unsqueeze(2))

        outputs_3d = torch.cat(outputs_3d, dim=2)

        if self.time_kernel_size == 1:
            return outputs_3d

        outputs_3d_tm1 = []
        for t in range(-1, x.size(2)-1, self.time_stride): 
            xt = x[:, :, t]
            out = self.conv_3d_tm1(xt)
            outputs_3d_tm1.append(out.unsqueeze(2))

        outputs_3d_tm1 = torch.cat(outputs_3d_tm1, dim=2)
        
        outputs_3d_tp1 = []
        for t in range(1, x.size(2)+1, self.time_stride): 
            xt = x[:, :, t%x.size(2)]
            out = self.conv_3d_tp1(xt)
            outputs_3d_tp1.append(out.unsqueeze(2))

        outputs_3d_tp1 = torch.cat(outputs_3d_tp1, dim=2)

        return outputs_3d_tm1 + outputs_3d + outputs_3d_tp1

class Conv4D(nn.Module):
    def __init__(self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride = [1,1,1,1],
        padding: Union[str] = 0, 
        dilation = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device=None,
        dtype=None):

        super().__init__()

        if not isinstance(stride, (tuple, list, np.ndarray)):
            stride = [stride] * 4

        
        if not isinstance(kernel_size, (tuple, list, np.ndarray)):
            kernel_size = [kernel_size] * 4

        if not isinstance(padding, (tuple, list, np.ndarray)):
            padding = [padding] * 4

        self.time_stride = stride[0]
        self.z_stride = stride[1]
        self._time_pedding_size = (kernel_size[0]-1)//2
        if padding_mode == "time_cyclic":
            self.time_padding_mode = "cyclic"
            padding_mode = "zeros"
        else:
            self.time_padding_mode = "zeros"

        self.time_kernel_size = kernel_size[0]
        
        if kernel_size[0] != 3 and kernel_size[0] != 1:
            print(kernel_size)
            raise NotImplementedError("Only kernel size[0] 3 is implemented for 4D convolution")
        self.conv_3d_tm1 = Conv3d(in_channels=in_channels, 
                                  out_channels=out_channels, 
                                  kernel_size=kernel_size[1:], 
                                  stride = stride[1:], 
                                  padding = padding[1:], 
                                  dilation = dilation, 
                                  groups = groups, 
                                  bias = False, 
                                  padding_mode = padding_mode, 
                                  device = device, 
                                  dtype = dtype)
        self.conv_3d = Conv3d(in_channels=in_channels, 
                                  out_channels=out_channels, 
                                  kernel_size=kernel_size[1:], 
                                  stride = stride[1:], 
                                  padding = padding[1:], 
                                  dilation = dilation, 
                                  groups = groups, 
                                  bias = bias, 
                                  padding_mode = padding_mode, 
                                  device = device, 
                                  dtype = dtype)
        self.conv_3d_tp1 = Conv3d(in_channels=in_channels, 
                                  out_channels=out_channels, 
                                  kernel_size=kernel_size[1:], 
                                  stride = stride[1:], 
                                  padding = padding[1:], 
                                  dilation = dilation, 
                                  groups = groups, 
                                  bias = False, 
                                  padding_mode = padding_mode, 
                                  device = device, 
                                  dtype = dtype)
        
    def forward(self, x):
        outputs_3d = []
        for t in range(0, x.size(2), self.time_stride): 
            xt = x[:, :, t]
            out = self.conv_3d(xt)
            outputs_3d.append(out.unsqueeze(2))

        outputs_3d = torch.cat(outputs_3d, dim=2)

        if self.time_kernel_size == 1:
            return outputs_3d

        outputs_3d_tm1 = []
        for t in range(-1, x.size(2)-1, self.time_stride): 
            xt = x[:, :, t]
            out = self.conv_3d_tm1(xt)
            outputs_3d_tm1.append(out.unsqueeze(2))

        outputs_3d_tm1 = torch.cat(outputs_3d_tm1, dim=2)
        
        outputs_3d_tp1 = []
        for t in range(1, x.size(2)+1, self.time_stride): 
            xt = x[:, :, t%x.size(2)]
            out = self.conv_3d_tp1(xt)
            outputs_3d_tp1.append(out.unsqueeze(2))

        outputs_3d_tp1 = torch.cat(outputs_3d_tp1, dim=2)

        return outputs_3d_tm1 + outputs_3d + outputs_3d_tp1



class Conv3Din4D(nn.Module):
    def __init__(self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride = [1,1,1,1],
        padding: Union[str] = 0, 
        dilation = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device=None,
        dtype=None):

        super().__init__()

        if not isinstance(stride, (tuple, list, np.ndarray)):
            stride = [stride] * 4

        if not isinstance(kernel_size, (tuple, list, np.ndarray)):
            kernel_size = [kernel_size] * 4

        if not isinstance(padding, (tuple, list, np.ndarray)):
            padding = [padding] * 4

        self.time_stride = stride[0]
        self.z_stride = stride[1]
        self._time_pedding_size = (kernel_size[0]-1)//2
        if padding_mode == "time_cyclic":
            self.time_padding_mode = "cyclic"
            padding_mode = "zeros"
        else:
            self.time_padding_mode = "zeros"
        
        self.conv_3d = Conv3d(in_channels, out_channels, kernel_size[1:], stride[1:], padding[1:], dilation, groups, bias, padding_mode, device, dtype)
        
    def forward(self, x):
        outputs_3d = []
        for t in range(0, x.size(2), self.time_stride): 
            xt = x[:, :, t]
            out = self.conv_3d(xt)
            outputs_3d.append(out.unsqueeze(2))

        outputs_3d = torch.cat(outputs_3d, dim=2)

        return outputs_3d
    

class NormOp4D(nn.Module):
    def __init__(self, output_channels, **norm_op_kwargs):
        super().__init__()
        self._norm_op = InstanceNorm3d(output_channels, **norm_op_kwargs)

    def forward(self, x):
        outputs = []
        for t in range(x.size(2)): 
            xt = x[:, :, t]
            out = self._norm_op(xt)
            outputs.append(out.unsqueeze(2))
        return torch.cat(outputs, dim=2)
    

class TranspConv4D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride = 1,
        padding = 0,
        output_padding = 0,
        groups: int = 1,
        bias: bool = True,
        dilation = 1,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.stride_t = stride[0]
        if padding_mode == "time_cyclic":
            padding_mode = "zeros"
        self._conv_op = nn.ConvTranspose3d(in_channels, out_channels, kernel_size[1:], stride[1:],
                                           padding, output_padding, groups, bias, dilation, padding_mode, device, dtype)

    def forward(self, x):
        outputs = []
        for t in range(x.size(2)): 
            xt = x[:, :, t]
            out = self._conv_op(xt)
            for i in range(self.stride_t):
                outputs.append(out.unsqueeze(2))
        return torch.cat(outputs, dim=2)








