import torch
import torch.nn as nn
from torch.nn import init
from torch.nn.functional import upsample
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import Parameter as P

# xla imports
import torch_xla.core.xla_model as xm
import torch_xla.distributed.data_parallel as dp
import torch_xla.debug.metrics as met
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.distributed.parallel_loader as pl
from argparse import ArgumentParser

class InterpolateNearest2d(nn.Module):
    """
    Custom implementation of nn.Upsample because pytroch/xla
    does not yet support scale_factor and needs to be provided with
    the output_size
    """

    def __init__(self, scale_factor=2):
        """
        Create an InterpolateNearest module

        Args:
            scale_factor (int, optional): Output size multiplier. Defaults to 2.
        """
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        """
        Interpolate x in "nearest" mode on its last 2 dimensions

        Args:
            x (torch.Tensor): input to interpolate

        Returns:
            torch.Tensor: upsampled tensor with shape
                (...x.shape, x.shape[-2] * scale_factor, x.shape[-1] * scale_factor)
        """
        return nn.functional.interpolate(
            x,
            size=(x.shape[-2] * self.scale_factor, x.shape[-1] * self.scale_factor),
            mode="nearest",
        )
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--upsample", action="store_true", default=False)
    parser.add_argument("--random", action="store_true", default=False)
    args = parser.parse_args()

    device = xm.xla_device()
    x = torch.randn(2,3,128,128, device=device)
    upsample = InterpolateNearest2d(scale_factor=2)
    
    if args.upsample:
        o = upsample(x)
    
    if args.random:
        x.random_(100)
        x.normal_(mean=0, std=0.02)

    print(met.metrics_report())

