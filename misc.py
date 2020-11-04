import torch
import torch.nn as nn
from torch.nn import init
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

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--upsample", action="store_true", default=False)
    parser.add_argument("--random", action="store_true", default=False)
    args = parser.parse_args()

    device = xm.xla_device()
    x = torch.randn(2,3,128,128, device=device)

    if args.upsample:
        o = F.interpolate(x, scale_factor=2)
    
    if args.random:
        x.random_(100)

    print(met..metrics_report())

