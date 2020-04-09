import os
import math

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def get_grid(h, w, b=0, norm=True):
    if norm:
        xgrid = np.arange(0, w) / w
        ygrid = np.arange(0, h) / h
    else:
        xgrid = np.arange(0, w)
        ygrid = np.arange(0, h)
    xv, yv = np.meshgrid(xgrid, ygrid, indexing='xy')
    grid = np.stack([xv, yv], axis=-1)[None]

    grid = torch.from_numpy(grid).float()
    if b > 0:
        grid = grid.expand(b, -1, -1, -1)  # [Batch, H, W, UV]
        return grid
    else:
        return grid[0]

def blur(tensor, kernel_size=15, sigma=3):
    x_cord = torch.arange(kernel_size)
    x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1)

    mean = (kernel_size - 1)/2.
    variance = sigma**2.

    gaussian_kernel = (1./(2.*math.pi*variance)) *\
                      torch.exp(
                          -torch.sum((xy_grid - mean)**2., dim=-1) /\
                          (2*variance)
                      ).to(tensor.device).view(1, 1, kernel_size, kernel_size)

    return F.conv2d(tensor,
                    gaussian_kernel.expand(3, 1, kernel_size, kernel_size),
                    groups=3,
                    padding=kernel_size // 2
                   )

def save_output(novel_img, path, step, interval='01', mode='jpg'):
    os.makedirs(path, exist_ok=True)
    if interval == '01':
        img = novel_img[0].permute(1, 2, 0).cpu().data.numpy() * 255
    elif interval == '-11':
        img = ((novel_img[0] + 1.) / 2.).permute(1, 2, 0).cpu().data.numpy() * 255
    else:
        raise ValueError('only 01 and -11 intervals are supported')
    Image.fromarray(img.clip(0, 255).astype(np.uint8)).save(os.path.join(path, f'{step}.{mode}'))
