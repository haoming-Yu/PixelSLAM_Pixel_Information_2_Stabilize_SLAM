import glob
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from common import as_intrinsics_matrix
from torch.utils.data import Dataset

def readEXR_onlydepth(filename):
    """
    Read depth data from EXR image file.

    Args:
        filename (str): File path.

    Returns:
        Y (numpy.array): Depth buffer in float32 format.
    """
    # move the import here since only CoFusion needs these package
    # sometimes installation of openexr is hard, you can run all other datasets even without openexr
    import Imath
    import OpenEXR as exr

    exrfile = exr.InputFile(filename)
    header = exrfile.header()
    dw = header['dataWindow']
    isize = (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1)

    channelData = dict()

    for c in header['channels']:
        C = exrfile.channel(c, Imath.PixelType(Imath.PixelType.FLOAT))
        C = np.fromstring(C, dtype=np.float32)
        C = np.reshape(C, isize)

        channelData[c] = C

    Y = None if 'Y' not in header['channels'] else channelData['Y']

    return Y

def get_dataset(cfg, args, device='cuda:0'):
    return dataset_dict[cfg['dataset']](cfg, args, device=device)

class BaseDataset(Dataset):
    def __init__(self, cfg, args, device='cuda:0'
                 ):
        super(BaseDataset, self).__init__()
        self.name = cfg['dataset']
        self.device = device
        self.png_depth_scale = cfg['cam']['png_depth_scale']

        self.H, self.W, self.fx, self.fy, self.cx, self.cy = cfg['cam']['H'], cfg['cam'][
            'W'], cfg['cam']['fx'], cfg['cam']['fy'], cfg['cam']['cx'], cfg['cam']['cy']
        
        self.distortion = np.array(
            cfg['cam']['distortion']
        ) if 'distortion' in cfg['cam'] else None

        self.crop_size = cfg['cam']['crop_size'] if 'crop_size' in cfg['cam'] else None

        if args.input_folder is None:
            self.input_folder = cfg['data']['input_folder']
        else:
            self.input_folder = args.input_folder

        self.crop_edge = cfg['cam']['crop_edge']

    def __len__(self):
        return self.n_img
    
    @staticmethod # you can call this method directly through class name.
    def set_edge_pixels_to_zero(depth_data, crop_edge):
        mask = torch.ones_like(depth_data)
        mask[:crop_edge, :] = 0
        mask[-crop_edge:, :] = 0
        mask[:, :crop_edge] = 0
        mask[:, -crop_edge:] = 0

        depth_data = depth_data * mask
        return depth_data

class Replica(BaseDataset):
    def __init__(self) -> None:
        super().__init__()

class ScanNet(BaseDataset):
    def __init__(self) -> None:
        super().__init__()

class TUM_RGBD(BaseDataset):
    def __init__(self) -> None:
        super().__init__()


dataset_dict = {
    "replica": Replica,
    "tumrgbd": TUM_RGBD,
    'scannet': ScanNet
}