import numpy as np
import random
import torch
import torch.nn.functional as F

from skimage.color import rgb2gray
from skimage import filters

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def clone_kf_dict(keyframes_dict):
    cloned_keyfraomes_dict = []
    for keyframe in keyframes_dict:
        cloned_keyframe = {}
        for key, value in keyframe.items():
            cloned_value = value.clone()
            cloned_keyframe[key] = cloned_value
        cloned_keyfraomes_dict.append(cloned_keyframe)
    return cloned_keyfraomes_dict