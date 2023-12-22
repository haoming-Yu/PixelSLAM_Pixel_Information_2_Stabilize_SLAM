import os
import shutil
import traceback
import subprocess
import time
import cv2
import numpy as np
import open3d as o3d
import torch

from ast import literal_eval
from colorama import Fore, Style
from torch.autograd import Variable
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from common import get_camera_from_tensor, get_samples, setup_seed