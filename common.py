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

def as_intrinsics_matrix(intrinsics):
    """
    Get matrix representation of intrinsics (fx, fy, cx, cy).
    """
    K = np.eye(3) # return a 2-D array with ones on the diagonal and zeros elsewhere
    K[0, 0] = intrinsics[0]
    K[1, 1] = intrinsics[1]
    K[0, 2] = intrinsics[2]
    K[1, 2] = intrinsics[3]
    # intrinsics[0] | 0             | intrinsics[2]
    # 0             | intrinsics[1] | intrinsics[3]
    # 0             | 0             | 1
    return K

def quad2rotation(quad):
    """
    Convert quaternion to rotation in batch. Since all operation in pytorch, support gradient passing.

    Args:
        quad (tensor, batch_size*4): quaternion.

    Returns:
        rot_mat (tensor, batch_size*3*3): rotation.
    """
    bs = quad.shape[0] # batch_size
    qr, qi, qj, qk = quad[:, 0], quad[:, 1], quad[:, 2], quad[:, 3]
    two_s = 2.0 / (quad * quad).sum(-1)
    rot_mat = torch.zeros(bs, 3, 3).to(quad.get_device())
    rot_mat[:, 0, 0] = 1 - two_s * (qj ** 2 + qk ** 2)
    rot_mat[:, 0, 1] = two_s * (qi * qj - qk * qr)
    rot_mat[:, 0, 2] = two_s * (qi * qk + qj * qr)
    rot_mat[:, 1, 0] = two_s * (qi * qj + qk * qr)
    rot_mat[:, 1, 1] = 1 - two_s * (qi ** 2 + qk ** 2)
    rot_mat[:, 1, 2] = two_s * (qj * qk - qi * qr)
    rot_mat[:, 2, 0] = two_s * (qi * qk - qj * qr)
    rot_mat[:, 2, 1] = two_s * (qj * qk + qi * qr)
    rot_mat[:, 2, 2] = 1 - two_s * (qi ** 2 + qj ** 2)
    return rot_mat

def get_camera_from_tensor(inputs):
    """
    Convert quaternion(four elements tuple) and translation to transformation matrix.

    Returns:
        tensor(N*3*4 if batch input or 3*4): Transformation matrix.
    """
    N = len(inputs.shape)
    if N == 1:
        inputs = inputs.unsqueeze(0) # not batch input
    quad, T = inputs[:, :4], inputs[:, 4:]  
    # Here inputs[:, :4] means first element to third element
    # Here inputs[:. 4:] means fourth element to last element
    R = quad2rotation(quad) # convert the camera quaternion 4 ->  into rotation matrix 3*3
    RT = torch.cat([R, T[:, :, None]], 2)
    if N == 1:
        RT = RT[0]
    return RT

def select_uv(i, j, n, depth, color, device='cuda:0'):
    """
    Select n pixels (u, v) from dense (u, v)
    """
    i = i.reshape(-1)
    j = j.reshape(-1)
    indices = torch.randint(i.shape[0], (n,), device=device)
    indices = indices.clamp(0, i.shape[0])

def get_sample_uv(H0, H1, W0, W1, n, depth, color, device='cuda:0'):
    """
    Sample n uv coordinates from an image region H0..(H1-1), W0..(W1-1)

    """
    depth = depth[H0:H1, W0:W1]
    color = color[H0:H1, W0:W1]
    i, j = torch.meshgrid(torch.linspace(
        W0, W1 - 1, W1 - W0).to(device), torch.linespace(H0, H1-1, H1-H0).to(device), indexing='ij'
    )
    i = i.t()
    j = j.t()
    # Get the transpose of i and j
    i, j, depth, color = select_uv()
    

def get_samples(H0, H1, W0, W1, n, fx, fy, cx, cy, c2w, depth, color, device,
                depth_filter=False, return_index=False, depth_limit=None):
    """
    Get n rays from the image region H0..H1, W0..W1.
    fx, fy, cx, cy: intrinsics.
    c2w is its camera pose and depth/color is the corresponding image tensor.
    """
    i, j, sample_depth, sample_color = get_sample_uv(
        
    )