# This file is used to record the tools function
# to process the camera coordinate.

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

def clone_kf_dict(keyframes_dict):
    cloned_keyfraomes_dict = []
    for keyframe in keyframes_dict:
        cloned_keyframe = {}
        for key, value in keyframe.items():
            cloned_value = value.clone()
            cloned_keyframe[key] = cloned_value
        cloned_keyfraomes_dict.append(cloned_keyframe)
    return cloned_keyfraomes_dict

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
    Do the random sampling.
    """
    i = i.reshape(-1)
    j = j.reshape(-1) 
    indices = torch.randint(i.shape[0], (n,), device=device) 
    # low is default to be 0, 
    # returns a tensor filled with random integers generated uniformly between low (inclusive) and high (exclusive).
    # n is the number of random sampling
    # and the range is [0, i.shape[0])
    indices = indices.clamp(0, i.shape[0]) # For this sentence, I think it's not necessary
    i = i[indices]
    j = j[indices]
    depth = depth.reshape(-1)
    color = color.reshape(-1, 3)
    depth = depth[indices]
    color = color[indices]
    return i, j, depth, color

def get_sample_uv(H0, H1, W0, W1, n, depth, color, device='cuda:0'):
    """
    Sample n uv coordinates from an image region H0..(H1-1), W0..(W1-1)

    """
    depth = depth[H0:H1, W0:W1]
    color = color[H0:H1, W0:W1]
    i, j = torch.meshgrid(torch.linspace(
        W0, W1 - 1, W1 - W0).to(device), torch.linspace(H0, H1-1, H1-H0).to(device), indexing='ij'
    )
    i = i.t()
    j = j.t()
    # Get the transpose of i and j
    i, j, depth, color = select_uv(i, j, n, depth, color, device=device)
    return i, j, depth, color
    
def get_rays_from_uv(i, j, c2w, fx, fy, cx, cy, device):
    """
    Get corresponding rays from input uv
    i,j are flattened.

    c2w: camera to world coordinate

    """
    if isinstance(c2w, np.ndarray):
        c2w = torch.from_numpy(c2w).to(device) # move c2w matrix to gpu

    dirs = torch.stack(
        [(i-cx)/fx, -(j-cy)/fy, -torch.ones_like(i)], -1
    ).to(device)
    # stack function will stack all the tensors along the last dimension.

    dirs = dirs.reshape(-1, 1, 3)

    rays_d = torch.sum(dirs * c2w[:3, :3], -1)

    rays_o = c2w[:3, -1].expand(rays_d.shape)
    return rays_o, rays_d


def get_samples(H0, H1, W0, W1, n, fx, fy, cx, cy, c2w, depth, color, device,
                depth_filter=False, return_index=False, depth_limit=None):
    """
    Get n rays from the image region H0..H1, W0..W1.
    fx, fy, cx, cy: intrinsics.
    c2w is its camera pose and depth/color is the corresponding image tensor.
    """
    i, j, sample_depth, sample_color = get_sample_uv(
        H0, H1, W0, W1, n, depth, color, device=device
    )
    rays_o, rays_d = get_rays_from_uv(i, j, c2w, fx, fy, cx, cy, device)
    if depth_filter:
        mask = sample_depth > 0
        if depth_limit is not None:
            mask = mask & (sample_depth < depth_limit)
        rays_o, rays_d, sample_depth, sample_color = rays_o[mask], rays_d[mask], sample_depth[mask], sample_color[mask]
        i, j = i[mask], j[mask]

    if return_index:
        return rays_o, rays_d, sample_depth, sample_color, i.to(torch.int64), j.to(torch.int64)
    return rays_o, rays_d, sample_depth, sample_color

def get_sample_uv_with_grad(H0, H1, W0, W1, n, image):
    """
    Sample n uv coordinates from an image region H0..H1, W0..W1
    image (numpy.ndarray): color image or estimated normal image

    First, we sample 5n pixel index from image. And then do the masking
    to get proper index within the range of H0, H1, W0, W1.

    And because of the range selection, the points will be less or equal than 5n
    Finally, do the random selection to get n points.

    This is the reason why 5*n is sampled. And the 5 here is just a hyper-parameter.
    """
    intensity = rgb2gray(image.cpu().numpy()) # Get gray picture from rgb picture to get evaluate the gradient
    grad_y = filters.sobel_h(intensity) # find horizontal edges of an image using the sobel transform
    grad_x = filters.sobel_v(intensity) # find vertical edges of an image using the sobel transform.
    # These operation above will return a sobel edge map
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)

    img_size = (image.shape[0], image.shape[1])
    selected_index = np.argpartition(grad_mag, -5*n, axis=None)[-5*n:]
    # np.argpartition will do the sorting job. And here the axis=None will use the flattened array
    # select the last 5n grad in ascent order, so -5*n can select the biggest 5n gradient points
    indices_h, indices_w = np.unravel_index(selected_index, img_size)
    # unravel_index is used to convert the flattened index into tuples of indexes, and the index is split into two lists.
    mask = (indices_h >= H0) & (indices_h < H1) & (
        indices_w >= W0) & (indices_w < W1)
    indices_h, indices_w = indices_h[mask], indices_w[mask]
    selected_index = np.ravel_multi_index(
        np.array((indices_h, indices_w)), img_size)
    samples = np.random.choice(
        range(0, indices_h.shape[0]), size=n, replace=False)

    return selected_index[samples]


def get_samples_with_pixel_grad(H0, H1, W0, W1, n_color, H, W, fx, fy, cx, cy, c2w, depth, color, device,
                                depth_filter=True, return_index=True, depth_limit=None):
    """
    Get n rays from the image region H0..H1, W0..W1 based on color gradients, normal map gradients and random selection
    H, W: height, width.
    fx, fy, cx, cy: intrinsics.
    c2w is its camera pose and depth/color is the corresponding image tensor.
    """
    assert (n_color > 0), 'invalid number of rays to sample.'

    index_color_grad, index_normal_grad = [], []
    if n_color > 0:
        index_color_grad = get_sample_uv_with_grad(
            H0, H1, W0, W1, n_color, color)

    merged_indices = np.union1d(index_color_grad, index_normal_grad)

    i, j = np.unravel_index(merged_indices.astype(int), (H, W))
    i, j = torch.from_numpy(j).to(device).float(), torch.from_numpy(
        i).to(device).float()  # (i-cx), on column axis
    # At the same time, switch the content of i, j
    rays_o, rays_d = get_rays_from_uv(i, j, c2w, fx, fy, cx, cy, device)
    i, j = i.long(), j.long() # self.long() <=> self.to(torch.int64)
    sample_depth = depth[j, i]
    sample_color = color[j, i]
    if depth_filter:
        mask = sample_depth > 0
        if depth_limit is not None:
            mask = mask & (sample_depth < depth_limit)
        rays_o, rays_d, sample_depth, sample_color = rays_o[
            mask], rays_d[mask], sample_depth[mask], sample_color[mask]
        i, j = i[mask], j[mask]

    if return_index:
        return rays_o, rays_d, sample_depth, sample_color, i.to(torch.int64), j.to(torch.int64)
    return rays_o, rays_d, sample_depth, sample_color

def random_select(l, k):
    """
    Random select k values from 0 to (l-1)

    """
    return list(np.random.permutation(np.array(range(l)))[:min(l, k)])

def get_tensor_from_camera(RT, Tquad=False):
    """
    Convert transformation matrix to quaternion and translation

    """
    gpu_id = -1
    if type(RT) == torch.Tensor:
        if RT.get_device() != -1:
            RT = RT.detach().cpu()
            # move the transformation matrix to cpu
            # detach it from the computational graph
            gpu_id = RT.get_device()
            # refresh the gpu_id 
        RT = RT.numpy() # use numpy to do the cpu computation
    R, T = RT[:3, :3], RT[:3, 3]

    from scipy.spatial.transform import Rotation
    rot = Rotation.from_matrix(R)
    quad = rot.as_quat()
    quad = np.roll(quad, 1)

    if Tquad:
        tensor = np.concatenate([T, quad], 0)
    else:
        tensor = np.concatenate([quad, T], 0)
    tensor = torch.from_numpy(tensor).float()
    if gpu_id != -1:
        # after the processing, move the tensor back to the gpus
        tensor = tensor.to(gpu_id)
    return tensor