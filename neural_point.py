import torch
import numpy as np

import faiss
import faiss.contrib.torch_utils
from utils.common import setup_seed, clone_kf_dict

class NeuralPointCloud(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.c_dim = cfg['model']['c_dim']
        self.cuda_id = cfg['mapping']['device'] # Here I change the 0 to mapping cuda device
        self.use_dynamic_radius = cfg['use_dynamic_radius']
        self.nn_num = cfg['pointcloud']['nn_num']

        self.nlist = cfg['pointcloud']['nlist']
        self.radius_add = cfg['pointcloud']['radius_add']
        self.radius_min = cfg['pointcloud']['radius_min']
        self.radius_query = cfg['pointcloud']['radius_query']
        self.fix_interval_when_add_along_ray = cfg['pointcloud']['fix_interval_when_add_along_ray']

        self.N_surface = cfg['rendering']['N_surface']
        self.N_add = cfg['pointcloud']['N_add'] # how many points to add at one location (front and behind gt_depth)
        self.near_end_surface = cfg['pointcloud']['near_end_surface']
        self.far_end_surface = cfg['pointcloud']['far_end_surface']

        self._cloud_pos = [] # (input_pos) * N_add
        self._input_pos = [] # to save locations of the depth input
        self._input_rgb = [] # to save locations of the rgb input at the depth input
        self._pts_num = 0    # number of points in neural point cloud
        self.geo_feats = None
        self.col_feats = None
        self.keyframe_dict = []
        
        self.resource = faiss.StandardGpuResources()
        self.index = faiss.index_cpu_to_gpu(self.resource,
                                            self.cuda_id,
                                            faiss.IndexIVFFlat(faiss.IndexFlatL2(3), 3, self.nlist, faiss.METRIC_L2))
        self.index.nprobe = cfg['pointcloud']['nprobe']
        # Here need to notice the usage about faiss
        # faiss.StandardGpuResources -> use a single GPU to do the trainning and searching
        # faiss.index_cpu_to_gpu -> move the index into the gpu
        # faiss.IndexFlatL2 -> brute force searching use L2 distance as a searching standard, here it is used as a quantizer to construct the inverted file searching
        # faiss.IndexIVFFlat -> use Inverted file indexing method to search for the nearest neighborhood. And the inner part is a clustering method
        # Here the argument 3 means the dimension of the target vertex -> (x, y, z)
        # self.nlist means the number of clustering center
        # And nprobe means the number of center we should search while searching
        # If nprobe is bigger, the result is more accurate
        # However, the speed for searching will be slower
        setup_seed(cfg["setup_seed"])

    # Here are the interfaces we provide for accessing the point cloud
        
    def cloud_pos(self, index=None):
        if index is None:
            return self._cloud_pos
        return self._cloud_pos[index]
    
    def input_pos(self):
        return self._input_pos
    
    def input_rgb(self):
        return self._input_rgb
    
    def pts_num(self):
        return self._pts_num
    
    def index_train(self, xb):
        assert torch.is_tensor(xb), 'use tensor to train FAISS index'
        self.index.train(xb)
        return self.index.is_trained
    
    def get_radius_query(self):
        return self.radius_query
    
    def get_geo_feats(self):
        return self.geo_feats
    
    def get_col_feats(self):
        return self.col_feats
    
    def update_geo_feats(self, feats, indices=None):
        assert torch.is_tensor(feats), 'use tensor to update features'
        if indices is not None:
            self.geo_feats[indices] = feats.detach().clone()
            # Note that here detach() method will make current tensor detached from current graph and will never require gradient.
            # clone() method is a deep copy method.
        else: # This is used for initialization or substitution.
            assert feats.shape[0] == self.geo_feats.shape[0], 'feature shape[0] mismatch'
            self.geo_feats = feats.detach().clone()

    def update_col_feats(self, feats, indices=None):
        assert torch.is_tensor(feats), 'use tensor to update features'
        if indices is not None:
            self.col_feats[indices] = feats.detach().clone()
        else:
            assert feats.shape[0] == self.col_feats.shape[0], 'feature shape[0] mismatch'
            self.col_feats = feats.detach().clone()

    def add_neural_points(self, batch_rays_o, batch_rays_d, batch_gt_depth, batch_gt_color,
                          train=False, is_pts_grad=False, dynamic_radius=None):
        """
        Add multiple neural points, will use depth filter when getting these samples.

        Args:
            batch_rays_o (tensor): ray origins (N,3)
            batch_rays_d (tensor): ray directions (N,3)
            batch_gt_depth (tensor): sensor depth (N,)
            batch_gt_color (tensor): sensor color (N,3)
            train (bool): whether to update the FAISS index
            is_pts_grad (bool): the points are chosen based on color gradient
            dynamic_radius (tensor): choose every radius differently based on its color gradient

        Returns:
            the return value should be the point number which is successfully added to the point-cloud after this calling
        """

        if batch_rays_o.shape[0]:
            mask = batch_gt_depth > 0 # first mask away the rays whose depth is 0
            batch_gt_color = batch_gt_color*255
            batch_rays_o, batch_rays_d, batch_gt_depth, batch_gt_color = \
                batch_rays_o[mask], batch_rays_d[mask], batch_gt_depth[mask], batch_gt_color[mask]
            
            pts_gt = batch_rays_o[..., None, :] + batch_rays_d[..., None, :] * batch_gt_depth[..., None, None]
            mask = torch.ones(pts_gt.shape[0], device=self.device).bool()
            pts_gt = pts_gt.reshape(-1, 3)

            if self.index.is_trained:
                _, _, neighbor_num_gt = self.find_neighbors_faiss(
                    pts_gt, step='add', is_pts_grad=is_pts_grad, dynamic_radius=dynamic_radius
                )
                mask = (neighbor_num_gt == 0) # need to mask away the points that have no neighbors
            # If the index has not been trained, the mask is initialized to be all one, mask nothing for now.

            self._input_pos.extend(pts_gt[mask].tolist())
            self._input_rgb.extend(batch_gt_color[mask].tolist())

            gt_depth_surface = batch_gt_depth.unsqueeze(-1).repeat(1, self.N_add) 
            # unsqueeze will insert a new dimension at the position specified. 
            # In this case, it is -1. Thus, a new dimension is inserted at the last dimension
            # And repeat the depth batch ray for each point in N_add
            t_vals_surface = torch.linspace(0.0, 1.0, steps=self.N_add, device=self.device)
            # linspace is used to create a one-dimensional tensor of size steps
            # Here t_vals_surface is [0.0, 0.5, 1.0]
            if self.fix_interval_when_add_along_ray:
                # add along ray, interval unrelated to depth
                intervals = torch.linspace(-0.04, 0.04, steps=self.N_add,
                                           device=self.device).unsqueeze(0)
                z_vals = gt_depth_surface + intervals
            else:
                z_vals = self.near_end_surface * gt_depth_surface * (1. - t_vals_surface) + \
                    self.far_end_surface * \
                    gt_depth_surface * t_vals_surface
                # Use this way to generate three points

            pts = batch_rays_o[..., None, :] + \
                batch_rays_d[..., None, :] * z_vals[..., :, None]
            
            pts  = pts[mask]
            pts = pts.reshape(-1, 3)

            self._cloud_pos += pts.tolist()
            self._pts_num += pts.shape[0]

            if self.geo_feats is None:
                # Use Gaussian noise to do the initialization
                self.geo_feats = torch.zeros(
                    [self._pts_num, self.c_dim], device=self.device
                ).normal_(mean=0, std=0.1)
                self.col_feats = torch.zeros(
                    [self._pts_num, self.c_dim], device=self.device
                ).normal_(mean=0, std=0.1)
            else:
                self.geo_feats = torch.cat([self.geo_feats,
                                            torch.zeros([pts.shape[0], self.c_dim], device=self.device).normal_(mean=0, std=0.1)], 0)
                self.col_feats = torch.cat([self.col_feats,
                                            torch.zeros([pts.shape[0], self.c_dim], device=self.device).normal_(mean=0, std=0.1)], 0)
                
            if train or not self.index.is_trained:
                self.index.train(pts)
            self.index.train(torch.tensor(self._cloud_pos, device=self.device)) 
            # After the addition of new points, we need to train the cluster again
            self.index.add(pts)
            return torch.sum(mask)
        else:
            return 0
        
    def find_neighbors_faiss(self, pos, step='add', retrain=False, is_pts_grad=False, dynamic_radius=None):
        """
        Query neighbors using faiss.

        Args:
            pos (tensor): points to find neighbors
            step (str): 'add'|'query'
            retrain (bool, optional): if to retrain the index cluster of IVF
            is_pts_grad: whether it's the points chosen based on color grad, will use smaller radius when looking for neighbors
            dynamic_radius (tensor, optional): choose every radius differently based on its color gradient

        Returns:
            D: distances to neighbors for the positions in pos
            I: indices of neighbors for the positions in pos
            neighbor_num: number of neighbors for the positions in pos
        """
        if (not self.index.is_trained) or retrain:
            self.index.train(self._cloud_pos)

        assert step in ['add', 'query']
        split_pos = torch.split(pos, 6500, dim=0) # split the points into batches, each batch has 6500 points or less than that
        D_list = []
        I_list = []
        for split_p in split_pos:
            D, I = self.index.search(split_p.float(), self.nn_num)
            D_list.append(D)
            I_list.append(I)
        D = torch.cat(D_list, dim=0)
        I = torch.cat(I_list, dim=0)

        if step == 'query':
            radius = self.radius_query
        else:
            if not is_pts_grad:
                radius = self.radius_add
            else:
                radius = self.radius_min
        
        if dynamic_radius is not None:
            assert pos.shape[0] == dynamic_radius.shape[0], 'shape mis-match for input points and dynamic radius'
            neighbor_num = (D < dynamic_radius.reshape(-1, 1)
                            ** 2).sum(axis=-1).int()
        else:
            neighbor_num = (D < radius**2).sum(axis=-1).int()

        return D, I, neighbor_num # Notice that here D is the squared distance
    
    def sample_near_pcl(self, rays_o, rays_d, near, far, num):
        """
        For pixels with 0 depth readings, preferably sample near point cloud.

        Args:
            rays_o (tensor): rays origin
            rays_d (tensor): rays direction
            near : near end for sampling along this ray
            far: far end
            num (int): sampling num between near and far

        Returns:
            z_vals (tensor): z values for zero valued depth pixels
            invalid_mask (bool): mask for zero valued depth pixels that are not close to neural point cloud
        """
        rays_o, rays_d = rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)
        n_rays = rays_d.shape[0]
        intervals = 25
        z_vals = torch.linspace(near, far, steps=intervals, device=self.device)
        pts = rays_o[..., None, :] + \
            rays_d[..., None, :] * z_vals[..., :, None]
        pts = pts.reshape(-1, 3)

        if torch.is_tensor(far):
            far = far.item() # item will only work for tensor with one element
        z_vals_section = np.linspace(near, far, intervals)
        z_vals_np = np.linspace(near, far, num)
        z_vals_total = np.tile(z_vals_np, (n_rays, 1)) # Repeat the z_val_np to be n_rays row

        pts_split = torch.split(pts, 65000) # batch number is 65000
        Ds, Is, neighbor_nums = [], [], []
        for pts_batch in pts_split:
            D, I, neighbor_num = self.find_neighbors_faiss(
                pts_batch, step='query'
            )
            D, I, neighbor_num = D.cpu().numpy(), I.cpu().numpy(), neighbor_num.cpu().numpy()
            Ds.append(D)
            Is.append(I)
            neighbor_nums.append(neighbor_num)
        





                





