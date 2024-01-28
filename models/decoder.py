import math
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.autograd.profiler as profiler
import util
import numpy as np

class ConvEncoder(nn.Module):
    """
    Basic, extremely simple convolutional encoder
    """

    def __init__(
        self,
        dim_in=3,
        norm_layer=util.get_norm_layer("group"),
        padding_type="reflect",
        use_leaky_relu=True,
        use_skip_conn=True,
    ):
        super().__init__()
        self.dim_in = dim_in
        self.norm_layer = norm_layer
        self.activation = nn.LeakyReLU() if use_leaky_relu else nn.ReLU()
        self.padding_type = padding_type
        self.use_skip_conn = use_skip_conn

        # TODO: make these configurable
        first_layer_chnls = 64
        mid_layer_chnls = 128
        last_layer_chnls = 128
        n_down_layers = 3
        self.n_down_layers = n_down_layers

        self.conv_in = nn.Sequential(
            nn.Conv2d(dim_in, first_layer_chnls, kernel_size=7, stride=2, bias=False),
            norm_layer(first_layer_chnls),
            self.activation,
        )

        chnls = first_layer_chnls
        for i in range(0, n_down_layers):
            conv = nn.Sequential(
                nn.Conv2d(chnls, 2 * chnls, kernel_size=3, stride=2, bias=False),
                norm_layer(2 * chnls),
                self.activation,
            )
            setattr(self, "conv" + str(i), conv)

            deconv = nn.Sequential(
                nn.ConvTranspose2d(
                    4 * chnls, chnls, kernel_size=3, stride=2, bias=False
                ),
                norm_layer(chnls),
                self.activation,
            )
            setattr(self, "deconv" + str(i), deconv)
            chnls *= 2

        self.conv_mid = nn.Sequential(
            nn.Conv2d(chnls, mid_layer_chnls, kernel_size=4, stride=4, bias=False),
            norm_layer(mid_layer_chnls),
            self.activation,
        )

        self.deconv_last = nn.ConvTranspose2d(
            first_layer_chnls, last_layer_chnls, kernel_size=3, stride=2, bias=True
        )

        self.dims = [last_layer_chnls]

    def forward(self, x):
        x = util.same_pad_conv2d(x, padding_type=self.padding_type, layer=self.conv_in)
        x = self.conv_in(x)

        inters = []
        for i in range(0, self.n_down_layers):
            conv_i = getattr(self, "conv" + str(i))
            x = util.same_pad_conv2d(x, padding_type=self.padding_type, layer=conv_i)
            x = conv_i(x)
            inters.append(x)

        x = util.same_pad_conv2d(x, padding_type=self.padding_type, layer=self.conv_mid)
        x = self.conv_mid(x)
        x = x.reshape(x.shape[0], -1, 1, 1).expand(-1, -1, *inters[-1].shape[-2:])

        for i in reversed(range(0, self.n_down_layers)):
            if self.use_skip_conn:
                x = torch.cat((x, inters[i]), dim=1)
            deconv_i = getattr(self, "deconv" + str(i))
            x = deconv_i(x)
            x = util.same_unpad_deconv2d(x, layer=deconv_i)
        x = self.deconv_last(x)
        x = util.same_unpad_deconv2d(x, layer=self.deconv_last)
        return x

# This class is referenced from pixelNeRF implementation
# In the original implementation, the feature extracted from image is directly 
# concatenated to the xyz coordinate to feed the MLP.
class SpatialEncoder(nn.Module):
    """
    2D (Spatial/Pixel-aligned/local) image encoder
    """

    def __init__(
        self,
        backbone="resnet34",
        pretrained=True,
        num_layers=4,
        index_interp="bilinear",
        index_padding="border",
        upsample_interp="bilinear",
        feature_scale=1.0,
        use_first_pool=True,
        norm_type="batch",
    ):
        """
        :param backbone Backbone network. Either custom, in which case
        model.custom_encoder.ConvEncoder is used OR resnet18/resnet34, in which case the relevant
        model from torchvision is used
        :param num_layers number of resnet layers to use, 1-5
        :param pretrained Whether to use model weights pretrained on ImageNet
        :param index_interp Interpolation to use for indexing
        :param index_padding Padding mode to use for indexing, border | zeros | reflection
        :param upsample_interp Interpolation to use for upscaling latent code
        :param feature_scale factor to scale all latent by. Useful (<1) if image
        is extremely large, to fit in memory.
        :param use_first_pool if false, skips first maxpool layer to avoid downscaling image
        features too much (ResNet only)
        :param norm_type norm type to applied; pretrained model must use batch
        """
        super().__init__()

        if norm_type != "batch":
            assert not pretrained

        self.use_custom_resnet = backbone == "custom"
        self.feature_scale = feature_scale
        self.use_first_pool = use_first_pool
        norm_layer = util.get_norm_layer(norm_type)

        if self.use_custom_resnet:
            print("WARNING: Custom encoder is experimental only")
            print("Using simple convolutional encoder")
            self.model = ConvEncoder(3, norm_layer=norm_layer)
            self.latent_size = self.model.dims[-1]
        else:
            print("Using torchvision", backbone, "encoder")
            self.model = getattr(torchvision.models, backbone)(
                weights='ResNet34_Weights.DEFAULT', norm_layer=norm_layer
            )
            # Following 2 lines need to be uncommented for older configs
            self.model.fc = nn.Sequential()
            self.model.avgpool = nn.Sequential()
            self.latent_size = [0, 64, 128, 256, 512, 1024, 1024, 1024, 1024][num_layers]

        self.num_layers = num_layers
        self.index_interp = index_interp
        self.index_padding = index_padding
        self.upsample_interp = upsample_interp
        self.register_buffer("latent", torch.empty(1, 1, 1, 1), persistent=False)
        self.register_buffer(
            "latent_scaling", torch.empty(2, dtype=torch.float32), persistent=False
        )
        # self.latent (B, L, H, W)

    def index(self, uv, cam_z=None, image_size=(), z_bounds=None):
        """
        Get pixel-aligned image features at 2D image coordinates
        :param uv (B, N, 2) image points (x,y)
        :param cam_z ignored (for compatibility)
        :param image_size image size, either (width, height) or single int.
        if not specified, assumes coords are in [-1, 1]
        :param z_bounds ignored (for compatibility)
        :return (B, L, N) L is latent size
        """
        with profiler.record_function("encoder_index"):
            if uv.shape[0] == 1 and self.latent.shape[0] > 1:
                uv = uv.expand(self.latent.shape[0], -1, -1)

            with profiler.record_function("encoder_index_pre"):
                if len(image_size) > 0:
                    if len(image_size) == 1:
                        image_size = (image_size, image_size)
                    scale = self.latent_scaling / image_size
                    uv = uv * scale - 1.0

            uv = uv.unsqueeze(2)  # (B, N, 1, 2)
            samples = F.grid_sample(
                self.latent,
                uv,
                align_corners=True,
                mode=self.index_interp,
                padding_mode=self.index_padding,
            )
            return samples[:, :, :, 0]  # (B, C, N)

    def forward(self, x):
        """
        For extracting ResNet's features.
        :param x image (B, C, H, W)
        :return latent (B, latent_size, H, W)
        """
        if self.feature_scale != 1.0:
            x = F.interpolate(
                x,
                scale_factor=self.feature_scale,
                mode="bilinear" if self.feature_scale > 1.0 else "area",
                align_corners=True if self.feature_scale > 1.0 else None,
                recompute_scale_factor=True,
            )
        x = x.to(device=self.latent.device)

        if self.use_custom_resnet:
            self.latent = self.model(x)
        else:
            x = self.model.conv1(x)
            x = self.model.bn1(x)
            x = self.model.relu(x)

            latents = [x]
            if self.num_layers > 1:
                if self.use_first_pool:
                    x = self.model.maxpool(x)
                x = self.model.layer1(x)
                latents.append(x)
            if self.num_layers > 2:
                x = self.model.layer2(x)
                latents.append(x)
            if self.num_layers > 3:
                x = self.model.layer3(x)
                latents.append(x)
            if self.num_layers > 4:
                x = self.model.layer4(x)
                latents.append(x)

            self.latents = latents
            align_corners = None if self.index_interp == "nearest " else True
            latent_sz = latents[0].shape[-2:]
            for i in range(len(latents)):
                latents[i] = F.interpolate(
                    latents[i],
                    latent_sz,
                    mode=self.upsample_interp,
                    align_corners=align_corners,
                )
            self.latent = torch.cat(latents, dim=1)
        self.latent_scaling[0] = self.latent.shape[-1]
        self.latent_scaling[1] = self.latent.shape[-2]
        self.latent_scaling = self.latent_scaling / (self.latent_scaling - 1) * 2.0
        return self.latent

"""
This module is not modified for now. 
Usage:
    Support concatenation of the GaussianFourierFeatureTransform
Problem:
    Need to know the exact number of the input and output's dimension and their exact meaning. And then add the information to the Usage row.
Future Change:
    Might use a PointNet++ Segmentation to subsitute the GaussianFourierFeatureTransform
"""
class GaussianFourierFeatureTransform(torch.nn.Module):
    """
    Modified based on the implementation of Gaussian Fourier feature mapping.

    "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains":
       https://arxiv.org/abs/2006.10739
       https://people.eecs.berkeley.edu/~bmild/fourfeat/index.html

    """

    def __init__(self, num_input_channels, mapping_size=93, scale=25, learnable=False, concat=True):
        super().__init__()
        self.concat = concat
        self.mapping_size = mapping_size
        self.scale = scale
        self.learnable = learnable
        if learnable:
            self._B = nn.Parameter(torch.randn(
                (num_input_channels, mapping_size)) * scale)
            # Here the num_imput_channels should be dim 
            # related to 32dim geo-feature or 32dim color-feature
            # And mapping_size is 93 as a default choice as a result of gaussian fourier feature transform
        else:
            self._B = torch.randn((num_input_channels, mapping_size)) * scale

    def forward(self, x):
        x = x.squeeze(0)
        assert x.dim() == 2, 'Expected 2D input (got {}D input)'.format(x.dim())
        x = (2*math.pi*x) @ self._B.to(x.device)
        if self.concat:
            return torch.cat((torch.sin(x), torch.cos(x)), dim=-1)
        else:
            return torch.sin(x)


"""
This module is not changed for now.
Usage:
    This module is actually a changed version of nn.Linear, and use relu as the activation method.
    Here the initialization method is xavier_uniform, but not kaiming initialization
Problem:    
    The reason why we use different Linear module initialization method 
    needs to be found out
"""
class DenseLayer(nn.Linear):
    def __init__(self, in_dim: int, out_dim: int, activation: str = "relu", *args, **kwargs) -> None:
        self.activation = activation
        super().__init__(in_dim, out_dim, *args, **kwargs)

    def reset_parameters(self) -> None:
        torch.nn.init.xavier_uniform_(
            self.weight, gain=torch.nn.init.calculate_gain(self.activation))
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

"""
This module is not changed for now
Usage:
    F_{theta} network in paper, here the F is a one-layer MLP, with 128 neurons and softpluss activations.
    And the input to the network is relative point vector and the neighboring points' features
    the output is final relative features used to construct color features.
Problem:
    What is the dimension of the input? 
    Need to find out whether all points are using the same network or many networks are constructed.
        -> figured out: there is only one such network in the geometry feature MLP
"""
class MLP_col_neighbor(nn.Module):
    # F_theta network in paper
    def __init__(self, c_dim, embedding_size_rel, hidden_size):
        super().__init__()
        self.linear1 = nn.Linear(c_dim + embedding_size_rel, hidden_size)
        self.linear2 = nn.Linear(hidden_size, c_dim)
        self.act_fn = nn.Softplus(beta=100)

        init.xavier_uniform_(self.linear1.weight)
        init.xavier_uniform_(self.linear2.weight)

    def forward(self, x):
        x = self.linear1(x)
        x = self.act_fn(x)
        x = self.linear2(x)
        return x

"""
This module is not chaged for now
Usage:
    This module is used as a framework to load the pretrained network in NICE-SLAM
    Therefore it has a grid_len argument to represent the voxel length of its corresponding feature grid.
Problem:
    Get to understand the whole architecture and the reason why it is constructed this way
    Input and output format
    Get to know how the gradient of the tracking process is passed. And give the gradient calculation.
"""
class MLP_geometry(nn.Module):
    """
    Decoder. Point coordinates not only used in sampling the feature grids, but also as MLP input.

    Args:
        c_dim (int): feature dimension.
        hidden_size (int): hidden size of Decoder network.
        n_blocks (int): number of layers.
        leaky (bool): whether to use leaky ReLUs.
        skips (list): list of layers to have skip connections.(layer index)
        pos_embedding_method (str): positional embedding method.
    """

    def __init__(self, cfg, c_dim=32,
                 hidden_size=128, n_blocks=5, leaky=False,
                 skips=[2], pos_embedding_method='fourier'):
        super().__init__()
        self.feat_name = 'geometry_feat'
        self.c_dim = c_dim
        self.n_blocks = n_blocks
        self.skips = skips
        self.weighting = cfg['pointcloud']['nn_weighting']
        self.use_dynamic_radius = cfg['use_dynamic_radius']
        self.min_nn_num = cfg['pointcloud']['min_nn_num']
        self.N_surface = cfg['rendering']['N_surface']
        self.img_encoder = SpatialEncoder(
            backbone=cfg['mapping']['img_encoder']['backbone'],
            pretrained=cfg['mapping']['img_encoder']['pretrained'],
            num_layers=cfg['mapping']['img_encoder']['num_layers'],
            index_interp=cfg['mapping']['img_encoder']['index_interp'],
            index_padding=cfg['mapping']['img_encoder']['index_padding'],
            upsample_interp=cfg['mapping']['img_encoder']['upsample_interp'],
            feature_scale=cfg['mapping']['img_encoder']['feature_scale'],
            use_first_pool=cfg['mapping']['img_encoder']['use_first_pool'],
            norm_type=cfg['mapping']['img_encoder']['norm_type']
        )

        if self.c_dim != 0:
            self.fc_c = nn.ModuleList([
                nn.Linear(self.c_dim, hidden_size) for i in range(n_blocks)
            ])
        
        if self.img_encoder.latent_size != 0:
            self.fc_img_feature = nn.ModuleList([
                nn.Linear(self.img_encoder.latent_size, hidden_size) for i in range(n_blocks)
            ])

        if pos_embedding_method == 'fourier':
            embedding_size = 93
            # the input dimension is always 3
            self.embedder = GaussianFourierFeatureTransform(
                3, mapping_size=embedding_size, scale=25, concat=False, learnable=True)

        # xyz coord. -> embedding size
        embedding_input = embedding_size
        self.pts_linears = nn.ModuleList(
            [DenseLayer(embedding_input, hidden_size, activation="relu")] +
            [DenseLayer(hidden_size, hidden_size, activation="relu") if i not in self.skips
             else DenseLayer(hidden_size + embedding_input, hidden_size, activation="relu") for i in range(self.n_blocks-1)])

        self.output_linear = DenseLayer(
            hidden_size, 1, activation="relu")

        if not leaky:
            self.actvn = torch.nn.Softplus(beta=100)
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

    def get_img_feature_at_pos(self, p, cur_c2w, fx, fy, cx, cy, cur_RGB, device):
        """
        cur_RGB should have the shape of (H, W), and only one picture is used here for now.
        """
        vertices = p.detach().cpu().numpy().reshape(-1, 3)
        c2w = cur_c2w.detach().cpu().numpy()
        w2c = np.linalg.inv(c2w)
        ones = np.ones_like(vertices[:, 0]).reshape(-1, 1)
        homo_vertices = np.concatenate(
            [vertices, ones], axis=1).reshape(-1, 4, 1)
        cam_cord_homo = w2c @ homo_vertices
        cam_cord = cam_cord_homo[:, :3]
        K = np.array([
            [fx, .0, cx],
            [.0, fy, cy],
            [.0, .0, 1.0]
        ]).reshape(3, 3)
        cam_cord[:, 0] *= -1
        uv = K @ cam_cord 
        z = uv[:, -1:] + 1e-5
        uv = uv[:, :2] / z
        uv = uv.astype(np.float32) # for now uv means pixel coordinate.
        uv = uv.reshape(1, uv.shape[0], uv.shape[1])

        W = cur_RGB.shape[0]
        H = cur_RGB.shape[1]
        cur_RGB = cur_RGB.reshape(1, 3, cur_RGB.shape[0], cur_RGB.shape[1])
        self.img_encoder(cur_RGB.float())
        uv = torch.tensor(uv).to(device)
        latent = self.img_encoder.index(uv, image_size=torch.tensor([W, H]).to(device)) # Now the size of latent should be (1, L, N_points)
        latent = latent.transpose(1, 2).reshape(-1, self.img_encoder.latent_size) # (N_points, L)

        return latent

    def get_feature_at_pos(self, npc, p, npc_feats, is_tracker=False, cloud_pos=None, with_prune=False, 
                           dynamic_r_query=None):
        assert torch.is_tensor(
            p), 'point locations for get_feature_at_pos should be tensor.'
        device = p.device
        p = p.reshape(-1, 3)
        D, I, neighbor_num = npc.find_neighbors_faiss(p.detach().clone(),
                                                      step='query',
                                                      dynamic_radius=dynamic_r_query)

        # faiss returns "D" in the form of squared distances. Thus we compare D to the squared radius
        radius_query_bound = npc.get_radius_query(
        )**2 if (with_prune or (not self.use_dynamic_radius)) else dynamic_r_query.reshape(-1, 1)**2
        if is_tracker:
            # re-calculate D to propagate gradients to the camera extrinsics
            nn_num = D.shape[1]
            D = torch.sum(torch.square(
                cloud_pos[I]-p.reshape(-1, 1, 3)), dim=-1)
            D = D.reshape(-1, nn_num)

        has_neighbors = neighbor_num > self.min_nn_num-1

        if self.weighting == 'distance':
            weights = 1.0/(D+1e-10)
        else:
            # try to avoid over-smoothing by e^(-x)
            weights = torch.exp(-20*torch.sqrt(D))
        weights[D > radius_query_bound] = 0.

        # (n_points, nn_num=8, 1)
        weights = F.normalize(weights, p=1, dim=1).unsqueeze(-1)

        # use fixed num of nearest nn
        # select neighbors within range, then interpolate feature by inverse distance weighting
        neighbor_feats = npc_feats[I]  # (n_points, nn_num=8, c_dim)

        c = weights * neighbor_feats
        c = c.sum(axis=1).reshape(-1, self.c_dim)
        # points with no neighbors are given a random feature vector
        # rays that have no neighbors are thus rendered with random feature vectors for depth
        c[~has_neighbors] = torch.zeros(
            [self.c_dim], device=device).normal_(mean=0, std=0.01)

        return c, has_neighbors  # (N_point,c_dim), mask for pts

    def forward(self, p, npc, npc_geo_feats, cur_c2w, fx, fy, cx, cy, cur_RGB, with_prune=False, pts_num=16, is_tracker=False, cloud_pos=None,
                pts_views_d=None, dynamic_r_query=None):
        """
        forward method of geometric decoder.

        Args:
            p (tensor): sampling locations, N*3
            npc (NerualPointCloud): shared npc object
            npc_geo_feats (tensor): cloned from npc. Contains the optimizable parameters during mapping
            with_prune (bool): label who are calling. If it is called by the prunning strategy, no need to search for neighbor.
            pts_num (int, optional): sampled pts num along each ray. Defaults to N_surface.
            is_tracker (bool, optional): whether called by tracker. Defaults to False.
            cloud_pos (tensor, optional): point cloud position. 
            pts_views_d (tensor): viewing directions
            dynamic_r_query (tensor, optional): if enabled dynamic radius, query radius for every pixel will be different.

        Returns:
            out (tensor): occupancies for the points p
            valid_ray_mask (bool): boolen tensor. True if at least half of all points along the ray have neighbors
            has_neighbors (bool): boolean tensor. False if at least two neighbors were not found for the point in question
        """

        c, has_neighbors = self.get_feature_at_pos(
            npc, p, npc_geo_feats, is_tracker, cloud_pos, with_prune, dynamic_r_query=dynamic_r_query)  # get (N,c_dim), e.g. (N,32)

        img_feature = self.get_img_feature_at_pos(
            p, cur_c2w, fx, fy, cx, cy, cur_RGB, device='cuda:0'
        ) # The size of img_feature should be latent size.

        valid_ray_mask = None
        # ray is not close to the current npc, choose bar here
        # a ray is considered valid if at least half of all points along the ray have neighbors.
        if not with_prune:
            valid_ray_mask = ~(
                torch.sum(has_neighbors.view(-1, pts_num), 1) < int(self.N_surface/2+1))

        p = p.float().reshape(1, -1, 3)

        embedded_pts = self.embedder(p)
        embedded_input = embedded_pts

        h = embedded_input
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if self.c_dim != 0:
                # hidden dim + (feature dim->hidden dim) -> hidden dim
                h = h + self.fc_c[i](c)
                h = h + self.fc_img_feature[i](img_feature)
                # so for hidden layers in the decoder, its input comes from both its feature and embedded location.
            if i in self.skips:
                h = torch.cat([embedded_input, h], -1)
        out = self.output_linear(h)

        # (N,1)->(N,) for occupancy
        out = out.squeeze(-1)
        return out, valid_ray_mask, has_neighbors
    
class Same(nn.Module):
    def __init__(self, mapping_size=3) -> None:
        super().__init__()
        self.mapping_size = mapping_size
    
    def forward(self, x):
        x = x.squeeze(0)
        return x
    
class MLP_exposure(nn.Module):
    # Exposure compensation MLP
    def __init__(self, latent_dim, hidden_size):
        super().__init__()
        self.linear1 = nn.Linear(latent_dim, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 12)
        self.act_fn = nn.Softplus(beta=100)

        init.normal_(self.linear1.weight, mean=0, std=0.01)
        init.normal_(self.linear2.weight, mean=0, std=0.01)

    def forward(self, x):
        x = self.linear1(x)
        x = self.act_fn(x)
        x = self.linear2(x)
        return x

class MLP_color(nn.Module):
    """
    Decoder. Point coordinates not only used in sampling the feature grids, but also as MLP input.

    Args:
        c_dim (int): feature dimension.
        hidden_size (int): hidden size of Decoder network.
        n_blocks (int): number of layers.
        leaky (bool): whether to use leaky ReLUs.
        skips (list): list of layers to have skip connections.
        grid_len (float): voxel length of its corresponding feature grid.
        pos_embedding_method (str): positional embedding method.
        use_view_direction (bool): whether to use view direction.
    """

    def __init__(self, cfg, c_dim=32,
                 hidden_size=128, n_blocks=5, leaky=False,
                 skips=[2], pos_embedding_method='fourier',
                 use_view_direction=False):
        super().__init__()
        self.feat_name = 'color_feat'
        self.c_dim = c_dim
        self.n_blocks = n_blocks
        self.skips = skips
        self.weighting = cfg['pointcloud']['nn_weighting']
        self.min_nn_num = cfg['pointcloud']['min_nn_num']
        self.use_dynamic_radius = cfg['use_dynamic_radius']
        self.N_surface = cfg['rendering']['N_surface']
        self.use_view_direction = use_view_direction
        self.encode_rel_pos_in_col = cfg['model']['encode_rel_pos_in_col']
        self.encode_exposure = cfg['model']['encode_exposure']
        self.encode_viewd = cfg['model']['encode_viewd']

        if c_dim != 0:
            self.fc_c = nn.ModuleList([
                nn.Linear(c_dim, hidden_size) for i in range(n_blocks)
            ])

        if pos_embedding_method == 'fourier':
            embedding_size = 20
            # the input dimension is always 3
            self.embedder = GaussianFourierFeatureTransform(
                3, mapping_size=embedding_size, scale=32)
            if self.use_view_direction:
                if self.encode_viewd:
                    self.embedder_view_direction = GaussianFourierFeatureTransform(
                        3, mapping_size=embedding_size, scale=32)
                else:
                    self.embedder_view_direction = Same(mapping_size=3)
            self.embedder_rel_pos = GaussianFourierFeatureTransform(
                3, mapping_size=10, scale=32, learnable=True)
        self.mlp_col_neighbor = MLP_col_neighbor(
            self.c_dim, 2*self.embedder_rel_pos.mapping_size, hidden_size)
        # Here the reason why we need to multiply 2 is that the embedder_view_direction use concatenation.
        if self.encode_exposure:
            self.mlp_exposure = MLP_exposure(
                cfg['model']['exposure_dim'], hidden_size)

        # xyz coord. -> embedding size
        embedding_input = 2*embedding_size # 40
        if self.use_view_direction:
            embedding_input += (2 if self.encode_viewd else 1) * \
                self.embedder_view_direction.mapping_size 
            # if encode_viewd, then embedding_input is 40 + 2 * 20 = 80
            # else, the embedding_input is 40 + 1 * 3 = 43
        self.pts_linears = nn.ModuleList(
            [DenseLayer(embedding_input, hidden_size, activation="relu")] +
            [DenseLayer(hidden_size, hidden_size, activation="relu") if i not in self.skips
             else DenseLayer(hidden_size + embedding_input, hidden_size, activation="relu") for i in range(n_blocks-1)])

        self.output_linear = DenseLayer(
            hidden_size, 3, activation="linear")

        if not leaky:
            self.actvn = torch.nn.Softplus(beta=100)
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

    def get_feature_at_pos(self, npc, p, npc_feats, is_tracker=False, cloud_pos=None,
                           dynamic_r_query=None):
        # Here the dynamic_r_query should be a list, which list the query radius of each point
        # And here npc should be abbreviation for neural point cloud. This should be a set or list to represent the point cloud
        # p should be the queried point, which is a (x, y, z) position.
        # cloud_pos should be a list which contains the position of each point in the point cloud.
        # is_tracker labels current process, and judge whether it is a tracker or a mapper.
        assert torch.is_tensor(
            p), 'point locations for get_feature_at_pos should be tensor.'
        device = p.device
        p = p.reshape(-1, 3)
        D, I, neighbor_num = npc.find_neighbors_faiss(p.detach().clone(),
                                                      step='query',
                                                      dynamic_radius=dynamic_r_query)
        # faiss returns "D" in the form of squared distances. Thus we compare D to the squared radius
        radius_query_bound = npc.get_radius_query(
        )**2 if not self.use_dynamic_radius else dynamic_r_query.reshape(-1, 1)**2
        if is_tracker:
            # re-calculate D to propagate gradients to the camera extrinsics
            nn_num = D.shape[1]
            D = torch.sum(torch.square(
                cloud_pos[I]-p.reshape(-1, 1, 3)), dim=-1)
            # Here the cloud_pos size should be (n_points, 3)
            # And the I's size should be (n_points, nn_num)
            # -> cloud_pos[I] size should be (n_points, nn_num, 3)
            # And D here should the Euclidean distance's square
            # size of D is (n_points, nn_num), each element is the distance number
            D = D.reshape(-1, nn_num)

        has_neighbors = neighbor_num > self.min_nn_num-1

        if self.weighting == 'distance':
            weights = 1.0/(D+1e-10)
        else:
            # try to avoid over-smoothing by e^(-x)
            weights = torch.exp(-20*torch.sqrt(D))
        weights[D > radius_query_bound] = 0.
        weights = F.normalize(weights, p=1, dim=1).unsqueeze(-1)

        # use fixed num of nearest nn
        # select neighbors within range, then interpolate feature by inverse distance weighting
        neighbor_feats = npc_feats[I]             # (n_points, nn_num=8, c_dim)
        if self.encode_rel_pos_in_col:
            neighbor_pos = cloud_pos[I]  # (N,nn_num,3)
            neighbor_rel_pos = neighbor_pos - p[:, None, :]
            embedding_rel_pos = self.embedder_rel_pos(
                neighbor_rel_pos.reshape(-1, 3))             # (N, nn_num, 20)
            neighbor_feats = torch.cat([embedding_rel_pos.reshape(neighbor_pos.shape[0], -1, self.embedder_rel_pos.mapping_size*2),
                                        neighbor_feats], dim=-1)  # (N, nn_num, 20+c_dim)
            neighbor_feats = self.mlp_col_neighbor(
                neighbor_feats)                  # (N, nn_num, c_dim)

        c = weights * neighbor_feats
        c = c.sum(axis=1).reshape(-1, self.c_dim)
        # points with no neighbors are given a random feature vector
        # rays that have no neighbors are thus rendered with random feature vectors for color
        c[~has_neighbors] = torch.zeros(
            [self.c_dim], device=device).normal_(mean=0, std=0.01)

        return c, has_neighbors  # (N_point,c_dim), mask for pts
    
    def forward(self, p, npc, npc_col_feats, is_tracker=False, cloud_pos=None, pts_views_d=None, dynamic_r_query=None, exposure_feat=None):
        """
        forwad method of decoder.

        Args:
            p (tensor): sampling locations, N*3
            npc (NerualPointCloud): shared npc object
            npc_col_feats (tensor): cloned from npc. Contains the optimizable parameters during mapping
            pts_num (int, optional): sampled pts num along each ray. Defaults to N_surface.
            is_tracker (bool, optional): whether is called by tracker.
            cloud_pos (tensor, optional): point cloud position, used when called by tracker to re-calculate D. 
            pts_views_d (tensor): viweing directions
            dynamic_r_query (tensor, optional): if enabled dynamic radius, query radius for every pixel will be different.
            exposure_feat (tensor): exposure feature vector. Needs to be the same for all points in the batch.

        Returns:
            predicted colors for points p
        """
        c, _ = self.get_feature_at_pos(
            npc, p, npc_col_feats, is_tracker, cloud_pos, dynamic_r_query=dynamic_r_query)
        p = p.float().reshape(1, -1, 3) # (1, N, 3)

        embedded_pts = self.embedder(p) # (1, N, 40)
        embedded_input = embedded_pts

        if self.use_view_direction:
            pts_views_d = F.normalize(pts_views_d, p=2, dim=1) # The shape should be (1, N, 3), now the normalization is done each sample
            embedded_views_d = self.embedder_view_direction(pts_views_d) # (1, N, 40)
            embedded_input = torch.cat(
                [embedded_pts, embedded_views_d], -1) # (1, N, 80) concatenation of the point's location embedding and viewing direction embeddings
        h = embedded_input
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = self.actvn(h)
            if self.c_dim != 0:
                # hidden dim + (feature dim->hidden dim) -> hidden dim
                h = h + self.fc_c[i](c)
                # so for hidden layers in the decoder, its input comes from both its feature and embedded location.
            if i in self.skips:
                h = torch.cat([embedded_input, h], -1)
        out = self.output_linear(h)
        if self.encode_exposure:
            if exposure_feat is not None:
                affine_tensor = self.mlp_exposure(exposure_feat)
                rot, trans = affine_tensor[:9].reshape(
                    3, 3), affine_tensor[-3:]
                out = torch.matmul(out, rot) + trans
                out = torch.sigmoid(out)
            else:
                # apply exposure compensation outside "self.renderer.render_batch_ray" call in mapper
                # this is done when multiple exposure feature vectors are needed for different rays
                # during mapping. Each keyframe has its own exposure feature vector, while the forward
                # function of the MLP_color class assumes that all rays have the same exposure feature
                # vector.
                return out
        else:
            out = torch.sigmoid(out)

        return out
    
class POINT(nn.Module):
    """    
    Decoder for point represented features.

    Args:
        c_dim (int): feature dimension.
        hidden_size (int): hidden size of decoder network
        pos_embedding_method (str): positional embedding method.
        use_view_direction (bool): use view direction or not.
    """

    def __init__(self, cfg, c_dim=32,
                 hidden_size=128,
                 pos_embedding_method='fourier', use_view_direction=False):
        super().__init__()

        self.geo_decoder = MLP_geometry(cfg=cfg, c_dim=c_dim,
                                        skips=[2], n_blocks=5, hidden_size=32,
                                        pos_embedding_method=pos_embedding_method)
        self.color_decoder = MLP_color(cfg=cfg, c_dim=c_dim,
                                       skips=[2], n_blocks=5, hidden_size=hidden_size,
                                       pos_embedding_method=pos_embedding_method,
                                       use_view_direction=use_view_direction)

    def forward(self, p, npc, stage, npc_geo_feats, npc_col_feats, pts_num=16, is_tracker=False, cloud_pos=None,
                pts_views_d=None, dynamic_r_query=None, exposure_feat=None, with_prune=False, cur_c2w=None, fx=None, fy=None, cx=None, cy=None, cur_RGB=None):
        """
            Output occupancy/color and associated masks for validity

        Args:
            p (tensor): point locations
            npc (tensor): NeuralPointCloud object.
            stage (str): listed below.
            npc_geo_feats (tensor): cloned from npc. Contains the optimizable parameters during mapping
            npc_col_feats (tensor): cloned from npc. Contains the optimizable parameters during mapping
            pts_num (int): number of points in sampled in each ray, used only by geo_decoder.
            is_tracker (bool): whether called by tracker.
            cloud_pos (tensor): (N,3)
            pts_views_d (tensor): used if color decoder encodes viewing directions.
            dynamic_r_query (tensor): (N,), used if dynamic radius enabled.
            exposure_feat (tensor): exposure feature vector. Needs to be the same for all points in the batch.

        Returns:
            raw (tensor): predicted color and occupancies for the points p
            ray_mask (tensor): boolen tensor. True if at least half of all points along the ray have neighbors
            point_mask (tensor): boolean tensor. False if at least two neighbors were not found for the point in question
        """
        device = f'cuda:{p.get_device()}' 
        match stage:
            case 'geometry':
                geo_occ, ray_mask, point_mask = self.geo_decoder(p, npc, npc_geo_feats, with_prune=with_prune,
                                                                 pts_num=pts_num, is_tracker=is_tracker, cloud_pos=cloud_pos,
                                                                 dynamic_r_query=dynamic_r_query, cur_c2w=cur_c2w, fx=fx, fy=fy, cx=cx, cy=cy, cur_RGB=cur_RGB)
                raw = torch.zeros(
                    geo_occ.shape[0], 4, device=device, dtype=torch.float)
                raw[..., -1] = geo_occ
                return raw, ray_mask, point_mask
            case 'color':
                geo_occ, ray_mask, point_mask = self.geo_decoder(p, npc, npc_geo_feats,
                                                                 pts_num=pts_num, is_tracker=is_tracker, cloud_pos=cloud_pos,
                                                                 dynamic_r_query=dynamic_r_query, cur_c2w=cur_c2w, fx=fx, fy=fy, cx=cx, cy=cy, cur_RGB=cur_RGB)
                raw = self.color_decoder(p, npc, npc_col_feats,                                # returned (N,4)
                                         is_tracker=is_tracker, cloud_pos=cloud_pos,
                                         pts_views_d=pts_views_d,
                                         dynamic_r_query=dynamic_r_query, exposure_feat=exposure_feat)
                raw = torch.cat([raw, geo_occ.unsqueeze(-1)], dim=-1)
                return raw, ray_mask, point_mask
