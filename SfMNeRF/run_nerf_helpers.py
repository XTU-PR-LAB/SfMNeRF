import torch
# torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.signal
import os
from torch.linalg import det


# Misc
img2mse = lambda x, y : torch.mean((x - y) ** 2)
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)


# Positional encoding (section 5.1)
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=0):
    if i == -1:
        return nn.Identity(), 3
    
    embed_kwargs = {
                'include_input' : True,
                'input_dims' : 3,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }
    
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim


# Model
class NeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=False):
        """ 
        """
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)])
        
        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W//2)])

        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])
        
        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W//2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, x):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)
        
            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        return outputs    

    def load_weights_from_keras(self, weights):
        assert self.use_viewdirs, "Not implemented if use_viewdirs=False"
        
        # Load pts_linears
        for i in range(self.D):
            idx_pts_linears = 2 * i
            self.pts_linears[i].weight.data = torch.from_numpy(np.transpose(weights[idx_pts_linears]))    
            self.pts_linears[i].bias.data = torch.from_numpy(np.transpose(weights[idx_pts_linears+1]))
        
        # Load feature_linear
        idx_feature_linear = 2 * self.D
        self.feature_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_feature_linear]))
        self.feature_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_feature_linear+1]))

        # Load views_linears
        idx_views_linears = 2 * self.D + 2
        self.views_linears[0].weight.data = torch.from_numpy(np.transpose(weights[idx_views_linears]))
        self.views_linears[0].bias.data = torch.from_numpy(np.transpose(weights[idx_views_linears+1]))

        # Load rgb_linear
        idx_rbg_linear = 2 * self.D + 4
        self.rgb_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear]))
        self.rgb_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear+1]))

        # Load alpha_linear
        idx_alpha_linear = 2 * self.D + 6
        self.alpha_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear]))
        self.alpha_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear+1]))



# Ray helpers
def get_rays(H, W, K, c2w):
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,-1].expand(rays_d.shape)
    return rays_o, rays_d


def get_rays_np(H, W, K, c2w):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))
    return rays_o, rays_d

def inverse_ndc_rays(H, W, focal, X):
    z = 2 * focal / (X[...,2] - 1.)
    x = -X[...,0] * z * W / (2. * focal)
    y = -X[...,1] * z * H / (2. * focal)
    unprojected_X = torch.stack([x,y,z], -1)
    return unprojected_X
    

def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = -(near + rays_o[...,2]) / rays_d[...,2]
    rays_o = rays_o + t[...,None] * rays_d
    
    # Projection
    o0 = -1./(W/(2.*focal)) * rays_o[...,0] / rays_o[...,2]
    o1 = -1./(H/(2.*focal)) * rays_o[...,1] / rays_o[...,2]
    o2 = 1. + 2. * near / rays_o[...,2]

    d0 = -1./(W/(2.*focal)) * (rays_d[...,0]/rays_d[...,2] - rays_o[...,0]/rays_o[...,2])
    d1 = -1./(H/(2.*focal)) * (rays_d[...,1]/rays_d[...,2] - rays_o[...,1]/rays_o[...,2])
    d2 = -2. * near / rays_o[...,2]
    
    rays_o = torch.stack([o0,o1,o2], -1)
    rays_d = torch.stack([d0,d1,d2], -1)
    
    return rays_o, rays_d


# Hierarchical sampling (section 5.2)
def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
    # Get pdf
    weights = weights + 1e-5 # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

    # Pytest, overwrite u with numpy's fixed random numbers
    if pytest:
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]
        if det:
            u = np.linspace(0., 1., N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = torch.Tensor(u)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[...,1]-cdf_g[...,0])
    denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)
    t = (u-cdf_g[...,0])/denom
    samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])

    return samples
''' Evaluation metrics (ssim, lpips)
'''
def rgb_ssim(img0, img1, max_val,
             filter_size=11,
             filter_sigma=1.5,
             k1=0.01,
             k2=0.03,
             return_map=False):
    # Modified from https://github.com/google/mipnerf/blob/16e73dfdb52044dcceb47cda5243a686391a6e0f/internal/math.py#L58
    assert len(img0.shape) == 3
    assert img0.shape[-1] == 3
    assert img0.shape == img1.shape

    # Construct a 1D Gaussian blur filter.
    hw = filter_size // 2
    shift = (2 * hw - filter_size + 1) / 2
    f_i = ((np.arange(filter_size) - hw + shift) / filter_sigma)**2
    filt = np.exp(-0.5 * f_i)
    filt /= np.sum(filt)

    # Blur in x and y (faster than the 2D convolution).
    def convolve2d(z, f):
        return scipy.signal.convolve2d(z, f, mode='valid')
        # this_type_str = type(z)
        # if this_type_str is np.ndarray:
        #     return scipy.signal.convolve2d(z, f, mode='valid')
        # else:
        #     return scipy.signal.convolve2d(z.cpu(), f, mode='valid')

    filt_fn = lambda z: np.stack([
        convolve2d(convolve2d(z[...,i], filt[:, None]), filt[None, :])
        for i in range(z.shape[-1])], -1)
    mu0 = filt_fn(img0)
    mu1 = filt_fn(img1)
    mu00 = mu0 * mu0
    mu11 = mu1 * mu1
    mu01 = mu0 * mu1
    sigma00 = filt_fn(img0**2) - mu00
    sigma11 = filt_fn(img1**2) - mu11
    sigma01 = filt_fn(img0 * img1) - mu01

    # Clip the variances and covariances to valid values.
    # Variance must be non-negative:
    sigma00 = np.maximum(0., sigma00)
    sigma11 = np.maximum(0., sigma11)
    sigma01 = np.sign(sigma01) * np.minimum(
        np.sqrt(sigma00 * sigma11), np.abs(sigma01))
    c1 = (k1 * max_val)**2
    c2 = (k2 * max_val)**2
    numer = (2 * mu01 + c1) * (2 * sigma01 + c2)
    denom = (mu00 + mu11 + c1) * (sigma00 + sigma11 + c2)
    ssim_map = numer / denom
    ssim = np.mean(ssim_map)
    return ssim_map if return_map else ssim

__LPIPS__ = {}
def init_lpips(net_name, device):
    assert net_name in ['alex', 'vgg']
    import lpips
    print(f'init_lpips: lpips_{net_name}')
    return lpips.LPIPS(net=net_name, version='0.1').eval().to(device)

def rgb_lpips(np_gt, np_im, net_name, device):
    if net_name not in __LPIPS__:
        __LPIPS__[net_name] = init_lpips(net_name, device)
    gt = torch.from_numpy(np_gt).permute([2, 0, 1]).contiguous().to(device)
    im = torch.from_numpy(np_im).permute([2, 0, 1]).contiguous().to(device)
    return __LPIPS__[net_name](gt, im, normalize=True).item()

def load_txt_file(data_file):
    if os.stat(data_file).st_size == 0:
        return []
    with open(data_file,'r') as in_file:
        txt = in_file.readlines()
        txt = [txt[i].split(' ') for i in range(len(txt))]                      
    for i in range(len(txt)):    
        for j in range(len(txt[0])):                                     
            txt[i][j] = txt[i][j].strip('[\n,\t]') 
            txt[i][j] = float(txt[i][j])
    return txt

def load_mask_rect(data_file):
    mask_rect = load_txt_file(data_file)
    mask_rect_array = np.array(mask_rect)  # [n, 6], followed as img0_index, img1_index, x,y, w,h
    mask_rect = {}
    for i in range(len(mask_rect_array)):
        mask_rect['{:03d}_{:03d}'.format(int(mask_rect_array[i][0]), int(mask_rect_array[i][1]))] = [mask_rect_array[i][2], mask_rect_array[i][3], mask_rect_array[i][4], mask_rect_array[i][5]]
    return mask_rect

def get_sift_rays(i, j, K, c2w):
    # i--x, j--y
    dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1) 
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,-1].expand(rays_d.shape) 
    return rays_o, rays_d

def bilinear_sampler(img, coord,y_first):
    # coord-[n, 2]--(y,x)
    # img-[h, w, 3]
    if y_first:
        coord_y = coord[:, 0]
        coord_x = coord[:, 1]
    else:
        coord_x = coord[:, 0]
        coord_y = coord[:, 1]    
    x0 = coord_x.floor()
    x1 = x0 + 1
    y0 = coord_y.floor()
    y1 = y0 + 1
    x0 = x0.long()
    x1 = x1.long()
    y0 = y0.long()
    y1 = y1.long()

    y_max = torch.ones_like(x0) * (img.shape[0] - 1)  # height-1
    x_max = torch.ones_like(x0) * (img.shape[1] - 1)  # width-1

    x0_safe = torch.where(x0 < 0, torch.zeros_like(x0), torch.where(x0 > x_max, x_max, x0) )
    y0_safe = torch.where(y0 < 0, torch.zeros_like(x0), torch.where(y0 > y_max, y_max, y0) )
    x1_safe = torch.where(x1 < 0, torch.zeros_like(x0), torch.where(x1 > x_max, x_max, x1) )
    y1_safe = torch.where(y1 < 0, torch.zeros_like(x0), torch.where(y1 > y_max, y_max, y1) )    

    wt_x0 = x1_safe - coord_x  
    wt_x1 = coord_x - x0_safe
    wt_y0 = y1_safe - coord_y
    wt_y1 = coord_y - y0_safe  

    im00 = img[y0_safe, x0_safe]  # left top
    im10 = img[y0_safe, x1_safe]  # right top
    im01 = img[y1_safe, x0_safe]  # left bottom
    im11 = img[y1_safe, x1_safe]  # right bottom

    w00 = wt_x0 * wt_y0
    w01 = wt_x0 * wt_y1
    w10 = wt_x1 * wt_y0
    w11 = wt_x1 * wt_y1
    
    output = w00.unsqueeze(-1) * im00 + w01.unsqueeze(-1) * im01 + w10.unsqueeze(-1) * im10 + w11.unsqueeze(-1) * im11
    return output

def cof1(M,index):
    zs = M[:index[0]-1,:index[1]-1]
    ys = M[:index[0]-1,index[1]:]
    zx = M[index[0]:,:index[1]-1]
    yx = M[index[0]:,index[1]:]
    s = torch.cat((zs,ys),axis=1)
    x = torch.cat((zx,yx),axis=1)
    return det(torch.cat((s,x),axis=0))

def alcof(M,index):
    return pow(-1,index[0]+index[1])*cof1(M,index)

def adj(M):
    result = torch.zeros((M.shape[0],M.shape[1]))
    for i in range(1,M.shape[0]+1):
        for j in range(1,M.shape[1]+1):
            result[j-1][i-1] = alcof(M,[i,j])
    return result

# 矩阵求逆
def invmat(M):
    return 1.0/det(M)*adj(M) 

def get_pose_inverse(pose):
    # poses-[n, 3, 4]
    temp_pose = []
    for i in range(pose.shape[0]):
        pose_i = pose[i, ...]  # [3, 4]
        R_inv = invmat(pose_i[:, :3])
        t = pose_i[:, 3] # [3, 1]
        t_inv = -(R_inv @ t)
        pose_i_inv = torch.cat([R_inv, t_inv.unsqueeze(-1)], dim=1)  #[3, 4]
        temp_pose.append(pose_i_inv)
        
    pose_inv = torch.stack(temp_pose, dim=0) # [n, 3, 4]
    return pose_inv 

def get_transfomation_matrix(ref_pose, local_pose):
    """ Create the transformation matrix between ref_pose and local_pose
    :param ref_pose, target_pose: [3, 4]
    :returns: 3 x 3 rotation, 3 x 1 translation
    """
    ref_r = ref_pose[:, :3] # [3, 3]
    ref_t = ref_pose[:, 3].unsqueeze(-1) # np.expand_dims(ref_pose[:, 3], -1) # [3, 1]     
    ref_r_inv = torch.linalg.inv (ref_r) 

    loc_r = local_pose[:, :3] # [3, 3]
    loc_t = local_pose[:, 3].unsqueeze(-1) # np.expand_dims(local_pose[:, 3], -1) # [3, 1]
    trans_r = torch.matmul(ref_r_inv, loc_r) # [3, 3]
    trans_t = torch.matmul(ref_r_inv, loc_t-ref_t) # [3, 1]
    trans_pose = torch.cat([trans_r, trans_t], 1) # np.concatenate([trans_r, trans_t], 1) # [3, 4]
    return trans_pose

def cam2pixel(cam_coords, intrinsics):
    """Transform coordinates in the camera frame to the pixel frame.
    Args:
        cam_coords: [B, 3]
        intrinsics - [3, 3]
    Returns:
        coordinates -- [B, 2]
    """
    b, _ = cam_coords.size()
    cam_coords_flat = cam_coords.reshape(b, 3, -1) # [b, 3, 1]
    b_intrinsics = intrinsics.unsqueeze(0).repeat(b, 1, 1) # [b, 3, 3]
    pcoords = b_intrinsics @ cam_coords_flat  # [b, 3, 1]
    X = pcoords[:, 0] # [B, 1]
    Y = pcoords[:, 1]
    Z = pcoords[:, 2].clamp(min=1e-3)

    X = X / Z # [B, 1]
    Y = Y / Z

    pixel_coords = torch.cat([X, Y], dim=1)  # [B, 2]
    return pixel_coords

def get_projected_pt(pts, poses, intrinsics):

    # pts-[N_sift, 3]
    # poses-[3, 4]
    # intrinsics-[3, 3]
    # return-[N_sift, 2]

    rot, tr = poses[:,:3], poses[:,-1:]  # [3, 3], [3, 1]
    rot = rot.repeat(pts.shape[0], 1, 1) # [N_sift, 3, 3]

    tr = tr.repeat(pts.shape[0], 1, 1) # [N_sift, 3, 1]
    
    pts = pts.reshape(-1, 3, 1).float() # [b=N_sift, 3, 1]
    
    pixel_coords = rot @ pts + tr    # [b= N_sift, 3]     
    pixel_coords = torch.cat([pixel_coords[:, 0, :], -pixel_coords[:, 1, :], -pixel_coords[:, 2, :]], dim = -1)  
    pixel_coords = cam2pixel(pixel_coords,intrinsics)  # [b=N_sift, 2]

    return pixel_coords

class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """
    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool   = nn.AvgPool2d(3, 1)
        self.mu_y_pool   = nn.AvgPool2d(3, 1)
        self.sig_x_pool  = nn.AvgPool2d(3, 1)
        self.sig_y_pool  = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x  = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y  = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)
    
def get_smooth_loss(disp, img):
    """Computes the smoothness loss for a disparity image
    The color image is used for edge-aware smoothness
    """
    grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
    grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

    grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)

    return grad_disp_x.mean() + grad_disp_y.mean()

def load_sift_correspondences(data_file):
    if not os.path.exists(data_file): 
        return []
    sift_correspondences = load_txt_file(data_file)
    return sift_correspondences

def epipolar_constraint(pts):   
    loss = torch.mean(((pts[0, ...] - pts[1, ...])) ** 2)
    return loss

def cal_epolar_loss(pt, mapped_pts):
    pts = torch.ones_like(mapped_pts) * pt
    error = torch.mean((pts - mapped_pts) ** 2, dim=1) # torch.mean() # torch.abs(pts - mapped_pts)
    error = torch.min(error)
    return error

def load_predicted_pts(data_file):
    if not os.path.exists(data_file):  
        return []
    depth = load_txt_file(data_file)
    depth = np.array(depth)
    return depth

def load_fundemental_matrix(data_file):
    fundemental_matrix = load_txt_file(data_file)
    fundemental_matrix = np.array(fundemental_matrix)  
    return fundemental_matrix.reshape(3, 3)

def del_file(path_data):
    for i in os.listdir(path_data) :
        file_data = path_data + "/" + i
        if os.path.isfile(file_data) == True:
            os.remove(file_data)
        else:
            del_file(file_data)
    os.rmdir(path_data)
    
def compute_similarity_transform(X, Y, compute_optimal_scale=False):
    """
    A port of MATLAB's `procrustes` function to Numpy.
    Adapted from http://stackoverflow.com/a/18927641/1884420

    Args
    X: array NxM of targets, with N number of points and M point dimensionality
    Y: array NxM of inputs
    compute_optimal_scale: whether we compute optimal scale or force it to be 1

    Returns:
    d: squared error after transformation
    Z: transformed Y
    T: computed rotation
    b: scaling
    c: translation
    """
    import numpy as np

    muX = X.mean(0)
    muY = Y.mean(0)

    X0 = X - muX
    Y0 = Y - muY

    ssX = (X0**2.).sum()
    ssY = (Y0**2.).sum()

    # centred Frobenius norm
    normX = np.sqrt(ssX)
    normY = np.sqrt(ssY)

    # scale to equal (unit) norm
    X0 = X0 / normX
    Y0 = Y0 / normY

    # optimum rotation matrix of Y
    A = np.dot(X0.T, Y0)
    U,s,Vt = np.linalg.svd(A,full_matrices=False)
    V = Vt.T
    T = np.dot(V, U.T)

    # Make sure we have a rotation
    detT = np.linalg.det(T)
    V[:,-1] *= np.sign( detT )
    s[-1]   *= np.sign( detT )
    T = np.dot(V, U.T)

    traceTA = s.sum()

    if compute_optimal_scale:  # Compute optimum scaling of Y.
        b = traceTA * normX / normY
        d = 1 - traceTA**2
        Z = normX*traceTA*np.dot(Y0, T) + muX
    else:  # If no scaling allowed
        b = 1
        d = 1 + ssY/ssX - 2 * traceTA * normY / normX
        Z = normY*np.dot(Y0, T) + muX

    c = muX - b*np.dot(muY, T)

    return d, Z, T, b, c

def eval_predicted_depth(pts, poses, gt_depth, gt_mask=None):
    rot, tr = poses[:,:3], poses[:,-1:]  # [3, 3], [3, 1]
    rot = rot.repeat(pts.shape[0], 1, 1) # [n, 3, 3]

    tr = tr.repeat(pts.shape[0], 1, 1) # [n, 3, 1]
    
    pts = pts.reshape(-1, 3, 1).float() # [n, 3, 1]
    
    pixel_coords = rot @ pts + tr    # [n, 3]     
    pixel_coords = torch.cat([pixel_coords[:, 0, :], -pixel_coords[:, 1, :], -pixel_coords[:, 2, :]], dim = -1)  
    out = pixel_coords[:, 2]
    out = out.cpu().numpy()
    if not gt_mask is None:
        out = out[gt_mask.astype(bool)][:, np.newaxis]
        gt_depth  = gt_depth[gt_mask.astype(bool)][:, np.newaxis]
        
    else:
        out = out[:, np.newaxis]
        gt_depth  = gt_depth[:, np.newaxis]
    _, Z, T, b, c = compute_similarity_transform(gt_depth,out,compute_optimal_scale=True)
    out = (b*out.dot(T))+c
    if not gt_mask is None:
        if sum(gt_mask) > 0:
            err = sum(abs(out.reshape(-1) - gt_depth.reshape(-1)))/sum(gt_mask)
        else:
            err = -1.
    else:
        err = sum(abs(out.reshape(-1) - gt_depth.reshape(-1))) / gt_depth.reshape(-1).shape[0]
    return err

def eval_predicted_depth2(predicted_depth, gt_depth, gt_mask):  
    predicted_depth = predicted_depth.reshape(-1)  
    out = predicted_depth.cpu().numpy()
    out = 1.0 / out
    out = out[gt_mask.astype(bool)][:, np.newaxis]
    gt_depth  = gt_depth[gt_mask.astype(bool)][:, np.newaxis]
    _, Z, T, b, c = compute_similarity_transform(gt_depth,out,compute_optimal_scale=True)
    out = (b*out.dot(T))+c
    if sum(gt_mask) > 0:
        err = sum(abs(out.reshape(-1) - gt_depth.reshape(-1)))/sum(gt_mask)
    else:
        err = -1.
    return err