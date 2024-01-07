import os, sys
import numpy as np
import imageio
import json
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange
import math

# import cv2
from numpy.linalg import inv
import cv2 as cv
import skimage
import matplotlib.pyplot as plt

from run_nerf_helpers import *

from load_llff import load_llff_data, load_DTU_depth, load_ScanNet_data, load_ScanNet_depth
from load_deepvoxels import load_dv_data
from load_blender import load_blender_data
from load_LINEMOD import load_LINEMOD_data


from visualization import *
from torch.utils.tensorboard import SummaryWriter
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
###
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
DEBUG = False


def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn
    def ret(inputs):
        return torch.cat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
    return ret


def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024*64):
    """Prepares inputs and applies network 'fn'.
    """
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    embedded = embed_fn(inputs_flat)

    if viewdirs is not None:
        input_dirs = viewdirs[:,None].expand(inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat([embedded, embedded_dirs], -1)

    outputs_flat = batchify(fn, netchunk)(embedded)
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs


def batchify_rays(rays_flat, chunk=1024*32, **kwargs):
    """Render rays in smaller minibatches to avoid OOM.
    """
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(rays_flat[i:i+chunk], **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k : torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret


def render(H, W, K, chunk=1024*32, rays=None, c2w=None, ndc=True,
                  near=0., far=1.,
                  use_viewdirs=False, c2w_staticcam=None,
                  **kwargs):
    """Render rays
    Args:
      H: int. Height of image in pixels.
      W: int. Width of image in pixels.
      focal: float. Focal length of pinhole camera.
      chunk: int. Maximum number of rays to process simultaneously. Used to
        control maximum memory usage. Does not affect final results.
      rays: array of shape [2, batch_size, 3]. Ray origin and direction for
        each example in batch.
      c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
      ndc: bool. If True, represent ray origin, direction in NDC coordinates.
      near: float or array of shape [batch_size]. Nearest distance for a ray.
      far: float or array of shape [batch_size]. Farthest distance for a ray.
      use_viewdirs: bool. If True, use viewing direction of a point in space in model.
      c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for 
       camera while using other c2w argument for viewing directions.
    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.
      disp_map: [batch_size]. Disparity map. Inverse of depth.
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      extras: dict with everything returned by render_rays().
    """
    if c2w is not None:
        # special case to render full image
        rays_o, rays_d = get_rays(H, W, K, c2w)
    else:
        # use provided ray batch
        rays_o, rays_d = rays

    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        if c2w_staticcam is not None:
            # special case to visualize effect of viewdirs
            rays_o, rays_d = get_rays(H, W, K, c2w_staticcam)
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1,3]).float()

    sh = rays_d.shape # [..., 3]
    if ndc:
        unprojected_rays_o = rays_o 
        unprojected_rays_d = rays_d
        # for forward facing scenes
        rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1,3]).float()
    rays_d = torch.reshape(rays_d, [-1,3]).float()

    near, far = near * torch.ones_like(rays_d[...,:1]), far * torch.ones_like(rays_d[...,:1])
    rays = torch.cat([rays_o, rays_d, near, far], -1)
    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)

    # Render and reshape
    all_ret = batchify_rays(rays, chunk, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map', 'disp_map', 'acc_map', 'pt_map', 'disp_mask'] 
    ret_list = [all_ret[k] for k in k_extract]
    
    if ndc:         
        unprojected_pt_map = inverse_ndc_rays(H, W, K[0][0], all_ret['pt_map'])
        # oz = unprojected_rays_o[..., 2]
        # dz = unprojected_rays_d[..., 2]        
        # t = all_ret['disp_map'] * oz
        # t = t / (dz - all_ret['disp_map'] * dz)
        # unprojected_pt_map = unprojected_rays_o + t[...,:,None] * unprojected_rays_d
        ret_list.append(unprojected_pt_map)
        
        
        # assert not torch.any(torch.isnan(unprojected_pt_map))
        unprojected_pt_map = inverse_ndc_rays(H, W, K[0][0], all_ret['pt0'])
        # t = all_ret['disp0'] * oz
        # t = t / (dz - all_ret['disp0'] * dz)
        # unprojected_pt_map = unprojected_rays_o + t[...,:,None] * unprojected_rays_d
        all_ret['unprojected_pt0'] = unprojected_pt_map
        # assert not torch.any(torch.isnan(unprojected_pt_map))
        
    else:
        ret_list.append(all_ret['pt_map'])
        all_ret['unprojected_pt0'] = all_ret['pt0']
    
    ret_dict = {k : all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]


def render_path(render_poses, hwf, K, chunk, render_kwargs, gt_imgs=None, savedir=None, render_factor=0):

    H, W, focal = hwf

    if render_factor!=0:
        # Render downsampled for speed
        H = H//render_factor
        W = W//render_factor
        focal = focal/render_factor

    rgbs = []
    disps = []

    t = time.time()
    for i, c2w in enumerate(tqdm(render_poses)):
        print(i, time.time() - t)
        t = time.time()
        rgb, disp, acc, pt, disp_mask, unprojected_pt, _ = render(H, W, K, chunk=chunk, c2w=c2w[:3,:4], **render_kwargs)
        rgbs.append(rgb.cpu().numpy())
        disps.append(disp.cpu().numpy())
        if i==0:
            print(rgb.shape, disp.shape)

        """
        if gt_imgs is not None and render_factor==0:
            p = -10. * np.log10(np.mean(np.square(rgb.cpu().numpy() - gt_imgs[i])))
            print(p)
        """

        if savedir is not None:
            rgb8 = to8b(rgbs[-1])
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)



    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)

    return rgbs, disps


def create_nerf(args):
    """Instantiate NeRF's MLP model.
    """
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed)

    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed)
    output_ch = 5 if args.N_importance > 0 else 4
    skips = [4]
    model = NeRF(D=args.netdepth, W=args.netwidth,
                 input_ch=input_ch, output_ch=output_ch, skips=skips,
                 input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
    grad_vars = list(model.parameters())

    model_fine = None
    if args.N_importance > 0:
        model_fine = NeRF(D=args.netdepth_fine, W=args.netwidth_fine,
                          input_ch=input_ch, output_ch=output_ch, skips=skips,
                          input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
        grad_vars += list(model_fine.parameters())

    network_query_fn = lambda inputs, viewdirs, network_fn : run_network(inputs, viewdirs, network_fn,
                                                                embed_fn=embed_fn,
                                                                embeddirs_fn=embeddirs_fn,
                                                                netchunk=args.netchunk)

    # Create optimizer
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    start = 0
    basedir = args.basedir
    expname = args.expname

    ##########################

    # Load checkpoints
    if args.ft_path is not None and args.ft_path!='None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if 'tar' in f]

    print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)
        
        # Load model
        model.load_state_dict(ckpt['network_fn_state_dict'])
        if model_fine is not None:
            model_fine.load_state_dict(ckpt['network_fine_state_dict'])

        start = ckpt['global_step']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        

    ##########################

    render_kwargs_train = {
        'network_query_fn' : network_query_fn,
        'perturb' : args.perturb,
        'N_importance' : args.N_importance,
        'network_fine' : model_fine,
        'N_samples' : args.N_samples,
        'network_fn' : model,
        'use_viewdirs' : args.use_viewdirs,
        'white_bkgd' : args.white_bkgd,
        'raw_noise_std' : args.raw_noise_std,
    }

    # NDC only good for LLFF-style forward facing data
    if args.no_ndc:
        print('Not ndc!')
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp

    render_kwargs_test = {k : render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer


def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)

    dists = z_vals[...,1:] - z_vals[...,:-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape)], -1)  # [N_rays, N_samples]

    dists = dists * torch.norm(rays_d[...,None,:], dim=-1)

    rgb = torch.sigmoid(raw[...,:3])  # [N_rays, N_samples, 3]
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[...,3].shape) * raw_noise_std

        # Overwrite randomly sampled data if pytest
        if pytest:
            np.random.seed(0)
            noise = np.random.rand(*list(raw[...,3].shape)) * raw_noise_std
            noise = torch.Tensor(noise)

    alpha = raw2alpha(raw[...,3] + noise, dists)  # [N_rays, N_samples]
    # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]

    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1)) 
    disp_mask = torch.ones_like(disp_map)

    acc_map = torch.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1.-acc_map[...,None])

    return rgb_map, disp_map, acc_map, weights, depth_map, disp_mask


def render_rays(ray_batch,
                network_fn,
                network_query_fn,
                N_samples,
                retraw=False,
                lindisp=False,
                perturb=0.,
                N_importance=0,
                network_fine=None,
                white_bkgd=False,
                raw_noise_std=0.,
                verbose=False,
                pytest=False):
    """Volumetric rendering.
    Args:
      ray_batch: array of shape [batch_size, ...]. All information necessary
        for sampling along a ray, including: ray origin, ray direction, min
        dist, max dist, and unit-magnitude viewing direction.
      network_fn: function. Model for predicting RGB and density at each point
        in space.
      network_query_fn: function used for passing queries to network_fn.
      N_samples: int. Number of different times to sample along each ray.
      retraw: bool. If True, include model's raw, unprocessed predictions.
      lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
      perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
        random points in time.
      N_importance: int. Number of additional times to sample along each ray.
        These samples are only passed to network_fine.
      network_fine: "fine" network with same spec as network_fn.
      white_bkgd: bool. If True, assume a white background.
      raw_noise_std: ...
      verbose: bool. If True, print more debugging info.
    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
      disp_map: [num_rays]. Disparity map. 1 / depth.
      acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
      raw: [num_rays, num_samples, 4]. Raw predictions from model.
      rgb0: See rgb_map. Output for coarse model.
      disp0: See disp_map. Output for coarse model.
      acc0: See acc_map. Output for coarse model.
      z_std: [num_rays]. Standard deviation of distances along ray for each
        sample.
    """
    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:,0:3], ray_batch[:,3:6] # [N_rays, 3] each
    viewdirs = ray_batch[:,-3:] if ray_batch.shape[-1] > 8 else None
    bounds = torch.reshape(ray_batch[...,6:8], [-1,1,2])
    near, far = bounds[...,0], bounds[...,1] # [-1,1]

    t_vals = torch.linspace(0., 1., steps=N_samples)
    if not lindisp:
        z_vals = near * (1.-t_vals) + far * (t_vals)
    else:
        z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

    z_vals = z_vals.expand([N_rays, N_samples])

    if perturb > 0.:
        # get intervals between samples
        mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        upper = torch.cat([mids, z_vals[...,-1:]], -1)
        lower = torch.cat([z_vals[...,:1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape)

        # Pytest, overwrite u with numpy's fixed random numbers
        if pytest:
            np.random.seed(0)
            t_rand = np.random.rand(*list(z_vals.shape))
            t_rand = torch.Tensor(t_rand)

        z_vals = lower + (upper - lower) * t_rand

    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]


#     raw = run_network(pts)
    raw = network_query_fn(pts, viewdirs, network_fn)
    rgb_map, disp_map, acc_map, weights, depth_map, disp_mask = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)
    pt_map = rays_o + rays_d * depth_map[...,:,None]  
    if N_importance > 0:

        rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map
        pt_map_0 = pt_map  
        disp_mask_0 = disp_mask 

        z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        z_samples = sample_pdf(z_vals_mid, weights[...,1:-1], N_importance, det=(perturb==0.), pytest=pytest)
        z_samples = z_samples.detach()

        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples + N_importance, 3]

        run_fn = network_fn if network_fine is None else network_fine
#         raw = run_network(pts, fn=run_fn)
        raw = network_query_fn(pts, viewdirs, run_fn)

        rgb_map, disp_map, acc_map, weights, depth_map, disp_mask = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)
        pt_map = rays_o + rays_d * depth_map[...,:,None]  

    ret = {'rgb_map' : rgb_map, 'disp_map' : disp_map, 'acc_map' : acc_map, 'pt_map' : pt_map, 'disp_mask' : disp_mask}
    if retraw:
        ret['raw'] = raw
    if N_importance > 0:
        ret['rgb0'] = rgb_map_0
        ret['disp0'] = disp_map_0
        ret['acc0'] = acc_map_0
        ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]
        ret['pt0'] = pt_map_0 
        ret['disp_mask_0'] = disp_mask_0 

    for k in ret:
        if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
            print(f"! [Numerical Error] {k} contains nan or inf.")

    return ret

def get_spare_train(i_train, sparse_input_num):
    med_idx = int(len(i_train)/2)
    start_no = med_idx - int(sparse_input_num/2)
    if start_no < 0:
        start_no = 0
    if start_no +  sparse_input_num >=  len(i_train):
        start_no = len(i_train) - sparse_input_num
    end_no = start_no + sparse_input_num
    sparse_train = []
    for i in range(start_no, end_no):        
        sparse_train.append(i_train[i])
    return sparse_train


def config_parser():

    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, 
                        help='config file path')
    parser.add_argument("--expname", type=str, default='leaves_test', 
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/', 
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='../NoExtNeRF/data/nerf_llff_data/leaves', 
                        help='input data directory')
    parser.add_argument("--siftdir", type=str, default='../NoExtNeRF/data/nerf_llff_data/leaves/new_sift_correspondences/', 
                        help='where to store ckpts and logs')   

    # training options
    parser.add_argument("--netdepth", type=int, default=8, 
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256, 
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=8, 
                        help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256, 
                        help='channels per layer in fine network')    
    parser.add_argument("--N_rand", type=int, default=32*32*4, 
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float, default=5e-4, 
                        help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=250, 
                        help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--chunk", type=int, default=1024*32, 
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024*64, 
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_batching", action='store_true', 
                        help='only take random rays from 1 image at a time')
    parser.add_argument("--no_reload", action='store_true', 
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None, 
                        help='specific weights npy file to reload for coarse network')
    
    
    parser.add_argument("--Patch_H", type=int, default=32, help='')  # 96, 80
    parser.add_argument("--Patch_W", type=int, default=32, help='') 
    parser.add_argument('--interpolation_render', type=bool, default=True) 
    parser.add_argument("--sift_threshold_for_sample", type=int, default=20, 
                        help='sift_threshold_for_sample') 
    parser.add_argument("--render_pixel_num", type=int, default=2048, # 1024
                        help='')
    parser.add_argument("--epolar_points_range", type=int, default=5)  
    parser.add_argument('--use_sparse_input', type=bool, default=True)
    parser.add_argument("--sparse_input_num", type=int, default=2)
    parser.add_argument("--epiline_threshold", type=int, default=2) 
    parser.add_argument('--use_predicted_pts', type=bool, default=False)
    parser.add_argument("--image_weight", type=float, default=1.0, 
                        help='image weight in the loss')
    parser.add_argument("--color_weight", type=float, default=0.001, 
                        help='color_weight in the loss')
    parser.add_argument("--ssim_weight", type=float, default=0.008, 
                        help='ssim_weight in the loss')
    parser.add_argument("--sift_weight", type=float, default=1.0, 
                        help='sift weight in the loss')
    parser.add_argument("--epolar_weight", type=float, default=0.0001, 
                        help='epipolar weight in the loss')    
    parser.add_argument("--smooth_weight", type=float, default=0.01, 
                        help='smooth weight in the loss') 
    parser.add_argument("--depth_weight", type=float, default=1.0,  
                        help='depth weight in the loss') 
    parser.add_argument("--max_sift_num", type=int, default=50)  


    # rendering options
    parser.add_argument("--N_samples", type=int, default=64, 
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true', 
                        help='use full 5D input instead of 3D')
    parser.add_argument("--i_embed", type=int, default=0, 
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10, 
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4, 
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0., 
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')

    parser.add_argument("--render_only", action='store_true', 
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true', 
                        help='render the test set instead of render_poses path')
    parser.add_argument("--render_factor", type=int, default=0, 
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')

    # training options
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops') 

    # dataset options
    parser.add_argument("--dataset_type", type=str, default='llff', 
                        help='options: llff / blender / deepvoxels')
    parser.add_argument("--testskip", type=int, default=8, 
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')

    ## deepvoxels flags
    parser.add_argument("--shape", type=str, default='greek', 
                        help='options : armchair / cube / greek / vase')

    ## blender flags
    parser.add_argument("--white_bkgd", action='store_true', 
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--half_res", action='store_true', 
                        help='load blender synthetic data at 400x400 instead of 800x800')

    ## llff flags
    parser.add_argument("--factor", type=int, default=8, 
                        help='downsample factor for LLFF images')#scn 9.9
    parser.add_argument("--no_ndc", action='store_true', 
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument("--lindisp", action='store_true', 
                        help='sampling linearly in disparity rather than depth')
    parser.add_argument("--spherify", action='store_true', 
                        help='set for spherical 360 scenes')
    parser.add_argument("--llffhold", type=int, default=8, 
                        help='will take every 1/N images as LLFF test set, paper uses 8')

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=100, 
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_img",     type=int, default=500, 
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=25000, 
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=20000, 
                        help='frequency of testset saving')
    parser.add_argument("--i_video",   type=int, default=50000, 
                        help='frequency of render_poses video saving')
    parser.add_argument("--i_eval", type=int, default=2000,
                        help='frequency of testset measuring') 
    parser.add_argument('--compute_extra_metrics', type=bool, default=True)

    return parser

def get_sample_dist(h, w):
    dist_list = []
    center_h = h / 2.0
    center_w = w / 2.0
    dist_max = 0
    for i in range(h):
        for j in range(w):
            if abs(i - center_h) < center_h * 0.7 and abs(j - center_w) < center_w * 0.7:
                dist = math.sqrt(math.pow(center_h *0.7, 2) + pow(center_w *0.7, 2)) 
            else:
                dist = math.sqrt(math.pow(i - center_h, 2) + pow(j - center_w, 2))
            dist_list.append([dist, i, j])
            if (dist > dist_max):
                dist_max = dist
            
    dist_list.sort(key=lambda x:x[0])
    dist_max = int(dist_max)
    return dist_list, dist_max

def nonuniform_sample(dist_list, dist_max):
    '''
    功能: 非均匀采样，根据其离中心的距离作为采样概率
    '''    
    if dist_max <=0:
        return [0,0]  
    selected_inds = np.random.choice(dist_max, size=[1])
    selected_coords = []
    for i in range(len(dist_list)-1,-1,-1):  
        if (dist_list[i][0] > selected_inds[0]):
            selected_coords.append(dist_list[i])
            
    final_selected_inds = np.random.choice(len(selected_coords), size=[1]) 
    x = selected_coords[final_selected_inds[0]][2]
    y = selected_coords[final_selected_inds[0]][1]
    return [x, y]  # x, y    
    


def get_sampled_coords( patch_h, patch_w, H, W):

    sampled_coords = []                 

    sample_x = np.random.choice(W-patch_w, size=[1])
    sample_y = np.random.choice(H-patch_h, size=[1])
    total_num = (W-patch_w) * (H-patch_h) 
    probs = []
    
    for i in range(patch_h):
        for j in range(patch_w):
            sampled_coords.append([i, j])
            total_num = total_num + max(patch_h-i, patch_w-j)
    for i in range(H-2*patch_h, H-patch_h):
        for j in range(W-2*patch_w, W-patch_w):
            sampled_coords.append([i, j])
            total_num = total_num + max(i + 2 * patch_h - H + 1, j + 2 * patch_w - W + 1)
    
       
    for i in range(patch_h):
        for j in range(patch_w):
            temp = max(patch_h-i, patch_w-j) / total_num
            probs.append(temp)            
    for i in range(H-2*patch_h, H-patch_h):
        for j in range(W-2*patch_w, W-patch_w):
            temp = max(i + 2 * patch_h - H + 1, j + 2 * patch_w - W + 1) / total_num
            probs.append(temp)
            
    sampled_coords.append([sample_y[0], sample_x[0]])
    temp = (W-patch_w) * (H-patch_h) / total_num
    probs.append(temp)
    return sampled_coords, probs


def get_sample_rect( patch_h, patch_w, mask_rect, coords, probs):
      
    final_selected_inds = np.random.choice(len(coords), size=[1], p=probs) 
    sample_x = coords[final_selected_inds[0]][1]
    sample_y = coords[final_selected_inds[0]][0]
    sampled_coords = []
    for i in range(patch_h):
        for j in range(patch_w):
            sampled_coords.append([sample_y + i, sample_x + j])
    sampled_coords = np.array(sampled_coords)
    sampled_coords = torch.Tensor(sampled_coords).to(device) # [rect_w*rect_w, 2]
    if (sample_x >= mask_rect[0]) and (sample_y >= mask_rect[1]): 
        ssim_mask = True
    else:
        ssim_mask = False
    if (mask_rect[0] + mask_rect[2] >= sample_x + patch_w) and (mask_rect[1] + mask_rect[3] >= sample_y + patch_h): 
        ssim_mask = ssim_mask and True
    else:
        ssim_mask = False
            
    return sampled_coords, ssim_mask


def calc_F(k0, r0, t0, k1, r1, t1):
    rr = r1 @ r0.T
    tt = (t1 - rr @ t0)
    st = np.float32([[0, -tt[2], tt[1]], [tt[2], 0, -tt[0]], [-tt[1], tt[0], 0]])
    F = inv(k1).T @ st @ rr @ inv(k0)
    return F

def cal_epolar_points(pos0, pos1, K, pts1, img0, img1, args, sample_range = 100): #):
    pos0 = pos0.cpu().numpy()
    pos1 = pos1.cpu().numpy()
    # K = K.cpu().numpy()
    pts1 = pts1.cpu().numpy()
    img0 = img0.cpu().numpy()
    img1 = img1.cpu().numpy()
    
    r0, t0 = pos0[:,:3], pos0[:,-1:]  
    r1, t1 = pos1[:,:3], pos1[:,-1:]
    F = calc_F(K, r0, t0, K, r1, t1)
    lines = cv.computeCorrespondEpilines(pts1.reshape(-1,1,2), 2,F)
    lines = lines.reshape(-1,3)
    r,c,_ = img0.shape
    H, W = r,c     

    threshold = args.epiline_threshold
    sample_range = int(W/5)   
    
    epolar_points = []
    notempty_epolar_points = []
    for r,pt1 in zip(lines,pts1):
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        if y0 < 0:
            x0,y0 = map(int, [-r[2]/r[0], 0])    
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        if y1 < 0:
            x1,y1 = map(int, [-r[2]/r[0], 0])
        cc, rr = skimage.draw.line(y0,x0,y1, x1) 
        points = []
        if y0 >= 0 and y1 >= 0:
            for i in range(len(cc)):
                if (rr[i] + sample_range >= pt1[0]) or (rr[i] <= pt1[0] + sample_range):
                    if cc[i] >= H or cc[i] < 0 or rr[i]>=W or rr[i]<0:
                        continue
                    if math.sqrt( (img1[int(pt1[1]), int(pt1[0]), 0] - img0[cc[i], rr[i], 0]) **2 +(img1[int(pt1[1]), int(pt1[0]), 1] - img0[cc[i], rr[i], 1]) **2+ (img1[int(pt1[1]), int(pt1[0]), 2] - img0[cc[i], rr[i], 2]) **2 ) < threshold:
                        points.append([rr[i],cc[i]])                    
            
    
        epolar_points.append(points)
        if len(points) > 0:
            notempty_epolar_points.append(points)
        
    return  epolar_points, notempty_epolar_points


def train():

    parser = config_parser()
    args = parser.parse_args()

    mask_rect = load_mask_rect(args.datadir + '/new_mask/mask_{:0>2d}_rect.txt'.format(int(args.factor)) ) # dict, [x,y, w,h]
    
    # Load data
    K = None
    if args.dataset_type == 'llff' or args.dataset_type == 'DTU' or args.dataset_type == 'ScanNet':
        images, poses, bds, render_poses, i_test = load_llff_data(args.datadir, args.factor,
                                                                recenter=True, bd_factor=.75,
                                                                spherify=args.spherify)
        if args.dataset_type == 'DTU':
            gt_depth, gt_mask = load_DTU_depth(args.datadir, args.factor)
            gt_depth = gt_depth.transpose(2, 0, 1)
            gt_mask = gt_mask.transpose(2, 0, 1)
        if args.dataset_type == 'ScanNet':
            gt_depth = load_ScanNet_depth(args.datadir, args.factor)
            
        hwf = poses[0,:3,-1]
        poses = poses[:,:3,:4]
        print('Loaded llff', images.shape, render_poses.shape, hwf, args.datadir)
        if not isinstance(i_test, list):
            i_test = [i_test]

        if args.llffhold > 0:
            print('Auto LLFF holdout,', args.llffhold)
            i_test = np.arange(images.shape[0])[::args.llffhold]

        i_val = i_test
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                        (i not in i_test and i not in i_val)])        

       

        print('DEFINING BOUNDS')
        if args.no_ndc:
            near = np.ndarray.min(bds) * .9
            far = np.ndarray.max(bds) * 1.
            
        else:
            near = 0.
            far = 1.
        print('NEAR FAR', near, far)

    elif args.dataset_type == 'blender':
        images, poses, render_poses, hwf, i_split = load_blender_data(args.datadir, args.half_res, args.testskip)
        print('Loaded blender', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        near = 2.
        far = 6.

        if args.white_bkgd:
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        else:
            images = images[...,:3]

    elif args.dataset_type == 'LINEMOD':
        images, poses, render_poses, hwf, K, i_split, near, far = load_LINEMOD_data(args.datadir, args.half_res, args.testskip)
        print(f'Loaded LINEMOD, images shape: {images.shape}, hwf: {hwf}, K: {K}')
        print(f'[CHECK HERE] near: {near}, far: {far}.')
        i_train, i_val, i_test = i_split

        if args.white_bkgd:
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        else:
            images = images[...,:3]

    elif args.dataset_type == 'deepvoxels':

        images, poses, render_poses, hwf, i_split = load_dv_data(scene=args.shape,
                                                                 basedir=args.datadir,
                                                                 testskip=args.testskip)

        print('Loaded deepvoxels', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        hemi_R = np.mean(np.linalg.norm(poses[:,:3,-1], axis=-1))
        near = hemi_R-1.
        far = hemi_R+1.

    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return

    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    if K is None:
        K = np.array([
            [focal, 0, 0.5*W],
            [0, focal, 0.5*H],
            [0, 0, 1]
        ])

    if args.render_test:
        render_poses = np.array(poses[i_test])

    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    # Create nerf model
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args)
    global_step = start

    bds_dict = {
        'near' : near,
        'far' : far,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    # Move testing data to GPU
    render_poses = torch.Tensor(render_poses).to(device)

    # Short circuit if only rendering out from trained modelrender_path
    if args.render_only:
        print('RENDER ONLY')
        with torch.no_grad():
            if args.render_test:
                # render_test switches to test poses
                images = images[i_test]
            else:
                # Default is smoother render_poses path
                images = None

            testsavedir = os.path.join(basedir, expname, 'renderonly_{}_{:06d}'.format('test' if args.render_test else 'path', start))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', render_poses.shape)

            rgbs, _ = render_path(render_poses, hwf, K, args.chunk, render_kwargs_test, gt_imgs=images, savedir=testsavedir, render_factor=args.render_factor)
            print('Done rendering', testsavedir)
            imageio.mimwrite(os.path.join(testsavedir, 'video.mp4'), to8b(rgbs), fps=30, quality=8)

            return

    # Prepare raybatch tensor if batching random rays
    # N_rand = args.N_rand
    if args.Patch_H > H:
        args.Patch_H = H
    if args.Patch_W > W:
        args.Patch_W = W  
    N_rand = args.Patch_H * args.Patch_W
      
    if args.epolar_points_range > N_rand:
        args.epolar_points_range = N_rand
        
    if args.use_sparse_input:
        i_sparse_train = get_spare_train(i_train, args.sparse_input_num)
        i_train = i_sparse_train
        
    # use_batching = not args.no_batching
    use_batching = False
    # if use_batching:
    # For random ray batching
    print('get rays')
    rays = np.stack([get_rays_np(H, W, K, p) for p in poses[:,:3,:4]], 0) # [N, ro+rd, H, W, 3]
    print('done, concats')
    rays_rgb = np.concatenate([rays, images[:,None]], 1) # [N, ro+rd+rgb, H, W, 3]
    rays_rgb = np.transpose(rays_rgb, [0,2,3,1,4]) # [N, H, W, ro+rd+rgb, 3]
    rays_rgb = np.stack([rays_rgb[i] for i in i_train], 0) # train images only
    rays_rgb = np.reshape(rays_rgb, [-1,3,3]) # [(N-1)*H*W, ro+rd+rgb, 3]
    rays_rgb = rays_rgb.astype(np.float32)
    print('rays_rgb_shape', rays_rgb.shape)
    print('shuffle rays')
    np.random.shuffle(rays_rgb)

    print('done')
    i_batch = 0
    
    # Move training data to GPU
    if use_batching:
        images = torch.Tensor(images).to(device)
    poses = torch.Tensor(poses).to(device)
    # if use_batching:
    rays_rgb = torch.Tensor(rays_rgb).to(device)

    ssim_fun = SSIM()
    ssim_fun = ssim_fun.to(device)
    
    N_iters = 300000 + 1
    print('Begin')
    print('TRAIN views are', i_train)
    print('TEST views are', i_test)
    print('VAL views are', i_val)

    tb_writer = SummaryWriter(log_dir=".//log") 

    # Summary writers
    # writer = SummaryWriter(os.path.join(basedir, 'summaries', expname))
    
    # dist_list, dist_max = get_sample_dist(H - args.Patch_H + 1, W -args.Patch_W + 1)
    sampled_coords, probs = get_sampled_coords(args.Patch_H, args.Patch_W, H, W)
    
    best_eval_psnr = 0
    best_ssim = 0
    start = start + 1
    
    # no_better_result_counter = 0 
    saved_ckpt_name = ""
    saved_disp_name = ""
    saved_testset_name = ""
    
    orginal_sift_weight = args.sift_weight
    orginal_epolar_weight = args.epolar_weight
    orginal_color_weight = args.color_weight
    orginal_ssim_weight = args.ssim_weight
    
    
        
    if args.use_predicted_pts:
        predicted_pts = {}
        i_predicted_train = i_train
            
        for i, v in enumerate(i_predicted_train):
            predicted_pts_file = 'predicted_pts_{:0>2d}_{:0>2d}.txt'.format(int(args.factor*2), v)
            predicted_pts_file = os.path.join(basedir, expname, predicted_pts_file)
            temp_predicted_pts = load_predicted_pts(predicted_pts_file)  # [h, w*3]
            temp_predicted_pts = np.reshape(temp_predicted_pts, [temp_predicted_pts.shape[0], -1, 3]) # [H, W, 3]
            predicted_pts['{:03d}'.format(v)] = temp_predicted_pts
    
    mid = (near + far)/2.0
    
    # for i in trange(start, N_iters):
    for i in range(start, N_iters): 
        time0 = time.time()
        
        # Random from two image
        sample_img = []
        sample_pose = []       
        valid_frame_inds = random.sample(list(i_train), 2)
        img_inds = []
        img_inds.append(valid_frame_inds[0])
        sample_pose.append(poses[img_inds[0], :3,:4]) 
        img_inds.append(valid_frame_inds[1])
        sample_pose.append(poses[img_inds[1], :3,:4])
        
        for j in img_inds:
            dst_img = images[j]
            dst_img = torch.Tensor(dst_img).to(device)
            sample_img.append(dst_img)    
        sample_pose_inv = get_pose_inverse( torch.stack(sample_pose, dim=0)) 
        # Sample random ray batch
        if use_batching:
            # Random over all images
            batch = rays_rgb[i_batch:i_batch+N_rand] # [B, 2+1, 3*?]
            batch = torch.transpose(batch, 0, 1)
            batch_rays, target_s = batch[:2], batch[2]

            i_batch += N_rand
            if i_batch >= rays_rgb.shape[0]:
                print("Shuffle data after an epoch!")
                rand_idx = torch.randperm(rays_rgb.shape[0])
                rays_rgb = rays_rgb[rand_idx]
                i_batch = 0

        else:

            img = sample_img[0] # [H,W,C]
            if args.interpolation_render:
                sample_mask_rect = mask_rect['{:03d}_{:03d}'.format(img_inds[0], img_inds[1])]                
                select_coords, ssim_mask = get_sample_rect(args.Patch_H, args.Patch_W, sample_mask_rect,sampled_coords, probs)
                offset_x = np.random.rand(select_coords.shape[0]) 
                offset_x = offset_x.reshape(select_coords.shape[0], 1)
                offset_y = np.random.rand(select_coords.shape[0]) 
                offset_y = offset_y.reshape(select_coords.shape[0], 1)
                offset = np.concatenate([offset_x, offset_y], axis=1) # [n,2]
                offset = torch.from_numpy(offset).to(device)
   
                select_coords = select_coords + offset  
                target_s = bilinear_sampler(img, select_coords, True)                       
                rays_o, rays_d = get_sift_rays(select_coords[:, 1], select_coords[:, 0], K, sample_pose[0])
                batch_rays = torch.stack([rays_o, rays_d], 0)
            else:          
                rays_o, rays_d = get_rays(H, W, K, sample_pose[0])    
                
                sample_mask_rect = mask_rect['{:03d}_{:03d}'.format(img_inds[0], img_inds[1])]              
                
                select_coords, ssim_mask = get_sample_rect(args.Patch_H, args.Patch_W, sample_mask_rect, sampled_coords, probs)
                select_coords = select_coords.long()
                rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                batch_rays = torch.stack([rays_o, rays_d], 0) # (2, N_rand, 3)
                target_s = img[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3) 
               
            sift_file = args.siftdir + 'sift_{:0>2d}_{:0>2d}_{:0>2d}.txt'.format(args.factor, img_inds[0], img_inds[1])                            
            sift_correspondences = load_sift_correspondences(sift_file) 
            sift_correspondences = np.array(sift_correspondences) 
            if len(sift_correspondences) > args.max_sift_num:                    
                select_inds = np.random.choice(len(sift_correspondences), size=[args.max_sift_num], replace=False)  # (N_rand,)
                sift_correspondences = sift_correspondences[select_inds]                   
            
                
            sift_correspondences = np.reshape(sift_correspondences, [-1, 2, 2]) # [n, 2, 2]
            n_sift = len(sift_correspondences)
            if n_sift > 0:
                sift_correspondences = np.transpose(sift_correspondences, [1,0,2]).astype(np.float32) # [2, n, 2]
                sift_correspondences = torch.Tensor(sift_correspondences).to(device)
                
                sift_ray_o = []
                sift_ray_d = []

                for j in range(2):
                    sift_point = sift_correspondences[j, :, :] # [n, 2]                    

                    rays_o, rays_d = get_sift_rays(sift_point[:, 0], sift_point[:, 1], K, sample_pose[j])
                    sift_ray_o.append(rays_o)
                    sift_ray_d.append(rays_d)

                sift_rays_o = torch.cat(sift_ray_o, 0) # (2*N_sift, 3)
                sift_rays_d = torch.cat(sift_ray_d, 0) # (2*N_sift, 3)
                batch_sift_rays = torch.stack([sift_rays_o, sift_rays_d], 0) # (2, 2*N_sift, 3)
                batch_rays = torch.cat([batch_rays, batch_sift_rays], dim=1) # [2, N_rand+2*N_sift, 3]
                
                for k in range(sift_correspondences.shape[0]):
                    sift_coords = sift_correspondences[k]
                    temp_target_s = bilinear_sampler(sample_img[k], sift_coords, False)
                    target_s = torch.cat([target_s, temp_target_s], dim=0)

            if args.epolar_weight > 0:
                rays_o, rays_d = get_rays(H, W, K, torch.Tensor(sample_pose[0]))  # (H, W, 3), (H, W, 3)
                coords = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H), torch.linspace(0, W-1, W)), -1)  # (H, W, 2)
                coords = torch.reshape(coords, [-1,2])  # (H * W, 2)
                select_inds = np.random.choice(coords.shape[0], size=[args.epolar_points_range], replace=False)  
                epolar_points = coords[select_inds].long()  # (args.epolar_points_range, 2)
                epolar_rays_o = rays_o[epolar_points[:, 0], epolar_points[:, 1]]  # (args.epolar_points_range, 3)
                epolar_rays_d = rays_d[epolar_points[:, 0], epolar_points[:, 1]]  # (args.epolar_points_range, 3)
                batch_epolar_rays = torch.stack([epolar_rays_o, epolar_rays_d], 0)
                batch_rays = torch.cat([batch_rays, batch_epolar_rays], dim=1) # [2, N_rand+2*N_sift+args.epolar_points_range, 3]
                temp_target_s = img[epolar_points[:, 0], epolar_points[:, 1]]  # (args.epolar_points_range, 3)
                target_s = torch.cat([target_s, temp_target_s], dim=0)
                
                epolar_points = torch.cat([epolar_points[:, 1].unsqueeze(-1), epolar_points[:, 0].unsqueeze(-1)], dim=-1)
                epolar_points, notempty_epolar_points = cal_epolar_points(sample_pose[1], sample_pose[0], K, epolar_points, sample_img[1], sample_img[0], args)                 
                epolar_points_index = []
                counter = N_rand + 2 * n_sift + args.epolar_points_range 
                epolar_points_index.append(counter)
                for k in range(len(epolar_points)):
                    counter = counter + len(epolar_points[k])
                    epolar_points_index.append(counter)
                
                epolar_points = np.array(notempty_epolar_points)

                if len(epolar_points) > 0:
                    epolar_points = np.concatenate(epolar_points)  
                    epolar_points = torch.Tensor(epolar_points).to(device)
                    rays_o, rays_d = get_sift_rays(epolar_points[:, 0], epolar_points[:, 1], K, sample_pose[1])
                    batch_epolar_rays = torch.stack([rays_o, rays_d], 0) # (2, epolar_points , 3)
                    batch_rays = torch.cat([batch_rays, batch_epolar_rays], dim=1) # [2, epolar_points + N_rand+2*N_sift, 3]

                    temp_target_s = bilinear_sampler(sample_img[1], epolar_points, False)
                    target_s = torch.cat([target_s, temp_target_s], dim=0)
             
            # Random over all images
            batch = rays_rgb[i_batch:i_batch+args.render_pixel_num] # [B, 2+1, 3*?]
            batch = torch.transpose(batch, 0, 1)
            temp_batch_rays, temp_target_s = batch[:2], batch[2]
            batch_rays = torch.cat([batch_rays, temp_batch_rays], dim=1)
            target_s = torch.cat([target_s, temp_target_s], dim=0)

            i_batch += args.render_pixel_num
            if i_batch >= rays_rgb.shape[0]:
                print("Shuffle data after an epoch!")
                rand_idx = torch.randperm(rays_rgb.shape[0])
                rays_rgb = rays_rgb[rand_idx]
                i_batch = 0
            
        #####  Core optimization loop  #####
        rgb, disp, acc, pt, disp_mask, unprojected_pt, extras = render(H, W, K, chunk=args.chunk, rays=batch_rays,
                                                verbose=i < 10, retraw=True,
                                                **render_kwargs_train)

        optimizer.zero_grad()
        img_loss = img2mse(rgb, target_s)
        trans = extras['raw'][...,-1]
        loss = args.image_weight * img_loss
        psnr = mse2psnr(img_loss)

        pixel_rgb = target_s[:N_rand,...]  # [N_rand, 3]
        
        if args.no_ndc:
            pixel_pts = pt[:N_rand,...] # [N_rand, 3]
        else:
            pixel_pts = unprojected_pt[:N_rand,...] 
   
        projected_pts = get_projected_pt(pixel_pts, sample_pose_inv[1], torch.Tensor(K).to(device))        
        
        mask_fname = os.path.join(args.datadir, 'new_mask/') + 'mask_{:0>2d}_{:0>2d}_{:0>2d}.png'.format(args.factor, img_inds[0], img_inds[1])
        mask_img = cv.imread(mask_fname) 
        mask_img = torch.Tensor(mask_img).to(device)
        mask_img = mask_img / 255
        mask_pixel = mask_img[select_coords[:, 0].long(), select_coords[:, 1].long()]
        pixel_rgb_rect = torch.reshape(pixel_rgb, [args.Patch_H, args.Patch_W, 3])  # [n, n, 3]
        b_consective = False
        if abs(img_inds[0] - img_inds[0])<=3:
            mask_pixel = torch.ones_like(pixel_rgb)     
            b_consective = True
        interplated_rgb = bilinear_sampler(sample_img[1], projected_pts, False)

        if torch.isnan(interplated_rgb).int().sum() == 0:
           
            color_loss = torch.mean(torch.abs(pixel_rgb - interplated_rgb) * mask_pixel )
            if color_loss >1.0e-10 and args.color_weight>0:
                loss = loss + args.color_weight * color_loss   
                
            if ssim_mask or b_consective:                
                interplated_rgb = torch.reshape(interplated_rgb, [args.Patch_H, args.Patch_W, 3])  # [ n, n, 3]
                ssim_loss = torch.mean(ssim_fun(pixel_rgb_rect.unsqueeze(0), interplated_rgb.unsqueeze(0)))
                if ssim_loss >1.0e-10 and args.ssim_weight>0:
                    loss = loss + args.ssim_weight * ssim_loss   
            else:
                ssim_loss = 0  
        else:
            color_loss = 0  
            ssim_loss = 0     

        img0_pts = pt[0: N_rand,...] # [N_rand, 3]
        img0_pts = torch.reshape(img0_pts, [args.Patch_H, args.Patch_W, 3]).unsqueeze(0) # [1, n, n, 3]
        img0_target = target_s[0: N_rand,...] # [N_rand, 3]
        img0_target = torch.reshape(img0_target, [args.Patch_H, args.Patch_W, 3]).unsqueeze(0) # [1, n, n, 3]
        smooth_loss = get_smooth_loss(img0_pts.permute(0, 3, 1, 2), img0_target.permute(0, 3, 1, 2))
        if smooth_loss >1.0e-10 and args.smooth_weight>0:
            loss = loss + args.smooth_weight * smooth_loss
            
   
        if args.use_predicted_pts:
            img_predicted_pts = predicted_pts['{:03d}'.format(img_inds[0])]
            img_predicted_pts = torch.Tensor(img_predicted_pts).to(device)
            predicted_select_coords = torch.Tensor(select_coords).to(device)
            predicted_select_coords = predicted_select_coords / 2.0 
            interplated_pts = bilinear_sampler(img_predicted_pts,  predicted_select_coords, True)  
            if torch.isnan(interplated_pts).int().sum() == 0:
                depth_loss = torch.mean(torch.abs(pt[:N_rand,...] - interplated_pts)) 
                if depth_loss >1.0e-10:
                    loss = loss + args.depth_weight * depth_loss 
            else:
                depth_loss = 0                 
        

        if n_sift > 0:
            sift_pts = pt[N_rand: N_rand + 2*n_sift,...] # [2*N_sift, 3]
            sift_pts = torch.reshape(sift_pts, [2, n_sift, 3])  
            sift_loss = epipolar_constraint(sift_pts)
            if sift_loss >1.0e-10:
                loss = loss + args.sift_weight * sift_loss 
        else:
            sift_loss = 0
        if args.epolar_weight > 0:   
            epolar_loss = 0
            epolar_pts = pt[N_rand + 2*n_sift: N_rand + 2*n_sift + args.epolar_points_range,...] 
            for k in range(epolar_pts.shape[0]):
                if epolar_points_index[k] != epolar_points_index[k+1]: 
                    mapped_epolar_pts = pt[epolar_points_index[k]: epolar_points_index[k+1],...]
                    epolar_loss = epolar_loss +  cal_epolar_loss(epolar_pts[k], mapped_epolar_pts) 
            if epolar_loss >1.0e-10:
                loss = loss + args.epolar_weight * epolar_loss  
   

        if 'rgb0' in extras:
            img_loss0 = img2mse(extras['rgb0'], target_s)
            if img_loss0 >1.0e-10 and args.image_weight>0:          
                loss = loss + args.image_weight * img_loss0 
            psnr0 = mse2psnr(img_loss0)
            if args.no_ndc:
                pixel_pts0 = extras['pt0'][:N_rand,...] # [N_rand, 3]
            else:
                pixel_pts0 = extras['unprojected_pt0'][:N_rand,...]
            # pixel_pts0 = extras['pt0'][:N_rand,...]
            pixel_mask0 = extras['disp_mask_0'][:N_rand,...] 
            projected_pts0 = get_projected_pt(pixel_pts0, sample_pose_inv[1], torch.Tensor(K).to(device))     
                                      
            interplated_rgb0 = bilinear_sampler(sample_img[1], projected_pts0, False)
            
            if torch.isnan(interplated_rgb0).int().sum() == 0:
                color_loss0 = torch.mean(torch.abs(pixel_rgb - interplated_rgb0) * mask_pixel)                
                if color_loss0 >1.0e-10 and args.color_weight>0:
                    loss = loss + args.color_weight * color_loss0  
                
                
                if ssim_mask or b_consective:                    
                    interplated_rgb0 = torch.reshape(interplated_rgb0, [args.Patch_H, args.Patch_W, 3])  # [n, n, 3]
                    ssim_loss0 = torch.mean(ssim_fun(pixel_rgb_rect.unsqueeze(0), interplated_rgb0.unsqueeze(0)))
                    if ssim_loss0 >1.0e-10 and args.ssim_weight>0:
                        loss = loss + args.ssim_weight * ssim_loss0 
                else:
                    ssim_loss0 = 0 
            else:
                color_loss0 = 0
                ssim_loss0 = 0                 
            
            img0_pts0 = torch.reshape(extras['pt0'][:N_rand,...], [args.Patch_H, args.Patch_W, 3]).unsqueeze(0) # [1, n, n, 3]
            img0_target = target_s[0: N_rand,...] # [N_rand, 3]
            img0_target = torch.reshape(img0_target, [args.Patch_H, args.Patch_W, 3]).unsqueeze(0) # [1, n, n, 3]
            smooth_loss0 = get_smooth_loss(img0_pts0.permute(0, 3, 1, 2), img0_target.permute(0, 3, 1, 2))
            if smooth_loss0 >1.0e-10 and args.smooth_weight>0: 
                loss = loss + args.smooth_weight * smooth_loss0                
 
            if args.use_predicted_pts:  
                if torch.isnan(interplated_pts).int().sum() == 0:              
                    depth_loss0 = torch.mean(torch.abs(extras['pt0'][:N_rand,...] - interplated_pts)) 
                    if depth_loss0 >1.0e-10:
                        loss = loss + args.depth_weight * depth_loss0  
                else:
                    depth_loss0 = 0 
               
            if n_sift > 0:
                sift_pts0 = extras['pt0'][N_rand: N_rand + 2*n_sift,...] # [2*N_sift, 3]
                sift_pts0 = torch.reshape(sift_pts0, [2, n_sift, 3])    
                sift_loss0 = epipolar_constraint(sift_pts0)
                if sift_loss0 >1.0e-10:
                    loss = loss + args.sift_weight * sift_loss0 
            else:
                sift_loss0 = 0
            if args.epolar_weight > 0:  
                epolar_loss0 = 0
                epolar_pts0 = extras['pt0'][N_rand + 2*n_sift: N_rand + 2*n_sift + args.epolar_points_range,...]
                for k in range(epolar_pts0.shape[0]):
                    if epolar_points_index[k] != epolar_points_index[k+1]:
                        mapped_epolar_pts0 = extras['pt0'][epolar_points_index[k]: epolar_points_index[k+1],...]
                        epolar_loss0 = epolar_loss0 +  cal_epolar_loss(epolar_pts0[k], mapped_epolar_pts0)
                if epolar_loss0 >1.0e-10: 
                    loss = loss + args.epolar_weight * epolar_loss0   
                
        loss.backward()
        optimizer.step()

        # NOTE: IMPORTANT!
        ###   update learning rate   ###
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate
        
        # only for 2 images input
        if args.sparse_input_num < 5:
            sift_decay_steps = args.lrate_decay * 500    
            args.sift_weight = orginal_sift_weight * (decay_rate ** (global_step / sift_decay_steps)) 
            args.epolar_weight = orginal_epolar_weight * (decay_rate ** (global_step / sift_decay_steps)) 
            
            args.color_weight = orginal_color_weight * (1.0 + global_step / N_iters * 50.0) 
            args.ssim_weight = orginal_ssim_weight * (1.0 + global_step / N_iters * 50.0) 
        ################################

        dt = time.time()-time0        

        if i%args.i_video==0 and i > 0:
            # Turn on testing mode
            with torch.no_grad():
                rgbs, disps = render_path(render_poses, hwf, K, args.chunk, render_kwargs_test)
            print('Done, saving', rgbs.shape, disps.shape)
            moviebase = os.path.join(basedir, expname, '{}_spiral_{:06d}_'.format(expname, i))
            imageio.mimwrite(moviebase + 'rgb.mp4', to8b(rgbs), fps=30, quality=8)
            imageio.mimwrite(moviebase + 'disp.mp4', to8b(disps / np.max(disps)), fps=30, quality=8)

        if i%args.i_testset==0 and i > 0:
            testsavedir = os.path.join(basedir, expname, 'testset_{:06d}'.format(i))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', poses[i_test].shape)
            with torch.no_grad():
                render_path(torch.Tensor(poses[i_test]).to(device), hwf, K, args.chunk, render_kwargs_test, gt_imgs=images[i_test], savedir=testsavedir)
            print('Saved test set')


    
        if i%args.i_print==0:            
            print("[TRAIN] Iter: {} Loss: {}  PSNR: {}".format(i, loss.item(), psnr.item()))
            tb_writer.add_scalar("loss",loss,global_step)
            tb_writer.add_scalar("psnr",psnr,global_step)            
            tb_writer.add_scalar("lr",optimizer.param_groups[0]['lr'],global_step)
            tb_writer.add_scalar("img_loss",img_loss,global_step)
            tb_writer.add_scalar("color_loss",color_loss,global_step)
            tb_writer.add_scalar("ssim_loss",ssim_loss,global_step)
            tb_writer.add_scalar("smooth_loss",smooth_loss,global_step)
            tb_writer.add_scalar("sift_loss",sift_loss,global_step)
            if args.epolar_weight > 0:
                tb_writer.add_scalar("epolar_loss",epolar_loss,global_step)
            tb_writer.add_scalar("sift_weight",args.sift_weight,global_step)
            tb_writer.add_scalar("color_weight",args.color_weight,global_step)
            if args.use_predicted_pts:
                tb_writer.add_scalar("depth_loss",depth_loss,global_step)
            

        if i%args.i_img==0:
            # Log a rendered validation view to Tensorboard
            img_i=np.random.choice(i_val)
            target = images[img_i]
            pose = poses[img_i, :3,:4]            
            
            with torch.no_grad():
                rgb, disp, acc, pt, disp_mask, unprojected_pt, extras = render(H, W, K, chunk=args.chunk, c2w=pose,
                                                    **render_kwargs_test)
            
            # psnr = mse2psnr(img2mse(rgb, target)) 
            temp = to8b(rgb.cpu().numpy())
            tb_writer.add_image('rgb', temp, dataformats='HWC')
            
            if args.dataset_type == 'DTU':
                temp = colorize(gt_depth[img_i])
                tb_writer.add_image('gt_depth', temp, dataformats='HWC')
                temp = colorize(gt_mask[img_i])
                tb_writer.add_image('gt_mask', temp, dataformats='HWC')
            
            if args.dataset_type == 'ScanNet':
                temp = colorize(gt_depth[img_i])
                tb_writer.add_image('gt_depth', temp, dataformats='HWC')
            
            disp = disp.cpu().numpy()
            disp = colorize(disp)
            if i%args.i_testset==0 and i > 0:
                for j in i_val:
                    disp_pose = poses[j, :3,:4]
                    with torch.no_grad():
                        disp_rgb, disp_disp, disp_acc, disp_pt, disp_mask, disp_unprojected_pt, disp_extras = render(H, W, K, chunk=args.chunk, c2w=disp_pose,
                                                            **render_kwargs_test)
                    disp_disp = disp_disp.cpu().numpy()
                    disp_disp = colorize(disp_disp)
                    dispsavedir = os.path.join(basedir, expname, 'disp_{:06d}'.format(i))
                    os.makedirs(dispsavedir, exist_ok=True)            
                    disp_disp = to8b(disp_disp)
                    filename = os.path.join(dispsavedir, '{:03d}.png'.format(j))
                    imageio.imwrite(filename, disp_disp)
            tb_writer.add_image('disp', disp.transpose(2, 0, 1))            

            tb_writer.add_image('acc', acc[np.newaxis,...])
            
            # tb_writer.add_scalar('psnr_holdout', psnr, global_step)
            tb_writer.add_image('rgb_holdout', target, dataformats='HWC')
            if args.N_importance > 0:
                temp = to8b(extras['rgb0'].cpu().numpy())                
                tb_writer.add_image('rgb0', temp, dataformats='HWC')
                tb_writer.add_image('disp0', extras['disp0'][np.newaxis,...])
                tb_writer.add_image('z_std', extras['z_std'][np.newaxis,...])    

        if i%args.i_eval==0:
            PSNRs = []
            ssims,l_alex,l_vgg, dep_error=[],[],[],[]
            for j, v in enumerate(i_val):
                target = images[v]
                pose = poses[v, :3,:4]
                with torch.no_grad():
                    rgb, disp, acc, pt, disp_mask, unprojected_pt, extras = render(H, W, K, chunk=args.chunk, c2w=pose,
                                                        **render_kwargs_test)
                
                psnr = mse2psnr(img2mse(rgb, torch.from_numpy(target).to(device)))
                PSNRs.append(psnr)
                if args.compute_extra_metrics:
                    """ this_type_str = type(rgb)
                    if this_type_str is not np.ndarray: """
                    rgb = np.array(rgb.cpu())
                    
                    ssim = rgb_ssim(rgb, target, 1)
                    l_a = rgb_lpips(target, rgb, 'alex', device)
                    l_v = rgb_lpips(target, rgb, 'vgg', device)
                    if args.dataset_type == 'DTU':
                        depth_error = eval_predicted_depth(unprojected_pt.reshape(-1, 3), sample_pose_inv[0], gt_depth[img_inds[0]].reshape(-1), gt_mask[img_inds[0]].reshape(-1))  
                        depth_error2 = eval_predicted_depth2(disp, gt_depth[img_inds[0]].reshape(-1), gt_mask[img_inds[0]].reshape(-1))  
                        dep_error.append(depth_error)
                    if args.dataset_type == 'ScanNet':
                        depth_threshold = np.full(gt_depth[img_inds[0]].shape, 1e-8) 
                        gt_mask = np.where(np.less_equal(depth_threshold, gt_depth[img_inds[0]]), np.ones_like(gt_depth[img_inds[0]]), np.zeros_like(gt_depth[img_inds[0]]) )
                        temp = gt_depth[img_inds[0]].reshape(-1)
                        depth_error = eval_predicted_depth(unprojected_pt.reshape(-1, 3), sample_pose_inv[0], gt_depth[img_inds[0]].reshape(-1), gt_mask.reshape(-1))                      
                        dep_error.append(depth_error)
                    ssims.append(ssim)
                    l_alex.append(l_a)
                    l_vgg.append(l_v)
            if PSNRs:
                eval_psnr = torch.stack(PSNRs, -1).mean()                
                if args.compute_extra_metrics:
                    eval_ssim = np.mean(np.asarray(ssims))
                    eval_l_a = np.mean(np.asarray(l_alex))
                    eval_l_v = np.mean(np.asarray(l_vgg))
                    if args.dataset_type == 'DTU' or args.dataset_type == 'ScanNet':
                        eval_dep_error = np.mean(np.asarray(dep_error))

                    if eval_psnr > best_eval_psnr :
                        
                        for j in i_val:
                            disp_pose = poses[j, :3,:4]
                            with torch.no_grad():
                                disp_rgb, disp_disp, disp_acc, disp_pt, disp_mask, disp_unprojected_pt, disp_extras = render(H, W, K, chunk=args.chunk, c2w=disp_pose,
                                                                    **render_kwargs_test)
                            disp_disp = disp_disp.cpu().numpy()
                            disp_disp = colorize(disp_disp)
                            dispsavedir = os.path.join(basedir, expname, 'disp_{:06d}'.format(i))
                            os.makedirs(dispsavedir, exist_ok=True)            
                            disp_disp = to8b(disp_disp)
                            filename = os.path.join(dispsavedir, '{:03d}.png'.format(j))
                            imageio.imwrite(filename, disp_disp)
                            
                            if args.dataset_type == 'DTU' or args.dataset_type == 'ScanNet':
                                dispsavedir = os.path.join(basedir, expname, 'gt_depth_{:06d}'.format(i))
                                os.makedirs(dispsavedir, exist_ok=True)            
                                gt_depth_j = colorize(gt_depth[j])
                                gt_depth_j = to8b(gt_depth_j)
                                filename = os.path.join(dispsavedir, '{:03d}.png'.format(j))
                                imageio.imwrite(filename, gt_depth_j)
                        
                        if os.path.exists(saved_disp_name):
                            del_file(saved_disp_name)
                        saved_disp_name = dispsavedir  
                        
                        testsavedir = os.path.join(basedir, expname, 'testset_{:06d}'.format(i))
                        os.makedirs(testsavedir, exist_ok=True)
                        print('test poses shape', poses[i_test].shape)
                        with torch.no_grad():
                            render_path(torch.Tensor(poses[i_test]).to(device), hwf, K, args.chunk, render_kwargs_test, gt_imgs=images[i_test], savedir=testsavedir)
                        print('Saved test set')
                        if os.path.exists(saved_testset_name):
                            del_file(saved_testset_name)
                        saved_testset_name = testsavedir 
                        
                        path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
                        torch.save({
                            'global_step': global_step,
                            'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                            'network_fine_state_dict': render_kwargs_train['network_fine'].state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                        }, path)
                        print('Saved checkpoints at', path) 
                        if os.path.exists(saved_ckpt_name):
                            os.remove(saved_ckpt_name)
                        saved_ckpt_name = path  
                        
                        
                        best_eval_psnr = eval_psnr 
                        
                        mean_name = os.path.join(basedir, expname, 'mean.txt')
                        with open(mean_name, "a") as f:
                            if args.dataset_type == 'DTU' or args.dataset_type == 'ScanNet':
                                np.savetxt(f, np.asarray([eval_psnr.cpu(), eval_ssim, eval_l_a, eval_l_v, eval_dep_error, i]), newline=' ', delimiter=',')
                            else:
                                np.savetxt(f, np.asarray([eval_psnr.cpu(), eval_ssim, eval_l_a, eval_l_v, i]), newline=' ', delimiter=',')
                            f.write("\n")
                        i_eval_train = i_train                        
                            
                        for l, v in enumerate(i_eval_train):
                            pose = poses[v, :3,:4]
                            with torch.no_grad():
                                rgb, disp, acc, pt, disp_mask, unprojected_pt, extras = render(H, W, K, chunk=args.chunk, c2w=pose,
                                                                    **render_kwargs_test)
                                this_type_str = type(pt)
                                if this_type_str is not np.ndarray:
                                    pt = np.array(pt.cpu())
                                saved_file = 'predicted_pts_{:0>2d}_{:0>2d}.txt'.format(int(args.factor), v)
                                saved_file = os.path.join(basedir, expname, saved_file)
                                with open(saved_file, 'w') as f:
                                    for i in range(pt.shape[0]):
                                        for j in range(pt.shape[1]):
                                            if j < pt.shape[1]-1:
                                                f.write('%f %f %f ' % (pt[i][j][0], pt[i][j][1], pt[i][j][2]))
                                            else:
                                                f.write('%f %f %f' % (pt[i][j][0], pt[i][j][1], pt[i][j][2]))
                                        f.write('\n')  
                                        
                    print("eval_psnr: {}".format(eval_psnr))
                    tb_writer.add_scalar('eval_psnr', eval_psnr, global_step)
       
        global_step += 1


if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    train()
