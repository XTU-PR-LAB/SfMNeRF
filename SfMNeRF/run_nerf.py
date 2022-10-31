# -*- coding: utf-8 -*-
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

import matplotlib.pyplot as plt

from run_nerf_helpers import *
from make_FMatrix_and_mask import *

from load_llff import load_llff_data
from load_deepvoxels import load_dv_data
from load_blender import load_blender_data
from load_LINEMOD import load_LINEMOD_data

from visualization import *

from torch.utils.tensorboard import SummaryWriter

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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
       rays_flat: [b, 3+3+1+1] o,d,near,far
    """
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(rays_flat[i:i+chunk], **kwargs)
        for k in ret: # k must be the key value
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k : torch.cat(all_ret[k], 0) for k in all_ret} 
    return all_ret

# ndc-Normalized Device Coordinate
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
        viewdirs = torch.reshape(viewdirs, [-1,3]).float() # [b, 3]

    sh = rays_d.shape # [..., 3]
    if ndc:
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
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:]) # list(sh[:-1])-batch_size, 
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    # k_extract = ['rgb_map', 'disp_map', 'acc_map']
    k_extract = ['rgb_map', 'disp_map', 'acc_map', 'pt_map', 'disp_mask'] 
    ret_list = [all_ret[k] for k in k_extract] # list
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
        rgb, disp, acc, pt, disp_mask, extras = render(H, W, K, chunk=chunk, c2w=c2w[:3,:4], **render_kwargs)  
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
    output_ch = 4  
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

        start = ckpt['global_step']
        
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 2000
        new_lrate = args.lrate * (decay_rate ** (start / decay_steps))
        optimizer.param_groups[0]['lr'] = new_lrate       

        # Load model
        model.load_state_dict(ckpt['network_fn_state_dict'])
        if model_fine is not None:
            model_fine.load_state_dict(ckpt['network_fine_state_dict'])     
    
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
    if args.dataset_type != 'llff' or args.no_ndc:
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
        raw: [num_rays, num_samples along ray, 4].[N_rays, N_samples, 3+1(rgb+alpha)] Prediction from model.
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

    alpha = raw2alpha(raw[...,3] + noise, dists)      
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    rgb_map = torch.sum(weights[...,None] * rgb, -2)  

    depth_map = torch.sum(weights * z_vals, -1) 
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.max(1e-10 * torch.ones(weights.shape[0]), torch.sum(weights, -1))) 
    depth_threshold = np.full(disp_map.shape, 1e-10)
    depth_threshold = torch.from_numpy(depth_threshold).float()
    disp_mask = torch.where(torch.lt(depth_threshold.cuda(), torch.sum(weights, -1)), torch.ones_like(disp_map, dtype=torch.float32), torch.zeros_like(disp_map, dtype=torch.float32) )
    acc_map = torch.sum(weights, -1) # [N_rays]

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
        z_vals = near * (1.-t_vals) + far * (t_vals) # 1.0 *near + t_vals * (far - near)
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
    raw = network_query_fn(pts, viewdirs, network_fn) # [N_rays, N_samples, 3+1(rgb+alpha)]
    rgb_map, disp_map, acc_map, weights, depth_map, disp_mask = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

    pt_map = rays_o + rays_d / disp_map[...,:,None] 
    

    if N_importance > 0: # refining procedure

        rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map
        pt_map_0 = pt_map  
        disp_mask_0 = disp_mask 

        z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        z_samples = sample_pdf(z_vals_mid, weights[...,1:-1], N_importance, det=(perturb==0.), pytest=pytest)
        z_samples = z_samples.detach() # detach-Returns a new Variable, detached from the current graph.

        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)  
        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples + N_importance, 3]

        run_fn = network_fn if network_fine is None else network_fine
#         raw = run_network(pts, fn=run_fn)
        raw = network_query_fn(pts, viewdirs, run_fn)  # [N_rays, N_samples, 3+1(rgb+alpha)]

        rgb_map, disp_map, acc_map, weights, depth_map, disp_mask = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)
        pt_map = rays_o + rays_d / disp_map[...,:,None]  
        
    
    
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
            print("! [Numerical Error]: contains nan or inf.")

    return ret

'''
    config_fern.txt
    factor = 8
    llffhold = 8

    N_rand = 1024
    N_samples = 64
    N_importance = 64

    use_viewdirs = True
    raw_noise_std = 1e0

'''
def config_parser():

    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, 
                        help='config file path')  
    parser.add_argument("--expname", type=str, default='orchids',  
                        help='experiment name')   
    parser.add_argument("--basedir", type=str, default='./logs/', 
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='../NoExtNeRF/data/nerf_llff_data/orchids', 
                        help='input data directory')  
    parser.add_argument("--ref_img_no", type=int, default=20,   
                        help="the index of refered image")  
    parser.add_argument("--siftdir", type=str, default='../NoExtNeRF/data/nerf_llff_data/orchids/orginal_sift_correspondences/', 
                        help='where to store ckpts and logs')   
    parser.add_argument('--pretrained-exppose', dest='pretrained_exp_pose', default=None, metavar='PATH',
                    help='path to pre-trained Exp Pose net model')
    
    # training options
    parser.add_argument("--netdepth", type=int, default=8, 
                        help='layers in network')    
    parser.add_argument("--netwidth", type=int, default=256, 
                        help='channels per layer')    
    parser.add_argument("--netdepth_fine", type=int, default=8, 
                        help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256, 
                        help='channels per layer in fine network')
    parser.add_argument("--N_rand", type=int, default=48*48,  
                        help='batch size (number of random rays per gradient step)') 
    parser.add_argument("--lrate", type=float, default=5e-4,  
                        help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=250,  
                        help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--chunk", type=int, default=1024*32,                         
                        help='number of rays processed in parallel, decrease if running out of memory')                        
    parser.add_argument("--netchunk", type=int, default=1024*64, 
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_batching", action='store_false', 
                        help='only take random rays from 1 image at a time')
    parser.add_argument("--no_reload", action='store_true',   
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None,   
                        help='specific weights npy file to reload for coarse network')
    parser.add_argument("--image_weight", type=float, default=1.0, 
                        help='image weight in the loss')
    parser.add_argument("--color_weight", type=float, default=0.001, 
                        help='color_weight in the loss')
    parser.add_argument("--ssim_weight", type=float, default=0.01, 
                        help='ssim_weight in the loss')
    parser.add_argument("--sift_weight", type=float, default=0.1, 
                        help='sift weight in the loss')   
    parser.add_argument("--smooth_weight", type=float, default=0.001,  
                        help='smooth weight in the loss')    
    parser.add_argument('--pose_lrate', '--learning-rate', default=5e-4, type=float, # 2e-4
                    metavar='LR', help='initial learning rate for pose net')
    parser.add_argument('--weight_decay', '--wd', default=0, type=float,
                    metavar='W', help='weight decay')
    parser.add_argument("--max_sift_num", type=int, default=20)  
    parser.add_argument('--interpolation_render', type=bool, default=True) 
    parser.add_argument("--sift_threshold_for_sample", type=int, default=20, 
                        help='sift_threshold_for_sample') 

    # rendering options
    parser.add_argument("--N_samples", type=int, default=64,                          
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=64,  
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_false', 
                        help='use full 5D input instead of 3D')
    parser.add_argument("--i_embed", type=int, default=0, 
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10, 
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4, 
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=1e0, 
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
                        help='downsample factor for LLFF images')
    parser.add_argument("--no_ndc", action='store_true', 
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument("--lindisp", action='store_true', 
                        help='sampling linearly in disparity rather than depth')
    parser.add_argument("--spherify", action='store_true', 
                        help='set for spherical 360 scenes')
    parser.add_argument("--llffhold", type=int, default=8, 
                        help='will take every 1/N images as LLFF test set, paper uses 8')

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=25, 
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_img",     type=int, default=500, #500
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=10000,  #10000
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=50000, 
                        help='frequency of testset saving')
    parser.add_argument("--i_video",   type=int, default=50000, 
                        help='frequency of render_poses video saving')
    parser.add_argument("--i_eval", type=int, default=5000,
                        help='frequency of testset measuring')
    parser.add_argument("--i_poses",     type=int, default=200, #500
                        help='frequency of saving predicted poses')
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

def nonuniform_sample(h, w, dist_list, dist_max):
    '''
    功能: 非均匀采样，根据其离中心的距离作为采样概率
    '''      
    selected_inds = np.random.choice(dist_max, size=[1])
    selected_coords = []
    for i in range(len(dist_list)-1,-1,-1):  
        if (dist_list[i][0] > selected_inds[0]):
            selected_coords.append(dist_list[i])
            
    final_selected_inds = np.random.choice(len(selected_coords), size=[1]) 
    x = selected_coords[final_selected_inds[0]][2]
    y = selected_coords[final_selected_inds[0]][1]
    return [x, y]  # x, y    
    
            

def get_sample_rect(img_h, img_w, rect_w, mask_rect, dist_list, dist_max):

    sample_x, sample_y = nonuniform_sample(img_h-rect_w, img_w-rect_w, dist_list, dist_max)
    sampled_coords = []
    for i in range(rect_w):
        for j in range(rect_w):
            sampled_coords.append([sample_y + i, sample_x + j])
    sampled_coords = np.array(sampled_coords)
    sampled_coords = torch.Tensor(sampled_coords).to(device) # [rect_w*rect_w, 2]
    if (sample_x >= mask_rect[0]) and (sample_y >= mask_rect[1]): 
        ssim_mask = True
    else:
        ssim_mask = False
    if (mask_rect[0] + mask_rect[2] >= sample_x + rect_w) and (mask_rect[1] + mask_rect[3] >= sample_y + rect_w): 
        ssim_mask = ssim_mask and True
    else:
        ssim_mask = False
            
    return sampled_coords, ssim_mask

def train():

    parser = config_parser()
    args = parser.parse_args()
    
    mask_rect = load_mask_rect(args.datadir + '/mask/mask_rect.txt') # dict, [x,y, w,h]
    # Load data
    K = None
    if args.dataset_type == 'llff':
        images, poses, bds, render_poses, i_test = load_llff_data(args.datadir, args.factor,
                                                                  recenter=True, bd_factor=.75,
                                                                  spherify=args.spherify)
        # images- [b, h, w, c]
        hwf = poses[0,:3,-1]  # [b, 3], intronics
        poses = poses[:,:3,:4] # [b, 3, 4], extronics
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

    # Short circuit if only rendering out from trained model
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
    N_rand = args.N_rand
    use_batching = not args.no_batching
    if use_batching:
        # For random ray batching
        print('get rays')
        rays = np.stack([get_rays_np(H, W, K, p) for p in poses[:,:3,:4]], 0) # [N, ro+rd, H, W, 3]
        print('done, concats')
        rays_rgb = np.concatenate([rays, images[:,None]], 1) # [N, ro+rd+rgb, H, W, 3]
        rays_rgb = np.transpose(rays_rgb, [0,2,3,1,4]) # [N, H, W, ro+rd+rgb, 3]
        rays_rgb = np.stack([rays_rgb[i] for i in i_train], 0) # train images only
        rays_rgb = np.reshape(rays_rgb, [-1,3,3]) # [(N-1)*H*W, ro+rd+rgb, 3]
        rays_rgb = rays_rgb.astype(np.float32)
        print('shuffle rays')
        np.random.shuffle(rays_rgb)

        print('done')
        i_batch = 0

    # Move training data to GPU
    if use_batching:
        images = torch.Tensor(images).to(device)
    poses = torch.Tensor(poses).to(device)
    if use_batching:
        rays_rgb = torch.Tensor(rays_rgb).to(device)

    ssim_fun = SSIM()
    ssim_fun = ssim_fun.to(device)
    
    N_iters = 20000000 + 1  
    print('Begin')
    print('TRAIN views are', i_train)
    print('TEST views are', i_test)
    print('VAL views are', i_val)
    
    tb_writer = SummaryWriter(log_dir=".//log")
    valid_frames = get_sfm_training_frames(i_train, images.shape[0], args)        
    
    dist_list, dist_max = get_sample_dist(H-1-int(math.sqrt(N_rand)), W-1-int(math.sqrt(N_rand)) )
    
    best_eval_psnr = 0
    best_ssim = 0
    start = start + 1

    for i in range(start, N_iters): 
        time0 = time.time()
        
        # Random from two image
        sample_img = []
        sample_pose = []       
        
        valid_frame_inds = np.random.randint(len(valid_frames)) 
        img_inds = valid_frames[valid_frame_inds] 
        sample_pose.append(poses[img_inds[0], :3,:4].detach()) 
        
        for j in img_inds:
            dst_img = images[j]
            dst_img = torch.Tensor(dst_img).to(device)
            sample_img.append(dst_img)         

        for j in range(2):            
            sample_pose.append(poses[img_inds[j+1], :3,:4])

        sample_pose_inv = get_pose_inverse( torch.stack(sample_pose, dim=0))            
        if N_rand is not None:                    
            img = sample_img[0] # [H,W,C]
            if args.interpolation_render:
                if img_inds[1] > img_inds[2]:
                    sample_mask_rect = mask_rect['{:03d}_{:03d}_{:03d}'.format(img_inds[0], img_inds[2], img_inds[1])]
                else:
                    sample_mask_rect = mask_rect['{:03d}_{:03d}_{:03d}'.format(img_inds[0], img_inds[1], img_inds[2])]
                select_coords, ssim_mask = get_sample_rect(H-1, W-1, int(math.sqrt(N_rand)), sample_mask_rect, dist_list, dist_max) 
                offset_x = np.random.rand(select_coords.shape[0]) 
                offset_x = offset_x.reshape(select_coords.shape[0], 1)
                offset_y = np.random.rand(select_coords.shape[0]) 
                offset_y = offset_y.reshape(select_coords.shape[0], 1)
                offset = np.concatenate([offset_x, offset_y], axis=1) # [n,2]
                offset = torch.from_numpy(offset).to(device)
                select_coords = select_coords + offset  
                target_s = bilinear_sampler(img, select_coords)      
                rays_o, rays_d = get_sift_rays(select_coords[:, 1], select_coords[:, 0], K, sample_pose[0])
                batch_rays = torch.stack([rays_o, rays_d], 0)
                
            else:
                rays_o, rays_d = get_rays(H, W, K, sample_pose[0])       
                
                if img_inds[1] > img_inds[2]:
                    sample_mask_rect = mask_rect['{:03d}_{:03d}_{:03d}'.format(img_inds[0], img_inds[2], img_inds[1])]
                else:
                    sample_mask_rect = mask_rect['{:03d}_{:03d}_{:03d}'.format(img_inds[0], img_inds[1], img_inds[2])]
                select_coords, ssim_mask = get_sample_rect(H-1, W-1, int(math.sqrt(N_rand)), sample_mask_rect, dist_list, dist_max) 
                select_coords = select_coords.long()
                rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                batch_rays = torch.stack([rays_o, rays_d], 0) # (2, N_rand, 3)
                target_s = img[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3) 

            if img_inds[1] < img_inds[2]:
                sift_file = args.siftdir + 'sift_{:0>2d}_{:0>2d}_{:0>2d}_{:0>2d}.txt'.format(args.factor, img_inds[0], img_inds[1], img_inds[2])  
            else:
                sift_file = args.siftdir + 'sift_{:0>2d}_{:0>2d}_{:0>2d}_{:0>2d}.txt'.format(args.factor, img_inds[0], img_inds[2], img_inds[1])                              
            sift_correspondences = load_sift_correspondences(sift_file) 
            sift_correspondences = np.array(sift_correspondences) 
            if len(sift_correspondences) > args.max_sift_num:             
                select_inds = np.random.choice(len(sift_correspondences), size=[args.max_sift_num], replace=False)  # (N_rand,)
                sift_correspondences = sift_correspondences[select_inds]                   
            
                
            sift_correspondences = np.reshape(sift_correspondences, [-1, 3, 2]) # [n, 3, 2]
            n_sift = len(sift_correspondences)
            if n_sift > 0:
                sift_correspondences = np.transpose(sift_correspondences, [1,0,2]).astype(np.float32) # [3, n, 2]
                sift_correspondences = torch.Tensor(sift_correspondences).to(device)
                
                sift_ray_o = []
                sift_ray_d = []

                for j in range(3):
                    sift_point = sift_correspondences[j, :, :] # [n, 2]                    

                    rays_o, rays_d = get_sift_rays(sift_point[:, 0], sift_point[:, 1], K, sample_pose[j])
                    sift_ray_o.append(rays_o)
                    sift_ray_d.append(rays_d)

                sift_rays_o = torch.cat(sift_ray_o, 0) # (3*N_sift, 3)
                sift_rays_d = torch.cat(sift_ray_d, 0) # (3*N_sift, 3)
                batch_sift_rays = torch.stack([sift_rays_o, sift_rays_d], 0) # (2, 3*N_sift, 3)
                batch_rays = torch.cat([batch_rays, batch_sift_rays], dim=1) # [2, N_rand+3*N_sift, 3]
                
                for k in range(sift_correspondences.shape[0]):
                    sift_coords = sift_correspondences[k]
                    sift_coords = torch.cat([sift_coords[:, 1].unsqueeze(-1), sift_coords[:, 0].unsqueeze(-1)], dim=-1) 
                    temp_target_s = bilinear_sampler(sample_img[k], sift_coords)
                    target_s = torch.cat([target_s, temp_target_s], dim=0)  
      
        #####  Core optimization loop  #####
        rgb, disp, acc, pt, disp_mask, extras = render(H, W, K, chunk=args.chunk, rays=batch_rays,
                                                verbose=i < 10, retraw=True,
                                                **render_kwargs_train)
       
        optimizer.zero_grad()
        img_loss = img2mse(rgb, target_s) 
        trans = extras['raw'][...,-1]
        loss = args.image_weight * img_loss 
        psnr = mse2psnr(img_loss)         
        

        pixel_rgb = target_s[:N_rand,...]  # [N_rand, 3]
        
        pixel_pts = pt[:N_rand,...] # [N_rand, 3]
        trans_pose1 = get_transfomation_matrix(sample_pose_inv[1, ...], sample_pose_inv[0, ...])
        trans_pose2 = get_transfomation_matrix(sample_pose_inv[2, ...], sample_pose_inv[0, ...])
        trans_pose = torch.cat([trans_pose1.unsqueeze(0), trans_pose2.unsqueeze(0)], dim=0)
        projected_pts = get_projected_pt(pixel_pts, trans_pose, torch.Tensor(K).to(device))         
        
        interplated_rgb1 = bilinear_sampler(sample_img[1], projected_pts[0]) 
        interplated_rgb2 = bilinear_sampler(sample_img[2], projected_pts[1]) 
        interplated_rgb = torch.stack([interplated_rgb1, interplated_rgb2], dim=0)
        
        if img_inds[1] < img_inds[2]:
            mask_fname = os.path.join(args.datadir, 'mask/') + 'mask_{:0>2d}_{:0>2d}_{:0>2d}_{:0>2d}.png'.format(args.factor, img_inds[0], img_inds[1], img_inds[2]) 
        else:
            mask_fname = os.path.join(args.datadir, 'mask/') + 'mask_{:0>2d}_{:0>2d}_{:0>2d}_{:0>2d}.png'.format(args.factor, img_inds[0], img_inds[2], img_inds[1]) 
        mask_img = cv2.imread(mask_fname) 
        mask_img = torch.Tensor(mask_img).to(device)
        mask_img = mask_img / 255
        mask_pixel = mask_img[select_coords[:, 0].long(), select_coords[:, 1].long()]
        color_loss = torch.mean(torch.abs(pixel_rgb - interplated_rgb[0]) * mask_pixel)
        color_loss = color_loss + torch.mean(torch.abs(pixel_rgb - interplated_rgb[1]) * mask_pixel)
        loss = loss + args.color_weight * color_loss # loss + color_loss / color_loss.detach()

        if ssim_mask:
            pixel_rgb_rect = torch.reshape(pixel_rgb, [int(math.sqrt(N_rand)), int(math.sqrt(N_rand)), 3]).unsqueeze(0).repeat(2, 1, 1, 1)  # [2, n, n, 3]
            interplated_rgb = torch.reshape(interplated_rgb, [2, int(math.sqrt(N_rand)), int(math.sqrt(N_rand)), 3])  # [2, n, n, 3]
            ssim_loss = torch.mean(ssim_fun(pixel_rgb_rect, interplated_rgb))
            loss = loss + args.ssim_weight * ssim_loss # loss + ssim_loss / ssim_loss.detach()
        else:
            ssim_loss = 0

        if n_sift > 0:
            sift_pts = pt[N_rand: N_rand + 3*n_sift,...] # [3*N_sift, 3]
            sift_pts = torch.reshape(sift_pts, [3, n_sift, 3])   
            sift_loss = epipolar_constraint(sift_pts)
            loss = loss + args.sift_weight * sift_loss  # loss + sift_loss / sift_loss.detach()
        else:
            sift_loss = 0
            
        img0_pts = pt[0: N_rand,...] 
        img0_pts = torch.reshape(img0_pts, [int(math.sqrt(N_rand)), int(math.sqrt(N_rand)), 3]).unsqueeze(0) # [1, n, n, 3]
        img0_target = torch.reshape(pixel_pts, [int(math.sqrt(N_rand)), int(math.sqrt(N_rand)), 3]).unsqueeze(0) # [1, n, n, 3]
        smooth_loss = get_smooth_loss(img0_pts.permute(0, 3, 1, 2), img0_target.permute(0, 3, 1, 2))
        loss = loss + args.smooth_weight * smooth_loss       
   
        if 'rgb0' in extras:            
            img_loss0 = img2mse(extras['rgb0'], target_s)
            psnr0 = mse2psnr(img_loss0)            
            loss = loss + args.image_weight * img_loss0 
                        
            pixel_pts0 = extras['pt0'][:N_rand,...] # [N_rand, 3]
            projected_pts0 = get_projected_pt(pixel_pts0, trans_pose, torch.Tensor(K).to(device))
            
            interplated_rgb0_1 = bilinear_sampler(sample_img[1], projected_pts0[0]) 
            interplated_rgb0_2 = bilinear_sampler(sample_img[2], projected_pts0[1]) 
            interplated_rgb0 = torch.stack([interplated_rgb0_1, interplated_rgb0_2], dim=0)            
           
            color_loss0 = torch.mean(torch.abs(pixel_rgb - interplated_rgb0[0]) * mask_pixel)
            color_loss0 = color_loss0 + torch.mean(torch.abs(pixel_rgb - interplated_rgb0[1]) * mask_pixel)
            loss = loss + args.color_weight * color_loss0
            
            if ssim_mask:
                interplated_rgb0 = torch.reshape(interplated_rgb0, [2, int(math.sqrt(N_rand)), int(math.sqrt(N_rand)), 3])  # [2, n, n, 3]
                ssim_loss0 = torch.mean(ssim_fun(pixel_rgb_rect, interplated_rgb0))
                loss = loss + args.ssim_weight * ssim_loss0 
            else:
                ssim_loss0 = 0          

            if n_sift > 0:
                sift_pts0 = extras['pt0'][N_rand: N_rand + 3*n_sift,...] # [3*N_sift, 3]
                sift_pts0 = torch.reshape(sift_pts0, [3, n_sift, 3])    
                sift_loss0 = epipolar_constraint(sift_pts0)
                loss = loss + args.sift_weight * sift_loss0 
            else:
                sift_loss0 = 0
                
            img0_pts0 = torch.reshape(pixel_pts0, [int(math.sqrt(N_rand)), int(math.sqrt(N_rand)), 3]).unsqueeze(0) # [1, n, n, 3]
            smooth_loss0 = get_smooth_loss(img0_pts0.permute(0, 3, 1, 2), img0_target.permute(0, 3, 1, 2))
            loss = loss + args.smooth_weight * smooth_loss0
             
    
        loss.backward()  
        optimizer.step()

        # NOTE: IMPORTANT!
        ###   update learning rate   ###
        decay_rate = 0.1
        decay_steps = args.lrate_decay  * 2000 
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        optimizer.param_groups[0]['lr'] = new_lrate               
        
        # Rest is logging
        if i%args.i_weights==0:
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
            if args.N_importance > 0:                
                torch.save({
                    'global_step': global_step,
                    'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                    'network_fine_state_dict': render_kwargs_train['network_fine'].state_dict(),                        
                    'optimizer_state_dict': optimizer.state_dict(),
                }, path)
            else:                
                torch.save({
                    'global_step': global_step,
                    'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),                                            
                    'optimizer_state_dict': optimizer.state_dict(),
                }, path)
            print('Saved checkpoints at', path)

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
            tb_writer.add_scalar("sift_loss",sift_loss,global_step) 
            tb_writer.add_scalar("img_loss",img_loss,global_step) 
            tb_writer.add_scalar("color_loss",color_loss,global_step) 
            tb_writer.add_scalar("ssim_loss",ssim_loss,global_step) 
            tb_writer.add_scalar("smooth_loss",smooth_loss,global_step) 
            tb_writer.add_scalar("sift_loss0",sift_loss0,global_step) 
            tb_writer.add_scalar("img_loss0",img_loss0,global_step) 
            tb_writer.add_scalar("color_loss0",color_loss0,global_step) 
            tb_writer.add_scalar("ssim_loss0",ssim_loss0,global_step)   
            tb_writer.add_scalar("smooth_loss0",smooth_loss0,global_step)                       
            tb_writer.add_scalar("psnr",psnr,global_step)            
            tb_writer.add_scalar("lr",optimizer.param_groups[0]['lr'],global_step)            
            if args.N_importance > 0:
                tb_writer.add_scalar("psnr0",psnr0,global_step)
        
        if i%args.i_img==0:
            
            # Log a rendered validation view to Tensorboard
            img_i=np.random.choice(i_val)
            target = images[img_i]
            pose = poses[img_i, :3,:4]            
            
            with torch.no_grad():
                rgb, disp, acc, pt, disp_mask, extras = render(H, W, K, chunk=args.chunk, c2w=pose,
                                                    **render_kwargs_test)
            
            psnr = mse2psnr(img2mse(rgb, torch.from_numpy(target).to(device)))
            temp = to8b(rgb.cpu().numpy())
            tb_writer.add_image('rgb', temp, dataformats='HWC')
            
            disp = disp.cpu().numpy()
            disp = colorize(disp)
            if i%args.i_testset==0 and i > 0:
                for j in i_val:
                    disp_pose = poses[j, :3,:4]
                    with torch.no_grad():
                        disp_rgb, disp_disp, disp_acc, disp_pt, disp_mask, disp_extras = render(H, W, K, chunk=args.chunk, c2w=disp_pose,
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
            
            tb_writer.add_scalar('psnr_holdout', psnr, global_step)
            tb_writer.add_image('rgb_holdout', target, dataformats='HWC')
            if args.N_importance > 0:
                temp = to8b(extras['rgb0'].cpu().numpy())                
                tb_writer.add_image('rgb0', temp, dataformats='HWC')
                tb_writer.add_image('disp0', extras['disp0'][np.newaxis,...])
                tb_writer.add_image('z_std', extras['z_std'][np.newaxis,...])      
        
        if i%args.i_eval==0:
            PSNRs = []
            ssims,l_alex,l_vgg=[],[],[]
            for i, v in enumerate(i_val):
                target = images[v]
                pose = poses[v, :3,:4]
                with torch.no_grad():
                    rgb, disp, acc, pt, disp_mask, extras = render(H, W, K, chunk=args.chunk, c2w=pose,
                                                        **render_kwargs_test)
                
                psnr = mse2psnr(img2mse(rgb, torch.from_numpy(target).to(device)))
                PSNRs.append(psnr)
                if args.compute_extra_metrics:
                    this_type_str = type(rgb)
                    if this_type_str is not np.ndarray:
                        rgb = np.array(rgb.cpu())
                    
                    ssim = rgb_ssim(rgb, target, 1)
                    l_a = rgb_lpips(target, rgb, 'alex', device)
                    l_v = rgb_lpips(target, rgb, 'vgg', device)
                    ssims.append(ssim)
                    l_alex.append(l_a)
                    l_vgg.append(l_v)
            if PSNRs:
                eval_psnr = torch.stack(PSNRs, -1).mean()                
                if args.compute_extra_metrics:
                    eval_ssim = np.mean(np.asarray(ssims))
                    eval_l_a = np.mean(np.asarray(l_alex))
                    eval_l_v = np.mean(np.asarray(l_vgg))
                    if eval_ssim > best_ssim:
                        best_ssim = eval_ssim
                        mean_name = os.path.join(basedir, expname, 'mean.txt')
                        with open(mean_name, "a") as f:
                            np.savetxt(f, np.asarray([eval_psnr.cpu(), eval_ssim, eval_l_a, eval_l_v]), newline=' ', delimiter=',')
                            f.write("\n")
                    if eval_psnr > best_eval_psnr:
                        mean_name = os.path.join(basedir, expname, 'mean.txt')
                        with open(mean_name, "a") as f:
                            np.savetxt(f, np.asarray([eval_psnr.cpu(), eval_ssim, eval_l_a, eval_l_v]), newline=' ', delimiter=',')
                            f.write("\n")
                else:
                    if eval_psnr> best_eval_psnr:
                        np.savetxt(mean_name, np.asarray([eval_psnr.cpu()])) 
                if eval_psnr> best_eval_psnr:
                    best_eval_psnr = eval_psnr      
                print("eval_psnr: {}".format(eval_psnr))
                tb_writer.add_scalar('eval_psnr', eval_psnr,global_step)
        
        global_step += 1


if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    train()
