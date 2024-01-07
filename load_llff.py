import numpy as np
import os, imageio
import cv2
import torch
import glob
import re
from PIL import Image


########## Slightly modified version of LLFF data loading code 
##########  see https://github.com/Fyusion/LLFF for original

def _minify(basedir, factors=[], resolutions=[]):
    needtoload = False
    for r in factors:
        imgdir = os.path.join(basedir, 'images_{}'.format(r))
        if not os.path.exists(imgdir):
            needtoload = True
    for r in resolutions:
        imgdir = os.path.join(basedir, 'images_{}x{}'.format(r[1], r[0]))
        if not os.path.exists(imgdir):
            needtoload = True
    if not needtoload:
        return
    
    from shutil import copy
    from subprocess import check_output
    
    imgdir = os.path.join(basedir, 'images')
    imgs = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir))]
    imgs = [f for f in imgs if any([f.endswith(ex) for ex in ['JPG', 'jpg', 'png', 'jpeg', 'PNG']])]
    imgdir_orig = imgdir
    
    wd = os.getcwd()

    for r in factors + resolutions:
        if isinstance(r, int):
            name = 'images_{}'.format(r)
            resizearg = '{}%'.format(100./r)
        else:
            name = 'images_{}x{}'.format(r[1], r[0])
            resizearg = '{}x{}'.format(r[1], r[0])
        imgdir = os.path.join(basedir, name)
        if os.path.exists(imgdir):
            continue
            
        print('Minifying', r, basedir)
        
        os.makedirs(imgdir)
        check_output('cp {}/* {}'.format(imgdir_orig, imgdir), shell=True)
        
        ext = imgs[0].split('.')[-1]
        args = ' '.join(['mogrify', '-resize', resizearg, '-format', 'png', '*.{}'.format(ext)])
        print(args)
        os.chdir(imgdir)
        check_output(args, shell=True)
        os.chdir(wd)
        
        if ext != 'png':
            check_output('rm {}/*.{}'.format(imgdir, ext), shell=True)
            print('Removed duplicates')
        print('Done')
            
        
        
        
def _load_data(basedir, factor=None, width=None, height=None, load_imgs=True):
    
    poses_arr = np.load(os.path.join(basedir, 'poses_bounds.npy')) # (imagenumber*17)
    poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1,2,0]) # (3, 5, imagenumber)
    bds = poses_arr[:, -2:].transpose([1,0]) # (2, imagenumber)
    
    img0 = [os.path.join(basedir, 'images', f) for f in sorted(os.listdir(os.path.join(basedir, 'images'))) \
            if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')][0]
    sh = imageio.imread(img0).shape # desk2 (4344, 5792, 3)
    
    sfx = '' 
    
    if factor is not None:
        sfx = '_{}'.format(factor)
        _minify(basedir, factors=[factor])
        factor = factor
    elif height is not None:
        factor = sh[0] / float(height)
        width = int(sh[1] / factor)
        _minify(basedir, resolutions=[[height, width]])
        sfx = '_{}x{}'.format(width, height)
    elif width is not None:
        factor = sh[1] / float(width)
        height = int(sh[0] / factor)
        _minify(basedir, resolutions=[[height, width]])
        sfx = '_{}x{}'.format(width, height)
    else:
        factor = 1
    
    imgdir = os.path.join(basedir, 'images' + sfx)
    if not os.path.exists(imgdir):
        print( imgdir, 'does not exist, returning' )
        return
    # imgfiles list
    imgfiles = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir)) if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
    if poses.shape[-1] != len(imgfiles):
        print( 'Mismatch between imgs {} and poses {} !!!!'.format(len(imgfiles), poses.shape[-1]) )
        return
    
    sh = imageio.imread(imgfiles[0]).shape # image shape, desk2(543, 724, 3)
    poses[:2, 4, :] = np.array(sh[:2]).reshape([2, 1]) # (3, 5, imagenumber)
    poses[2, 4, :] = poses[2, 4, :] * 1./factor
    
    if not load_imgs:
        return poses, bds
    
    def imread(f):
        if f.endswith('png'):
            return imageio.imread(f, ignoregamma=True)
        else:
            return imageio.imread(f)
        
    imgs = imgs = [imread(f)[...,:3]/255. for f in imgfiles]
    imgs = np.stack(imgs, -1) #(543, 727,3 ,1 151) 
    
    print('Loaded image data', imgs.shape, poses[:,-1,0])
    return poses, bds, imgs

    
            
            
    

def normalize(x):
    return x / np.linalg.norm(x)

def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m

def ptstocam(pts, c2w):
    tt = np.matmul(c2w[:3,:3].T, (pts-c2w[:3,3])[...,np.newaxis])[...,0]
    return tt

def poses_avg(poses):

    hwf = poses[0, :3, -1:]

    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)
    
    return c2w



def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, rots, N):
    render_poses = []
    rads = np.array(list(rads) + [1.])
    hwf = c2w[:,4:5]
    
    for theta in np.linspace(0., 2. * np.pi * rots, N+1)[:-1]:
        c = np.dot(c2w[:3,:4], np.array([np.cos(theta), -np.sin(theta), -np.sin(theta*zrate), 1.]) * rads) 
        z = normalize(c - np.dot(c2w[:3,:4], np.array([0,0,-focal, 1.])))
        render_poses.append(np.concatenate([viewmatrix(z, up, c), hwf], 1))
    return render_poses
    


def recenter_poses(poses):

    poses_ = poses+0
    bottom = np.reshape([0,0,0,1.], [1,4])
    c2w = poses_avg(poses)
    c2w = np.concatenate([c2w[:3,:4], bottom], -2)
    bottom = np.tile(np.reshape(bottom, [1,1,4]), [poses.shape[0],1,1])
    poses = np.concatenate([poses[:,:3,:4], bottom], -2)

    poses = np.linalg.inv(c2w) @ poses
    poses_[:,:3,:4] = poses[:,:3,:4]
    poses = poses_
    return poses


#####################


def spherify_poses(poses, bds):
    
    p34_to_44 = lambda p : np.concatenate([p, np.tile(np.reshape(np.eye(4)[-1,:], [1,1,4]), [p.shape[0], 1,1])], 1)
    
    rays_d = poses[:,:3,2:3]
    rays_o = poses[:,:3,3:4]

    def min_line_dist(rays_o, rays_d):
        A_i = np.eye(3) - rays_d * np.transpose(rays_d, [0,2,1])
        b_i = -A_i @ rays_o
        pt_mindist = np.squeeze(-np.linalg.inv((np.transpose(A_i, [0,2,1]) @ A_i).mean(0)) @ (b_i).mean(0))
        return pt_mindist

    pt_mindist = min_line_dist(rays_o, rays_d)
    
    center = pt_mindist
    up = (poses[:,:3,3] - center).mean(0)

    vec0 = normalize(up)
    vec1 = normalize(np.cross([.1,.2,.3], vec0))
    vec2 = normalize(np.cross(vec0, vec1))
    pos = center
    c2w = np.stack([vec1, vec2, vec0, pos], 1)

    poses_reset = np.linalg.inv(p34_to_44(c2w[None])) @ p34_to_44(poses[:,:3,:4])

    rad = np.sqrt(np.mean(np.sum(np.square(poses_reset[:,:3,3]), -1)))
    
    sc = 1./rad
    poses_reset[:,:3,3] *= sc
    bds *= sc
    rad *= sc
    
    centroid = np.mean(poses_reset[:,:3,3], 0)
    zh = centroid[2]
    radcircle = np.sqrt(rad**2-zh**2)
    new_poses = []
    
    for th in np.linspace(0.,2.*np.pi, 120):

        camorigin = np.array([radcircle * np.cos(th), radcircle * np.sin(th), zh])
        up = np.array([0,0,-1.])

        vec2 = normalize(camorigin)
        vec0 = normalize(np.cross(vec2, up))
        vec1 = normalize(np.cross(vec2, vec0))
        pos = camorigin
        p = np.stack([vec0, vec1, vec2, pos], 1)

        new_poses.append(p)

    new_poses = np.stack(new_poses, 0)
    
    new_poses = np.concatenate([new_poses, np.broadcast_to(poses[0,:3,-1:], new_poses[:,:3,-1:].shape)], -1)
    poses_reset = np.concatenate([poses_reset[:,:3,:4], np.broadcast_to(poses[0,:3,-1:], poses_reset[:,:3,-1:].shape)], -1)
    
    return poses_reset, new_poses, bds
    

def load_llff_data(basedir, factor=8, recenter=True, bd_factor=.75, spherify=False, path_zflat=False):
    

    poses, bds, imgs = _load_data(basedir, factor=factor) # factor=8 downsamples original imgs by 8x // DTU set for width=500
    print('Loaded', basedir, bds.min(), bds.max())
    
    # Correct rotation matrix ordering and move variable dim to axis 0
    poses = np.concatenate([poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1)
    poses = np.moveaxis(poses, -1, 0).astype(np.float32) #（151， 3， 5）
    imgs = np.moveaxis(imgs, -1, 0).astype(np.float32)
    images = imgs #（151， 543，727， 3）
    bds = np.moveaxis(bds, -1, 0).astype(np.float32)
    
    # Rescale if bd_factor is provided
    sc = 1. if bd_factor is None else 1./(bds.min() * bd_factor)
    poses[:,:3,3] *= sc
    bds *= sc
    
    if recenter:
        poses = recenter_poses(poses)
        
    if spherify:
        poses, render_poses, bds = spherify_poses(poses, bds)

    else:
        
        c2w = poses_avg(poses)
        print('recentered', c2w.shape)
        print(c2w[:3,:4])

        ## Get spiral
        # Get average pose
        up = normalize(poses[:, :3, 1].sum(0))

        # Find a reasonable "focus depth" for this dataset
        close_depth, inf_depth = bds.min()*.9, bds.max()*5.
        dt = .75
        mean_dz = 1./(((1.-dt)/close_depth + dt/inf_depth))
        focal = mean_dz

        # Get radii for spiral path
        shrink_factor = .8
        zdelta = close_depth * .2
        tt = poses[:,:3,3] # ptstocam(poses[:3,3,:].T, c2w).T
        rads = np.percentile(np.abs(tt), 90, 0)
        c2w_path = c2w
        N_views = 120
        N_rots = 2
        if path_zflat:
#             zloc = np.percentile(tt, 10, 0)[2]
            zloc = -close_depth * .1
            c2w_path[:3,3] = c2w_path[:3,3] + zloc * c2w_path[:3,2]
            rads[2] = 0.
            N_rots = 1
            N_views/=2

        # Generate poses for spiral path
        render_poses = render_path_spiral(c2w_path, up, rads, focal, zdelta, zrate=.5, rots=N_rots, N=N_views)
        
        
    render_poses = np.array(render_poses).astype(np.float32)

    c2w = poses_avg(poses)
    print('Data:')
    print(poses.shape, images.shape, bds.shape)
    
    dists = np.sum(np.square(c2w[:3,3] - poses[:,:3,3]), -1)
    i_test = np.argmin(dists)
    print('HOLDOUT view is', i_test)
    
    images = images.astype(np.float32)
    poses = poses.astype(np.float32)

    return images, poses, bds, render_poses, i_test

def read_pfm(filename):
    file = open(filename, 'rb')
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    file.close()
    return data, scale

def read_mask_hr(filename):
    img = Image.open(filename)
    np_img = np.array(img, dtype=np.float32)
    np_img = (np_img > 10).astype(np.float32)
    return np_img


def load_DTU_depth(basedir, factor=8):
    img0 = [os.path.join(basedir, 'images', f) for f in sorted(os.listdir(os.path.join(basedir, 'images'))) \
            if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')][0]
    
    scan_filename = os.path.basename(basedir)
    father_dir = os.path.dirname(basedir)
    father_dir = os.path.dirname(father_dir)
    
    all_depths = []  
    all_masks = [] 
    
    for vid in range(len(img0)):
        depth_filename_hr = os.path.join(father_dir, 'Depths_raw', scan_filename, 'depth_map_{:0>4}.pfm'.format(vid))
        depth_raw = np.array(read_pfm(depth_filename_hr)[0], dtype=np.float32)
        h, w = depth_raw.shape
        depths = cv2.resize(depth_raw, (w // factor, h // factor), interpolation=cv2.INTER_NEAREST)
        all_depths.append(depths) 
        mask_filename_hr = os.path.join(father_dir, 'Depths_raw', scan_filename, 'depth_visual_{:0>4}.png'.format(vid))
        mask_img = read_mask_hr(mask_filename_hr)
        h, w = mask_img.shape
        mask_img = cv2.resize(mask_img, (w // factor, h // factor), interpolation=cv2.INTER_NEAREST)
        all_masks.append(mask_img) 

    depths = np.stack(all_depths)
    depths = depths.transpose(1, 2, 0)
    masks = np.stack(all_masks)
    masks = masks.transpose(1, 2, 0)
    
    sh = imageio.imread(img0).shape # desk2 (4344, 5792, 3)
    
    sfx = '' # (default=8)
    
    if factor is not None:
        sfx = '_{}'.format(factor)
        factor = factor
    elif height is not None:
        factor = sh[0] / float(height)
        width = int(sh[1] / factor)
        sfx = '_{}x{}'.format(width, height)
    elif width is not None:
        factor = sh[1] / float(width)
        height = int(sh[0] / factor)
        sfx = '_{}x{}'.format(width, height)
    else:
        factor = 1
    
    imgdir = os.path.join(basedir, 'images' + sfx)
    if not os.path.exists(imgdir):
        print( imgdir, 'does not exist, returning' )
        return
    # imgfiles list
    imgfiles = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir)) if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
    if len(imgfiles) < depths.shape[-1]:
        depths = depths[:, :, :len(imgfiles)] 
        masks = masks[:, :, :len(imgfiles)]
    return depths, masks


def load_ScanNet_depth(basedir, factor=8):
    imgs = [os.path.join(basedir, 'images', f) for f in sorted(os.listdir(os.path.join(basedir, 'images'))) \
            if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
    depth_files = [os.path.join(basedir, 'depth', f) for f in sorted(os.listdir(os.path.join(basedir, 'depth'))) \
            if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
    all_depths = []     
    img_h, img_w, _ = imageio.imread(imgs[0]).shape # 
    for vid in range(len(imgs)):
        depth_raw = imageio.imread(depth_files[vid], ignoregamma=True)
        # h, w = depth_raw.shape
        depths = cv2.resize(depth_raw, (img_w // factor, img_h // factor), interpolation=cv2.INTER_NEAREST)
        all_depths.append(depths)  
    depths = np.stack(all_depths)
    # depths = depths.transpose(1, 2, 0)  
    
    return depths


def _load_ScanNet_data(basedir, factor=None, width=None, height=None, load_imgs=True):
    
    # poses_arr = np.load(os.path.join(basedir, 'poses_bounds.npy')) 
    # poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1,2,0]) # (3, 5, imagenumber)
    # bds = poses_arr[:, -2:].transpose([1,0]) # (2, imagenumber)
    
    imgs = [os.path.join(basedir, 'images', f) for f in sorted(os.listdir(os.path.join(basedir, 'images'))) \
            if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
    
    depth_files = [os.path.join(basedir, 'depth', f) for f in sorted(os.listdir(os.path.join(basedir, 'depth'))) \
            if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
    
    pose_files = [os.path.join(basedir, 'pose', f) for f in sorted(os.listdir(os.path.join(basedir, 'pose'))) \
            if f.endswith('txt')]

    
    all_poses = [] 
    all_depths = [] 
    all_bds = []
    intrinic_filename = os.path.join(basedir, 'intrinsic/intrinsic_color.txt')
    with open(intrinic_filename) as f:
        lines = f.readlines()
        lines = [line.rstrip() for line in lines]
    # extrinsics: line [1,5), 4x4 matrix
    K = np.fromstring(' '.join(lines[0:4]), dtype=np.float32, sep=' ').reshape((4, 4)) 
    K = K[0:3, 0:3]    
    img_h, img_w, _ = imageio.imread(imgs[0]).shape # 
    for vid in range(len(imgs)):
        depth_raw = imageio.imread(depth_files[vid], ignoregamma=True)
        # h, w = depth_raw.shape
        depths = cv2.resize(depth_raw, (img_w // factor, img_h // factor), interpolation=cv2.INTER_NEAREST)
        all_depths.append(depths)  
        
        bds = np.array([[np.min(depth_raw)], [np.max(depth_raw)]],dtype =np.float64)
        all_bds.append(bds)
                 
        

        with open(pose_files[vid]) as f:
            lines = f.readlines()
            lines = [line.rstrip() for line in lines]
        extrinsics = np.fromstring(' '.join(lines[0:4]), dtype=np.float32, sep=' ').reshape((4, 4))
    
        focal = (K[0][0] + K[1][1])/2.0
        poses_temp = np.concatenate([extrinsics[:3, :], np.array([[img_h], [img_w], [focal]])], axis=1)
        all_poses.append(poses_temp)  # [3, 5]
    poses = np.stack(all_poses)
    poses = poses.transpose(1, 2, 0)
    depths = np.stack(all_depths)
    depths = depths.transpose(1, 2, 0)
    bds = np.stack(all_bds).squeeze(-1)
    bds = bds.transpose(1, 0)

    
    sh = imageio.imread(imgs[0]).shape # desk2 (4344, 5792, 3)
    
    sfx = '' # (default=8)
    
    if factor is not None:
        sfx = '_{}'.format(factor)
        _minify(basedir, factors=[factor])
        factor = factor
    elif height is not None:
        factor = sh[0] / float(height)
        width = int(sh[1] / factor)
        _minify(basedir, resolutions=[[height, width]])
        sfx = '_{}x{}'.format(width, height)
    elif width is not None:
        factor = sh[1] / float(width)
        height = int(sh[0] / factor)
        _minify(basedir, resolutions=[[height, width]])
        sfx = '_{}x{}'.format(width, height)
    else:
        factor = 1
    
    imgdir = os.path.join(basedir, 'images' + sfx)
    if not os.path.exists(imgdir):
        print( imgdir, 'does not exist, returning' )
        return
    # imgfiles list
    imgfiles = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir)) if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
    if len(imgfiles) < poses.shape[-1]:
        poses = poses[:, :, :len(imgfiles)]    
        depths = depths[:, :, :len(imgfiles)] 
        bds = bds[:, :len(imgfiles)]
    if poses.shape[-1] != len(imgfiles):
        print( 'Mismatch between imgs {} and poses {} !!!!'.format(len(imgfiles), poses.shape[-1]) )
        return
    
    sh = imageio.imread(imgfiles[0]).shape # image shape, desk2(543, 724, 3)
    poses[:2, 4, :] = np.array(sh[:2]).reshape([2, 1]) # (3, 5, imagenumber)
    poses[2, 4, :] = poses[2, 4, :] * 1./factor
    
    K[0,0] = K[0,0]/factor
    K[1,1] = K[1,1]/factor
    K[0,2] = K[0,2]/factor
    K[1,2] = K[1,2]/factor
    
    if not load_imgs:
        return poses, bds
    
    def imread(f):
        if f.endswith('png'):
            return imageio.imread(f, ignoregamma=True)
        else:
            return imageio.imread(f)
        
    imgs = imgs = [imread(f)[...,:3]/255. for f in imgfiles]
    imgs = np.stack(imgs, -1) #(543, 727,3 ,1 151) 
    
    print('Loaded image data', imgs.shape, poses[:,-1,0])
    return poses, bds, imgs, K, depths

def load_ScanNet_data(basedir, factor=8, recenter=True, bd_factor=.75, spherify=False, path_zflat=False):
    

    poses, bds, imgs, K, depths = _load_ScanNet_data(basedir, factor=factor) # factor=4 downsamples original imgs by 4x for DTU
    print('Loaded', basedir, bds.min(), bds.max())
    
    # Correct rotation matrix ordering and move variable dim to axis 0
    poses = np.concatenate([poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1)
    poses = np.moveaxis(poses, -1, 0).astype(np.float32) #（151， 3， 5）
    imgs = np.moveaxis(imgs, -1, 0).astype(np.float32)
    images = imgs #（151， 543，727， 3）
    bds = np.moveaxis(bds, -1, 0).astype(np.float32)

    sc = 1.
    poses[:,:3,3] *= sc
    bds *= sc
    
    if recenter:
        poses = recenter_poses(poses)
        
    if spherify:
        poses, render_poses, bds = spherify_poses(poses, bds)

    else:
        
        c2w = poses_avg(poses)
        print('recentered', c2w.shape)
        print(c2w[:3,:4])

        ## Get spiral
        # Get average pose
        up = normalize(poses[:, :3, 1].sum(0))

        # Find a reasonable "focus depth" for this dataset
        close_depth, inf_depth = bds.min()*.9, bds.max()*5.
        dt = .75
        mean_dz = 1./(((1.-dt)/close_depth + dt/inf_depth))
        focal = mean_dz

        # Get radii for spiral path
        shrink_factor = .8
        zdelta = close_depth * .2
        tt = poses[:,:3,3] # ptstocam(poses[:3,3,:].T, c2w).T
        rads = np.percentile(np.abs(tt), 90, 0)
        c2w_path = c2w
        N_views = 120
        N_rots = 2
        if path_zflat:
#             zloc = np.percentile(tt, 10, 0)[2]
            zloc = -close_depth * .1
            c2w_path[:3,3] = c2w_path[:3,3] + zloc * c2w_path[:3,2]
            rads[2] = 0.
            N_rots = 1
            N_views/=2

        # Generate poses for spiral path
        render_poses = render_path_spiral(c2w_path, up, rads, focal, zdelta, zrate=.5, rots=N_rots, N=N_views)
        
        
    render_poses = np.array(render_poses).astype(np.float32)

    c2w = poses_avg(poses)
    print('Data:')
    print(poses.shape, images.shape, bds.shape)
    
    dists = np.sum(np.square(c2w[:3,3] - poses[:,:3,3]), -1)
    i_test = np.argmin(dists)
    print('HOLDOUT view is', i_test)
    
    images = images.astype(np.float32)
    poses = poses.astype(np.float32)

    return images, poses, bds, render_poses, i_test, K, depths


def read_DTU_cam_file(filename):
    with open(filename) as f:
        lines = f.readlines()
        lines = [line.rstrip() for line in lines]
    # extrinsics: line [1,5), 4x4 matrix
    extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
    # intrinsics: line [7-10), 3x3 matrix
    intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))

    return intrinsics, extrinsics #, depth_min, depth_interval


def read_depth(filename):
    # read pfm depth file
    return np.array(read_pfm(filename)[0], dtype=np.float32)

def _load_DTU_data(basedir, factor=None, width=None, height=None, load_imgs=True):
    
    
    img0 = [os.path.join(basedir, 'images', f) for f in sorted(os.listdir(os.path.join(basedir, 'images'))) \
            if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')][0]
    
    scan_filename = os.path.basename(basedir)
    father_dir = os.path.dirname(basedir)
    father_dir = os.path.dirname(father_dir)
    
    all_poses = [] 
    all_depths = []  
    all_masks = [] 
   
    for vid in range(len(img0)):
        depth_filename_hr = os.path.join(father_dir, 'Depths_raw', scan_filename, 'depth_map_{:0>4}.pfm'.format(vid))
        depth_raw = np.array(read_pfm(depth_filename_hr)[0], dtype=np.float32)
        h, w = depth_raw.shape
        depths = cv2.resize(depth_raw, (w // factor, h // factor), interpolation=cv2.INTER_NEAREST)
        all_depths.append(depths) 
        mask_filename_hr = os.path.join(father_dir, 'Depths_raw', scan_filename, 'depth_visual_{:0>4}.png'.format(vid))
        mask_img = read_mask_hr(mask_filename_hr)
        h, w = mask_img.shape
        mask_img = cv2.resize(mask_img, (w // factor, h // factor), interpolation=cv2.INTER_NEAREST)
        all_masks.append(mask_img) 
        
        
        camera_filename = os.path.join(father_dir, 'Cameras', '{:0>8}_cam.txt').format(vid)
        K, extrinsics = read_DTU_cam_file(camera_filename)
        focal = (K[0][0] + K[1][1])/2.0
        poses_temp = np.concatenate([extrinsics[:3, :], np.array([[h], [w], [focal]])], axis=1)
        all_poses.append(poses_temp)  # [3, 5]
    poses = np.stack(all_poses)
    poses = poses.transpose(1, 2, 0)
    depths = np.stack(all_depths)
    depths = depths.transpose(1, 2, 0)
    masks = np.stack(all_masks)
    masks = masks.transpose(1, 2, 0)
    bds = np.array([[2.5], [425.0]],dtype =np.float64)
    bds= bds.repeat(poses.shape[2], axis=1)

    
    sh = imageio.imread(img0).shape # desk2 (4344, 5792, 3)
    
    sfx = '' # (default=8)
    
    if factor is not None:
        sfx = '_{}'.format(factor)
        _minify(basedir, factors=[factor])
        factor = factor
    elif height is not None:
        factor = sh[0] / float(height)
        width = int(sh[1] / factor)
        _minify(basedir, resolutions=[[height, width]])
        sfx = '_{}x{}'.format(width, height)
    elif width is not None:
        factor = sh[1] / float(width)
        height = int(sh[0] / factor)
        _minify(basedir, resolutions=[[height, width]])
        sfx = '_{}x{}'.format(width, height)
    else:
        factor = 1
    
    imgdir = os.path.join(basedir, 'images' + sfx)
    if not os.path.exists(imgdir):
        print( imgdir, 'does not exist, returning' )
        return
    # imgfiles list
    imgfiles = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir)) if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
    if len(imgfiles) < poses.shape[-1]:
        poses = poses[:, :, :len(imgfiles)]    
        depths = depths[:, :, :len(imgfiles)] 
        masks = masks[:, :, :len(imgfiles)]
        bds = bds[:, :len(imgfiles)]
    if poses.shape[-1] != len(imgfiles):
        print( 'Mismatch between imgs {} and poses {} !!!!'.format(len(imgfiles), poses.shape[-1]) )
        return
    
    sh = imageio.imread(imgfiles[0]).shape # image shape, desk2(543, 724, 3)
    poses[:2, 4, :] = np.array(sh[:2]).reshape([2, 1]) # (3, 5, imagenumber)
    poses[2, 4, :] = poses[2, 4, :] * 1./factor
    
    K[0,0] = K[0,0]/factor
    K[1,1] = K[1,1]/factor
    K[0,2] = K[0,2]/factor
    K[1,2] = K[1,2]/factor
    
    if not load_imgs:
        return poses, bds
    
    def imread(f):
        if f.endswith('png'):
            return imageio.imread(f, ignoregamma=True)
        else:
            return imageio.imread(f)
        
    imgs = imgs = [imread(f)[...,:3]/255. for f in imgfiles]
    imgs = np.stack(imgs, -1) #(543, 727,3 ,1 151) 
    
    print('Loaded image data', imgs.shape, poses[:,-1,0])
    return poses, bds, imgs, K, depths, masks

def load_DTU_data(basedir, factor=8, recenter=True, bd_factor=.75, spherify=False, path_zflat=False):
    

    poses, bds, imgs, K, depths, masks = _load_DTU_data(basedir, factor=factor) # factor=4 downsamples original imgs by 4x for DTU
    print('Loaded', basedir, bds.min(), bds.max())
    
    # Correct rotation matrix ordering and move variable dim to axis 0
    poses = np.concatenate([poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1)
    poses = np.moveaxis(poses, -1, 0).astype(np.float32) #（151， 3， 5）
    imgs = np.moveaxis(imgs, -1, 0).astype(np.float32)
    images = imgs #（151， 543，727， 3）
    bds = np.moveaxis(bds, -1, 0).astype(np.float32)
    
    # Rescale if bd_factor is provided
    sc = 1. if bd_factor is None else 1./(bds.min() * bd_factor)
    poses[:,:3,3] *= sc
    bds *= sc
    
    if recenter:
        poses = recenter_poses(poses)
        
    if spherify:
        poses, render_poses, bds = spherify_poses(poses, bds)

    else:
        
        c2w = poses_avg(poses)
        print('recentered', c2w.shape)
        print(c2w[:3,:4])

        ## Get spiral
        # Get average pose
        up = normalize(poses[:, :3, 1].sum(0))

        # Find a reasonable "focus depth" for this dataset
        close_depth, inf_depth = bds.min()*.9, bds.max()*5.
        dt = .75
        mean_dz = 1./(((1.-dt)/close_depth + dt/inf_depth))
        focal = mean_dz

        # Get radii for spiral path
        shrink_factor = .8
        zdelta = close_depth * .2
        tt = poses[:,:3,3] # ptstocam(poses[:3,3,:].T, c2w).T
        rads = np.percentile(np.abs(tt), 90, 0)
        c2w_path = c2w
        N_views = 120
        N_rots = 2
        if path_zflat:
#             zloc = np.percentile(tt, 10, 0)[2]
            zloc = -close_depth * .1
            c2w_path[:3,3] = c2w_path[:3,3] + zloc * c2w_path[:3,2]
            rads[2] = 0.
            N_rots = 1
            N_views/=2

        # Generate poses for spiral path
        render_poses = render_path_spiral(c2w_path, up, rads, focal, zdelta, zrate=.5, rots=N_rots, N=N_views)
        
        
    render_poses = np.array(render_poses).astype(np.float32)

    c2w = poses_avg(poses)
    print('Data:')
    print(poses.shape, images.shape, bds.shape)
    
    dists = np.sum(np.square(c2w[:3,3] - poses[:,:3,3]), -1)
    i_test = np.argmin(dists)
    print('HOLDOUT view is', i_test)
    
    images = images.astype(np.float32)
    poses = poses.astype(np.float32)

    return images, poses, bds, render_poses, i_test, K, depths, masks









