# -*- coding: utf-8 -*-
from __future__ import division
import argparse
import numpy as np
import os
import cv2

from load_llff import load_llff_data
from run_nerf_helpers import *

parser = argparse.ArgumentParser()
parser.add_argument("--datadir", type=str, default='../NoExtNeRF/data/ScanNet/scene0653_00', help="where the dataset is stored")
parser.add_argument("--factor", type=int, default=8, help="downsample factor for LLFF images")  #DTU-4, ScanNet-2, LLFF-8

parser.add_argument("--spherify", action='store_true', 
                        help='set for spherical 360 scenes') 
parser.add_argument("--llffhold", type=int, default=8, 
                        help='will take every 1/N images as LLFF test set, paper uses 8')
parser.add_argument("--dataset_type", type=str, default='llff', 
                        help='options: llff / blender / deepvoxels')
args = parser.parse_args()

def skew(x):
    """ Create a skew symmetric matrix *A* from a 3d vector *x*.
        Property: np.cross(A, v) == np.dot(x, v)
    :param x: 3d vector
    :returns: 3 x 3 skew symmetric matrix from *x*
    """
    return np.array([
        [0, -x[2], x[1]],
        [x[2], 0, -x[0]],
        [-x[1], x[0], 0]
    ])

def fundamental_matrix_from_rt(rot_mat, translation, intrinsics):
    translation_ssm = skew(translation)
    essential_mat = np.matmul(translation_ssm, rot_mat) 
    intrinsics_inv = np.linalg.inv (intrinsics)
    fundamental_mat = np.matmul(intrinsics_inv.T, essential_mat)
    fundamental_mat = np.matmul(fundamental_mat, intrinsics_inv)
    return fundamental_mat

def fundamental_matrix_from_E(essential_mat, intrinsics):
    intrinsics_inv = np.linalg.inv (intrinsics)
    fundamental_mat = np.matmul(intrinsics_inv.T, essential_mat)
    fundamental_mat = np.matmul(fundamental_mat, intrinsics_inv)
    return fundamental_mat

def cal_transfomation_matrix(ref_pose, local_pose):
    """ Create the transformation matrix between ref_pose and local_pose
    :param ref_pose, target_pose: [3, 4]
    :returns: 3 x 3 rotation, 3 x 1 translation
    """
    ref_r = ref_pose[:, :3] # [3, 3]
    ref_t = np.expand_dims(ref_pose[:, 3], -1) # [3, 1]     
    ref_r_inv = np.linalg.inv (ref_r) 

    loc_r = local_pose[:, :3] # [3, 3]
    loc_t = np.expand_dims(local_pose[:, 3], -1) # [3, 1]
    trans_r = np.matmul(ref_r_inv, loc_r) # [3, 3]
    trans_t = np.matmul(ref_r_inv, loc_t-ref_t) # [3, 1]

    return trans_r, trans_t

def save_fundemental_matrix(file_name, f_matrix):
    with open(file_name, 'w') as f:
        f.write('%.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f\n' % (f_matrix[0][0], f_matrix[0][1], f_matrix[0][2], f_matrix[1][0], f_matrix[1][1], f_matrix[1][2], f_matrix[2][0], f_matrix[2][1], f_matrix[2][2])) 

def load_sift_pairs(i_train):
    sift = {}
    siftdir =  os.path.join(args.datadir, 'new_sift_correspondences/')
    if not os.path.exists(siftdir):
        print("sift_correspondences not exist!")
        exit(0)
    for i in i_train:
        for j in i_train:
            if i != j:
                sift_file = siftdir + 'sift_{:0>2d}_{:0>2d}_{:0>2d}.txt'.format(int(args.factor), i, j)                
                sift_correspondences = load_sift_correspondences(sift_file) # [n, 4]
                sift_correspondences = np.array(sift_correspondences)
                sift['{:03d}_{:03d}'.format(i,j)] = sift_correspondences
                    
    return sift

def make_fundemental_matrix_v2(sift, K, img_number):
    dir_name = os.path.join(args.datadir, 'fundemental_matrix/')
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    for i in range(img_number):
        for j in range(img_number):
            if i != j:
                if i < j:
                    temp_i = i
                    temp_j = j
                else:
                    temp_i = j
                    temp_j = i
                
                sift_number = 0
                for k  in range(img_number):
                    if k != i and k!=j:                                 
                        sift_correspondences = sift['{:03d}_{:03d}_{:03d}'.format(k,temp_i,temp_j)]
                        sift_correspondences = np.reshape(sift_correspondences, [-1, 3, 2]) # [n, 3, 2]
                        if len(sift_correspondences) > sift_number:
                            sift_with_large_number = sift_correspondences
                            sift_number = len(sift_correspondences)
                if i < j:
                    pts1 = sift_with_large_number[:, 1, :]  # [n, 2]
                    pts2 = sift_with_large_number[:, 2, :]  # [n, 2]
                else:
                    pts1 = sift_with_large_number[:, 2, :]  # [n, 2]
                    pts2 = sift_with_large_number[:, 1, :]  # [n, 2]
                E, mask = cv2.findEssentialMat(pts1, pts2, cameraMatrix=K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
                fundamental_mat = fundamental_matrix_from_E(E, K)
                saved_file = dir_name + 'fundemental_matrix_v2_{:0>2d}_{:0>2d}_{:0>2d}.txt'.format((args.factor), i, j)
                save_fundemental_matrix(saved_file, fundamental_mat)

def make_fundemental_matrix(poses):
    hwf = poses[0,:3,-1]  # [b, 3], intronics
    poses = poses[:,:3,:4] # [b, 3, 4], extronics
       
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    K = np.array([
        [focal, 0, 0.5*W],
        [0, focal, 0.5*H],
        [0, 0, 1]
    ])
    
    dir_name = os.path.join(args.datadir, 'fundemental_matrix/')
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
        
    for i in range(len(poses)):
        for j in range(len(poses)):
            if i != j:
                r, t = cal_transfomation_matrix(poses[i], poses[j])
                fundamental_mat = fundamental_matrix_from_rt(r, t, K)
                saved_file = dir_name + 'fundemental_matrix_{:0>2d}_{:0>2d}_{:0>2d}.txt'.format((int(args.factor)), i, j)
                save_fundemental_matrix(saved_file, fundamental_mat)

def make_mask(sift, img, i_train):
    dir_name = os.path.join(args.datadir, 'new_mask/')
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    height, width, channel = img.shape
    mask_rect = []
    for i in i_train:
        for j in i_train:
            if i!=j:
                sift_correspondences = sift['{:03d}_{:03d}'.format(i,j)]
                if len(sift_correspondences) > 0:
                    sift_correspondences = sift_correspondences[:, :2] # select the sift of image k, and image k as the reference image
                    sift_correspondences = np.expand_dims(sift_correspondences, axis=1) # [n, 1, 2]
                    sift_correspondences = sift_correspondences.astype(int)
                    rect = cv2.boundingRect(sift_correspondences)  # [x,y, w,h] 
                    mask_img = np.zeros((height, width, channel), dtype=img.dtype)
                    temp = mask_img[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]
                    temp2 = np.zeros((temp.shape[0], temp.shape[1], channel), dtype=img.dtype) + 255
                    mask_img[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]] = temp2
                else:
                    mask_img = np.zeros((height, width, channel), dtype=img.dtype)
                    rect = np.zeros(4, dtype=np.int32)
                mask_file = dir_name + 'mask_{:0>2d}_{:0>2d}_{:0>2d}.png'.format(int(args.factor), i, j) 
                cv2.imwrite(mask_file, mask_img)
                mask_rect.append([[i,j], rect])
    
    mask_rect_file_name = dir_name + 'mask_{:0>2d}_rect.txt'.format(int(args.factor))
    with open(mask_rect_file_name, 'w') as f:
        for i in range(len(mask_rect)):
            f.write('%d %d %d %d %d %d\n' % (mask_rect[i][0][0], mask_rect[i][0][1], mask_rect[i][1][0], mask_rect[i][1][1], mask_rect[i][1][2], mask_rect[i][1][3])) # [k,i, x,y,w,h]

def val_fundemental_matrix(sift, img_number):
    dir_name = os.path.join(args.datadir, 'fundemental_matrix/')
    for j in range(1, img_number):
        file_name = dir_name + 'fundemental_matrix_{:0>2d}_{:0>2d}_{:0>2d}.txt'.format(int(args.factor), 0, j)
        fundemental_matrix = load_fundemental_matrix(file_name)
        for k in range(1, img_number):
            if k != j:
                sift_correspondences = sift['{:03d}_{:03d}_{:03d}'.format(k,0,j)]
                sift_correspondences = np.reshape(sift_correspondences, [-1, 3, 2]) # [n, 3, 2]
                sift_np_one = np.ones((len(sift_correspondences), 1), dtype=sift_correspondences.dtype)
                p = sift_correspondences[:, 1, :]  # [n, 2]
                p = np.concatenate([p, sift_np_one], 1) # [n, 3]
                p = np.expand_dims(p, axis=-1)  # [n, 3, 1]
                p = p.transpose((0, 2, 1))  # [n, 1, 3]
                q = sift_correspondences[:, 2, :]  # [n, 2]
                q = np.concatenate([q, sift_np_one], 1) # [n, 3]
                q = np.expand_dims(q, axis=-1)  # [n, 3, 1]                
                epipolar_error = np.matmul(p, fundemental_matrix)
                epipolar_error = np.matmul(epipolar_error, q)
                print(np.mean(np.abs(epipolar_error))) 
                break
        
def cal_epipolar_loss(p, q, fundemental_matrix):
    sift_np_one = torch.ones((len(p), 1), dtype=p.dtype)
    p = torch.cat([p, sift_np_one], 1) # [n, 3]
    p = p.unsqueeze(-1) # np.expand_dims(p, axis=-1)  # [n, 3, 1]
    p = p.permute((0, 2, 1))  # [n, 1, 3]
    q = torch.cat([q, sift_np_one], 1) # [n, 3]
    q = q.unsqueeze(-1) # np.expand_dims(q, axis=-1)  # [n, 3, 1]                
    epipolar_error = torch.matmul(p, fundemental_matrix)
    epipolar_error = torch.matmul(epipolar_error, q)
    epipolar_error = torch.mean(torch.abs(epipolar_error))
    return  epipolar_error           
    

def make_fundemental_matrix_and_mask(): 
    images, poses, bds, render_poses, i_test = load_llff_data(args.datadir, args.factor,
                                                                  recenter=True, bd_factor=.75,
                                                                  spherify=args.spherify) 
    
    if not isinstance(i_test, list):
            i_test = [i_test] 
    if args.llffhold > 0:
        print('Auto LLFF holdout,', args.llffhold)
        i_test = np.arange(images.shape[0])[::args.llffhold]
    i_val = i_test
    i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                    (i not in i_test and i not in i_val)]) 
    
    hwf = poses[0,:3,-1]  # [b, 3], intronics
    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    K = np.array([
        [focal, 0, 0.5*W],
        [0, focal, 0.5*H],
        [0, 0, 1]
    ])
    make_fundemental_matrix(poses)
    siftdir =  os.path.join(args.datadir, 'new_sift_correspondences/')
    sift = load_sift_pairs(i_train)    
    make_mask(sift, images[0], i_train)

make_fundemental_matrix_and_mask()      