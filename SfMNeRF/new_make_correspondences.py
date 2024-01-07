from __future__ import division
import argparse
import scipy.misc
import numpy as np
from glob import glob
# from joblib import Parallel, delayed
import os
import cv2
import json
from load_llff import load_llff_data
from shutil import copyfile

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_dir", type=str, default='../data/nerf_llff_data/fortress', help="where the dataset is stored")
parser.add_argument("--factor", type=float, default=32, help="downsample factor for LLFF images") 
parser.add_argument("--testskip", type=int, default=8, 
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')
parser.add_argument("--dataset_type", type=str, default='llff', 
                        help='options: llff / blender / deepvoxels')
parser.add_argument("--spherify", action='store_true', 
                        help='set for spherical 360 scenes')
parser.add_argument("--llffhold", type=int, default=8, 
                        help='will take every 1/N images as LLFF test set, paper uses 8')
args = parser.parse_args()

def cal_sift_descript(img_file):
    sift_kp_des = []
    sift = cv2.xfeatures2d.SIFT_create()
    for i in range(len(img_file)):
       img = cv2.imread(img_file[i])
       kp, des = sift.detectAndCompute(img, None) 
       sift_kp_des.append([kp, des])
    return sift_kp_des, img.shape


def cal_correspondences(src_kp_des, dst1_kp_des, img_shape):
    
    distance_threshold = 0.75  
    max_threshold = 5
    src_kp = src_kp_des[0]
    src_des = src_kp_des[1]
    dst1_kp = dst1_kp_des[0]
    dst1_des = dst1_kp_des[1]


    bf = cv2.BFMatcher()
    matches1 = bf.knnMatch(src_des, dst1_des, k=2)  # src_img, dst1_img
    valid_matches1 = []
    min_dist = 1.0e10
    for m, n in matches1:
        if m.distance > distance_threshold * n.distance: 
            continue
        dist = m.distance
        if dist < min_dist:
            min_dist = dist
    for m, n in matches1:
        if m.distance > distance_threshold * n.distance or m.distance > max_threshold * max(min_dist, 10.0): 
            continue        
        valid_matches1.append(m)            
    
    max_resolution = img_shape[0] if img_shape[0] > img_shape[1] else img_shape[1]
    
    inlier_points = []
    if len(valid_matches1) > 10:
        src_pts = np.float32([src_kp[m.queryIdx].pt for m in valid_matches1]).reshape(-1, 1, 2)  
        dst_pts = np.float32([dst1_kp[m.trainIdx].pt for m in valid_matches1]).reshape(-1, 1, 2)  

        Homograpgy, status = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 0.004 * max_resolution)  
        for i in range(len(status)):
            if status[i] > 0:
                inlier_points.append(valid_matches1[i])
    valid_matches1= inlier_points
    
            

    
    total_matches = []
    for m in valid_matches1:
        total_matches.append([np.float32(src_kp[m.queryIdx].pt), np.float32(dst1_kp[m.trainIdx].pt)])

    return  total_matches
   
    

def make_correspondence_file(imgfiles, i_val, i_train):  
    dump_dir_name = os.path.join(args.dataset_dir, 'new_sift_correspondences/')  
    if not os.path.exists(dump_dir_name):
        os.mkdir(dump_dir_name)
    img_number = len(imgfiles)
    kp_des, img_shape = cal_sift_descript(imgfiles)
    best_matched_frame_idx = []
    for k in range(img_number):
        if not k in i_train:
            continue
        best_matches = []
        best_matched_idx = 0
        for i in range(img_number): # 
                if i != k and i in i_train:                     
                    total_matches = cal_correspondences(kp_des[k], kp_des[i], img_shape)
                    if len(total_matches) > len(best_matches):
                        best_matches =  total_matches
                        best_matched_idx = i           
                    print("k:{}, i:{}".format(k, i))
                    saved_file = dump_dir_name + 'sift_{:0>2d}_{:0>2d}_{:0>2d}.txt'.format(int(args.factor), k, i)
                    with open(saved_file, 'w') as f:
                        for m in total_matches:
                            f.write('%f %f %f %f\n' % (m[0][0]/args.factor, m[0][1]/args.factor, m[1][0]/args.factor, m[1][1]/args.factor)) # m[0].x, m[0].y, m[1].x, m[1].y
        best_matched_frame_idx.append(best_matched_idx)            
        
    saved_file = dump_dir_name + 'best_matched_frame_idx.txt'
    with open(saved_file, 'w') as f:
        for m in best_matched_frame_idx:
            f.write('%d\n' % (m))
    
def make_llff_correspondence_file():  
    i_val, i_train = get_train_and_test_idx() 
    dataset_dir = os.path.join(args.dataset_dir, 'images/')
    if not os.path.exists(dataset_dir):
        os.mkdir(dataset_dir)
    imgfiles = [os.path.join(dataset_dir, f) for f in sorted(os.listdir(dataset_dir)) if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
    make_correspondence_file(imgfiles, i_val, i_train)
    
def get_train_and_test_idx():
    if args.dataset_type == 'llff':
        images, poses, bds, render_poses, i_test = load_llff_data(args.dataset_dir, args.factor,
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
    return i_val, i_train 
 
def make_blender_image_dir():
    splits = ['train', 'test']
    metas = {}
    for s in splits:
        with open(os.path.join(args.dataset_dir, 'transforms_{}.json'.format(s)), 'r') as fp:
            metas[s] = json.load(fp)

    all_fnames = []
    counts = []
    for s in splits:
        meta = metas[s]
        if s=='train' or args.testskip==0:
            skip = 1
        else:
            skip = args.testskip
            
        for frame in meta['frames'][::skip]:
            fname = os.path.join(args.dataset_dir, frame['file_path'] + '.png')
            all_fnames.append(fname)
            
        counts.append(len(all_fnames))
    
    image_dir = os.path.join(args.dataset_dir, 'images/')
    if not os.path.exists(image_dir):
        os.mkdir(image_dir)
    for i in range(counts[1]):
        if i < counts[0]: # image in the train set
            dest_fname = image_dir + 'train_{}.png'.format(i)
        else:
            dest_fname = image_dir + 'val_{}.png'.format((i-counts[0])*args.testskip) 
        copyfile(all_fnames[i], dest_fname)    

def make_blender_correspondence_file():
    make_blender_image_dir()
    make_llff_correspondence_file()

make_llff_correspondence_file()
# make_blender_correspondence_file()