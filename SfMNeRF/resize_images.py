from __future__ import division
import argparse
import scipy.misc
import numpy as np
from glob import glob
import os
import cv2


import random


parser = argparse.ArgumentParser()
parser.add_argument("--dataset_dir", type=str, default='D:/raw_data/scene0653_00/images', help="where the dataset is stored")
parser.add_argument("--dump_root", type=str, default='D:/raw_data/scene0653_00/images_2/', help="Where to dump the data")
parser.add_argument("--factor", type=int, default=2, help="downsample factor for LLFF images")

args = parser.parse_args()

def resize_imgs():

    imgfiles = [os.path.join(args.dataset_dir, f) for f in sorted(os.listdir(args.dataset_dir)) if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
    img_number = len(imgfiles)
    if not os.path.exists(args.dump_root):
        os.mkdir(args.dump_root)
    for k in range(img_number):
        image_name = imgfiles[k]
        src_image = cv2.imread(image_name)  
        resized_image = cv2.resize(src_image, (0, 0), fx=1.0/args.factor, fy=1.0/args.factor)
        save_image_name = args.dump_root + 'image{:0>3d}.png'.format(k)
        cv2.imwrite(save_image_name, resized_image)


resize_imgs()

