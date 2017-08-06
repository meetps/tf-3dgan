#!/usr/bin/env python
import utils.dataIO as d
import numpy as np
import os
import sys
import argparse

def ProgParser():
    parser = argparse.ArgumentParser(
            formatter_class=argparse.RawTextHelpFormatter, 
            description="[Usage Example]\n"
                        "    python %(prog)s train_sample/biasfreee_3000.pkl --idx 0 --thres 0.7"
            )
    parser.add_argument('path', metavar='PATH', type=str,
            help='sample path (.pkl)')
    parser.add_argument('--idx', '-idx', type=int, default=0,
            help='index of voxels (0~31) [default: %(default)s]')
    parser.add_argument('--thres', '-thres', type=float, default=0.5,
            help='voxel threshold to show (0 < threshold < 1) [default: %(default)s]')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = ProgParser()
    path = args.path
    index = args.idx
    threshold = args.thres

    obj = np.load(path, encoding='latin1')
    voxels = np.squeeze(obj[index] > threshold)
    d.plotMeshFromVoxels(voxels, threshold=threshold)
