import torch
from simpleTSNEPredict import SimpleRegNetwork
import argparse
from plainTrajData import PlainTrajData
from torch.utils.data import DataLoader
import random
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
from collections import defaultdict
import pandas as pd

MASTER_COLOR_LIST=[[[255,0,0], [255,143,0], [110,7,7], [125,70,0]],
                   [[255,151,0], [243,255,0], [92,55,0], [98,103,0]],
                   [[255,255,0], [71,255,0], [88,88,0], [17,60,0]],
                   [[0,255,91], [0,247,255], [0,55,20], [0,42,43]],
                   [[0,84,255], [130,0,255], [0,28,85], [32,0,62]]]

def makeTSNELabel(args, prefix):
    global GT_TSNE_VALUES #all raw tsne XY and kmeans cluster labels
    global TSNE_N_CUTOFFS #the labels that belong to each of the N
    global TSNE_BOUNDS #the [max, min] for the raw tsne for each N
    GT_TSNE_VALUES = pd.DataFrame(columns=['tsne_X','tsne_Y','kmeans'])
    TSNE_N_CUTOFFS = {}
    TSNE_BOUNDS = {}
    max_label = 0
    for i in range(1,args.maxN+1):
        data = pd.read_csv('data/'+prefix+'_'+str(i)+'thresh_'+str(args.input_window)+'window.csv')
        temp = data.filter(['tsne_X', 'tsne_Y', 'kmeans'])
        TSNE_BOUNDS[i]=[[temp['tsne_X'].max(),temp['tsne_Y'].max()],[temp['tsne_X'].min(),temp['tsne_Y'].min()]]
        temp['kmeans']=temp['kmeans']+max_label
        GT_TSNE_VALUES = GT_TSNE_VALUES.append(temp)
        max_label = temp['kmeans'].max()+1
        temp = temp['kmeans'].unique()
        temp.sort()
        TSNE_N_CUTOFFS[i] = temp


def getTSNELabel(tsne):
    min_coord = np.argmin(np.sum((GT_TSNE_VALUES.filter(['tsne_X', 'tsne_Y']).values - tsne.numpy())**2, axis=1))
    label = GT_TSNE_VALUES['kmeans'].iloc[min_coord]
    return [[label]] # needs 2 dims to pass to the cvae

def get_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--num_clusters', default=50, type=int, help='number of clusters for kmeans')
    parser.add_argument('--input_window', default=8, type=int, help='number of frames for the input data')
    parser.add_argument('--output_window', default=0, type=int, help='number of frames for the output data')
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--xgrid_num', default=10, type=int)
    parser.add_argument('--ygrid_num', default=10, type=int)
    parser.add_argument('--maxN', default=5, type=int)
    parser.add_argument('--social_thresh', default=0.2, type=float)#0.9 for trajData
    parser.add_argument('--prefix', default='diffsData', type=str )
    args = parser.parse_args()
    return args


args = get_args()
makeTSNELabel(args, args.prefix)
breakpoint()