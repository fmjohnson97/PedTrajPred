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

CLUSTERS_PER_N = {1:10, 2:20, 3:35, 4:40, 5:40}

def makeTSNELabel(args, prefix):
    global GT_TSNE_VALUES #all raw tsne XY and kmeans cluster labels
    global TSNE_N_CUTOFFS #the labels that belong to each of the N
    global TSNE_BOUNDS #the [max, min] for the raw tsne for each N
    GT_TSNE_VALUES = pd.DataFrame(columns=['tsne_X','tsne_Y','kmeans', 'frames'])
    TSNE_N_CUTOFFS = {}
    TSNE_BOUNDS = {}
    max_label = 0
    for i in range(1,args.maxN+1):
        # breakpoint()
        data = pd.read_csv('data/'+prefix+'_'+str(i)+'thresh_'+str(args.input_window)+'window.csv')
        temp = data.filter(['tsne_X', 'tsne_Y', 'kmeans', 'frames'])
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


def find_nearest(data, coord):
    point =np.argmin(np.sum((data - coord)**2, axis=-1))
    return point


def graphPreds(args):
    dataset = input("Which dataset? (ETH, ETH_Hotel, UCY_Zara1, UCY_Zara2) \n")
    inds = [i for i in range(len(GT_TSNE_VALUES)) if "'"+dataset+"'" in GT_TSNE_VALUES['frames'].iloc[i]]
    npoints = int(input("How many points will you click?"))
    vals = GT_TSNE_VALUES.iloc[inds]

    CLICKS=[]
    for i in range(1,args.maxN+1):
        plotInds = [v for v in range(len(vals)) if vals['kmeans'].iloc[v] in TSNE_N_CUTOFFS[i]]
        plotVals = vals.iloc[plotInds]
        plotVals = plotVals.filter(['tsne_X', 'tsne_Y']).values
        plt.figure()
        plt.scatter(plotVals[:,0], plotVals[:,1])
        plt.title('Ground Truth TSNE for '+str(i)+" People")
        plt.waitforbuttonpress()
        points_clicked = plt.ginput(npoints, show_clicks=True)
        plt.show()
        CLICKS.append(points_clicked)


    breakpoint()
    points=[]
    gtpoints=[]
    for coords in points_clicked:
        points.append(find_nearest(preds, coords))
        gtpoints.append(find_nearest(real,coords))
    #     plt.scatter(preds[points[-1]][0], preds[points[-1]][1], c='k')
    #     plt.scatter(real[gtpoints[-1]][0], real[gtpoints[-1]][1], c='r')
    # plt.show()

        # breakpoint()
        colors=['b','r','g','k','m']
        for i, point in enumerate(points):
            plt.figure()
            # breakpoint()
            # for pos in inputs[point].reshape(-1, 8, 2):
            for j, group in enumerate(inputs[point].reshape(args.num_people, -1, args.window, 2)):
                for pos in group:
                    plt.plot(pos[:,0], pos[:,1], c=colors[j])
                    plt.scatter(pos[0][0], pos[0][1], c=colors[j])
                    # plt.title('Point '+str(i))
            plt.title('Point '+str(i))
        for i, point in enumerate(gtpoints):
            plt.figure()
            # breakpoint()
            # for pos in inputs[point].reshape(-1, 8, 2):
            for j, group in enumerate(inputs[point].reshape(args.num_people, -1, args.window, 2)):
                for pos in group:
                    plt.plot(pos[:, 0], pos[:, 1], c=colors[j])
                    plt.scatter(pos[0][0], pos[0][1], c=colors[j])
                    # plt.title('GT Point '+str(i))
            plt.title('GT Point ' + str(i))
        plt.figure()
        plt.scatter(preds[:, 0], preds[:, 1])
        plt.title(str(args.num_people) + " diffsData TSNE Predictions")
        for i in range(len(points)):
            plt.scatter(preds[points[i]][0], preds[points[i]][1], c='k')
            plt.scatter(real[gtpoints[i]][0], real[gtpoints[i]][1], c='r')
        plt.show()


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
    parser.add_argument('--prefix', default='noNorm_diffsData', type=str )
    args = parser.parse_args()
    return args


args = get_args()
makeTSNELabel(args, args.prefix)
graphPreds(args)
breakpoint()