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
from trajAugmentations import TrajAugs
from PIL import Image
from OpenTraj.utils import world2image
from simpleNetworkPerCluster import SimplestNet

def get_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--num_clusters', default=50, type=int, help='number of clusters for kmeans')
    parser.add_argument('--input_window', default=8, type=int, help='number of frames for the input data')
    parser.add_argument('--output_window', default=12, type=int, help='number of frames for the output data')
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--xgrid_num', default=24, type=int)
    parser.add_argument('--ygrid_num', default=32, type=int)
    parser.add_argument('--maxN', default=3, type=int)
    parser.add_argument('--social_thresh', default=0.9, type=float)#0.9 for trajData
    args = parser.parse_args()
    return args

def makeTSNELabel(maxN, input_window):
    # global GT_TSNE_VALUES
    global TSNE_N_CUTOFFS
    global TSNE_BOUNDS
    # GT_TSNE_VALUES = pd.DataFrame(columns=['tsne_X','tsne_Y','kmeans'])
    TSNE_N_CUTOFFS = {}
    TSNE_BOUNDS = {}
    max_label = 0
    for i in range(1,maxN+1):
        # breakpoint()
        data = pd.read_csv('allDiffsData_RotAug_'+str(i)+'thresh_'+str(input_window)+'window.csv')
        temp = data.filter(['tsne_X', 'tsne_Y', 'newClusters'])
        class_bounds =[]
        for b in range(int(temp['newClusters'].max())+1):
            bounds=temp[temp['newClusters']==b]
            coords = np.array([[bounds['tsne_X'].max(),bounds['tsne_Y'].max()],[bounds['tsne_X'].min(),bounds['tsne_Y'].min()]])
            sum_x = np.sum(coords[:, 0])
            sum_y = np.sum(coords[:, 1])
            class_bounds.append([sum_x / 2, sum_y / 2])

        # TSNE_BOUNDS[i]=[[temp['tsne_X'].max(),temp['tsne_Y'].max()],[temp['tsne_X'].min(),temp['tsne_Y'].min()]]
        TSNE_BOUNDS[i]=class_bounds
        temp['newClusters']=temp['newClusters']+max_label
        # GT_TSNE_VALUES = GT_TSNE_VALUES.append(temp)
        max_label = temp['newClusters'].max()+1
        temp = temp['newClusters'].unique()
        temp.sort()
        TSNE_N_CUTOFFS[i] = temp

def get_spaced_inds(n, max_value, row_len):
    interval = int(max_value / n)
    inds = [I for I in range(0, max_value, interval)]
    return [[i//row_len,i%row_len] for i in inds]


def getNewGroups(pos, diffs, args):
    # hard coding grid to be 3:4 (rows:columns) since that's aspect ratio of the images

    groupDict=defaultdict(int)
    # breakpoint()
    for i,p in enumerate(pos): # blue, orange, green, red, purple, brown
        dists = torch.sum(torch.sum((pos-p)**2,axis=-1)**.5, axis=-1)
        # print(dists)
        # breakpoint()
        inds=np.where(dists<args.social_thresh)
        for ind in inds:
            if len(ind)<=args.maxN:
                groupDict[tuple(ind)]+=1
        # minDists = dists #np.sum(np.sum(dists ** 2, axis=-1), axis=-1)
        # if minDists.shape[0] == args.maxN:
        #     # breakpoint()
        #     closest = [j for j in range(args.maxN)]
        # else:
        #     idx = np.argpartition(minDists, args.maxN)
        #     closest = [ind.item() for ind in idx[:args.maxN]]
        #     closest.sort()
        # groupDict[tuple(closest)]+=1

    groups=list(groupDict.keys())
    if len(groups)<1:
        for i, p in enumerate(pos):
            minDists = dists  # np.sum(np.sum(dists ** 2, axis=-1), axis=-1)
            if minDists.shape[0] == args.maxN:
                # breakpoint()
                closest = [j for j in range(args.maxN)]
            else:
                idx = np.argpartition(minDists, args.maxN)
                closest = [ind.item() for ind in idx[:args.maxN]]
                closest.sort()
            groupDict[tuple(closest)]+=1

    groups = list(groupDict.keys())

    # breakpoint()
    remove=[]
    for i,g in enumerate(groups):
        if sum([all([x in temp for x in g]) for temp in groups])>1:
            remove.append(i)

    # breakpoint()
    remove.reverse()
    for r in remove:
        groups.pop(r)
    if len(groups)<1:
        breakpoint()

    new_pos=[]
    new_diffs=[]
    new_allDiffs=[]
    for g in groups:
        new_pos.append(pos[np.array(list(g))])
        new_diffs.append(diffs[np.array(list(g))])
        allDiffs = []
        for i in range(new_pos[-1].shape[0]):
            # breakpoint()
            temp = np.concatenate((new_pos[-1][:i,:args.input_window,:], new_pos[-1][i+1:,:args.input_window,:]), axis=0)
            # hold = np.sum(new_pos[-1] ** 2, axis=1)
            # heading = np.arccos(np.sum(x[:-1, :] * x[1:, :], axis=1) / (hold[:-1] * hold[1:]) ** .5)
            if len(temp) > 0:
                temp = new_pos[-1][i,:args.input_window,:][1:] - temp[:, :-1, :]
                # breakpoint()
                allDiffs.append(
                    np.hstack(np.concatenate((np.diff(new_pos[-1][i,:args.input_window,:], axis=0).reshape(1, -1, 2) * 15, temp), axis=0)))
            else:
                # breakpoint()
                allDiffs.append(np.diff(new_pos[-1][i,:args.input_window,:], axis=0))
        new_allDiffs.append(torch.tensor(np.stack(allDiffs)).flatten())

    # breakpoint()
    return new_pos, new_allDiffs, new_diffs, groups


if __name__ == '__main__':
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    makeTSNELabel(args.maxN, args.input_window)
    nets=[]
    N=np.array(range(1,args.maxN+1))
    for i in N:
        net = SimpleRegNetwork(i*i * (args.input_window-1) * 2).eval()
        net.load_state_dict(torch.load('/Users/faith_johnson/GitRepos/PedTrajPred/simpleRegNet_allDiffsData_RotAug_'+str(i)+'people_'+str(args.input_window)+'window.pt'))
        nets.append(net)

    tsne_preds=[]
    inputs=[]
    people_per_frame=[]
    frames=[]
    traj_preds = []
    # colorMap=['maroon','r','tab:orange','y','lime','g','b','indigo','tab:purple','m']

    for name in ['ETH', 'ETH_Hotel', 'UCY_Zara1', 'UCY_Zara2']:
        dataset = PlainTrajData(name, input_window=args.input_window, output_window=args.output_window, maxN=args.maxN)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        print("Processing",name)
        tsne_preds=[]
        for data in tqdm(loader):
            if data['pos'].nelement() > 0:
                # breakpoint()
                if data['diffs'].shape[-3]<args.maxN:
                    # breakpoint()
                    net=nets[data['diffs'].shape[-3]-1]
                    pos = data['pos'][0][:,:args.input_window,:].float()#.flatten()
                    target = data['pos'][0][:,args.input_window:,:]
                    allDiffs = []
                    for i in range(pos.shape[0]):
                        temp = np.concatenate((pos[:i], pos[i + 1:]), axis=0)
                        if len(temp) > 0:
                            temp = pos[i][1:] - temp[:, -1, :]
                            allDiffs.append(np.concatenate((np.diff(pos[i], axis=0) * 15, temp),axis=-1))
                        else:
                            allDiffs.append(np.diff(pos[i], axis=0))
                    # breakpoint()
                    pos = torch.tensor(np.stack(allDiffs)).flatten()
                    people_per_frame.append(data['diffs'].shape[-3])
                    with torch.no_grad():
                        output = net(pos)
                        c = np.argmin(np.sum((np.array(TSNE_BOUNDS[people_per_frame[-1]]) - output.numpy()) ** 2, axis=-1))
                        trajnet = SimplestNet(args).to(device)
                        trajnet.load_state_dict(torch.load('simpleNetTSNE_allDiffsData_RotAug_' + str(people_per_frame[-1]) + '_cluster' + str(c) + '.pt', map_location=device))
                        traj, latent = trajnet(data['pos'][0, :, :args.input_window, :].reshape(people_per_frame[-1], -1).float())
                    tsne_preds.append([output.detach()])
                    inputs.append(data['pos'].flatten())
                    traj_preds.append(traj.detach())
                    frames.append(data['frames'])
                else:
                    # breakpoint()
                    pos,allDiffs, diffs, groups = getNewGroups(data['pos'][0], data['diffs'][0], args)
                    people=[]
                    preds=[]
                    ins=[]
                    temp_frame=[]
                    temp_trajs=[]
                    for i, p in enumerate(allDiffs):
                        net = nets[pos[i].shape[-3]-1]
                        people.append(pos[i].shape[-3])
                        with torch.no_grad():
                            # breakpoint()
                            output = net(p.flatten().float())
                            c = np.argmin(np.sum((np.array(TSNE_BOUNDS[people[-1]]) - output.numpy()) ** 2, axis=-1))
                            trajnet = SimplestNet(args).to(device)
                            trajnet.load_state_dict(torch.load('simpleNetTSNE_allDiffsData_RotAug_' + str(people[-1]) + '_cluster' + str(c) + '.pt',map_location=device))
                            traj, latent = trajnet(pos[i][:, :args.input_window, :].reshape(people[-1], -1).float())
                        ins.append(pos[i])
                        temp_trajs.append(traj.detach())
                        preds.append(output.detach())
                        temp_frame.append(data['frames'])
                    output = np.max(np.stack(preds), axis=0)
                    tsne_preds.append(preds)
                    inputs.append(ins)
                    people_per_frame.append(people)
                    frames.append(temp_frame)
                    traj_preds.append(temp_trajs)

        # breakpoint()
        print('Plotting Trajectories')
        augs=TrajAugs()
        for i, tpred in tqdm(enumerate(tsne_preds)):
            for j, t in enumerate(tpred):
                n=people_per_frame[i]
                pos = inputs[i]
                frame = frames[i]
                traj = traj_preds[i]
                if type(n) is list:
                    n=n[j]
                    pos = pos[j]
                    frame=frame[j]
                    traj = traj[j]
                # breakpoint()
                if n>1:
                    dataset = PlainTrajData(frame[0][0], input_window=args.input_window, output_window=args.output_window,maxN=args.maxN)
                    ims=dataset.getImages({'frames':torch.stack(frame[1:]).flatten().tolist()})
                    plt.figure()
                    im = Image.fromarray(ims[0])
                    im = im.rotate(-90, expand=True)
                    plt.imshow(im)#, origin='lower')
                    # pos=augs.rotate(pos.reshape(n,args.input_window,2),0)
                    pos = pos.reshape(n,-1,2) * (dataset.max - dataset.min) + dataset.min
                    for p in pos:
                        p = world2image(p,dataset.H)
                        plt.plot(p[:args.input_window,0], p[:args.input_window,1], c='b')
                        plt.plot(p[args.input_window:, 0], p[args.input_window:, 1], c='g')
                    traj = traj.reshape(n,-1,2) * (dataset.max - dataset.min) + dataset.min
                    for t in traj:
                        t=world2image(t,dataset.H)
                        plt.plot(t[:, 0], t[:, 1], c='tab:orange')
                    plt.figure()
                    plt.imshow(im, origin='lower')
                    plt.figure()
                    plt.axis([0, 480, 0 ,640])
                    for p in pos:
                        p = world2image(p, dataset.H)
                        plt.plot(p[:args.input_window, 0], p[:args.input_window, 1], c='b')
                        plt.plot(p[args.input_window:, 0], p[args.input_window:, 1], c='g')
                    for t in traj:
                        t=world2image(t,dataset.H)
                        plt.plot(t[:, 0], t[:, 1], c='tab:orange')
                    plt.show()

                # breakpoint()
        background=Image.open(name+'.png')
        plt.show()



