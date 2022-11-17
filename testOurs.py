import random
import numpy as np
import torch
import torch.nn as nn
import argparse

from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from collections import defaultdict

from plainTrajData import PlainTrajData
from simpleTSNEPredict import SimpleRegNetwork
from simpleNetworkPerCluster import SimplestNet

CLUSTER_NUM=[0,10,29,33]

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_window', default=8, type=int, help='number of frames for the input data')
    parser.add_argument('--output_window', default=12, type=int, help='number of frames for the output data')
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--epochs', default=2500, type=int)
    parser.add_argument('--maxN', default=3, type=int)
    parser.add_argument('--lr', default=1e-2, type=float)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--social_thresh', default=1.2, type=float)
    parser.add_argument('--prefix', default='', type=str)
    parser.add_argument('--name', default='', type=str)

    args = parser.parse_args()
    return args

def graph(args, inputs, predictions=None, name=None, index=None):
    plt.figure()
    plt.axis([0, 1, 0, 1])
    if predictions is None:
        plt.scatter(inputs[:,0], inputs[:,1])
    else:
        # breakpoint()
        if index is None:
            ind = random.choice(list(range(len(inputs))))
        else:
            ind = index
        # for pos in inputs[ind].reshape(-1, args.input_window,2):
        for pos in inputs[ind]:#.reshape(-1, args.obs_length+args.pred_length, 2):
            plt.scatter(pos[:args.obs_length,0], pos[:args.obs_length,1], c='b')
            plt.scatter(pos[args.obs_length:,0], pos[args.obs_length:,1], c='g')
        # for pos in predictions[ind].reshape(-1, args.input_window,2):
        for pos in predictions[ind]:#.reshape(-1, args.obs_length+args.pred_length-1, 2):
        # for pos in predictions[ind].reshape(-1, 2):
            plt.plot(pos[-args.pred_length:,0], pos[-args.pred_length:,1], c='tab:orange')
            plt.scatter(pos[-args.pred_length][0], pos[-args.pred_length][1], c='tab:orange')
            # plt.scatter(pos[0], pos[1], c='tab:orange')

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
            temp = np.concatenate((new_pos[-1][:i], new_pos[-1][i + 1:]), axis=0)
            # hold = np.sum(new_pos[-1] ** 2, axis=1)
            # heading = np.arccos(np.sum(x[:-1, :] * x[1:, :], axis=1) / (hold[:-1] * hold[1:]) ** .5)
            if len(temp) > 0:
                temp = new_pos[-1][i][1:] - temp[:, :-1, :]
                # breakpoint()
                allDiffs.append(
                    np.hstack(np.concatenate((np.diff(new_pos[-1][i], axis=0).reshape(1, -1, 2) * 15, temp), axis=0)))
            else:
                # breakpoint()
                allDiffs.append(np.diff(new_pos[-1][i], axis=0))
        new_allDiffs.append(torch.tensor(np.stack(allDiffs)).flatten())

    # breakpoint()
    return new_pos, new_allDiffs, new_diffs, groups

@torch.no_grad()
def test(args, tsne_models, traj_models):
    loss_func = nn.MSELoss()
    avgLoss=[]
    avgUnNormedLoss=[]
    fde=[]
    unNormedFDE=[]
    preds = []
    inputs = []
    for name in [args.name]:#, 'ETH_Hotel', 'UCY_Zara1', 'UCY_Zara2']:
        dataset = PlainTrajData(name, input_window=args.input_window, output_window=args.output_window, maxN=5, split=None)
        loader = DataLoader(dataset, batch_size=1, shuffle=False)
        for data in loader:
            if data['pos'].nelement() > 0:
                if data['diffs'].shape[-3] < 3:
                    breakpoint()
                    scene = data['pos'][0].transpose(0,1).float()
                    outputs= model(scene[:args.input_window])
                    loss = loss_func(outputs, scene[1:])
                    avgLoss.append(loss.item())
                    fde.append(loss_func(outputs[-1], scene[-1]).item())
                    unNormedOutput = outputs * (dataset.max-dataset.min) + dataset.min #(np.array(positions)-self.min)/(self.max-self.min)
                    unNormedScene = scene[1:] * (dataset.max - dataset.min) + dataset.min
                    unNormedLoss = loss_func(unNormedOutput, unNormedScene)
                    avgUnNormedLoss.append(unNormedLoss.item())
                    unNormedFDE.append(loss_func(unNormedOutput[-1], unNormedScene[-1]).item())
                    preds.append(torch.transpose(outputs, 0,1).numpy())
                    inputs.append(data['pos'][0].float())
                else:
                    breakpoint()
                    pos, groups = getNewGroups(data['pos'][0], data['diffs'][0], args)
                    for i, p in enumerate(pos):
                        scene = p.transpose(0, 1).float().to(device)
                        outputs = model(scene[:args.obs_length], None,torch.tensor([0, scene.shape[0]]),scene[args.obs_length:])
                        loss = loss_func(outputs, scene[1:])
                        avgLoss.append(loss.item())
                        fde.append(loss_func(outputs[-1], scene[-1]).item())
                        unNormedOutput = outputs * (dataset.max - dataset.min) + dataset.min  # (np.array(positions)-self.min)/(self.max-self.min)
                        unNormedScene = scene[1:] * (dataset.max - dataset.min) + dataset.min
                        unNormedLoss = loss_func(unNormedOutput, unNormedScene)
                        avgUnNormedLoss.append(unNormedLoss.item())
                        unNormedFDE.append(loss_func(unNormedOutput[-1], unNormedScene[-1]).item())
                        preds.append(torch.transpose(outputs, 0, 1).numpy())
                        inputs.append(p.float())

    print('Avg Test Loss =',np.mean(avgLoss))
    print('FDE = ',np.mean(fde))
    print('UnNormed Avg Test Loss =', np.mean(avgUnNormedLoss))
    print('UnNormed FDE = ',np.mean(unNormedFDE))
    return preds, inputs

'''             ##########################################              '''
args = get_args()
tsne_nets=[]
traj_nets=[[],[],[]]
N=np.array(range(1,args.maxN+1))

total_parameters = 0
for i in N:
    net = SimpleRegNetwork(i*i * (args.input_window-1) * 2).eval()
    net.load_state_dict(torch.load('/Users/faith_johnson/GitRepos/PedTrajPred/'+args.prefix+'simpleRegNet_allDiffsData_RotAug_'+str(i)+'people_'+str(args.input_window)+'window.pt'))
    net.eval()
    tsne_nets.append(net)
    total_parameters+=sum(p.numel() for p in net.parameters() if p.requires_grad)
    print('Parameters TSNE Network',i,':',sum(p.numel() for p in net.parameters() if p.requires_grad))

for n in range(1,args.maxN+1):
    for c in range(CLUSTER_NUM[n]):
        try:
            net = SimplestNet(args)
            net.load_state_dict(torch.load(args.prefix+'simpleNetTSNE_allDiffsData_RotAug_' + str(n) + '_cluster' + str(c) + '.pt',map_location=torch.device('cpu')))
            net.eval()
            traj_nets[n-1].append(net)
            total_parameters += sum(p.numel() for p in net.parameters() if p.requires_grad)
            print('Parameters',n,' Traj Network', c, ':', sum(p.numel() for p in net.parameters() if p.requires_grad))
        except:
            traj_nets[n-1].append(None)

print('Total Parameters:',total_parameters)
test(args, tsne_nets, traj_nets)