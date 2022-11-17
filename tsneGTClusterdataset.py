import random

from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd
import numpy as np
from plainTrajData import PlainTrajData
from sameSizeData import SameSizeData
from collections import defaultdict
from simpleTSNEPredict import SimpleRegNetwork


class TSNEClusterGT(Dataset):
    def __init__(self, N, cluster_num, path, split='train'):
        self.data=pd.read_csv(path)
        # self.data=pd.read_csv('just_Zara1_GT.csv')
        #'tsne_X', 'tsne_Y', 'N', 'cluster', 'pos'
        # breakpoint()
        self.data = self.data.iloc[(self.data['N'] == N).values]
        self.data = self.data.iloc[(self.data['cluster']==cluster_num).values]
        self.cluster_size=[0,10,29,33]
        self.N=N
        self.cluster_num=cluster_num
        if split=='test':
            self.data = self.data.iloc[int(len(self.data) * .5):]
        elif split=='final_test':
            print(len(self.data))
        else:
            if split=='train':
                self.data=self.data.iloc[:int(len(self.data)*.5)]
            print(len(self.data))
            self.augmentToIncreaseData()
            print(len(self.data))


    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        # breakpoint()
        _, tsneX, tsneY, N, cluster, pos, dataset= self.data.iloc[item].tolist()
        pos = pos[1:-1].strip().split('\n')
        pos = [x.strip().split(' ') for x in pos]
        pos = np.hstack(pos)
        pos = [float(x.strip()) for x in pos if len(x.strip()) > 0]

        return torch.tensor(pos).float(), torch.tensor([tsneX, tsneY]).float()

    def augmentToIncreaseData(self):
        extra_data=[]
        temp_data=self.data.values
        random.shuffle(temp_data)
        for row in temp_data:
            if len(extra_data)>2000:
                break
            _, tsneX, tsneY, N, cluster, pos, dataset = row.tolist()
            pos = pos[1:-1].strip().split('\n')
            pos = [x.strip().split(' ') for x in pos]
            pos = np.hstack(pos)
            pos = np.array([float(x.strip()) for x in pos if len(x.strip()) > 0]).reshape(int(N),-1,2)
            min_vals=np.eye(2)*np.min(np.min(pos,axis=1),axis=0)
            max_vals=np.eye(2)*(1-np.max(np.max(pos,axis=1),axis=0))
            min_val_offsets=[]
            max_val_offsets=[]
            for i in range(1,6):
                # breakpoint()
                if min_vals[0][0]>.25 and min_vals[1][1]>.25:
                    min_val_offsets.append(min_vals/6 * i)
                    min_val_offsets.append(np.diag(min_vals)/6 * i)
                elif min_vals[0][0]>.25:
                    min_val_offsets.append(min_vals[0] / 6 * i)
                elif min_vals[1][1]>.25:
                    min_val_offsets.append(min_vals[1] / 6 * i)
                if max_vals[0][0] > .25 and max_vals[1][1] > .25:
                    max_val_offsets.append(max_vals / 6 * i)
                    max_val_offsets.append(np.diag(max_vals) / 6 * i)
                elif max_vals[0][0] > .25:
                    max_val_offsets.append(max_vals[0] / 6 * i)
                elif max_vals[1][1] > .25:
                    max_val_offsets.append(max_vals[1] / 6 * i)

            # scaling_factors = [0.95, 1.05]
            if len(min_val_offsets)>0:
                min_val_offsets=np.vstack(min_val_offsets)
                for offset in min_val_offsets:
                    temp=pos-offset*.98
                    extra_data.append([_,tsneX,tsneY,N,cluster,str(temp.reshape(-1)),dataset])
                    # for factor in scaling_factors:
                    #     extra_data.append([_,tsneX,tsneY,N,cluster,str((temp*factor).reshape(-1)),dataset])
            if len(max_val_offsets)>0:
                max_val_offsets = np.vstack(max_val_offsets)
                for offset in max_val_offsets:
                    temp = pos + offset * .98
                    extra_data.append([_, tsneX, tsneY, N, cluster, str(temp.reshape(-1)), dataset])
                    # for factor in scaling_factors:
                    #     extra_data.append([_, tsneX, tsneY, N, cluster, str((temp * factor).reshape(-1)),dataset])
        # breakpoint()
        temp = pd.DataFrame(extra_data, columns=self.data.columns)
        self.data=pd.concat((self.data,temp),axis=0)


def getNewGroups(pos, social_thresh, maxN, diffs=None):
    # hard coding grid to be 3:4 (rows:columns) since that's aspect ratio of the images
    groupDict=defaultdict(int)
    for i,p in enumerate(pos): # blue, orange, green, red, purple, brown
        dists = np.sum(np.sum((pos-p)**2,axis=-1)**.5, axis=-1)
        # print(dists)
        inds=np.where(dists<social_thresh)
        for ind in inds:
            if len(ind)<=maxN:
                groupDict[tuple(ind)]+=1
    # breakpoint()
    groups=list(groupDict.keys())
    if len(groups)<1:
        totals = np.array(list(range(pos.shape[0])))
        inds = [list(range(x,x+maxN)) for x in range(len(totals)-maxN)]
        for i in inds:
            groups.append(totals[i])

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
    # breakpoint()
    for g in groups:
        new_pos.append(pos[np.array(list(g))])
        allDiffs=[]
        for i in range(new_pos[-1].shape[0]):
            temp = np.concatenate((new_pos[-1][:i], new_pos[-1][i + 1:]), axis=0)
            # hold = np.sum(new_pos[-1] ** 2, axis=1)
            # heading = np.arccos(np.sum(x[:-1, :] * x[1:, :], axis=1) / (hold[:-1] * hold[1:]) ** .5)
            if len(temp) > 0:
                temp = new_pos[-1][i][1:] - temp[:, :-1, :]
                # breakpoint()
                allDiffs.append(np.hstack(np.concatenate((np.diff(new_pos[-1][i], axis=0).reshape(1, -1, 2) * 15, temp), axis=0)))
            else:
                # breakpoint()
                allDiffs.append(np.diff(new_pos[-1][i], axis=0))
        new_diffs.append(torch.tensor(np.stack(allDiffs)))
    # breakpoint()

    return new_pos, new_diffs, groups

def makeTSNELabel(maxN, input_window, prefix):
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
        TSNE_N_CUTOFFS[i] = [int(t) for t in temp]

def getClusterGT(input_window, maxN, prefix):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tsne_nets = []
    N = np.array(range(1, maxN + 1))
    for i in N:
        temp = SimpleRegNetwork(i*i * (input_window - 1) * 2)  # .eval()
        temp.load_state_dict(torch.load(
            '/Users/faith_johnson/GitRepos/PedTrajPred/simpleRegNet_allDiffsData_RotAug_' +
            str(i) + 'people_' + str(input_window) + 'window.pt'))
        temp.eval()
        tsne_nets.append(temp.to(device))

    CLUSTER_NUM = [0,10,29,33]
    temp=[]
    preds=[[],[],[]]
    for name in ['ETH','ETH_Hotel','UCY_Zara1',]:#'UCY_Zara2',
        # dataset = PlainTrajData(name, input_window=input_window, output_window=12)
        dataset = SameSizeData(name, input_window=input_window, output_window=12)
        loader = DataLoader(dataset, batch_size=1, shuffle=False)
        for data in loader:
            if data['pos'].nelement() > 0:
                scene = data['pos'][0].numpy()
                if scene.shape[0] > maxN:
                    # breakpoint()
                    scene, diffs_list, groups = getNewGroups(scene, 1.6, maxN, data['allDiffs'][0])
                else:
                    scene = [scene]
                    diffs_list = [data['allDiffs'][0]]
                    # breakpoint()
                for i,s in enumerate(scene):
                    diffs = diffs_list[i]#torch.tensor(np.diff(s, axis=1))
                    with torch.no_grad():
                        # TSNE_N_CUTOFFS, TSNE_BOUNDS[class](max,min)
                        tsne_net = tsne_nets[s.shape[0] - 1]
                        tsne = tsne_net(diffs[:, :(input_window - 1), :].flatten().float())
                        preds[s.shape[0]-1].append(tsne.numpy())
                        bound_coords = np.array(TSNE_BOUNDS[s.shape[0]])
                        tsne_class = np.argmin(np.sum((tsne.numpy() - bound_coords) ** 2, axis=-1))# + sum(CLUSTER_NUM[:s.shape[0]])
                    temp.append([tsne.numpy()[0],tsne.numpy()[1],s.shape[0],tsne_class.item(),s.flatten(), name])
    # breakpoint()
    frame = pd.DataFrame(temp,columns=['tsne_X', 'tsne_Y', 'N', 'cluster', 'pos', 'dataset'], dtype=float)
    frame.to_csv(prefix+'allDiffsData_RotAug_TSNEGT.csv')
    from matplotlib import pyplot as plt
    # breakpoint()
    # for i in range(len(preds)):
    #     temp=np.stack(preds[i])
    #     plt.scatter(temp[:,0],temp[:,1])
    #     plt.title('Predicted Traj Clusters, N='+str(i))
    #     plt.show()
    # frame.to_csv('just_Zara2_GT.csv')

if __name__ == '__main__':
    makeTSNELabel(3,8,'noZara2_fullDict_')
    getClusterGT(8,3, 'noZara2_fullDict_')
    # x=TSNEClusterGT(1,0)
    # x.__getitem__(10)
