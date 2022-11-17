import torch
from tsneGTClusterdataset import TSNEClusterGT
import torch.nn as nn
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
import random
from trajAugmentations import TrajAugs
from simpleTSNEPredict import SimpleRegNetwork
from collections import defaultdict
from sklearn.manifold import TSNE
from sameSizeData import SameSizeData
from plainTrajData import PlainTrajData
import pandas as pd


# CLUSTERS_PER_N = {1:5, 2:13, 3:24} #diffsData8, diffsData16
# CLUSTERS_PER_N = {1:6, 2:28, 3:24} #allDiffs8, allDiffs16
# CLUSTERS_PER_N = {1:6, 2:19, 3:24} # allDiffs8Closest,
# CLUSTERS_PER_N = {1:6, 2:, 3:} # allDiffs8Long,
CLUSTERS_PER_N = {1:10, 2:29, 3:33} # allDiffs8Long30 45 60,

# CLUSTERS_PER_N = {1:14, 2:18, 3:22} diffsVel
# CLUSTERS_PER_N = {1:11, 2:18, 3:24} diffsVelAng
# CLUSTERS_PER_N = {1:16, 2:18, 3:24} angAllDiffs
CLUSTER_NUM=[0,10,29,33]


class SimplestNet(nn.Module):
    def __init__(self, args):
        super(SimplestNet, self).__init__()
        self.fc1=nn.Linear((args.input_window)*2,32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64+32, 32)
        self.outfc=nn.Linear(32+32+64, args.output_window*2)
        # self.fc1 = nn.Linear((args.input_window) * 4, 64)
        # self.fc2 = nn.Linear(64, 64)
        # self.fc3 = nn.Linear(64+64, 32)
        # self.outfc=nn.Linear(32+2*64, args.output_window*2)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        res1=self.relu(self.fc1(x))
        res2=self.relu(self.fc2(res1))
        res3 = self.relu(self.fc3(torch.cat((res2,res1), dim=-1)))
        x=self.outfc(torch.cat((res3,res2,res1), dim=-1))
        return x, torch.cat((res3,res2,res1), dim=-1)
        # return res1

class SimplestUNet(nn.Module):
    def __init__(self, args):
        super(SimplestUNet, self).__init__()
        self.efc1 = nn.Linear(args.input_window*2, 16)
        self.efc2 = nn.Linear(16, 8)
        self.efc3 = nn.Linear(16+8, 2)
        self.dfc1 = nn.Linear(2, 4)
        self.dfc2 = nn.Linear(4+8, 16)
        self.dfc3 = nn.Linear(16+16, args.input_window*2)
        self.relu = nn.PReLU()

    def forward(self, x):
        res1 = self.relu(self.efc1(x))
        res2 = self.relu(self.efc2(res1))
        latent = self.relu(self.efc3(torch.cat((res2,res1), dim=-1)))
        res3 = self.relu(self.dfc1(latent))
        res4 = self.relu(self.dfc2(torch.cat((res3,res2), dim=-1)))
        x = self.dfc3(torch.cat((res4,res1), dim=-1))
        return x, latent

class SimplestAutoEncoder(nn.Module):
    def __init__(self, args):
        super(SimplestAutoEncoder, self).__init__()
        self.efc1 = nn.Linear(args.input_window*2, 16)
        self.efc2 = nn.Linear(16, 8)
        self.efc3 = nn.Linear(16+8, 8)
        self.dfc1 = nn.Linear(8, 16)
        self.dfc2 = nn.Linear(16+8, 16)
        # self.dfc3 = nn.Linear(8+16, args.input_window*2) # for in-->in prediction
        self.dfc3 = nn.Linear(8 + 16, args.output_window * 2) #for in-->out prediction
        # self.dfc3 = nn.Linear(8 + 16, 2)

        # self.efc1 = nn.Linear(args.input_window * 2, 16)
        # self.efc2 = nn.Linear(16, 32)
        # self.efc3 = nn.Linear(16+32, 16)
        # self.dfc1 = nn.Linear(16, 32)
        # self.dfc2 = nn.Linear(16+32, 64)
        # self.dfc3 = nn.Linear(64+16, args.input_window*2)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        res1 = self.relu(self.efc1(x))
        res2 = self.relu(self.efc2(res1))
        latent = self.relu(self.efc3(torch.cat((res2,res1), dim=-1)))
        res3 = self.relu(self.dfc1(latent))
        res4 = self.relu(self.dfc2(torch.cat((res3,latent), dim=-1)))
        x = self.dfc3(torch.cat((res4,latent), dim=-1))
        return x, latent

class SimplestConvNet(nn.Module):
    def __init__(self, args):
        super(SimplestConvNet, self).__init__()
        self.conv1 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(2,32)
        self.fc2 = nn.Linear(32*8,args.output_window*2)
        self.relu1 = nn.PReLU()
        # self.relu2 = nn.PReLU()
        # self.relu3 = nn.PReLU()
        # self.relu4 = nn.PReLU()
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(32)
        self.bn4 = nn.BatchNorm2d(32)

    def forward(self, x):
        # breakpoint()
        x = self.relu1(self.fc1(x))
        x = torch.transpose(x,-1,0)
        x = x.unsqueeze(0)
        x = self.bn1(self.conv1(x))
        x = self.bn2(self.conv2(x))
        x = self.bn3(self.conv3(x))
        x = self.bn4(self.conv4(x))
        x = x.squeeze()
        x = torch.transpose(x, -1, 0)
        x = self.fc2(x.reshape(-1, x.shape[1]*x.shape[2]))
        return x

class SimplestVAE(nn.Module):
    def __init__(self, args):  # , device):
        super(SimplestVAE, self).__init__()
        self.efc1 = nn.Linear(args.input_window*2, 16)
        self.efc2 = nn.Linear(16, 16)
        self.mu_dense = nn.Linear(16+16, 8)
        self.log_var_dense = nn.Linear(16+16, 8)
        self.dfc1= nn.Linear(8, 16)
        self.dfc2 = nn.Linear(16+8, 16)
        self.output_fc = nn.Linear(16+8, args.input_window * 2)
        # self.output_fc = nn.Linear(16, args.output_window * 2)
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout()

    def encode(self, x, label=None):
        res1 = self.relu(self.efc1(x))
        res2 = self.relu(self.efc2(self.dropout(res1)))
        mu = self.mu_dense(torch.cat((res2,res1), dim=-1))
        log_var = self.log_var_dense(torch.cat((res2,res1), dim=-1))
        return mu, log_var

    def decode(self, x, label=None):
        res1 = self.relu(self.dfc1(x))
        res2 = self.relu(self.dfc2(self.dropout(torch.cat((res1,x), dim=-1))))
        output = torch.sigmoid(self.output_fc(torch.cat((res2, x), dim=-1)))
        return output

    def sample_z(self, mu, log_var):
        # breakpoint()
        eps = torch.randn(log_var.shape[0], log_var.shape[1])  # .to(self.device)
        return mu + log_var*eps

    def forward(self, x, label=None):
        # breakpoint()
        mu, log_var = self.encode(x)  # , label)
        z = self.sample_z(mu, torch.exp(0.5 * log_var))
        output = self.decode(z)  # , label)
        return output, mu, log_var, z

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

def train(args, net, device, N, cluster_num):
    min_loss = np.inf
    trajAugs = TrajAugs()
    dataset = TSNEClusterGT(N, cluster_num, args.prefix+'allDiffsData_RotAug_TSNEGT.csv', split=None)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    opt=torch.optim.Adam(net.parameters(), lr=args.lr)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs * len(dataset), 1e-12)
    loss_func = nn.MSELoss() #nn.BCELoss() #
    last_loss=[np.inf]
    for e in tqdm(range(args.epochs)):
        avgLoss=[]
        # breakpoint()
        if len(loader)<1:
            break
        for data in loader:
            pos, tsne = data
            pos = pos.reshape(N, args.input_window+args.output_window, 2)
            # pos = trajAugs.augment(pos)
            # breakpoint()
            if N==3:
                max_inds=torch.max(pos[:,0,:],axis=0)[1]
                min_inds = torch.min(pos[:, 0, :], axis=0)[1]
                offset=pos[set(range(3)).difference(set([max_inds[0].item(),min_inds[0].item()])).pop(),0,:]
            elif N==2:
                min_inds = torch.min(pos[:, 0, :], axis=0)[1]
                offset = pos[min_inds[0], 0, :]
            else:
                offset=pos[0,0,:]
            pos=pos-offset
            output, latent = net(pos[:,:args.input_window,:].reshape(-1, (args.input_window) * 2).float().to(device))
            opt.zero_grad()
            loss = loss_func(output, pos[:, args.input_window:, :].reshape(-1,args.output_window*2).float().to(device))
            avgLoss.append(loss.item())
            loss.backward()
            opt.step()
            scheduler.step()
        if e%10==0:
            print("Epoch",e,': Loss =',np.mean(avgLoss))
        if np.mean(avgLoss)<min_loss:
            torch.save(net.state_dict(),args.prefix+'simpleNetTSNE_allDiffsData_RotAug_'+str(N)+'_cluster'+str(cluster_num)+'.pt')
            min_loss=np.mean(avgLoss)
        if np.mean(avgLoss)<1.0e-4:# or abs(np.mean(last_loss[-500:])-np.mean(avgLoss))<1e-5:
            # breakpoint()
            torch.save(net.state_dict(),args.prefix+'simpleNetTSNE_allDiffsData_RotAug_'+str(N)+'_cluster'+str(cluster_num)+'.pt')
            print("Epoch",e,': Loss =',np.mean(avgLoss))
            break
        last_loss.append(np.mean(avgLoss))
    return net

def test(args, net, device, N, cluster_num):
    loss_func = nn.MSELoss() #nn.BCELoss() #
    preds=[]
    inputs=[]
    latents=[]
    avgLoss = []
    dataset = TSNEClusterGT(N, cluster_num, args.prefix+'allDiffsData_RotAug_TSNEGT.csv',split='test')#args.prefix+'allDiffsData_RotAug_TSNEGT.csv')#
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    for data in loader:
        pos, tsne = data
        pos = pos.reshape(N, args.input_window+args.output_window, 2)
        if N == 3:
            max_inds = torch.max(pos[:, 0, :], axis=0)[1]
            min_inds = torch.min(pos[:, 0, :], axis=0)[1]
            offset = pos[set(range(3)).difference(set([max_inds[0].item(), min_inds[0].item()])).pop(), 0, :]
        elif N == 2:
            min_inds = torch.min(pos[:, 0, :], axis=0)[1]
            offset = pos[min_inds[0], 0, :]
        else:
            offset = pos[0, 0, :]
        pos = pos - offset
        with torch.no_grad():
            output, latent = net(pos[:, :args.input_window, :].reshape(-1, (args.input_window) * 2).to(device))
            loss = loss_func(output, pos[:, args.input_window:, :].reshape(-1, args.output_window * 2).to(device))
        if loss.item()>1:
            print(N,cluster_num)
            print('\t',output)
            print('\t',pos)
        avgLoss.append(loss.item())
        inputs.append(pos.numpy()+offset.numpy())
        # latents.append(latent.numpy())
        # breakpoint()
        output=output.reshape(-1,args.output_window,2).cpu().numpy() + offset.numpy()
        preds.append(output.reshape(-1,args.output_window*2))
    print('Avg Test Loss =',np.mean(avgLoss))
    return preds, inputs, latents, avgLoss

def testAll(args, nets, device, maxN):
    loss_func = nn.MSELoss()  # nn.BCELoss() #
    preds = []
    inputs = []
    latents = []
    avgLoss = []
    groups=[]
    for n in range(1,maxN+1):
        for c in range(CLUSTER_NUM[n]):
            dataset = TSNEClusterGT(n, c)
            loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
            for data in loader:
                pos, tsne = data
                pos = pos.reshape(n, args.input_window + args.output_window, 2)
                if n == 3:
                    max_inds = torch.max(pos[:, 0, :], axis=0)[1]
                    min_inds = torch.min(pos[:, 0, :], axis=0)[1]
                    offset = pos[set(range(3)).difference(set([max_inds[0].item(), min_inds[0].item()])).pop(), 0, :]
                elif n == 2:
                    min_inds = torch.min(pos[:, 0, :], axis=0)[1]
                    offset = pos[min_inds[0], 0, :]
                else:
                    offset = pos[0, 0, :]
                pos = pos - offset
                with torch.no_grad():
                    minLoss=np.inf
                    minOutput=[]
                    group=[c,c]
                    for i,net in enumerate(nets[n-1]):
                        output, latent = net(pos[:, :args.input_window, :].reshape(-1, (args.input_window) * 2).to(device))
                        loss = loss_func(output, pos[:, args.input_window:, :].reshape(-1, args.output_window * 2).to(device))
                        # print(loss.item())
                        if loss.item()<minLoss:
                            minLoss=loss.item()
                            minOutput=output
                            group[1]=i
                avgLoss.append(minLoss)
                inputs.append(pos.numpy()+offset.numpy())
                groups.append(groups)
                # latents.append(latent.numpy())
                # breakpoint()
                minOutput=minOutput.reshape(-1, args.output_window, 2).cpu().numpy() + offset.numpy()
                preds.append(minOutput.reshape(-1, args.output_window * 2))
    print('Avg Test Loss =', np.mean(avgLoss))
    print('Correspondences:',sum([g[0]==g[1] for g in groups]),'out of',len(groups))
    return preds, inputs, latents, avgLoss, groups

def getNewGroups(pos, diffs, args):
    groupDict=defaultdict(int)
    # breakpoint()
    for i,p in enumerate(pos): # blue, orange, green, red, purple, brown
        dists = torch.sum(torch.sum((pos-p)**2,axis=-1)**.5, axis=-1)
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
            temp = np.concatenate((new_pos[-1][:i,:args.input_window,:], new_pos[-1][i+1:,:args.input_window,:]), axis=0)
            if len(temp) > 0:
                temp = new_pos[-1][i,:args.input_window,:][1:] - temp[:, :-1, :]
                allDiffs.append(
                    np.hstack(np.concatenate((np.diff(new_pos[-1][i,:args.input_window,:], axis=0).reshape(1, -1, 2) * 15, temp), axis=0)))
            else:
                allDiffs.append(np.diff(new_pos[-1][i,:args.input_window,:], axis=0))
        new_allDiffs.append(torch.tensor(np.stack(allDiffs)).flatten())

    return new_pos, new_allDiffs, new_diffs, groups

def makeTSNELabel(maxN, input_window):
    # global GT_TSNE_VALUES
    global TSNE_N_CUTOFFS
    global TSNE_BOUNDS
    # GT_TSNE_VALUES = pd.DataFrame(columns=['tsne_X','tsne_Y','kmeans'])
    TSNE_N_CUTOFFS = {}
    TSNE_BOUNDS = {}
    max_label = 0
    for i in range(1,maxN+1):
        # breakpoint() #args.prefix+
        data = pd.read_csv(args.prefix+'allDiffsData_RotAug_'+str(i)+'thresh_'+str(input_window)+'window.csv')
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

@torch.no_grad()
def testOne(args, device, name):
    makeTSNELabel(args.maxN, args.input_window)
    # dataset = SameSizeData(name, input_window=args.input_window, output_window=args.output_window)
    # breakpoint()
    #ETH: 1427, Hotel: 1147, Zara1: 844, Zara2: 1030
    dataset = PlainTrajData(name, input_window=9, output_window=args.output_window, maxN=5, split='test')
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    nets = []
    N = np.array(range(1, args.maxN + 1))
    for i in N:
        net = SimpleRegNetwork(i * i * (args.input_window - 1) * 2).eval()
        net.load_state_dict(torch.load(
            '/Users/faith_johnson/GitRepos/PedTrajPred/simpleRegNet_allDiffsData_RotAug_' + str(i) + 'people_' + str(
                args.input_window) + 'window.pt'))#'+args.prefix+'
        nets.append(net)
    tsne_preds=[]
    traj_preds=[]
    inputs=[]
    frames=[]
    total_loss=[]
    fde=[]
    avgUnNormedLoss=[]
    unNormedFDE=[]
    loss_func = nn.MSELoss()
    for data in tqdm(loader):
        if data['pos'].nelement() > 0:
            # breakpoint()
            data['pos']=data['pos'][:,:,:-1,:]
            if data['diffs'].shape[-3] < args.maxN:
                # breakpoint()
                net = nets[data['diffs'].shape[-3] - 1]
                pos = data['pos'][0][:, :args.input_window, :].float()  # .flatten()
                target = data['pos'][0][:, args.input_window:, :]
                allDiffs = []
                for i in range(pos.shape[0]):
                    temp = np.concatenate((pos[:i], pos[i + 1:]), axis=0)
                    if len(temp) > 0:
                        temp = pos[i][1:] - temp[:, -1, :]
                        allDiffs.append(np.concatenate((np.diff(pos[i], axis=0) * 15, temp), axis=-1))
                    else:
                        allDiffs.append(np.diff(pos[i], axis=0))
                # breakpoint()
                pos = torch.tensor(np.stack(allDiffs)).flatten()
                output = net(pos)
                c = np.argmin(np.sum((np.array(TSNE_BOUNDS[data['diffs'].shape[-3]]) - output.numpy()) ** 2, axis=-1))
                trajnet = SimplestNet(args).to(device)
                try:
                    trajnet.load_state_dict(torch.load(args.prefix+
                        'simpleNetTSNE_allDiffsData_RotAug_' + str(data['diffs'].shape[-3]) + '_cluster' + str(c) + '.pt',
                        map_location=device))
                except:
                    breakpoint()
                pos = data['pos'][0]
                # if data['diffs'].shape[-3]>3:
                #     breakpoint()
                # if data['diffs'].shape[-3] == 3:
                #     max_inds = torch.max(pos[:, 0, :], axis=0)[1]
                #     min_inds = torch.min(pos[:, 0, :], axis=0)[1]
                #     offset = pos[set(range(3)).difference(set([max_inds[0].item(), min_inds[0].item()])).pop(), 0, :]
                # elif data['diffs'].shape[-3] == 2:
                #     min_inds = torch.min(pos[:, 0, :], axis=0)[1]
                #     offset = pos[min_inds[0], 0, :]
                # else:
                #     offset = pos[0, 0, :]
                offset = torch.tensor([0, 0])
                pos = pos - offset
                traj, latent = trajnet(pos[:, :args.input_window, :].reshape(-1, (args.input_window) * 2).float().to(device))
                loss = loss_func(traj, pos[:, args.input_window:, :].reshape(-1, args.output_window * 2).float().to(device))
                if loss.item() > 0.001:
                    # print(data['index'], data['diffs'].shape[-3], c)
                    # print('\t', traj)
                    temp_loss_track = [[data['diffs'].shape[-3], c, loss, traj]]
                    if data['diffs'].shape[-3]==1:
                        sample_num=10
                    else:
                        sample_num=10
                    for new_c in random.sample(list(range(CLUSTER_NUM[data['diffs'].shape[-3]])), sample_num):
                        try:
                            trajnet.load_state_dict(torch.load(
                                args.prefix + 'simpleNetTSNE_allDiffsData_RotAug_' + str(data['diffs'].shape[-3]) + '_cluster' + str(
                                    new_c) + '.pt', map_location=device))
                            new_traj, latent = trajnet(pos[:, :args.input_window, :].reshape(-1, (args.input_window) * 2).float().to(device))
                            new_loss = loss_func(new_traj, pos[:, args.input_window:, :].reshape(-1,args.output_window * 2).float().to(
                                device))
                            temp_loss_track.append([data['diffs'].shape[-3], new_c, new_loss, new_traj])
                        except Exception as e:
                            print(e)
                            pass
                    losses=[l[2] for l in temp_loss_track]
                    min_loss_ind = np.argmin(losses)
                    loss=temp_loss_track[min_loss_ind][2]
                    traj=temp_loss_track[min_loss_ind][-1]
                    # print(data['diffs'].shape[-3], c, '-->', temp_loss_track[min_loss_ind][1])
                    # breakpoint()
                # if (data['diffs'].shape[-3]==2 and c==2) or data['diffs'].shape[-3]!=2:
                total_loss.append(loss.item())
                tsne_preds.append([output.detach()])
                inputs.append(data['pos'].flatten())
                temp_traj = traj.detach().cpu().reshape(-1, args.output_window, 2).cpu().numpy() + offset.numpy()
                traj_preds.append(temp_traj.reshape(-1, args.output_window * 2))
                frames.append(data['frames'])
                fde.append(loss_func(traj.reshape(-1, args.output_window, 2)[:, -1, :], pos[:, -1, :]).item())
                unNormedOutput = traj.reshape(-1, args.output_window, 2) * (dataset.max - dataset.min) + dataset.min  # (np.array(positions)-self.min)/(self.max-self.min)
                unNormedScene = pos[:, args.input_window:, :] * (dataset.max - dataset.min) + dataset.min
                unNormedLoss = loss_func(unNormedOutput, unNormedScene)
                avgUnNormedLoss.append(unNormedLoss.item())
                unNormedFDE.append(loss_func(unNormedOutput[:, -1, :], unNormedScene[:, -1, :]).item())
            else:
                # breakpoint()
                pos, allDiffs, diffs, groups = getNewGroups(data['pos'][0], data['diffs'][0], args)
                people = []
                # preds = []
                # ins = []
                temp_unNormLoss = []
                temp_unNormFDE = []
                pred_loss=[]
                temp_fde=[]
                for i, p in enumerate(allDiffs):
                    net = nets[pos[i].shape[-3] - 1]
                    people.append(pos[i].shape[-3])
                    output = net(p.flatten().float())
                    c = np.argmin(np.sum((np.array(TSNE_BOUNDS[people[-1]]) - output.numpy()) ** 2, axis=-1))
                    trajnet = SimplestNet(args).to(device)
                    try:
                        trajnet.load_state_dict(torch.load(args.prefix+
                            'simpleNetTSNE_allDiffsData_RotAug_' + str(people[-1]) + '_cluster' + str(c) + '.pt',
                            map_location=device))
                    except:
                        # trajnet.load_state_dict(torch.load(args.prefix +'simpleNetTSNE_allDiffsData_RotAug_' + str(
                        #     people[-1]) + '_cluster' + str(c-1) + '.pt',
                        #     map_location=device))
                        breakpoint()
                    if people[-1]>3:
                        breakpoint()
                    # if people[-1] == 3:
                    #     max_inds = torch.max(pos[i][:, 0, :], axis=0)[1]
                    #     min_inds = torch.min(pos[i][:, 0, :], axis=0)[1]
                    #     offset = pos[i][set(range(3)).difference(set([max_inds[0].item(), min_inds[0].item()])).pop(), 0,
                    #              :]
                    # elif people[-1] == 2:
                    #     min_inds = torch.min(pos[i][:, 0, :], axis=0)[1]
                    #     offset = pos[i][min_inds[0], 0, :]
                    # else:
                    #     offset = pos[i][0, 0, :]
                    offset=torch.tensor([0,0])
                    pos[i] = pos[i] - offset
                    traj, latent = trajnet(pos[i][:, :args.input_window, :].reshape(-1, (args.input_window) * 2).float().to(device))
                    loss = loss_func(traj, pos[i][:, args.input_window:, :].reshape(-1, args.output_window * 2).float().to(device))
                    # breakpoint()
                    if loss.item() > 0.001:
                        # print(data['index'], people[-1], c)
                        # print('\t',traj)
                        temp_loss_track = [[people[-1], c, loss, traj]]
                        if people[-1] == 1:
                            sample_num = 10
                        else:
                            sample_num = 10
                        for new_c in random.sample(list(range(CLUSTER_NUM[people[-1]])), sample_num):
                            try:
                                trajnet.load_state_dict(torch.load(
                                    args.prefix + 'simpleNetTSNE_allDiffsData_RotAug_' + str(people[-1]) + '_cluster' + str(
                                        new_c) + '.pt', map_location=device))
                                new_traj, latent = trajnet(pos[i][:, :args.input_window, :].reshape(-1, (args.input_window) * 2).float().to(device))
                                new_loss = loss_func(new_traj, pos[i][:, args.input_window:, :].reshape(-1,args.output_window * 2).float().to( device))
                                temp_loss_track.append([people[-1], new_c, new_loss, new_traj])
                            except Exception as e:
                                print(e)
                                pass
                        losses = [l[2] for l in temp_loss_track]
                        min_loss_ind = np.argmin(losses)
                        loss = temp_loss_track[min_loss_ind][2]
                        traj = temp_loss_track[min_loss_ind][-1]
                        # print(people[-1], c, '-->', temp_loss_track[min_loss_ind][1])
                        # breakpoint()
                    inputs.append(pos[i])
                    pred_loss.append(loss.item())
                    temp_traj = traj.detach().cpu().reshape(-1, args.output_window, 2).cpu().numpy() + offset.numpy()
                    traj_preds.append(temp_traj.reshape(-1, args.output_window * 2))
                    tsne_preds.append(output.detach())
                    frames.append(data['frames'])
                    # breakpoint()
                    temp_fde.append(loss_func(traj.reshape(people[-1],args.output_window,2)[:,-1,:],pos[i][:,-1,:]).item())
                    unNormedOutput = traj.reshape(people[-1],args.output_window,2) * (dataset.max - dataset.min) + dataset.min  # (np.array(positions)-self.min)/(self.max-self.min)
                    unNormedScene = pos[i][:, args.input_window:, :] * (dataset.max - dataset.min) + dataset.min
                    unNormedLoss = loss_func(unNormedOutput, unNormedScene)
                    temp_unNormLoss.append(unNormedLoss.item())
                    temp_unNormFDE.append(loss_func(unNormedOutput[:,-1,:], unNormedScene[:,-1,:]).item())
                # if (people[-1]==2 and c==2) or people[-1]!=2:
                total_loss.append(np.mean(pred_loss))
                fde.append(np.mean(temp_fde))
                avgUnNormedLoss.append(np.mean(temp_unNormLoss))
                unNormedFDE.append(np.mean(temp_unNormFDE))
                # inputs.append(ins)
                # frames.append(temp_frame)
                # traj_preds.append(temp_trajs)
    # breakpoint()
    print(name,'Avg Test Loss =', np.mean(total_loss))
    print(name, 'Avg FDE =', np.mean(fde))
    print(name, 'Avg unnormed Test Loss =', np.mean(avgUnNormedLoss))
    print(name, 'Avg unnormed FDE =', np.mean(unNormedFDE))
    return tsne_preds, inputs, traj_preds, frames

def graph(args, inputs, predictions=None, name=None, index=None, n=None, c=None, i=None):
    plt.figure()
    plt.axis([0,1,0,1])
    if predictions is None:
        plt.scatter(inputs[:,0], inputs[:,1])
    else:
        # breakpoint()
        if index is None:
            ind = random.choice(list(range(len(inputs))))
        else:
            ind=index
        # if sum([p < 0 for p in predictions[ind].reshape(-1)])>0:
        #     breakpoint()
        # if sum([p > 1 for p in predictions[ind].reshape(-1)])>0:
        #     breakpoint()
        # if sum([p < 0 for p in inputs[ind].reshape(-1)])>0:
        #     breakpoint()
        # if sum([p > 1 for p in inputs[ind].reshape(-1)])>0:
        #     breakpoint()
        # for pos in inputs[ind].reshape(-1, args.input_window,2):
        for pos in inputs[ind].reshape(-1, args.input_window+args.output_window, 2):
            plt.scatter(pos[:args.input_window,0], pos[:args.input_window,1], c='b')
            plt.scatter(pos[args.input_window:,0], pos[args.input_window:,1], c='g')
        # for pos in predictions[ind].reshape(-1, args.input_window,2):
        # for pos in predictions[ind].reshape(-1, args.output_window, 2):
        for pos in predictions[ind].reshape(-1, 2):
            # plt.plot(pos[:,0], pos[:,1], c='tab:orange')
            # plt.scatter(pos[0][0], pos[0][1], c='tab:orange')
            plt.scatter(pos[0], pos[1], c='tab:orange')
        plt.legend(['Input','Ground truth', 'Predictions'])
    plt.title(name)
    # if n and c and i:
    #     plt.savefig('/Users/faith_johnson/Documents/ECCV 2022/final_traj_results/'+str(n)+'_'+str(c)+'_'+str(i)+'.png')


if __name__=='__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args = get_args()

    if args.train:
        for n in range(1, args.maxN + 1):
            for c in range(0, CLUSTER_NUM[n]):
                print('People:', n, 'Cluster:', c)
                net = SimplestNet(args).to(device)
                try:
                    net.load_state_dict(torch.load(args.prefix+'simpleNetTSNE_allDiffsData_RotAug_'+str(n)+'_cluster'+str(c)+'.pt', map_location=device))
                except:
                    pass
                net = train(args, net, device, n, c)
                #preds, inputs, latents, loss = test(args, net, device, n, c)
                # if len(latents) > 0:
                #     tsne = TSNE()
                #     latents = tsne.fit_transform(latents)
                #     graph(args, latents, name='Learned Latent Space')
                #
                # for i in range(10):
                #     graph(args, inputs, preds, name='Inputs vs Predictions')
                #
                # plt.show()

    tsne_preds, inputs, traj_preds, frames = testOne(args, device, args.name)
    # for i in range(20):
    #     graph(args, inputs, traj_preds, name='Cluster Networks: Inputs vs Predictions',index=i)
    # plt.show()
    exit(0)
    test_loss=[]
    for n in range(1, args.maxN + 1):
        for c in range(CLUSTER_NUM[n]):
            print('People:',n,'Cluster:',c)
            net = SimplestNet(args).to(device)
            try:
                net.load_state_dict(torch.load(args.prefix+'simpleNetTSNE_allDiffsData_RotAug_'+str(n)+'_cluster'+str(c)+'.pt', map_location=device))
                net.eval()
                preds, inputs, latents, loss = test(args, net, device, n, c)
                test_loss.extend(loss)
                # if len(latents) > 0:
                #     tsne = TSNE()
                #     latents = tsne.fit_transform(latents)
                    # graph(args, latents, name='Learned Latent Space')

                # for i in range(5):
                #     graph(args, inputs, preds, name=str(n)+' People, Cluster '+str(c)+', Inputs vs Predictions', n=n,c=c,i=i)
                # plt.show()
            except Exception as e:
                print(e)
                continue
            # breakpoint()
    print('Total Avg Loss:', np.mean(test_loss))
    plt.show()

    # nets=[[],[],[]]
    # for n in range(1, args.maxN + 1):
    #     for c in range(CLUSTER_NUM[n]):
    #         print('People:', n, 'Cluster:', c)
    #         net = SimplestNet(args).to(device)
    #         try:
    #             net.load_state_dict(torch.load('simpleNetTSNE_'+str(n)+'_cluster'+str(c)+'.pt', map_location=device))
    #         except:
    #             continue
    #         net.eval()
    #         nets[n-1].append(net)
    #
    # preds, inputs, latents, loss, groups = testAll(args, nets, device, args.maxN)
    # for i in range(30):
    #     graph(args, inputs, preds, name=str(n)+' People, '+str(c)+'Cluster, Inputs vs Predictions')
    # plt.show()