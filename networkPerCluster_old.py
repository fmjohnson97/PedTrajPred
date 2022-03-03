import sys
sys.path.append('../')
import torch
from plainTrajData import PlainTrajData
import torch.nn as nn
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
import random
from sklearn.manifold import TSNE
from trajAugmentations import TrajAugs
from simpleTSNEPredict import SimpleRegNetwork
from tsneGTdataset import TSNEGT
import pandas as pd
from collections import defaultdict

CLUSTERS_PER_N = {1:5, 2:5, 3:5, 4:5}

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
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--maxN', default=3, type=int)
    parser.add_argument('--lr', default=1e-2, type=float)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--social_thresh', default=1.6, type=float)

    args = parser.parse_args()
    return args

def train(args, nets, device, tsne_nets):
    min_loss = [[np.inf],[np.inf],[np.inf],[np.inf],[np.inf],[np.inf],[np.inf],[np.inf],[np.inf],[np.inf],[np.inf],[np.inf],[np.inf],[np.inf],[np.inf]]
    trajAugs = TrajAugs()
    opts=[]
    schedulers = []
    stop_training=[]
    for i in range(len(nets)):
        opts.append(torch.optim.Adam(nets[i].parameters(), lr=args.lr))
        schedulers.append(torch.optim.lr_scheduler.CosineAnnealingLR(opts[-1], args.epochs * 1000, 1e-12))
    loss_func = nn.MSELoss() #nn.BCELoss() #
    for e in tqdm(range(args.epochs)):
        avgLoss=[[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
        for name in ['ETH', 'ETH_Hotel', 'UCY_Zara1', 'UCY_Zara2']:
            dataset = PlainTrajData(name, input_window=args.input_window, output_window=args.output_window, maxN=args.maxN)
            loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
            for data in loader:
                if data['pos'].nelement()>0:
                    scene = data['pos'][0] # torch.tensor(trajAugs.augment(data['pos'][0].numpy()))
                    if scene.shape[0]>args.maxN:
                        scene, groups = getNewGroups(scene, args)
                        # breakpoint()
                    else:
                        scene = [scene]
                    for s in scene:
                        diffs = torch.tensor(np.diff(s, axis=1))
                        # vels = torch.cat((torch.zeros(scene.shape[0],1,2),diffs[:,:args.input_window-1,:]),axis=1)
                        # input_scene = torch.cat((scene[:,:args.input_window,:],vels),axis=-1)
                        with torch.no_grad():
                            #TSNE_N_CUTOFFS, TSNE_BOUNDS[class](max,min)
                            tsne_net = tsne_nets[s.shape[0] - 1]
                            tsne = tsne_net(diffs[:, :(args.input_window - 1), :].flatten().float().to(device))
                            bound_coords = np.array(TSNE_BOUNDS[s.shape[0]])
                            tsne_class = np.argmin(np.sum((tsne.cpu().numpy()-bound_coords)**2,axis=-1))+5*(s.shape[0]-1)
                            # print(tsne_class)
                            if tsne_class in stop_training:
                                continue
                        output, latent = nets[tsne_class](s[:,:args.input_window,:].reshape(-1, (args.input_window) * 2).float().to(device))
                        opts[tsne_class].zero_grad()
                        loss = loss_func(output, s[:, args.input_window:, :].reshape(-1,args.output_window*2).float().to(device))
                        avgLoss[tsne_class].append(loss.item())
                        loss.backward()
                        opts[tsne_class].step()
                        schedulers[tsne_class].step()
        print("Epoch",e,': Loss =',np.mean([np.mean(a) for a in avgLoss]))
        for i, avg in enumerate(avgLoss):
            if np.mean(avg)<min_loss[i]:
                # for i in range(1, args.maxN + 1):
                #     torch.save('simpleRegNet_noNorm_diffsData_' + str(i) + 'people_' + str(args.input_window) + 'window.pt')
                torch.save(nets[i].state_dict(),'simpleNetTSNE_cluster'+str(i)+'.pt')
                min_loss[tsne_class]=np.mean(avg)
                if np.mean(avg)<1e-4:
                    stop_training.append(i)
        print('Training Accomplished for ',stop_training)
    return nets

def test(args, nets, device, tsne_nets):
    loss_func = nn.MSELoss() #nn.BCELoss() #
    avgLoss=[[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
    preds = []
    inputs = []
    latents = []
    for name in ['ETH', 'ETH_Hotel', 'UCY_Zara1', 'UCY_Zara2']:
        dataset = PlainTrajData(name, input_window=args.input_window, output_window=args.output_window, maxN=args.maxN, split='test')
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        for data in loader:
            if data['pos'].nelement() > 0:
                scene = data['pos'][0] #torch.tensor(trajAugs.augment(data['pos'][0].numpy()))
                if scene.shape[0] > args.maxN:
                    scene, groups = getNewGroups(scene, args)
                    # breakpoint()
                else:
                    scene = [scene]
                for s in scene:
                    diffs = torch.tensor(np.diff(s, axis=1))
                    # vels = torch.cat((torch.zeros(scene.shape[0],1,2),diffs[:,:args.input_window-1,:]),axis=1)
                    # input_scene = torch.cat((scene[:,:args.input_window,:],vels),axis=-1)
                    with torch.no_grad():
                        # TSNE_N_CUTOFFS, TSNE_BOUNDS[class](max,min)
                        tsne_net = tsne_nets[s.shape[0] - 1]
                        tsne = tsne_net(diffs[:, :(args.input_window - 1), :].flatten().float().to(device))
                        bound_coords = np.array(TSNE_BOUNDS[s.shape[0]])
                        tsne_class = np.argmin(np.sum((tsne.cpu().numpy() - bound_coords) ** 2, axis=-1)) + 5 * (s.shape[0] - 1)
                        # print(tsne_class, s.shape)
                        output, latent = nets[tsne_class](s[:, :args.input_window, :].reshape(-1, (args.input_window) * 2).float().to(device))
                        loss = loss_func(output,s[:, args.input_window:, :].reshape(-1, args.output_window * 2).float().to(device))
                    avgLoss[tsne_class].append(loss.item())
                    preds.append(output.numpy())
                    inputs.append(s.numpy())
                    latents.extend(latent.numpy())
    # breakpoint()
    print('Avg Test Loss =',np.mean([np.mean(a) for a in avgLoss]))
    for i in range(len(avgLoss)):
        print('\t Avg Loss '+str(i)+' Cluster:',np.mean(avgLoss[i]))
    return preds, inputs, latents

def graph(args, inputs, predictions=None, name=None):
    plt.figure()
    if predictions is None:
        plt.scatter(inputs[:,0], inputs[:,1])
    else:
        # breakpoint()
        ind = random.choice(list(range(len(inputs))))
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
    plt.title(name)

def makeTSNELabel():
    # global GT_TSNE_VALUES
    global TSNE_N_CUTOFFS
    global TSNE_BOUNDS
    # GT_TSNE_VALUES = pd.DataFrame(columns=['tsne_X','tsne_Y','kmeans'])
    TSNE_N_CUTOFFS = {}
    TSNE_BOUNDS = {}
    max_label = 0
    for i in range(1,args.maxN+1):
        # breakpoint()
        data = pd.read_csv('/Users/faith_johnson/GitRepos/PedTrajPred/equalClusterData/diffsData_'+str(i)+'thresh_'+str(args.input_window)+'window.csv')
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

def getNewGroups(pos, args):
    # hard coding grid to be 3:4 (rows:columns) since that's aspect ratio of the images
    groupDict=defaultdict(int)
    for i,p in enumerate(pos): # blue, orange, green, red, purple, brown
        dists = torch.sum(torch.sum((pos-p)**2,axis=-1)**.5, axis=-1)
        # print(dists)
        inds=np.where(dists<args.social_thresh)
        for ind in inds:
            if len(ind)<=args.maxN:
                groupDict[tuple(ind)]+=1
    # breakpoint()
    groups=list(groupDict.keys())
    if len(groups)<1:
        totals = np.array(list(range(pos.shape[0])))
        inds = [list(range(x,x+args.maxN)) for x in range(len(totals)-args.maxN)]
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
    for g in groups:
        new_pos.append(pos[np.array(list(g))])

    return new_pos, groups

if __name__=='__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args = get_args()
    makeTSNELabel()

    tsne_nets = []
    N = np.array(range(1, args.maxN + 1))
    for i in N:
        temp = SimpleRegNetwork(i * (args.input_window - 1) * 2)  # .eval()
        temp.load_state_dict(torch.load(
            '/Users/faith_johnson/GitRepos/PedTrajPred/simpleRegNet_diffsData_' +
            str(i) + 'people_' + str(args.input_window) + 'window.pt',map_location=torch.device('cpu')))
        temp.eval()
        tsne_nets.append(temp.to(device))

    if args.train:
        nets = []
        for i in range(5 * args.maxN):
            # net = SimplestAutoEncoder(args).to(device)
            # net = SimplestUNet(args)
            temp = SimplestNet(args).to(device)
            # net = SimplestConvNet(args)
            # net = SimplestVAE(args)
            # net = CNNTrajNet(args)
            temp.load_state_dict(torch.load('simpleNetTSNE_cluster'+str(i)+'.pt',map_location=torch.device('cpu')))
            nets.append(temp)
        # net.load_state_dict(torch.load('simpleAutoEnc_output.pt'))
        nets = train(args, nets, device, tsne_nets)

    nets = []
    for i in range(5 * args.maxN):
        # net = SimplestAutoEncoder(args).to(device)
        # net = SimplestUNet(args)
        temp = SimplestNet(args).to(device)
        # net = SimplestConvNet(args)
        # net = SimplestVAE(args)
        # net = CNNTrajNet(args)
        temp.load_state_dict(torch.load('/Users/faith_johnson/GitRepos/PedTrajPred/equalClusterData/simpleNetTSNE_cluster'+str(i)+'.pt',map_location=torch.device('cpu')))
        temp.eval()
        nets.append(temp)

    preds, inputs, latents = test(args, nets, device, tsne_nets)
    # breakpoint()
    if len(latents)>0:
        tsne=TSNE()
        latents=tsne.fit_transform(latents)
        graph(args, latents, name='Learned Latent Space')

    for i in range(10):
        graph(args, inputs, preds, name='Inputs vs Predictions')

    plt.show()
