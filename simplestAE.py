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
from social_cnn_pytorch.main_scripts.train import CNNTrajNet
from trajAugmentations import TrajAugs
from simpleTSNEPredict import SimpleRegNetwork
from sameSizeData import SameSizeData

CLUSTERS_PER_N = {1:10, 2:20, 3:35, 4:40, 5:40}

class TSNENet(nn.Module):
    def __init__(self, input_size):
        super(TSNENet, self).__init__()
        self.fc1=nn.Linear(input_size,16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 8)
        self.outfc=nn.Linear(8, 2)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        res1=self.relu(self.fc1(x))
        res2=self.relu(self.fc2(res1))
        res3 = self.relu(self.fc3(res2))
        x=self.outfc(res3)
        return x

class SimplestAutoEncoder(nn.Module):
    def __init__(self, args):
        super(SimplestAutoEncoder, self).__init__()
        # self.efc1 = nn.Linear(args.input_window*2, 16)
        # self.efc2 = nn.Linear(16, 8)
        # self.efc3 = nn.Linear(16+8, 8)
        # self.dfc1 = nn.Linear(8, 16)
        # self.dfc2 = nn.Linear(16+8, 16)
        # # self.dfc3 = nn.Linear(8+16, args.input_window*2) # for in-->in prediction
        # self.dfc3 = nn.Linear(8 + 16, args.output_window * 2) #for in-->out prediction
        # self.dfc3 = nn.Linear(8 + 16, 2) # for tsne???

        self.efc1 = nn.Linear(args.input_window * 2 * args.traj_thresh, 64)
        self.efc2 = nn.Linear(64, 32)
        self.efc3 = nn.Linear(64+32, 16)
        self.dfc1 = nn.Linear(16, 32)
        self.dfc2 = nn.Linear(16+32, 64)
        self.dfc3 = nn.Linear(64+16, args.output_window*2*args.traj_thresh)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        # breakpoint()
        res1 = self.relu(self.efc1(x))
        res2 = self.relu(self.efc2(res1))
        latent = self.relu(self.efc3(torch.cat((res2,res1), dim=-1)))
        res3 = self.relu(self.dfc1(latent))
        res4 = self.relu(self.dfc2(torch.cat((res3,latent), dim=-1)))
        x = self.dfc3(torch.cat((res4,latent), dim=-1))
        return x, latent

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
    parser.add_argument('--traj_thresh', default=2, type=int)
    args = parser.parse_args()
    return args

def train(args, net, device, tsne_net, learned_tsne):
    min_loss = np.inf
    trajAugs = TrajAugs()
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs * 10000, 1e-12)
    loss_func = nn.MSELoss() #nn.BCELoss() #
    for e in tqdm(range(args.epochs)):
        avgLoss=[]
        for name in ['ETH', 'ETH_Hotel', 'UCY_Zara1', 'UCY_Zara2']:
            dataset = SameSizeData(name, input_window=args.input_window, output_window=args.output_window,
                               trajThresh=args.traj_thresh)
            loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
            for data in loader:
                if data['pos'].nelement()>0:
                    # breakpoint()
                    scene = trajAugs.augment(data['pos'][0].numpy())
                    with torch.no_grad():
                        tsne = tsne_net(data['diffs'][0][:, :(args.input_window - 1), :].flatten().float())
                    # breakpoint()
                    output, latent = net(scene[:, :args.input_window, :].reshape(-1, (args.input_window) * 2 * args.traj_thresh).float().to(device))
                    pred_tsne=learned_tsne(latent)
                    opt.zero_grad()
                    loss1 = loss_func(output, scene[:, args.input_window:, :].reshape(-1, (args.output_window) * 2 * args.traj_thresh).float().to(device))
                    loss2 = loss_func(pred_tsne,tsne.unsqueeze(0))
                    loss=loss1+loss2
                    loss.backward()
                    opt.step()
                    scheduler.step()
                    avgLoss.append(loss.item())
        print("Epoch",e,': Loss =',np.mean(avgLoss))
        if np.mean(avgLoss)<min_loss:
            torch.save(net.state_dict(),'simpleAE_'+str(args.traj_thresh)+'.pt')
            torch.save(learned_tsne.state_dict(), 'learned_tsne_'+str(args.traj_thresh)+'.pt')
            min_loss=np.mean(avgLoss)
    return net, learned_tsne

@torch.no_grad()
def test(args, net, device, learned_tsne):
    net.eval()
    loss_func = nn.MSELoss() #nn.BCELoss() #
    avgLoss=[]
    preds = []
    inputs = []
    latents = []
    for name in ['ETH', 'ETH_Hotel', 'UCY_Zara1', 'UCY_Zara2']:
        dataset = SameSizeData(name, input_window=args.input_window, output_window=args.output_window,
                               trajThresh=args.traj_thresh)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        for data in loader:
            if data['pos'].nelement() > 0 and data['pos'].shape[1] == args.traj_thresh:
                # breakpoint()
                scene = data['pos'][0]
                output, latent = net(scene[:, :args.input_window, :].reshape(-1, (args.input_window) * 2 * args.traj_thresh).float().to(device))
                pred_tsne = learned_tsne(latent)
                loss1 = loss_func(output,scene[:, args.input_window:, :].reshape(-1, (args.input_window) * 2 * args.traj_thresh).float().to(device))
                loss2 = loss_func(pred_tsne, tsne.unsqueeze(0))
                loss = loss1 + loss2
                avgLoss.append(loss.item())
                preds.append(output.numpy())
                inputs.append(data['pos'][0].float())
                latents.extend(latent.numpy())
    print('Avg Test Loss =',np.mean(avgLoss))
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

if __name__=='__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args = get_args()
    net = SimplestAutoEncoder(args).to(device)
    learned_tsne=TSNENet(16)

    tsne_net= SimpleRegNetwork(args.traj_thresh*(args.input_window-1)*2)
    tsne_net.load_state_dict(torch.load('simpleRegNet_diffsData_' + str(args.traj_thresh) + 'people_' + str(args.input_window)+'window.pt'))
    tsne_net=tsne_net.eval()

    if args.train:
        # net.load_state_dict(torch.load('simpleAE_'+str(args.traj_thresh)+'.pt'))
        net, learned_tsne = train(args, net.to(device), device, tsne_net, learned_tsne)

    net.load_state_dict(torch.load('simpleAE_'+str(args.traj_thresh)+'.pt'))
    learned_tsne.load_state_dict(torch.load('learned_tsne.pt'))
    net.eval()
    net = net.to(device)

    preds, inputs, latents = test(args, net, device, learned_tsne)
    # breakpoint()
    if len(latents)>0:
        tsne=TSNE()
        latents=tsne.fit_transform(latents)
        graph(args, latents, name='Learned Latent Space')

    for i in range(10):
        graph(args, inputs, preds, name='Inputs vs Predictions')

    plt.show()
