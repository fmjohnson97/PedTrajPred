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

class SimplestNet(nn.Module):
    def __init__(self, args):
        super(SimplestNet, self).__init__()
        self.fc1=nn.Linear(args.input_window*2,64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(64+32, 32)
        self.outfc=nn.Linear(32+32+64, args.output_window*2)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        res1=self.relu(self.fc1(x))
        res2=self.relu(self.fc2(res1))
        res3 = self.relu(self.fc3(torch.cat((res2,res1), dim=-1)))
        x=self.outfc(torch.cat((res3,res2,res1), dim=-1))
        return x, torch.cat((res3,res2,res1), dim=-1)

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

        # self.efc1 = nn.Linear(args.input_window * 2, 16)
        # self.efc2 = nn.Linear(16, 32)
        # self.efc3 = nn.Linear(16+32, 16)
        # self.dfc1 = nn.Linear(16, 32)
        # self.dfc2 = nn.Linear(16+32, 64)
        # self.dfc3 = nn.Linear(64+16, args.input_window*2)
        self.relu = nn.PReLU()

    def forward(self, x):
        res1 = self.relu(self.efc1(x))
        res2 = self.relu(self.efc2(res1))
        latent = self.relu(self.efc3(torch.cat((res2,res1), dim=-1)))
        res3 = self.relu(self.dfc1(latent))
        res4 = self.relu(self.dfc2(torch.cat((res3,latent), dim=-1)))
        x = self.dfc3(torch.cat((res4,latent), dim=-1))
        return x, latent

class SimplestARNet(nn.Module):
    def __init__(self, args):
        super(SimplestARNet, self).__init__()
        self.fc1 = nn.Linear(args.input_window*2, args.output_window*2)
        # self.conv1 = nn.Conv1d(args.input_window*2, args.output_window*2)

    def forward(self, x):
        return self.fc1(x)
        # return self.conv1(x)

class SimplestVAE(nn.Module):
    pass

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_window', default=8, type=int, help='number of frames for the input data')
    parser.add_argument('--output_window', default=12, type=int, help='number of frames for the output data')
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--maxN', default=5, type=int)
    parser.add_argument('--lr', default=1e-2, type=float)
    parser.add_argument('--train', action='store_true')

    args = parser.parse_args()
    return args

def train(args, net):
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs * 10000, 1e-12)
    loss_func = nn.MSELoss()
    for e in tqdm(range(args.epochs)):
        avgLoss=[]
        for name in ['ETH', 'ETH_Hotel', 'UCY_Zara1', 'UCY_Zara2']:
            dataset = PlainTrajData(name, input_window=args.input_window, output_window=args.output_window, maxN=args.maxN)
            loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
            for data in loader:
                if data['pos'].nelement()>0:
                    # breakpoint()
                    output = net(data['pos'][0, :, :args.input_window, :].reshape(-1, args.input_window*2).float())
                    # output, latent = net(data['pos'][0, :, :args.input_window, :].reshape(-1, args.input_window * 2).float())
                    # output, latent = net(data['pos'][0].reshape(-1, args.input_window * 2).float())
                    opt.zero_grad()
                    # loss = loss_func(output, data['pos'][0].reshape(-1, args.input_window * 2).float())
                    # loss = loss_func(output, data['pos'][0, :, args.input_window:, :].reshape(-1,args.output_window * 2).float())
                    loss = loss_func(output, data['pos'][0, :, args.input_window:, :].reshape(-1,args.output_window*2).float())
                    loss.backward()
                    opt.step()
                    scheduler.step()
                    avgLoss.append(loss.item())
        print("Epoch",e,': Loss =',np.mean(avgLoss))
        torch.save(net.state_dict(),'simpleARNet.pt')
    return net

def test(args, net):
    net.eval()
    loss_func = nn.MSELoss()
    avgLoss=[]
    preds = []
    inputs = []
    latents = []
    for name in ['ETH', 'ETH_Hotel', 'UCY_Zara1', 'UCY_Zara2']:
        dataset = PlainTrajData(name, input_window=args.input_window, output_window=args.output_window, maxN=args.maxN, split='test')
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        for data in loader:
            if data['pos'].nelement() > 0:
                # breakpoint()
                with torch.no_grad():
                    output = net(data['pos'][0, :, :args.input_window, :].reshape(-1, args.input_window*2).float())
                    # output, latent = net(data['pos'][0, :, :args.input_window, :].reshape(-1, args.input_window * 2).float())
                    # output, latent = net(data['pos'][0].reshape(-1, args.input_window * 2).float())
                    # loss = loss_func(output, data['pos'][0].reshape(-1, args.input_window * 2).float())
                    # loss = loss_func(output, data['pos'][0, :, args.input_window:, :].reshape(-1,args.output_window * 2).float())
                    loss = loss_func(output,data['pos'][0, :, args.input_window:, :].reshape(-1, args.output_window*2).float())
                avgLoss.append(loss.item())
                preds.append(output.numpy())
                # inputs.append(data['pos'][0].reshape(-1,args.input_window*2).numpy())
                inputs.append(data['pos'][0].reshape(-1, (args.input_window+args.output_window) * 2).numpy())
                # latents.extend(latent.numpy())
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
        for pos in predictions[ind].reshape(-1, args.output_window, 2):
            plt.plot(pos[:,0], pos[:,1], c='tab:orange')
            plt.scatter(pos[0][0], pos[0][1], c='tab:orange')
    plt.title(name)

if __name__=='__main__':
    args = get_args()
    # net=SimplestAutoEncoder(args)
    # net = SimplestUNet(args)
    # net = SimplestNet(args)
    net = SimplestARNet(args)

    if args.train:
        # net.load_state_dict(torch.load('simpleAutoEnc_output.pt'))
        net = train(args, net)
    else:
        net.load_state_dict(torch.load('simpleARNet.pt'))

    preds, inputs, latents = test(args, net)
    # breakpoint()
    if len(latents)>0:
        tsne=TSNE()
        latents=tsne.fit_transform(latents)
        graph(args, latents, name='Learned Latent Space')

    for i in range(10):
        graph(args, inputs, preds, name='Inputs vs Predictions')

    plt.show()
