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
from TrajNet.trajnetplusplusbaselines.trajnetbaselines.lstm.data_load_utils import prepare_data
import trajnetplusplustools
from TrajNet.trajnetplusplusbaselines.trajnetbaselines.lstm.utils import center_scene

class SimplestNet(nn.Module):
    def __init__(self, args):
        super(SimplestNet, self).__init__()
        self.fc1=nn.Linear((args.input_window)*2,32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64+32, 32)
        self.outfc=nn.Linear(32+32+64, args.output_window*2)
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
    parser.add_argument('--maxN', default=5, type=int)
    parser.add_argument('--lr', default=1e-2, type=float)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--path', default='/Users/faith_johnson/GitRepos/PedTrajPred/TrajNet/trajnetplusplusbaselines/DATA_BLOCK/trajdata', type=str)
    parser.add_argument('--sample', default=1.0, type=float,
                        help='sample ratio when loading train/val scenes')
    parser.add_argument('--goals', action='store_true',
                        help='flag to consider goals of pedestrians')
    # CNNTraj parameters
    parser.add_argument('--input_size', default=8, type=int, help='number of frames for the input data')
    parser.add_argument('--output_size', default=12, type=int, help='number of frames for the output data')
    parser.add_argument('--embedding_size', default=32, type=int, help='size of the embedding')
    parser.add_argument('--dropout_rate', default=0.06, type=float)

    args = parser.parse_args()
    return args

def train(args, net, device, tsne_nets):
    trajAugs = TrajAugs()
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs * 10000, 1e-12)
    loss_func = nn.MSELoss() #nn.BCELoss() #
    for e in tqdm(range(args.epochs)):
        avgLoss=[]
        train_scenes, train_goals, _ = prepare_data(args.path, subset='/train/', sample=args.sample, goals=args.goals)
        for scene_i, (filename, scene_id, paths) in enumerate(train_scenes):
            breakpoint()
            scene = trajnetplusplustools.Reader.paths_to_xy(paths)
            scene, rot, center = center_scene(scene, args.input_window+args.output_window)
            scene = torch.tensor(scene)
            scene = torch.transpose(scene, 0, 1)
            # tsne_net = tsne_nets[scene.shape[0] - 1]
            # tsne = tsne_net(data['diffs'][0][:, :(args.input_window - 1), :].flatten().float())
            # breakpoint()
            # input_scene = torch.cat((scene[:, :args.input_window, :], torch.ones(scene.shape[0],1,2)*nn.functional.normalize(tsne.unsqueeze(0))), axis = 1)
            output, latent = net(scene[:,:args.input_window,:].reshape(-1, (args.input_window) * 2).float().to(device))
            opt.zero_grad()
            loss = loss_func(output, scene[:, args.input_window:args.output_window+args.input_window, :].reshape(-1,args.output_window*2).float().to(device))
            loss.backward()
            opt.step()
            scheduler.step()
            avgLoss.append(loss.item())
        print("Epoch",e,': Loss =',np.mean(avgLoss))
        torch.save(net.state_dict(),'simpleNet.pt')
    return net

def test(args, net, device, tsne_nets):
    net.eval()
    loss_func = nn.MSELoss() #nn.BCELoss() #
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
                    output, latent = net(data['pos'][0, :, :args.input_window, :].reshape(-1, args.input_window*2).float().to(device))
                    loss = loss_func(output, data['pos'][0, :, args.input_window:, :].reshape(-1,args.output_window*2).float().to(device))
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
    # net = SimplestAutoEncoder(args).to(device)
    # net = SimplestUNet(args)
    net = SimplestNet(args).to(device)
    # net = SimplestConvNet(args)
    # net = SimplestVAE(args)
    # net = CNNTrajNet(args)

    tsne_nets = []
    N = np.array(range(1, args.maxN + 1))
    # for i in N:
    #     temp = SimpleRegNetwork(i * (args.input_window - 1) * 2).eval()
    #     temp.load_state_dict(torch.load(
    #         '/Users/faith_johnson/GitRepos/PedTrajPred/weights/simpleRegNet_diffsData_' + str(i) + 'people_' + str(
    #             args.input_window) + 'window.pt'))
    #     tsne_nets.append(temp.eval())

    if args.train:
        # net.load_state_dict(torch.load('simpleAutoEnc_output.pt'))
        net = train(args, net.to(device), device, tsne_nets)
    else:
        net.load_state_dict(torch.load('simpleNet.pt'))
        net = net.to(device)

    preds, inputs, latents = test(args, net, device, tsne_nets)
    # breakpoint()
    if len(latents)>0:
        tsne=TSNE()
        latents=tsne.fit_transform(latents)
        graph(args, latents, name='Learned Latent Space')

    for i in range(10):
        graph(args, inputs, preds, name='Inputs vs Predictions')

    plt.show()
