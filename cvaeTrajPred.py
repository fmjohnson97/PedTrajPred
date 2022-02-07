# --alg=CVAE \
# --controlled_capacity_increase=true \
# --traverse_z=true \
# --traverse_c=true \
# --encoder=SimpleGaussianConv64 \
# --decoder=SimpleConv64 \
# --label_tiler=MultiTo2DChannel \
# --z_dim=8 \
# --w_kld=5 \
# --lr_G=0.0004 \
# --include_labels 1 \

import argparse
import torch
from plainTrajData import PlainTrajData
from torch.utils.data import DataLoader
import numpy as np
from simpleTSNEPredict import SimpleRegNetwork
from socialHeatmapMultiN import getNewGroups
import pandas as pd
from cvaeModels import CVAE_Net
from tqdm import tqdm

# from disentanglement_pytorch.models.cvae import CVAEModel
# from disentanglement_pytorch.architectures.encoders.linear import ShallowGaussianLinear, DeepGaussianLinear
# from disentanglement_pytorch.architectures.encoders.simple_conv64 import SimpleConv64, SimpleGaussianConv64
# from disentanglement_pytorch.architectures.decoders.linear import ShallowLinear, DeepLinear
# from disentanglement_pytorch.architectures.decoders.simple_conv64 import SimpleConv64
# from disentanglement_pytorch.architectures.others.tiler_networks import SingleTo2DChannel

CLUSTERS_PER_N = {1:10, 2:20, 3:35, 4:40, 5:40}

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_clusters', default=145, type=int, help='number of clusters for kmeans')
    parser.add_argument('--input_window', default=8, type=int, help='number of frames for the input data')
    parser.add_argument('--output_window', default=12, type=int, help='number of frames for the output data')
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--latent_dim', default=16, type=int)
    parser.add_argument('--maxN', default=5, type=int)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--social_thresh', default=0.2, type=float)  # 0.9 for trajData
    parser.add_argument('--lr', default=1e-3, type=float)

    args = parser.parse_args()
    return args


def makeTSNELabel(prefix):
    global GT_TSNE_VALUES
    global TSNE_N_CUTOFFS
    GT_TSNE_VALUES = pd.DataFrame(columns=['tsne_X','tsne_Y','kmeans'])
    TSNE_N_CUTOFFS = {}
    max_label = 0
    for i in range(1,args.maxN+1):
        data = pd.read_csv('data/'+prefix+'_'+str(i)+'thresh_'+str(args.input_window)+'window.csv')
        temp = data.filter(['tsne_X', 'tsne_Y', 'kmeans'])
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


def zeroPad(pos, args):
    npad = args.maxN - pos.shape[0]
    extra = torch.zeros(npad, args.input_window, 2)
    return torch.cat((pos,extra), dim=-2)


def train(args, net, tsne_nets):
    loss_func = torch.nn.BCELoss() #torch.nn.MSELoss() #torch.nn.L1Loss() #
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    totalData = 0
    for name in ['ETH', 'ETH_Hotel', 'UCY_Zara1', 'UCY_Zara2']:
        dataset = PlainTrajData(name, input_window=args.input_window, output_window=args.output_window, maxN=args.maxN)
        totalData+=len(dataset)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs * totalData, 1e-12)
    avg_loss = []
    for e in tqdm(range(args.epochs)):
        ep_loss=[]
        for name in ['ETH', 'ETH_Hotel', 'UCY_Zara1', 'UCY_Zara2']:
            dataset = PlainTrajData(name, input_window=args.input_window, output_window=args.output_window, maxN=args.maxN)
            loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
            for data in loader:
                if data['pos'].nelement() > 0:
                    pos, groups = getNewGroups(data['diffs'][0], args)
                    # breakpoint()
                    for i, p in enumerate(pos):
                        with torch.no_grad():
                            tsne_net = tsne_nets[p.shape[-3] - 1]
                            tsne = tsne_net(p[:,:(args.input_window-1),:].flatten().float())
                            # tsne_label = getTSNELabel(tsne)
                        target = data['pos'][0][np.array(list(groups[i]))]
                        output, mu, log_sigma, z = net(target[:,:args.input_window,:].reshape(args.input_window, -1, 2).float(), torch.stack([tsne]*p.shape[-3]).float())#torch.tensor(tsne_label))
                        # breakpoint()
                        opt.zero_grad()
                        # reconstruction loss
                        # people are using binary cross entropy here
                        rc_loss = loss_func(output,target[:,args.input_window:,:].reshape(-1, args.output_window*2).float())
                        #kl loss
                        kl_loss = -0.5 * torch.sum(-torch.exp(log_sigma) - mu.pow(2) + 1. + log_sigma) / args.batch_size
                        loss = rc_loss + kl_loss
                        loss.backward()
                        opt.step()
                        scheduler.step()
                        ep_loss.append(loss.item())

        print('Epoch',e,': Loss =',np.mean(ep_loss))
        avg_loss.append(np.mean(ep_loss))
        torch.save(net.state_dict(), 'cvaeTraj.pt')

if __name__ == '__main__':
    args = get_args()
    # encoder = ShallowGaussianLinear(args.latent_dim, num_channels=2, seq_len= args.input_window, num_people=args.maxN)
    # decoder = ShallowLinear(args.latent_dim, num_channels=2, seq_len= args.input_window, num_people=args.maxN)
    # tiler = SingleTo2DChannel(seq_len= args.input_window, num_people=args.maxN)
    # net = CVAEModel(encoder, decoder, tiler, num_classes=args.num_clusters)
    net=CVAE_Net(args)

    tsne_nets=[]
    N=np.array(range(1,args.maxN+1))
    for i in N:
        temp = SimpleRegNetwork(i * (args.input_window-1) * 2).eval()
        temp.load_state_dict(torch.load('/Users/faith_johnson/GitRepos/PedTrajPred/weights/simpleRegNet_diffsData_'+str(i)+'people_'+str(args.input_window)+'window.pt'))
        tsne_nets.append(temp.eval())

    makeTSNELabel('diffsData')

    train(args, net, tsne_nets)

