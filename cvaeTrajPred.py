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

from disentanglement_pytorch.models.cvae import CVAEModel
from disentanglement_pytorch.architectures.encoders.linear import ShallowGaussianLinear, DeepGaussianLinear
from disentanglement_pytorch.architectures.encoders.simple_conv64 import SimpleConv64, SimpleGaussianConv64
from disentanglement_pytorch.architectures.decoders.linear import ShallowLinear, DeepLinear
from disentanglement_pytorch.architectures.decoders.simple_conv64 import SimpleConv64
from disentanglement_pytorch.architectures.others.tiler_networks import SingleTo2DChannel

CLUSTERS_PER_N = {1:10, 2:20, 3:35, 4:40, 5:40}

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_clusters', default=98, type=int, help='number of clusters for kmeans')
    parser.add_argument('--input_window', default=8, type=int, help='number of frames for the input data')
    parser.add_argument('--output_window', default=0, type=int, help='number of frames for the output data')
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--latent_dim', default=16, type=int)
    parser.add_argument('--maxN', default=5, type=int)
    parser.add_argument('--epochs', default=50, type=int)

    parser.add_argument('--ygrid_num', default=10, type=int)
    parser.add_argument('--social_thresh', default=0.2, type=float)#0.9 for trajData
    args = parser.parse_args()
    return args

def zeroPad(pos, args):
    npad = args.maxN - pos.shape[0]
    extra = torch.zeros(npad, args.input_window, 2)
    return torch.cat((pos,extra), dim=-2)

def train(args, net):
    for e in range(args.epochs):
        for name in ['ETH', 'ETH_Hotel', 'UCY_Zara1', 'UCY_Zara2']:
            dataset = PlainTrajData(name, input_window=args.input_window, output_window=args.output_window, maxN=args.maxN)
            loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
            for data in loader:
                tsne =
                output=net(data['pos'])


args = get_args()
encoder = ShallowGaussianLinear(args.latent_dim, num_channels=2, seq_len= args.input_window, num_people=args.maxN)
decoder = ShallowLinear(args.latent_dim, num_channels=2, seq_len= args.input_window, num_people=args.maxN)
tiler = SingleTo2DChannel(seq_len= args.input_window, num_people=args.maxN)
net = CVAEModel(encoder, decoder, tiler, num_classes=args.num_clusters)

tsne_nets=[]
N=np.array(range(1,args.maxN+1))
for i in N:
    temp = SimpleRegNetwork(i * args.input_window-1 * 2).eval()
    temp.load_state_dict(torch.load('/Users/faith_johnson/GitRepos/PedTrajPred/weights/simpleRegNet_diffsData_'+str(i)+'people_'+str(args.input_window)+'window.pt'))
    tsne_nets.append(temp)

train(args, net, tsne_nets)

