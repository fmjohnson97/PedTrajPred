import torch
import torch.nn as nn
import argparse
import numpy as np
from torch.utils.data import DataLoader
from glob import glob
import random
from matplotlib import pyplot as plt

from plainTrajData import PlainTrajData
from TrajNet.trajnetplusplusbaselines.trajnetbaselines.lstm.lstm import LSTM
from TrajNet.trajnetplusplusbaselines.trajnetbaselines.lstm.non_gridbased_pooling import NN_Pooling, HiddenStateMLPPooling, AttentionMLPPooling, DirectionalMLPPooling
from TrajNet.trajnetplusplusbaselines.trajnetbaselines.lstm.non_gridbased_pooling import NN_LSTM, TrajectronPooling, SAttention_fast
from TrajNet.trajnetplusplusbaselines.trajnetbaselines.lstm.more_non_gridbased_pooling import NMMP
from TrajNet.trajnetplusplusbaselines.trajnetbaselines.lstm.gridbased_pooling import GridBasedPooling


def get_args():
    parser=argparse.ArgumentParser()
    parser.add_argument('--get_latents', action='store_true',
                        help='signals to run the input through the network instead of reading stored latents from file')
    parser.add_argument('--get_tsne', action='store_true', help='signals to do tsne')

    # Data Params
    parser.add_argument('--path', default='trajnetplusplusbaselines/DATA_BLOCK/trajdata', help='glob expression for data files')
    parser.add_argument('--sample', default=1.0, type=float, help='sample ratio when loading train/val scenes')
    parser.add_argument('--goals', action='store_true', help='flag to consider goals of pedestrians')
    parser.add_argument('--augment', action='store_true', help='perform rotation augmentation')
    parser.add_argument('--obs_length', default=9, type=int, help='observation length')
    parser.add_argument('--pred_length', default=12, type=int, help='prediction length')

    # Training Params
    parser.add_argument('--loss', default='pred', choices=('L2', 'pred'), help='loss objective, L2 loss (L2) and Gaussian loss (pred)')

    # Model Params
    parser.add_argument('--type', default='social',
                        choices=('vae','vanilla', 'occupancy', 'directional', 'social', 'hiddenstatemlp', 's_att_fast',
                                 'directionalmlp', 'nn', 'attentionmlp', 'nn_lstm', 'traj_pool', 'nmmp', 'dir_social'),
                        help='type of interaction encoder')
    parser.add_argument('--hidden-dim', type=int, default=128,
                                 help='LSTM hidden dimension')
    parser.add_argument('--coordinate-embedding-dim', type=int, default=64,
                                 help='coordinate embedding dimension')
    parser.add_argument('--goal_dim', type=int, default=64,
                                 help='goal embedding dimension')

    # Pooling Params
    parser.add_argument('--pool_dim', type=int, default=256,
                                 help='output dimension of interaction vector')
    parser.add_argument('--vel_dim', type=int, default=32,
                                 help='embedding dimension for relative velocity')
    parser.add_argument('--mp_iters', default=5, type=int,
                                 help='message passing iterations in NMMP')
    parser.add_argument('--spatial_dim', type=int, default=32,
                                 help='embedding dimension for relative position')
    parser.add_argument('--no_vel', action='store_true',
                                 help='flag to not consider relative velocity of neighbours')
    parser.add_argument('--cell_side', type=float, default=0.6,
                  help='cell size of real world (in m) for grid-based pooling')
    parser.add_argument('--n', type=int, default=12,
                                 help='number of cells per side for grid-based pooling')
    parser.add_argument('--front', action='store_true',
                                 help='flag to only consider pedestrian in front during grid-based pooling')
    parser.add_argument('--embedding_arch', default='one_layer',
                  help='interaction encoding arch for gridbased pooling')
    parser.add_argument('--pool_constant', default=0, type=int,
                                 help='background value (when cell empty) of gridbased pooling')
    parser.add_argument('--norm', default=0, type=int,
                                 help='normalization scheme for input batch during grid-based pooling')
    parser.add_argument('--latent_dim', type=int, default=16,
                                 help='latent dimension of encoding hidden dimension during social pooling')
    parser.add_argument('--layer_dims', type=int, nargs='*', default=[512],
                                 help='interaction module layer dims for gridbased pooling')

    # VAE Parameters
    parser.add_argument('--k', type=int, default=1, help='number of samples for reconstruction loss')
    parser.add_argument('--desire', action='store_true', help='flag to use kld version of DESIRE')
    parser.add_argument('--noise_dim', type=int, default=64, help='noise dim of VAE')

    return parser.parse_args()

def test(args, net, device):
    net.eval()
    loss_func = nn.MSELoss()
    avgLoss=[]
    preds = []
    inputs = []
    for name in ['ETH', 'ETH_Hotel', 'UCY_Zara1', 'UCY_Zara2']:
        dataset = PlainTrajData(name, input_window=args.obs_length, output_window=args.pred_length, maxN=5, split='test')
        loader = DataLoader(dataset, batch_size=1, shuffle=False)
        for data in loader:
            if data['pos'].nelement() > 0:
                # breakpoint()
                with torch.no_grad():
                    scene = data['pos'][0].transpose(0,1).float().to(device)
                    rel_outputs, outputs, CLUSTER_LATENT = model(scene[:args.obs_length], None, torch.tensor([0,scene.shape[0]]), scene[args.obs_length:])
                    loss = loss_func(outputs, scene[1:])
                avgLoss.append(loss.item())
                preds.append(torch.transpose(outputs, 0,1).numpy())
                inputs.append(data['pos'][0].float())
    print('Avg Test Loss =',np.mean(avgLoss))
    return preds, inputs

def graph(args, inputs, predictions=None, name=None):
    plt.figure()
    if predictions is None:
        plt.scatter(inputs[:,0], inputs[:,1])
    else:
        # breakpoint()
        ind = random.choice(list(range(len(inputs))))
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
    plt.title(name)


args = get_args()

pool = None
pretrained_pool = None
if args.type == 'hiddenstatemlp':
    pool = HiddenStateMLPPooling(hidden_dim=args.hidden_dim, out_dim=args.pool_dim,
                                 mlp_dim_vel=args.vel_dim)
elif args.type == 'nmmp':
    pool = NMMP(hidden_dim=args.hidden_dim, out_dim=args.pool_dim, k=args.mp_iters)
elif args.type == 'attentionmlp':
    pool = AttentionMLPPooling(hidden_dim=args.hidden_dim, out_dim=args.pool_dim,
                               mlp_dim_spatial=args.spatial_dim, mlp_dim_vel=args.vel_dim)
elif args.type == 'directionalmlp':
    pool = DirectionalMLPPooling(out_dim=args.pool_dim)
elif args.type == 'nn':
    pool = NN_Pooling(n=args.neigh, out_dim=args.pool_dim, no_vel=args.no_vel)
elif args.type == 'nn_lstm':
    pool = NN_LSTM(n=args.neigh, hidden_dim=args.hidden_dim, out_dim=args.pool_dim)
elif args.type == 'traj_pool':
    pool = TrajectronPooling(hidden_dim=args.hidden_dim, out_dim=args.pool_dim)
elif args.type == 's_att_fast':
    pool = SAttention_fast(hidden_dim=args.hidden_dim, out_dim=args.pool_dim)
elif args.type not in ['vanilla', 'vae']:
    pool = GridBasedPooling(type_=args.type, hidden_dim=args.hidden_dim,
                            cell_side=args.cell_side, n=args.n, front=args.front,
                            out_dim=args.pool_dim, embedding_arch=args.embedding_arch,
                            constant=args.pool_constant, pretrained_pool_encoder=pretrained_pool,
                            norm=args.norm, layer_dims=args.layer_dims, latent_dim=args.latent_dim)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
file = glob('TrajNet/trajnetplusplusbaselines/OUTPUT_BLOCK/trajdata/lstm_' + str(args.type) + '_None.pkl.*.state')
model = LSTM(pool=pool,
             embedding_dim=args.coordinate_embedding_dim,
             hidden_dim=args.hidden_dim,
             goal_flag=args.goals,
             goal_dim=args.goal_dim)
if len(file)>0:
    file.sort()
    print("Loading Model Dict")
    with open(file[-1], 'rb') as f:
        checkpoint = torch.load(f, map_location=torch.device('cpu'))
    pretrained_state_dict = checkpoint['state_dict']
    model.load_state_dict(pretrained_state_dict, strict=True)
else:
    print('!!! Starting from Random !!!')

model = model.to(device)

preds, inputs = test(args, model, device)

for i in range(10):
    graph(args, inputs, preds, name=args.type+' LSTM -- Inputs vs Predictions')

plt.show()