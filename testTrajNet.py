import torch
import torch.nn as nn
import argparse
import numpy as np
from torch.utils.data import DataLoader
from glob import glob
import random
from matplotlib import pyplot as plt
from collections import defaultdict
import copy

from plainTrajData import PlainTrajData
from TrajNet.trajnetplusplusbaselines.trajnetbaselines.lstm.lstm import LSTM
from TrajNet.trajnetplusplusbaselines.trajnetbaselines.vae.vae import VAE
from TrajNet.trajnetplusplusbaselines.trajnetbaselines.sgan.sgan import SGAN, LSTMDiscriminator, LSTMGenerator
from TrajNet.trajnetplusplusbaselines.trajnetbaselines.lstm.non_gridbased_pooling import NN_Pooling, HiddenStateMLPPooling, AttentionMLPPooling, DirectionalMLPPooling
from TrajNet.trajnetplusplusbaselines.trajnetbaselines.lstm.non_gridbased_pooling import NN_LSTM, TrajectronPooling, SAttention_fast
from TrajNet.trajnetplusplusbaselines.trajnetbaselines.lstm.more_non_gridbased_pooling import NMMP
from TrajNet.trajnetplusplusbaselines.trajnetbaselines.lstm.gridbased_pooling import GridBasedPooling

# Social LSTM args: --type social --augment --n 16 --embedding_arch two_layer --layer_dims 1024
# DLSTM args: --type directional --augment
#Vanilla VAE args: none
#SGAN args: --type directional --augment
def get_args():
    parser=argparse.ArgumentParser()
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--epochs', default=5, type=int)

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
    parser.add_argument('--type', default='vanilla',
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
    # parser.add_argument('--noise_dim', type=int, default=64, help='noise dim of VAE')

    #SGAN parameters
    parser.add_argument('--no_noise', action='store_true',help='flag to not add noise (i.e. deterministic model)')
    parser.add_argument('--noise_dim', type=int, default=16, help='dimension of noise z')
    parser.add_argument('--noise_type', default='gaussian',choices=('gaussian', 'uniform'),help='type of noise to be added')
    parser.add_argument('--g_steps', default=1, type=int,help='number of steps of generator training')
    parser.add_argument('--d_steps', default=1, type=int, help='number of steps of discriminator training')
    parser.add_argument('--g_step_size', default=10, type=int, help='step_size of generator scheduler')
    parser.add_argument('--d_step_size', default=10, type=int,help='step_size of discriminator scheduler')
    return parser.parse_args()

def getNewGroups(pos, diffs, args):
    # hard coding grid to be 3:4 (rows:columns) since that's aspect ratio of the images

    groupDict=defaultdict(int)
    # breakpoint()
    for i,p in enumerate(pos): # blue, orange, green, red, purple, brown
        dists = torch.sum(torch.sum((pos-p)**2,axis=-1)**.5, axis=-1)
        # print(dists)
        # breakpoint()
        inds=np.where(dists<0.9)
        for ind in inds:
            if len(ind)<=3:
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
        # breakpoint()
        return [],[]

    new_pos=[]
    new_diffs=[]
    new_allDiffs=[]
    for g in groups:
        new_pos.append(pos[np.array(list(g))])
    # breakpoint()
    return new_pos, groups #, new_allDiffs, new_diffs, groups

def train(args, model, device, save_name):
    model.train()
    loss_func = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs * 9000, 1e-16)
    for e in range(args.epochs):
        avgLoss=[]
        for name in ['ETH', 'ETH_Hotel', 'UCY_Zara1', 'UCY_Zara2']:
            dataset = PlainTrajData(name, input_window=args.obs_length, output_window=args.pred_length, maxN=5, split='train')
            loader = DataLoader(dataset, batch_size=1, shuffle=False)
            for data in loader:
                if data['pos'].nelement() > 0:
                    if data['diffs'].shape[-3] < 3:
                        # breakpoint()
                        scene = data['pos'][0].transpose(0,1).float().to(device)
                        #For LSTMs
                        rel_outputs, outputs, CLUSTER_LATENT = model(scene[:args.obs_length], None, torch.tensor([0,scene.shape[0]]), scene[args.obs_length:])
                        # for VAEs and SGANs
                        # all_outputs = model(scene[:args.obs_length], None, torch.tensor([0, scene.shape[0]]),
                        #                     scene[args.obs_length:])
                        # rel_outputs, outputs = all_outputs[0][0], all_outputs[1][0]
                        opt.zero_grad()
                        # for LSTMS and VAEs
                        loss = loss_func(outputs, scene[:-1])
                        # for SGANs
                        # loss = loss_func(outputs, scene[2:])
                        loss.backward()
                        opt.step()
                        scheduler.step()
                        avgLoss.append(loss.item())
                    else:
                        # breakpoint()
                        pos, groups = getNewGroups(data['pos'][0], data['diffs'][0], args)
                        opt.zero_grad()
                        loss=0
                        for i, p in enumerate(pos):
                            scene = p.transpose(0, 1).float().to(device)
                            #For LSTMs
                            rel_outputs, outputs, CLUSTER_LATENT = model(scene[:args.obs_length], None,
                                                                         torch.tensor([0, scene.shape[0]]),
                                                                         scene[args.obs_length:])
                            # for VAEs and SGANs
                            # all_outputs = model(scene[:args.obs_length], None, torch.tensor([0, scene.shape[0]]),
                            #                     scene[args.obs_length:])
                            # rel_outputs, outputs = all_outputs[0][0], all_outputs[1][0]
                            # for LSTMS and VAEs
                            loss += loss_func(outputs, scene[:-1])
                            # for SGANs
                            # loss += loss_func(outputs, scene[2:])
                        loss/=len(pos)
                        loss.backward()
                        avgLoss.append(loss.item())
                        opt.step()
                        scheduler.step()

        print('Epoch',e,': Avg Loss =',np.mean(avgLoss))
        torch.save(model.state_dict(),save_name)
    return model

@torch.no_grad()
def test(args, model, device):
    model.eval()
    loss_func = nn.MSELoss()
    avgLoss=[]
    avgUnNormedLoss=[]
    fde=[]
    unNormedFDE=[]
    preds = []
    inputs = []
    for name in ['ETH', 'ETH_Hotel', 'UCY_Zara1', 'UCY_Zara2']:
        dataset = PlainTrajData(name, input_window=args.obs_length, output_window=args.pred_length, maxN=5, split='test')
        loader = DataLoader(dataset, batch_size=1, shuffle=False)
        for data in loader:
            if data['pos'].nelement() > 0:
                if data['diffs'].shape[-3] < 3:
                    # breakpoint()
                    scene = data['pos'][0].transpose(0,1).float().to(device)
                    #for LSTMs
                    # rel_outputs, outputs, CLUSTER_LATENT = model(scene[:args.obs_length], None, torch.tensor([0,scene.shape[0]]), scene[args.obs_length:])
                    #for VAEs and SGANs
                    all_outputs= model(scene[:args.obs_length], None, torch.tensor([0,scene.shape[0]]), scene[args.obs_length:])
                    rel_outputs, outputs = all_outputs[0][0], all_outputs[1][0]
                    # for LSTMS and VAEs
                    # loss = loss_func(outputs, scene[1:])
                    # for SGANs
                    loss = loss_func(outputs, scene[2:])
                    avgLoss.append(loss.item())
                    fde.append(loss_func(outputs[-1], scene[-1]).item())
                    unNormedOutput = outputs * (dataset.max-dataset.min) + dataset.min #(np.array(positions)-self.min)/(self.max-self.min)
                    # for LSTMS and VAEs
                    # unNormedScene = scene[1:] * (dataset.max - dataset.min) + dataset.min
                    # for SGANs
                    unNormedScene = scene[2:] * (dataset.max - dataset.min) + dataset.min
                    unNormedLoss = loss_func(unNormedOutput, unNormedScene)
                    avgUnNormedLoss.append(unNormedLoss.item())
                    unNormedFDE.append(loss_func(unNormedOutput[-1], unNormedScene[-1]).item())
                    preds.append(torch.transpose(outputs, 0,1).numpy())
                    inputs.append(data['pos'][0].float())
                else:
                    # breakpoint()
                    pos, groups = getNewGroups(data['pos'][0], data['diffs'][0], args)
                    for i, p in enumerate(pos):
                        scene = p.transpose(0, 1).float().to(device)
                        #for LSTMs
                        # rel_outputs, outputs, CLUSTER_LATENT = model(scene[:args.obs_length], None,
                        #                                              torch.tensor([0, scene.shape[0]]),
                        #                                              scene[args.obs_length:])
                        #for VAEs and SGANs
                        all_outputs = model(scene[:args.obs_length], None,torch.tensor([0, scene.shape[0]]),scene[args.obs_length:])
                        rel_outputs, outputs = all_outputs[0][0], all_outputs[1][0]
                        # breakpoint()
                        #for LSTMS and VAEs
                        # loss = loss_func(outputs, scene[1:])
                        #for SGANs
                        loss = loss_func(outputs, scene[2:])
                        avgLoss.append(loss.item())
                        fde.append(loss_func(outputs[-1], scene[-1]).item())
                        unNormedOutput = outputs * (dataset.max - dataset.min) + dataset.min  # (np.array(positions)-self.min)/(self.max-self.min)
                        #for LSTMS and VAEs
                        # unNormedScene = scene[1:] * (dataset.max - dataset.min) + dataset.min
                        #for SGANs
                        unNormedScene = scene[2:] * (dataset.max - dataset.min) + dataset.min
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
            plt.scatter(pos[-args.pred_length:,0], pos[-args.pred_length:,1], c='tab:orange')
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

# breakpoint()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# file = glob('TrajNet/trajnetplusplusbaselines/OUTPUT_BLOCK/trajdata2/lstm_' + str(args.type) + '_None.pkl.*.state')
# save_name='lstm_' + str(args.type) + '_None.pkl.*.state'
# model = LSTM(pool=pool,
#              embedding_dim=args.coordinate_embedding_dim,
#              hidden_dim=args.hidden_dim,
#              goal_flag=args.goals,
#              goal_dim=args.goal_dim)

# file = glob('TrajNet/trajnetplusplusbaselines/OUTPUT_BLOCK/trajdata2/vae_' + str(args.type) + '_None.pkl.*.state')
# save_name = 'vae_' + str(args.type) + '_None.pkl.*.state'
# model = VAE(pool=pool,
#             embedding_dim=args.coordinate_embedding_dim,
#             hidden_dim=args.hidden_dim,
#             goal_flag=args.goals,
#             goal_dim=args.goal_dim,
#             num_modes=args.k,
#             desire_approach=args.desire,
#             latent_dim=args.noise_dim)

file = glob('TrajNet/trajnetplusplusbaselines/OUTPUT_BLOCK/trajdata2/sgan_' + str(args.type) + '_None.pkl.*.state')
save_name = 'sgan_' + str(args.type) + '_None.pkl.*.state'
lstm_generator = LSTMGenerator(embedding_dim=args.coordinate_embedding_dim, hidden_dim=args.hidden_dim,
                                   pool=pool, goal_flag=args.goals, goal_dim=args.goal_dim, noise_dim=args.noise_dim,
                                   no_noise=args.no_noise, noise_type=args.noise_type)
lstm_discriminator = LSTMDiscriminator(embedding_dim=args.coordinate_embedding_dim,
                                           hidden_dim=args.hidden_dim, pool=copy.deepcopy(pool),
                                           goal_flag=args.goals, goal_dim=args.goal_dim)
model = SGAN(generator=lstm_generator, discriminator=lstm_discriminator, g_steps=args.g_steps,
                 d_steps=args.d_steps, k=args.k)

if len(file)>0:
    file.sort()
    print("Loading Model Dict")
    print('Using',file[-1])
    # with open(file[-1], 'rb') as f:
    with open(save_name, 'rb') as f:
        checkpoint = torch.load(f, map_location=torch.device('cpu'))
    pretrained_state_dict = checkpoint#['state_dict']
    model.load_state_dict(pretrained_state_dict, strict=True)
    print('Total Model Parameters:',sum(p.numel() for p in model.parameters() if p.requires_grad))
else:
    print('!!! Starting from Random !!!')

model = model.to(device)

if args.train:
    model = train(args, model, device, save_name)
preds, inputs = test(args, model, device)

for i in range(20):
    graph(args, inputs, preds, name=args.type+' LSTM -- Inputs vs Predictions', index=i)

plt.show()