import gc
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import numpy as np
from torch.utils.data import DataLoader
from glob import glob
import random
from matplotlib import pyplot as plt
from collections import defaultdict

from plainTrajData import PlainTrajData
from sophie.models import  TrajectoryGenerator, TrajectoryDiscriminator
from sophie.utils import relative_to_abs
from sophie.utils import gan_g_loss, gan_d_loss, l2_loss, displacement_error, final_displacement_error
from sophie.constants import *
# from sophie.train import generator_step, discriminator_step

def discriminator_step(batch, generator, discriminator, d_loss_fn, optimizer_d):
    breakpoint()
    batch = [tensor.cuda() for tensor in batch]
    (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, vgg_list) = batch
    losses = {}
    loss = torch.zeros(1).to(pred_traj_gt)

    generator_out = generator(obs_traj, obs_traj_rel, vgg_list)

    pred_traj_fake_rel = generator_out
    pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1, :, 0, :])

    traj_real = torch.cat([obs_traj[:, :, 0, :], pred_traj_gt], dim=0)
    traj_real_rel = torch.cat([obs_traj_rel[:, :, 0, :], pred_traj_gt_rel], dim=0)
    traj_fake = torch.cat([obs_traj[:, :, 0, :], pred_traj_fake], dim=0)
    traj_fake_rel = torch.cat([obs_traj_rel[:, :, 0, :], pred_traj_fake_rel], dim=0)

    scores_fake = discriminator(traj_fake, traj_fake_rel)
    scores_real = discriminator(traj_real, traj_real_rel)

    data_loss = d_loss_fn(scores_real, scores_fake)
    losses['D_data_loss'] = data_loss.item()
    loss += data_loss
    losses['D_total_loss'] = loss.item()

    optimizer_d.zero_grad()
    loss.backward()
    optimizer_d.step()
    return losses

def generator_step(batch, generator, discriminator, g_loss_fn, optimizer_g):
    breakpoint()
    batch = [tensor.cuda() for tensor in batch]
    (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, vgg_list) = batch
    losses = {}
    loss = torch.zeros(1).to(pred_traj_gt)

    g_l2_loss_rel = []
    for _ in range(BEST_K):
        generator_out = generator(obs_traj, obs_traj_rel, vgg_list)

        pred_traj_fake_rel = generator_out
        pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1, :, 0, :])

        g_l2_loss_rel.append(l2_loss(
            pred_traj_fake_rel,
            pred_traj_gt_rel,
            mode='raw'))

    npeds = obs_traj.size(1)
    g_l2_loss_sum_rel = torch.zeros(1).to(pred_traj_gt)
    g_l2_loss_rel = torch.stack(g_l2_loss_rel, dim=1)
    _g_l2_loss_rel = torch.sum(g_l2_loss_rel, dim=0)
    _g_l2_loss_rel = torch.min(_g_l2_loss_rel) / (npeds * PRED_LEN)
    g_l2_loss_sum_rel += _g_l2_loss_rel
    losses['G_l2_loss_rel'] = g_l2_loss_sum_rel.item()
    loss += g_l2_loss_sum_rel

    traj_fake = torch.cat([obs_traj[:, :, 0, :], pred_traj_fake], dim=0)
    traj_fake_rel = torch.cat([obs_traj_rel[:, :, 0, :], pred_traj_fake_rel], dim=0)

    scores_fake = discriminator(traj_fake, traj_fake_rel)
    discriminator_loss = g_loss_fn(scores_fake)

    loss += discriminator_loss
    losses['G_discriminator_loss'] = discriminator_loss.item()
    losses['G_total_loss'] = loss.item()

    optimizer_g.zero_grad()
    loss.backward()
    optimizer_g.step()

    return losses

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight)

def get_dtypes():
    return torch.cuda.LongTensor, torch.cuda.FloatTensor


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
    parser.add_argument('--obs_length', default=8, type=int, help='observation length')
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
    parser.add_argument('--noise_dim', type=int, default=64, help='noise dim of VAE')

    #SGAN parameters
    parser.add_argument('--no_noise', action='store_true',help='flag to not add noise (i.e. deterministic model)')
    # parser.add_argument('--noise_dim', type=int, default=16, help='dimension of noise z')
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

def train(args, discriminator, generator, device, save_name_d, save_name_g):
    discriminator.train()
    generator.train()
    optimizer_g = optim.Adam(generator.parameters(), lr=G_LR)
    optimizer_d = optim.Adam(discriminator.parameters(), lr=D_LR)

    iterations_per_epoch = 9000 / D_STEPS
    NUM_ITERATIONS = int(iterations_per_epoch * args.epochs)

    t, epoch = 0, 0
    while t < NUM_ITERATIONS:
        avgLoss=[]
        gc.collect()
        d_steps_left = D_STEPS
        g_steps_left = G_STEPS
        epoch += 1
        print('Starting epoch {}'.format(epoch))
        for name in ['ETH', 'ETH_Hotel', 'UCY_Zara1', 'UCY_Zara2']:
            dataset = PlainTrajData(name, input_window=args.obs_length, output_window=args.pred_length, maxN=5, split='train')
            loader = DataLoader(dataset, batch_size=1, shuffle=False)
            for data in loader:
                if data['pos'].nelement() > 0:
                    if data['diffs'].shape[-3] < 3:
                        # breakpoint()
                        scene = data['pos'][0].transpose(0,1).float().to(device)
                        #each scene/batch should contain
                        #obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, vgg_list
                        batch=[scene[:args.obs_length,:,:],scene[args.obs_length:,:,:],
                               scene[:args.obs_length,:,:],scene[args.obs_length:,:,:],]
                        if d_steps_left > 0:
                            losses_d = discriminator_step(scene, generator,
                                                          discriminator, gan_d_loss,
                                                          optimizer_d)
                            d_steps_left -= 1
                        elif g_steps_left > 0:
                            losses_g = generator_step(scene, generator,
                                                      discriminator, gan_g_loss,
                                                      optimizer_g)
                            g_steps_left -= 1

                        if d_steps_left > 0 or g_steps_left > 0:
                            continue

                        avgLoss.append(sum(losses_g)+sum(losses_d))

                    else:
                        # breakpoint()
                        pos, groups = getNewGroups(data['pos'][0], data['diffs'][0], args)
                        for i, p in enumerate(pos):
                            scene = p.transpose(0, 1).float().to(device)
                            if d_steps_left > 0:
                                losses_d = discriminator_step(scene, generator,
                                                              discriminator, gan_d_loss,
                                                              optimizer_d)
                                d_steps_left -= 1
                            elif g_steps_left > 0:
                                losses_g = generator_step(scene, generator,
                                                          discriminator, gan_g_loss,
                                                          optimizer_g)
                                g_steps_left -= 1

                            if d_steps_left > 0 or g_steps_left > 0:
                                continue

                        avgLoss.append(sum(losses_g)+sum(losses_d))
            t += 1
            d_steps_left = D_STEPS
            g_steps_left = G_STEPS
            if t >= NUM_ITERATIONS:
                print('Epoch', epoch, ': Avg Loss =', np.mean(avgLoss))
                torch.save(discriminator.state_dict(), save_name_d)
                torch.save(generator.state_dict(), save_name_g)
                break

        print('Epoch',epoch,': Avg Loss =',np.mean(avgLoss))
        torch.save(discriminator.state_dict(), save_name_d)
        torch.save(generator.state_dict(), save_name_g)

    return discriminator, generator

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
                    loss = loss_func(outputs, scene[1:])
                    # for SGANs
                    # loss = loss_func(outputs, scene[2:])
                    avgLoss.append(loss.item())
                    fde.append(loss_func(outputs[-1], scene[-1]).item())
                    unNormedOutput = outputs * (dataset.max-dataset.min) + dataset.min #(np.array(positions)-self.min)/(self.max-self.min)
                    # for LSTMS and VAEs
                    unNormedScene = scene[1:] * (dataset.max - dataset.min) + dataset.min
                    # for SGANs
                    # unNormedScene = scene[2:] * (dataset.max - dataset.min) + dataset.min
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
                        loss = loss_func(outputs, scene[1:])
                        #for SGANs
                        # loss = loss_func(outputs, scene[2:])
                        avgLoss.append(loss.item())
                        fde.append(loss_func(outputs[-1], scene[-1]).item())
                        unNormedOutput = outputs * (dataset.max - dataset.min) + dataset.min  # (np.array(positions)-self.min)/(self.max-self.min)
                        #for LSTMS and VAEs
                        unNormedScene = scene[1:] * (dataset.max - dataset.min) + dataset.min
                        #for SGANs
                        # unNormedScene = scene[2:] * (dataset.max - dataset.min) + dataset.min
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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

generator = TrajectoryGenerator()
generator.apply(init_weights)
generator.float().train()
print('Here is the generator:')
print(generator)

discriminator = TrajectoryDiscriminator()
discriminator.apply(init_weights)
discriminator.float().train()
print('Here is the discriminator:')
print(discriminator)




save_name_d='sophie_' + str(discriminator) + '_None.pkl.*.state'
save_name_g='sophie_' + str(discriminator) + '_None.pkl.*.state'
file=glob(save_name_g)
if len(file)>0:
    print("Loading Model Dict")
    print('Using',file[-1])
    with open(save_name_g, 'rb') as f:
        checkpoint = torch.load(f, map_location=torch.device('cpu'))
    generator.load_state_dict(checkpoint, strict=True)
    params_g=sum(p.numel() for p in generator.parameters() if p.requires_grad)
    print('Generator Model Parameters:',params_g)

    print('Using', save_name_d)
    with open(save_name_d, 'rb') as f:
        checkpoint = torch.load(f, map_location=torch.device('cpu'))
    discriminator.load_state_dict(checkpoint, strict=True)
    params_d = sum(p.numel() for p in generator.parameters() if p.requires_grad)
    print('Generator Model Parameters:', params_d)
    print('total params: ',params_d+params_g)
else:
    print('!!! Starting from Random !!!')

discriminator = discriminator.to(device)
generator = generator.to(device)

if args.train:
    discriminator, generator = train(args, discriminator, generator, device, save_name_d, save_name_g)
preds, inputs = test(args, model, device)

for i in range(20):
    graph(args, inputs, preds, name=args.type+' LSTM -- Inputs vs Predictions', index=i)

plt.show()