import argparse
import torch
from glob import glob
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from sklearn.manifold import TSNE
import random

import trajnetplusplustools
from trajnetbaselines.lstm.data_load_utils import prepare_data
from trajnetbaselines.lstm.non_gridbased_pooling import NN_Pooling, HiddenStateMLPPooling, AttentionMLPPooling, DirectionalMLPPooling
from trajnetbaselines.lstm.non_gridbased_pooling import NN_LSTM, TrajectronPooling, SAttention_fast
from trajnetbaselines.lstm.more_non_gridbased_pooling import NMMP
from trajnetbaselines.lstm.gridbased_pooling import GridBasedPooling
from trajnetbaselines.lstm.lstm import LSTM, LSTMPredictor, drop_distant
from trajnetbaselines.lstm.utils import center_scene, random_rotation, visualize_scene
from trajnetbaselines.vae.vae import VAE

def get_latents(args, model, train_scenes):
    MASTER_LATENT_LIST_LIST = []
    MASTER_PREDICTIONS=[]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for scene_i, (filename, scene_id, paths) in tqdm(enumerate(train_scenes[:10])):
        ## make new scene
        scene = trajnetplusplustools.Reader.paths_to_xy(paths)

        ## get goals
        if train_goals is not None:
            scene_goal = np.array(train_goals[filename][scene_id])
        else:
            scene_goal = np.array([[0, 0] for path in paths])

        # ToDO: get rid of this eventually bc you want to do both close and distant preds?
        ## Drop Distant
        scene, mask = drop_distant(scene)
        scene_goal = scene_goal[mask]

        ##process scene
        # normalize the scene
        scene, _, _, scene_goal = center_scene(scene, args.obs_length, goals=scene_goal)
        # random rotate the trajectories
        if args.augment:
            scene, scene_goal = random_rotation(scene, goals=scene_goal)
        # add noise to the input
        # if self.augment_noise:
        #     scene = augmentation.add_noise(scene, thresh=0.02, ped='neigh')
        input_scene = torch.tensor(scene).to(device).float()
        input_goal = torch.tensor(scene_goal).to(device).float()
        input_split = torch.tensor([0, scene.shape[1]]).to(device).long()
        # outputs: Tensor [seq_length, num_tracks, 2]
        #    Absolute positions of all pedestrians
        # Rel_outputs: Tensor [seq_length, num_tracks, 5]
        #    Velocities of all pedestrians
        with torch.no_grad():
            if args.type == 'vae':
                rel_outputs, _, z_distr_xy, z_distr_x = model(input_scene[:args.obs_length], input_goal, input_split,
                                                                   input_scene[args.obs_length:])
                CLUSTER_LATENT = [z_distr_xy, z_distr_x]
                outputs = rel_outputs
            else:
                rel_outputs, outputs, CLUSTER_LATENT = model(input_scene[:args.obs_length], input_goal, input_split,
                                                         input_scene[args.obs_length:])
        MASTER_LATENT_LIST.append(CLUSTER_LATENT)
        MASTER_PREDICTIONS.append(outputs)

        # if scene_i == 0:
        #     input_scene = input_scene.reshape(-1, args.obs_length + args.pred_length, 2)
        #     outputs = outputs.reshape(-1, args.obs_length + args.pred_length - 1, 2).detach()
        #     for i, pos in enumerate(input_scene):
        #         plt.scatter(pos[:, 0], pos[:, 1], c='b')
        #         plt.scatter(outputs[i, :, 0], outputs[i, :, 1], c='r')
        #         plt.show()

    return MASTER_LATENT_LIST, MASTER_PREDICTIONS

def save_latents_lstm(args, latents):
    hiddens=[]
    cells = []
    for l in latents:
        hiddens.append(np.stack([temp[0][0] for temp in l]))
        cells.append(np.stack([temp[1][0] for temp in l]))
    hiddens=np.stack(hiddens)
    np.save(args.type + '_hiddens.npy', hiddens)
    cells=np.stack(cells)
    np.save(args.type + '_cells.npy', cells)
    return hiddens, cells

def save_latents_vae(args, latents):
    xy=[]
    x=[]
    for l in latents:
        xy.append(l[0].numpy())
        x.append(l[1].numpy())
    breakpoint()
    xy=np.stack(xy)
    np.save(args.type + '_xyLatent.npy', xy)
    x=np.stack(x)
    np.save(args.type + '_xLatent.npy', x)

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

args = get_args()

# Get data
train_scenes, train_goals, train_flag = prepare_data(args.path, subset='/train/', sample=args.sample, goals=args.goals)

# setup the model
# create interaction/pooling modules
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

# create forecasting model
if args.type=='vae':
    file = glob('trajnetplusplusbaselines/OUTPUT_BLOCK/trajdata/' + str(args.type) + '_vanilla_None.pkl.*.state')
    model = VAE(pool=pool,
                embedding_dim=args.coordinate_embedding_dim,
                hidden_dim=args.hidden_dim,
                goal_flag=args.goals,
                goal_dim=args.goal_dim,
                num_modes=args.k,
                desire_approach=args.desire,
                latent_dim=args.noise_dim)
else:
    file = glob('trajnetplusplusbaselines/OUTPUT_BLOCK/trajdata/lstm_' + str(args.type) + '_None.pkl.*.state')
    model = LSTM(pool=pool,
                 embedding_dim=args.coordinate_embedding_dim,
                 hidden_dim=args.hidden_dim,
                 goal_flag=args.goals,
                 goal_dim=args.goal_dim)

# load any presaved checkpoints
if len(file)>0:
    file.sort()
    print("Loading Model Dict")
    with open(file[-1], 'rb') as f:
        checkpoint = torch.load(f, map_location=torch.device('cpu'))
    pretrained_state_dict = checkpoint['state_dict']
    model.load_state_dict(pretrained_state_dict, strict=True)
else:
    print('!!! Starting from Random !!!')

# loop over all scenes to collect the latent variables
MASTER_LATENT_LIST=[]
MASTER_PREDICTIONS=[]
if args.get_latents:
    MASTER_LATENT_LIST, MASTER_PREDICTIONS = get_latents(args, model, train_scenes)
    if args.type =='vae':
        # hiddens is xy latent var
        # cells is x latent var
        # not sure what the difference is but they're differentiated inside the vae forward
        hiddens, cells = save_latents_vae(args, MASTER_LATENT_LIST)
    else:
        hiddens, cells = save_latents_lstm(args, MASTER_LATENT_LIST)#, MASTER_PREDICTIONS)
else:
    if args.type == 'vae':
        exit(-1)
        # hiddens = np.load()
        # cells = np.load()
    else:
        hiddens = np.load(args.type + '_hiddens.npy')
        # cells = np.load(args.type + '_cells.npy')

if args.get_tsne:
    tsne=TSNE()
    hiddens=hiddens.reshape(-1,(args.obs_length - 1)*args.hidden_dim)
    # cells = cells.reshape(-1,(args.obs_length - 1)*args.hidden_dim)
    # temp = random.sample(list(hiddens), k=500)
    embedding = tsne.fit_transform(hiddens)
else:
    hidden_embedding = np.load(args.type+'_hidden_embedding.npy')
    cells_embedding = np.load(args.type+'_cells_embedding.npy')

for pos in cells_embedding[:2000]:
    plt.scatter(pos[0],pos[1], c='b', alpha=0.3)
plt.title('TSNE of '+args.type+' Hidden Embedding')
plt.show()