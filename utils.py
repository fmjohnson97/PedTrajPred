from TrajNet.TrajNetData import TrajNetData
from TrajNet.trajnetplusplusbaselines.trajnetbaselines.lstm.data_load_utils import prepare_data
from OpenTraj.opTrajData import OpTrajData
from trajAugmentations import TrajAugs
import torch
import numpy as np
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

def getData(args):
    if args.data=='trajnet++':
        train_scenes, train_goals, _ = prepare_data('TrajNet/trajnetplusplusbaselines/DATA_BLOCK/trajdata',
                                                    subset='/train/', sample=args.sample, goals=args.goals)
        data=[TrajNetData(train_scenes)]
    elif args.data=='MOT':
        pass
    elif args.data=='opentraj':
        data=[]
        for name in ['ETH', 'ETH_Hotel', 'UCY_Zara1', 'UCY_Zara2']:
            data.append(OpTrajData(name, args.mode, image=args.image,input_window=args.num_frames, output_window=0))
    else:
        print(args.data,' is an invalid dataset')
        raise NotImplementedError

    return data

def batchify(inputs, args):
    # takes one trajectory and gets a window of input and output
    # then increments by one and does it again all the way down the traj
    # breakpoint()
    inds=[range(x,x+args.seq_len) for x in range(len(inputs)-(args.seq_len+args.targ_len))]
    traj = inputs[inds,:]
    inds = [range(x.stop,x.stop+args.targ_len) for x in inds]
    targs = inputs[inds,:]
    return traj, targs


def processData(args, inputs, doAugs=True):
    traj_augs=TrajAugs(args=args)
    trajectory, ims = inputs
    ### way #1; idk if I like this
    trajectory = 2.*( trajectory- torch.min(trajectory))/np.ptp(trajectory)-1

    ### way #2; this seems better maybe? with or without the *2 -1
    # min_val, min_ind = torch.min(trajectory, axis=1)
    # max_val, max_ind = torch.max(trajectory, axis=1)
    # 2 * (trajectory - min_val) / (max_val - min_val) - 1

    ### way #3; seems most standard so lets use this for now
    torch.nn.functional.normalize(trajectory)

    if trajectory.shape[1]<args.seq_len:
        return None, None, trajectory, ims
    trajectory = trajectory.numpy()[0]
    if doAugs is True:
        traj1=traj_augs.augment(trajectory)
        traj2=traj_augs.augment(trajectory)
        trim = len(traj1) % args.seq_len
        return torch.tensor(traj1[:-trim].reshape(-1,args.seq_len,2)), torch.tensor(traj2[:-trim].reshape(-1,args.seq_len,2)), trajectory, ims
    else:
        trim = len(trajectory) % args.seq_len
        return torch.tensor(trajectory[:-trim].reshape(-1,args.seq_len,2)), trajectory, ims

def plotTSNEclusters(prototypes, args, title="Clusters", saveName='clusters.png'):
    tsne = TSNE()
    data = tsne.fit_transform(prototypes)
    kmeans = KMeans(n_clusters=args.num_clusters, random_state=0).fit(data)
    centers = kmeans.cluster_centers_
    fig=plt.figure()
    plt.scatter(data[:, 0], data[:, 1], c=kmeans.labels_, alpha=0.5)
    plt.scatter(centers[:, 0], centers[:, 1], c='r')
    plt.title(title)
    plt.savefig(saveName)
    return data, centers
