from tsneGTdataset import TSNEGT
import argparse
from torch.utils.data import DataLoader
import torch
import numpy as np
from matplotlib import pyplot as plt
from simpleTSNEPredict import SimpleRegNetwork
from trajAugmentations import TrajAugs


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_people', default=2, type=int, help='number of people to include in each group')
    parser.add_argument('--path', default='/Users/faith_johnson/GitRepos/PedTrajPred/', type=str,
                        help='path to the csv for the dataset data')
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--window', default=8, type=int)
    args = parser.parse_args()
    return args

@torch.no_grad()
def test(net, loader):
    aug=TrajAugs()
    total_output = []
    total_labels = []
    total_pos = []
    total_rot = []
    for i,data in enumerate(loader):
        if i>len(loader)/2:
            break
        # breakpoint()
        pos, tsne, kmeans, originalPos = data
        rot_traj = [aug.rotate(x.reshape(args.num_people,8,2), 30) for x in originalPos]
        # breakpoint()
        # if sum(sum(rot_traj<0))>0:
        #     breakpoint()
        total_rot.extend(rot_traj)
        for i,r in enumerate(rot_traj):
            allDiffs = []
            scale = 15
            for j, x in enumerate(r):
                temp = np.concatenate((r[:j], r[j + 1:]), axis=0)
                if len(temp) > 0:
                    temp = x[1:] - temp[:, :-1, :]
                    # breakpoint()
                    allDiffs.append(np.hstack(np.concatenate((np.diff(x, axis=0).reshape(1, -1, 2) * scale, temp), axis=0)))
                else:
                    # breakpoint()
                    allDiffs.append(np.diff(x, axis=0))
            allDiffs = np.stack(allDiffs)
            rot_traj[i]=torch.tensor(allDiffs).flatten()
        # breakpoint()
        output = net(torch.stack(rot_traj))
        total_output.extend(output)
        total_labels.extend(tsne)
        total_pos.extend(originalPos)

    return total_output, total_labels, total_pos, total_rot


if __name__=='__main__':
    args = get_args()
    dataset = TSNEGT(args.path + 'allDiffsDataLong45_' + str(args.num_people) + 'thresh_' + str(args.window) + 'window.csv')
    # dataset = TSNEGT(args.path + 'socData_' + str(args.num_people) + 'thresh_'+ str(args.num_people) +'group_8window.csv', args.num_clusters)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    net = SimpleRegNetwork(args.num_people * args.num_people * (args.window - 1) * 2)  # SimpleCatNetwork(32, args.num_clusters) #
    net.load_state_dict(torch.load('simpleRegNet_allDiffsDataLong45_' + str(args.num_people) + 'people_' + str(args.window) + 'window.pt'))
    output, labels, pos, rot_pos = test(net, loader)

    # breakpoint()

    plt.figure()
    output = np.vstack(output)
    labels = np.vstack(labels)
    plt.scatter(labels[:, 0], labels[:, 1])
    plt.title('Ground Truth TSNE, N=' + str(args.num_people))

    plt.figure()
    plt.scatter(output[:, 0], output[:, 1])
    plt.title('Predicted TSNE, N=' + str(args.num_people))

    plt.figure()
    plt.scatter(labels[:, 0], labels[:, 1], alpha=0.5)
    plt.scatter(output[:, 0], output[:, 1], alpha=0.5)
    plt.legend(['real', 'predicted'])
    plt.title('Real vs Predicted TSNE Embeddings, N=' + str(args.num_people))
    plt.show()