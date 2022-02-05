from torch.utils.data import Dataset, DataLoader
import cv2
from torchvision.transforms import Compose, Resize, GaussianBlur
import numpy as np
from OpenTraj.opentraj.toolkit.loaders.loader_eth import load_eth
from OpenTraj.opentraj.toolkit.loaders.loader_crowds import load_crowds
from collections import defaultdict
from OpenTraj.utils import world2image
import torch
from sklearn.manifold import TSNE
from openTrajData import OpenTrajData
import argparse
from utils import social_batchify

class TSNEOpenTrajData(Dataset):
    def __init__(self, args):
        self.datasets=['ETH', 'ETH_Hotel', 'UCY_Zara1', 'UCY_Zara2']

        self.mode = args.mode
        self.image = args.image
        self.input_window = args.num_frames
        self.output_window = args.seq_len
        self.filter=args.filterMissing

        self.get_data()
        self.get_embedding()

    def get_data(self):
        input_data = []
        output_data = []
        data_frames = []
        for name in self.datasets:
            dataset = OpenTrajData(name, self.mode, image=self.image, input_window=self.input_window,
                                   output_window=self.output_window, filter=self.filter)
            for d in dataset:
                if len(d['pos'].shape) > 2:
                    traj, target, srcMask = social_batchify(d['pos'], args)
                else:
                    traj = []

                if len(traj) > 0:
                    input_data.append(traj.reshape(-1, self.input_window*2))
                    output_data.append(target)
                    data_frames.append(d['frames'])
        self.input_data=input_data
        self.output_data=output_data
        self.data_frames= data_frames


    def get_embedding(self):
        breakpoint()
        

    def __len__(self):
        if self.mode=='by_human':
            return len(self.groups)
        elif self.mode =='by_frame':
            return len(self.embedding)

    def __getitem__(self, item):
        data=[]
        return data


if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Social Transformer")

    #####################
    #### Data Params ####
    #####################
    parser.add_argument('--data', default='opentraj', choices=('trajnet++', 'MOT', 'opentraj'),
                        help='which dataset to use')
    parser.add_argument("--include_aug", type=bool, default=False,
                        help="whether to return the augmentation functions used on the data")
    parser.add_argument("--num_augs", default=1, type=int,
                        help="number of data augmentations to use (randomly chosen)")
    parser.add_argument('--num_frames', default=8, type=int,
                        help='number of frames of people to use for target for opentraj')
    parser.add_argument('--mode', default='by_frame', choices=('by_human', 'by_frame'),
                        help='whether to group by person or by frame for opentraj')
    parser.add_argument("--image", type=bool, default=False,
                        help="whether to return images or not")
    parser.add_argument('--socialN', default=3, type=int)
    parser.add_argument("--filterMissing", type=bool, default=True,
                        help="whether to return images or not")
    parser.add_argument('--saveName', default='social_simpleTransfPredictions',
                        help='prefix to name all the save files')

    #####################
    #### Arch Params ####
    #####################
    parser.add_argument("--feat_dim", default=8, type=int, help="feature dimension")
    parser.add_argument("--hidden_dim", default=1024, type=int, help="hidden dimension")
    parser.add_argument('--seq_len', default=12, type=int, help='num steps in each traj output')
    parser.add_argument('--num_clusters', default=10, type=int, help='num knn clusters')
    parser.add_argument('--nheads', default=2, type=int, help='num attention heads')
    parser.add_argument('--nlayers', default=2, type=int, help='how deep the transformer will be')
    parser.add_argument("--dropout", default=0.1, type=float, help="percentage for dropout (<1)")

    ######################
    #### Train Params ####
    ######################
    parser.add_argument("--epochs", default=100, type=int,
                        help="number of total epochs to run")
    parser.add_argument("--batch_size", default=1, type=int,
                        help="batch size per gpu, i.e. how many unique instances per gpu")
    parser.add_argument("--seed", type=int, default=31, help="seed")
    parser.add_argument("--warmup_epochs", default=10, type=int, help="number of warmup epochs")
    parser.add_argument("--start_warmup", default=0, type=float,
                        help="initial warmup learning rate")
    parser.add_argument("--base_lr", default=4.8, type=float, help="base learning rate")
    parser.add_argument("--final_lr", type=float, default=1e-5, help="final learning rate")

    args = parser.parse_args()

    x=TSNEOpenTrajData(args)