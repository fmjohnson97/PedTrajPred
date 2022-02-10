from torch.utils.data import Dataset
from trajAugmentations import TrajAugs
from glob import glob
import pandas as pd
import numpy as np
from sameSizeData import SameSizeData
from collections import defaultdict
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE


class SyntheticData(Dataset):
    def __init__(self,n=2):
        self.data=[]
        self.n=n # number of people in each group
        self.augs=TrajAugs()
        self.candidates=np.array([ ]) #ToDO: fill these in
        # self.candidateFiles=glob('data/*')
        # if len(self.candidateFiles)<1:
        #     self.candidates = self.getCandidates()
        #     self.augment()
        # else:
        #     self.candidates=pd.DataFrame()
        #     for file in self.candidateFiles:
        #         self.candidates=self.candidates.append(pd.read_csv(file))
        #         self.augment()
        #         breakpoint()


    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    def augment(self):
        breakpoint()
        self.data.append(self.augs(self.candidates))

    def find_nearest(self, data, coord):
        temp = np.abs(data - coord)
        minx = np.argmin(temp[:, 0])
        miny = np.argmin(temp[:, 1])
        if np.sum(temp[minx]) < np.sum(temp[miny]):
            point = minx
        else:
            point = miny

        return point

    def getCandidates(self):
        # load the TSNE data
        data = defaultdict(list)
        for name in ['ETH', 'ETH_Hotel', 'UCY_Zara1', 'UCY_Zara2']:
            dataset = SameSizeData(name, output_window=0, group_size=self.n, trajThresh=self.n, socialThresh=self.n)
            for i, d in enumerate(dataset):
                # if i>len(dataset)/2:
                #     break

                if len(d['pos']) > 0:
                    data['pos'].append(d['pos'].flatten())
                    data['posFrames'].append(d['frames'])

                # if len(d['deltas']) > 0:
                #     data['deltas'].append(d['deltas'].flatten())
                #     data['groupIDs'].append(d['groupIDs'])
                #     data['groupFrames'].append(d['frames'])

        tsne=TSNE()
        trajData = tsne.fit_transform(data['pos']) #np.loadtxt('trajData_' + str(self.n) + 'thresh_' + str(8) + 'window.npy')

        assert(len(trajData)==len(data['pos']))

        # socData = np.loadtxt('socData_' + str(self.n) + 'thresh_' + str(self.n) + 'group_' + str(8) + 'window.npy')

        kmeans = KMeans(n_clusters=50, random_state=0).fit(trajData)
        centers = kmeans.cluster_centers_
        plt.figure()
        plt.scatter(trajData[:, 0], trajData[:, 1], c=kmeans.labels_, alpha=0.5)
        plt.scatter(centers[:, 0], centers[:, 1], c='r')
        plt.title(str(self.n) + " Traj Data, " + str(40) + " Clusters, Len " + str(8))
        # plt.title('socData_' + str(args.social_thresh) + 'thresh_' + str(args.group_size) + 'group_' + str(args.input_window + args.output_window)+
        #     "window_Clusters" + str(args.num_clusters))
        plt.waitforbuttonpress()
        points_clicked = plt.ginput(30, show_clicks=True, timeout=-1)
        plt.show()

        points = []
        plt.scatter(trajData[:, 0], trajData[:, 1], c=kmeans.labels_, alpha=0.5)
        plt.scatter(centers[:, 0], centers[:, 1], c='r')
        plt.title(str(self.n) + " Traj Data, " + str(40) + " Clusters, Len " + str(8))
        # plt.title('socData_' + str(args.social_thresh) + 'thresh_' + str(args.group_size) + 'group_' + str(args.input_window + args.output_window)+
        #     "window_Clusters" + str(args.num_clusters))
        for coords in points_clicked:
            points.append(self.find_nearest(trajData, coords))
            plt.scatter(trajData[points[-1]][0], trajData[points[-1]][1], c='k', alpha=1)
        plt.show()

        breakpoint()
        # fdataset=SameSizeData(frames[point][0], image='mask', output_window=args.output_window, group_size=args.group_size, distThresh=args.dist_thresh,trajThresh=args.traj_thresh, socialThresh=args.social_thresh)
        for i, point in enumerate(points):
            plt.figure()
            for pos in data['pos'][point].reshape(-1, 8, 2):
                plt.plot(pos[:, 0], pos[:, 1])
                plt.scatter(pos[0][0], pos[0][1])
                plt.title('Point ' + str(i))
        plt.show()

        breakpoint()
        use = input('Points to use?').split(',')
        candidates=[]
        for i,point in enumerate(use):
            breakpoint()


        breakpoint()

x=SyntheticData(2)