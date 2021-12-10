from torch.utils.data import Dataset, DataLoader
import cv2
from torchvision.transforms import Compose, Resize, GaussianBlur
import numpy as np
from OpenTraj.opentraj.toolkit.loaders.loader_eth import load_eth
from OpenTraj.opentraj.toolkit.loaders.loader_crowds import load_crowds
from collections import defaultdict
from OpenTraj.utils import world2image
import torch

class OpenTrajData(Dataset):
    def __init__(self, dataset, mode, image=False, input_window=8, output_window=12, filter=True):
        # self.root='/Users/faith_johnson/GitRepos/OpenTraj/'
        self.root = '/Users/faith_johnson/GitRepos/PedTrajPred/OpenTraj/'
        self.name = dataset
        # H path, Vid Path, load_X arguments
        self.paths = {
            'ETH': ['datasets/ETH/seq_eth/H.txt', 'datasets/ETH/seq_eth/video.avi', '/datasets/ETH/seq_eth/obsmat.txt'],
            'ETH_Hotel': ['datasets/ETH/seq_hotel/H.txt', 'datasets/ETH/seq_hotel/video.avi',
                          '/datasets/ETH/seq_hotel/obsmat.txt'],
            'UCY_Zara1': ['datasets/UCY/zara01/H.txt', 'datasets/UCY/zara01/video.avi',
                          'datasets/UCY/zara01/annotation.vsp'],
            'UCY_Zara2': ['datasets/UCY/zara02/H.txt', 'datasets/UCY/zara02/video.avi',
                          'datasets/UCY/zara02/annotation.vsp']}

        self.mode = mode
        self.image = image
        self.transforms = Compose([GaussianBlur(5)])
        self.input_window = input_window
        self.output_window = output_window
        self.filter=filter

        try:
            paths = self.paths[dataset]
        except Exception as e:
            print(e)
            print('Unsupported Dataset for OpTrajData:', dataset)
            breakpoint()

        self.H = np.loadtxt(self.root + paths[0])
        self.video = cv2.VideoCapture(self.root + paths[1])

        if 'ETH' in dataset:
            self.H = np.linalg.inv(self.H)
            self.dataset = load_eth(self.root + paths[2])
        elif 'UCY' in dataset:
            print('WARNING: MASKS/IMAGES ARE NOT CURRENTLY SUPPORTED FOR THIS DATASET:', dataset)
            self.dataset = load_crowds(self.root + paths[2], use_kalman=False, homog_file=self.root + paths[0])

        if self.mode == 'by_human':
            self.trajectory = self.dataset.get_trajectories()
            self.groups = self.trajectory.indices

    def __len__(self):
        if self.mode=='by_human':
            return len(self.groups)
        elif self.mode =='by_frame':
            return self.dataset.data['frame_id'].nunique()-self.output_window-self.input_window

    def __getitem__(self, item):
        if self.mode == 'by_human':
            data = self.getOneHumanTraj(item)
        elif self.mode=='by_frame':
            data = self.getFrames(item,self.output_window+self.input_window)
            #turns the dict into an array if not filter
            #or if filter, it gets rid of people not present in the entire sequence of frames
            data = self.filterAndFlatten(data)
        if self.image:
            data['frames']=self.getImages(data)
        data['index']=[item]
        return data

    def getImages(self,data):
        breakpoint()
        inds = data['frames']
        frames=[]
        ret, f = self.video.read()
        imsize=f.shape
        for t,i in enumerate(inds):
            if self.image=='mask':
                f = np.zeros(imsize)
                locs=data['pos'][:,t,:]
                for l in locs:
                    pix_loc = world2image(np.array([l]), self.H)
                    cv2.circle(f, tuple(pix_loc[0]), 5, (255, 255, 255), -1)
                f=self.transforms(torch.FloatTensor(f))
            else:
                self.video.set(1, i)  # 1 is for CV_CAP_PROP_POS_FRAMES
                ret, f = self.video.read()
            frames.append(f)
        return frames

    def filterAndFlatten(self, data):
        peopleIDs, posDict, frames = data['peopleIDs'], data['pos'], data['frames']
        positions=[]

        for i,k in enumerate(posDict.keys()):
            if not self.filter or len(posDict[k])==self.output_window+self.input_window:
                positions.append(posDict[k])

        data['pos']=np.array(positions)
        if len(positions)==0:
            data['frames']=[]
        return data

    def getOneHumanTraj(self,item):
        dataset={}
        group=list(self.groups.keys())[item]
        data=self.trajectory.get_group(group)
        dataset['frames']=data['frame_id'].tolist()
        dataset['pos'] = data.filter(['pos_x', 'pos_y']).to_numpy()
        return dataset

    def getFrames(self, item, frameNumber):
        #breakpoint()
        peopleIDs, posDict, frames = [], defaultdict(list), []
        for i in range(frameNumber):
            frameID = [self.dataset.data['frame_id'].unique()[item+i]]
            frames.extend(frameID)
            people = self.dataset.get_frames(frameID)[0]
            peopleIDs.append(people['agent_id'].to_numpy())
            pos=people.filter(['pos_x', 'pos_y']).to_numpy()
            for j,p in enumerate(peopleIDs[-1]):
                posDict[p].append(pos[j])

        dataset={'peopleIDs':peopleIDs, 'pos':posDict, 'frames':frames}
        return dataset


if __name__=='__main__':
    x = OpenTrajData(dataset='ETH_Hotel', image=True, mode='by_frame')
    d=DataLoader(x,batch_size=1,shuffle=False)
    data=x.__getitem__(1) #78 gets none
    # # import pdb; pdb.set_trace()
    # for pid, pos, targ, img in d:
    #     try:
    #         # import pdb; pdb.set_trace()
    #         cv2.imshow('',img[0].numpy()[0])
    #         cv2.waitKey(100)
    #         # print(pos)
    #         # print(targ)
    #     except:
    #         import pdb; pdb.set_trace()
    # fram=data['frames']
    # for f in fram:
    #     # import pdb; pdb.set_trace()
    #     cv2.imshow('',f)
    #     cv2.waitKey(10)
