from torch.utils.data import Dataset, DataLoader
import cv2
from torchvision.transforms import Compose, Resize, GaussianBlur
import numpy as np
from OpenTraj.opentraj.toolkit.loaders.loader_eth import load_eth
from OpenTraj.opentraj.toolkit.loaders.loader_crowds import load_crowds
from collections import defaultdict
from OpenTraj.utils import world2image
import torch
import random
from TrajNet.trajnetplusplusbaselines.trajnetbaselines.lstm.utils import center_scene
from itertools import combinations

class FirstPOVData(Dataset):
    def __init__(self, dataset, image=False, input_window=8, output_window=12, trajThresh=1):
        '''
            dataset: selects which data paths to use from below
            mode: either chooses the trajectories "by_frame" or "by_human"
            image: False if need none; 'mask' if want masks; True if want raw images
            input_window: number of time steps in the input data
            output_window: number of time steps in the output data
            group_size: number of people to consider as being a "group" when computing distances
            distThresh: max avg distance a group's members can have while still being considered "close to each other"
            trajTrhesh: max number of people returned in one data point when considering raw trajectories; None means don't filter
            socialThresh: max number of social group distances returned in one data point; None means don't filter
        '''
        self.root = '/Users/faith_johnson/GitRepos/PedTrajPred/OpenTraj/'
        self.name = dataset
        # H path, Vid Path, load_X arguments
        # UCY_Zara1 [17.00800773 16.75168712] [-1.36153733 -1.14055993]
        # ETH_Hotel [4.289431  4.2920379] [-10.253669 -10.005824]
        # ETH [13.868879 13.868879] [-6.5106892 -6.2595176]
        # UCY_Zara2 [17.09842293 16.99113599] [-1.40575035 -1.27186937]
        self.paths = {
            'ETH': ['datasets/ETH/seq_eth/H.txt', 'datasets/ETH/seq_eth/video.avi', '/datasets/ETH/seq_eth/obsmat.txt',
                    (14, 14), (-7, -7)],  # (480, 640), (13.8689, 13.8689), (-6.5107, -6.2595),
            'ETH_Hotel': ['datasets/ETH/seq_hotel/H.txt', 'datasets/ETH/seq_hotel/video.avi',
                          '/datasets/ETH/seq_hotel/obsmat.txt', (5, 5), (-11, -11)],
            # (576, 720), (5, 3), (-6,-9),
            'UCY_Zara1': ['datasets/UCY/zara01/H.txt', 'datasets/UCY/zara01/video.avi',
                          'datasets/UCY/zara01/annotation.vsp', (18, 17), (-2, -2)],
            'UCY_Zara2': ['datasets/UCY/zara02/H.txt', 'datasets/UCY/zara02/video.avi',
                          'datasets/UCY/zara02/annotation.vsp', (18, 17), (-2, -2)]}

        self.image = image
        self.transforms = Compose([GaussianBlur(5)])
        self.input_window = input_window
        self.output_window = output_window
        self.traj_thresh=trajThresh
        self.max_group_size=22 # at seq len = 8 (so input=8, output=0); actually by dataset its [22, 15, 18, 17]

        try:
            paths = self.paths[dataset]
            self.min=np.array(paths[4])
            self.max=np.array(paths[3])
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


    def __len__(self):
        return self.dataset.data['frame_id'].nunique()-self.output_window-self.input_window

    def __getitem__(self, item):
        data = self.getFrames(item,self.output_window+self.input_window)
        #turns the dict into an array if not filter
        #or if filter, it gets rid of people not present in the entire sequence of frames
        data = self.filterLength(data)
        if len(data['pos'])>=self.traj_thresh:
            # breakpoint()
            data = self.getDistTrajs(data)
            # data = self.parameterize(data)
            data = self.newDiffs(data)
            # trims down # of raw trajectory based groups returned
            if len(data['pos']) > self.traj_thresh:
                data = self.collateRemainders(data)
            else:
                data['diffsFrames']=data['frames']
        else:
            data = self.reset(data,['allDiffs', 'angDiffs'])

        if self.image:
            data['frames']=self.getImages(data)
        data['index']=[item]
        return data

    def collateRemainders(self, data):
        # breakpoint()
        # get traj_thresh nearest neighbors for each person and use that as 1 sample
        newDiffs=[]
        gtPos = []
        gtFrames = []
        for i in range(len(data['allDiffs'])):
            dists = data['pos'][i]-data['pos']
            minDists = np.sum(np.sum(dists**2, axis=-1), axis=-1)
            if self.traj_thresh == 1:
                # breakpoint()
                minDists[i]=np.inf
                closest=np.argmin(minDists)
            elif data['pos'].shape[0]==self.traj_thresh+1:
                # breakpoint()
                closest = [j for j in range(self.traj_thresh+1) if j!=i]
            else:
                idx = np.argpartition(minDists, self.traj_thresh+1)
                closest=[ind for ind in idx[:self.traj_thresh+1] if ind!=i]
            newDiffs.append(data['allDiffs'][i][closest])
            gtPos.append(data['pos'][closest])
            gtFrames.extend([data['frames']]*len(data['allDiffs']))
        # breakpoint()
        data['allDiffs'] = np.stack(newDiffs)
        data['diffsPos'] = np.stack(gtPos)
        data['diffsFrames'] = np.stack(gtFrames)

        # # randomly choose self.traj_thresh trajectories from the total in the scene
        # temp=[(i,data['pos'][i]) for i in range(len(data['pos']))]
        # choices=random.sample(temp, k=self.traj_thresh)
        # ids=[data['peopleIDs'][x[0]] for x in choices]
        # diffs=[data['diffs'][x[0]] for x in choices]
        # pos=[x[1] for x in choices]
        # data['peopleIDs']=np.array(ids)
        # data['pos']=np.array(pos)
        # data['diffs']=np.array(diffs)

        # # get all combinations of people
        # data['pos']=np.array(list(combinations(data['pos'],self.traj_thresh)))
        # data['peopleIDs']=np.array(list(combinations(data['peopleIDs'], self.traj_thresh)))

        return data


    def reset(self, data, keys=None):
        for k in data.keys():
            data[k]=np.array([])

        if keys is not None:
            for k in keys:
                data[k]=[]
        return data

    def getDistTrajs(self, data):
        # gets the difference between the starting point and all the other points in the trajectory
        diff=[]
        for x in data['pos']:
            diff.append(np.diff(x, axis=0))
        data['diffs']=np.stack(diff)
        return data

    def newDiffs(self, data):
        # breakpoint()
        allDiffs=[]
        headingDiffs=[]
        scale = 15
        for i,x in enumerate(data['pos']):
            temp = np.concatenate((data['pos'][:i], data['pos'][i + 1:]), axis=0)
            hold = np.sum(x**2, axis=1)
            heading = np.arccos(np.sum(x[:-1,:]*x[1:,:], axis=1) / (hold[:-1]*hold[1:])**.5)
            if len(temp)>0:
                temp = x[1:] - temp[:, :-1, :]
                # breakpoint()
                allDiffs.append(np.concatenate((np.diff(x, axis=0).reshape(1,-1,2)*scale, temp), axis=0))
            else:
                # breakpoint()
                allDiffs.append(np.diff(x, axis=0))
                # allDiffs.append(np.concatenate((x.reshape(1,-1,2)[:,:-1,:],np.diff(x, axis=0).reshape(1,-1,2)*scale, heading.reshape(1,-1,1)), axis=-1))
            headingDiffs.append(np.concatenate((np.diff(x, axis=0).reshape(1, -1, 2)*scale, heading.reshape(1, -1, 1)), axis=-1))

        # breakpoint()
        data['allDiffs'] = np.stack(allDiffs)
        data['angDiffs'] = np.stack(headingDiffs)
        data['diffsPos']=data['pos']
        return data

    def getImages(self,data):
        # breakpoint()
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
                #f=self.transforms(torch.FloatTensor(f))
            else:
                self.video.set(1, i)  # 1 is for CV_CAP_PROP_POS_FRAMES
                ret, f = self.video.read()
            frames.append(f)
        return frames


    def filterLength(self, data):
        peopleIDs, posDict, frames = data['peopleIDs'], data['pos'], data['frames']
        positions=[]
        people=[]
        for i,k in enumerate(posDict.keys()):
            if len(posDict[k])==self.output_window+self.input_window:
                positions.append(posDict[k])
                people.append(k)
        if len(positions)>0:
            data['pos']=(np.array(positions)-self.min)/(self.max-self.min)
            # positions = np.stack(positions)
            # positions = (np.array(positions)-self.min)/(self.max-self.min)
            # pos, rot, center = center_scene(np.transpose(positions, (1, 0, 2)), positions.shape[1])
            # data['pos'] = np.transpose(pos, (1, 0, 2))
            data['peopleIDs'] = np.array(people)
            # data['rot'] = rot
            # data['center'] = center
        else:
            data=self.reset(data)
        return data

    def getFrames(self, item, frameNumber):
        #breakpoint()
        peopleIDs, posDict, frames = [], defaultdict(list), [self.name]
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

    def parameterize(self, data):
        temp=[]
        t=np.array(list(range(self.input_window+self.output_window)))/(self.input_window+self.output_window)
        A=np.stack([t**2, t, [1]*(self.input_window+self.output_window)]).T
        for x in data['pos']:
            q,res, r, s = np.linalg.lstsq(A,x, rcond=None)
            temp.append(q)
        data['spline']=np.stack(temp)
        return data


if __name__=='__main__':
    x = FirstPOVData(dataset='ETH_Hotel', output_window=0, trajThresh=3)
    d=DataLoader(x,batch_size=1,shuffle=False)
    # data=x.__getitem__(1) #78 gets none
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
    # breakpoint()
    # fram=data['frames']
    # for f in fram:
    #     # import pdb; pdb.set_trace()
    #     cv2.imshow('',f.numpy())
    #     cv2.waitKey(1000)
    for name in ['ETH', 'ETH_Hotel', 'UCY_Zara1', 'UCY_Zara2']:
        x = FirstPOVData(dataset=name, output_window=0, trajThresh=3)
        d = DataLoader(x, batch_size=1, shuffle=False)
        for p in d:
            print(p['pos'].shape)

    breakpoint()
