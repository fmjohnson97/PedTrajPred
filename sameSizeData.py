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

class SameSizeData(Dataset):
    def __init__(self, dataset, mode='by_frame', image=False, input_window=8, output_window=12, group_size=2,
                 distThresh=0.01, trajThresh=None, socialThresh=None):
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

        self.mode = mode
        self.image = image
        self.transforms = Compose([GaussianBlur(5)])
        self.input_window = input_window
        self.output_window = output_window
        self.group_size=group_size
        self.dist_thresh=distThresh
        self.traj_thresh=trajThresh
        self.social_thresh=socialThresh
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
            data = self.filterLength(data)
            if len(data['pos'])>1:
                # breakpoint()
                data = self.getSocialDistances(data)
                data = self.getDistTrajs(data)
                data = self.parameterize(data)
                if self.social_thresh is not None:
                    # trims down # of social groups returned
                    data = self.trimSocialGroups(data)
                if self.traj_thresh is not None:
                    # trims down # of raw trajectory based groups returned
                    data = self.collateRemainders(data)
                if len(data['pos'])==0 and len(data['deltas'])==0:
                    data['frames']=np.array([])
            else:
                data = self.reset(data,['deltas', 'groupIDs', 'distTraj', 'diffs', 'spline'])

        if self.image:
            data['frames']=self.getImages(data)
        data['index']=[item]
        return data

    def collateRemainders(self, data):
        if len(data['pos'])> self.traj_thresh:
            # randomly choose self.traj_thresh trajectories from the total in the scene
            temp=[(i,data['pos'][i]) for i in range(len(data['pos']))]
            choices=random.sample(temp, k=self.traj_thresh)
            ids=[data['peopleIDs'][x[0]] for x in choices]
            distTraj=[data['distTraj'][x[0]] for x in choices]
            diffs=[data['diffs'][x[0]] for x in choices]
            spline = [data['spline'][x[0]] for x in choices]
            pos=[x[1] for x in choices]
            data['peopleIDs']=np.array(ids)
            data['pos']=np.array(pos)
            data['distTraj']=np.array(distTraj)
            data['diffs']=np.array(diffs)
            data['spline']=np.array(spline)
            # data['pos']=np.array(list(combinations(data['pos'],self.traj_thresh)))
            # data['peopleIDs']=np.array(list(combinations(data['peopleIDs'], self.traj_thresh)))
        elif len(data['pos'])< self.traj_thresh:
            data['pos']=np.array([])
            data['peopleIDs']=np.array([])
            data['distTraj']=np.array([])
            data['diffs'] = np.array([])
            data['spline']=np.array([])

        return data

    def trimSocialGroups(self, data):
        if len(data['deltas'])<self.social_thresh:
            data['deltas']=np.array([])
            data['groupIDs']=np.array([])
        elif len(data['deltas'])>self.social_thresh:
            # breakpoint()
            # randomly choose self.social_thresh groupings from the total in the scene
            temp = [(i, data['deltas'][i]) for i in range(len(data['deltas']))]
            choices = random.sample(temp, k=self.social_thresh)
            ids = [data['groupIDs'][x[0]] for x in choices]
            pos = [x[1] for x in choices]
            data['groupIDs'] = np.array(ids)
            data['deltas'] = np.array(pos)
            # data['plotPos']=np.array([data['pos'][x-1] for x in ids])
            # data['deltas'] = np.array(list(combinations(data['deltas'], self.social_thresh)))
            # data['groupIDs'] = np.array(list(combinations(data['groupIDs'], self.social_thresh)))
        temp=[]
        conversion = {}
        for i, p in enumerate(data['peopleIDs']):
            conversion[p] = i
        for id in data['groupIDs']:
            inds=[conversion[i] for i in id]
            temp.append(data['pos'][inds])
        # breakpoint()
        if len(temp)>0:
            data['plotPos']=np.stack(temp)
        else:
            data['plotPos']=np.array([])
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
        temp=[]
        diff=[]
        for x in data['pos']:
            temp.append(x-x[0])
            diff.append(np.diff(x,axis=0))
        data['distTraj']=np.stack(temp)
        data['diffs']=np.stack(diff)
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

    def getSocialDistances(self, data):
        ''' So Far, going with avg distance over the trajectory needs to be below a threshold '''
        people = data['peopleIDs']
        conversion={}
        for i, p in enumerate(people):
            conversion[p]=i
        potential_groups=list(combinations(people, 2)) # get distance between every combo of 2 people
        diffs = [data['pos'][conversion[x[0]]] - data['pos'][conversion[x[1]]] for x in potential_groups]
        if self.group_size >2:
            # went with average distance between each pair in the group
            groups=list(combinations(people, self.group_size))
            subgroups=[list(combinations(x,2)) for x in groups]
            inds = [[potential_groups.index(x) for x in y] for y in subgroups]
            diffs = np.array([np.sum(np.array(diffs)[x], axis=0) for x in inds])/3.0
        else:
            groups=potential_groups

        thresh = [np.mean(np.sum(x ** 2, axis=-1)) for x in diffs]
        diffs=np.array(diffs)[np.array(thresh)<=self.dist_thresh]
        data['deltas']=diffs
        data['groupIDs']=np.array(groups)[np.array(thresh)<=self.dist_thresh]
        return data

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

    def getOneHumanTraj(self,item):
        dataset={}
        group=list(self.groups.keys())[item]
        data=self.trajectory.get_group(group)
        dataset['frames']=data['frame_id'].tolist()
        dataset['pos'] = data.filter(['pos_x', 'pos_y']).to_numpy()
        return dataset

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
    x = SameSizeData(dataset='ETH_Hotel', output_window=0, group_size=3, socialThresh=3, trajThresh=3)
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
        x = SameSizeData(dataset=name, output_window=0, group_size=2, socialThresh=3, trajThresh=3)
        d = DataLoader(x, batch_size=1, shuffle=False)
        for p in d:
            print(p['pos'].shape)
            print(p['deltas'].shape)
            breakpoint()

    breakpoint()
