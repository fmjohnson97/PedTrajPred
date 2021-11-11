from torch.utils.data import Dataset
from OpenTraj.opentraj.toolkit.loaders.loader_eth import load_eth
from OpenTraj.opentraj.toolkit.loaders.loader_crowds import load_crowds
import numpy as np
import torch
import cv2
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, GaussianBlur
from OpenTraj.utils import world2image
from collections import defaultdict

class OpTrajData(Dataset):
    def __init__(self,dataset='ETH',mode='by_frame', image=False,input_window=4, output_window=8):
        super(OpTrajData,self).__init__()
        # self.root='/Users/faith_johnson/GitRepos/OpenTraj/'
        self.root='/home/faith/PycharmProjects/PedTrajPredDict/OpenTraj/'
        self.name=dataset
        # H path, Vid Path, load_X arguments
        self.paths={'ETH':['datasets/ETH/seq_eth/H.txt','datasets/ETH/seq_eth/video.avi','/datasets/ETH/seq_eth/obsmat.txt'],
                    'ETH_Hotel':['datasets/ETH/seq_hotel/H.txt','datasets/ETH/seq_hotel/video.avi','/datasets/ETH/seq_hotel/obsmat.txt'],
                    'UCY_Zara1':['datasets/UCY/zara01/H.txt','datasets/UCY/zara01/video.avi','datasets/UCY/zara01/annotation.vsp'],
                    'UCY_Zara2':['datasets/UCY/zara02/H.txt','datasets/UCY/zara02/video.avi','datasets/UCY/zara02/annotation.vsp']}
                    #students 3 might take some extra pixel computations, so not including for now
                    #'UCY_Stud3':['datasets/UCY/zara01/H.txt','datasets/UCY/zara01/video.avi','datasets/UCY/zara01/annotation.vsp','datasets/UCY/zara01/H.txt']}

        self.mode=mode
        self.image=image
        self.transforms=Compose([GaussianBlur(5)])
        self.input_window = input_window
        self.output_window = output_window
        try:
            paths=self.paths[dataset]
        except Exception as e:
            print(e)
            print('Unsupported Dataset for OpTrajData:',dataset)
            import pdb; pdb.set_trace()
        self.H = np.loadtxt(self.root + paths[0])
        self.video=cv2.VideoCapture(self.root+paths[1])
        if 'ETH' in dataset:
            self.H=np.linalg.inv(self.H)
            self.dataset = load_eth(self.root + paths[2])
        elif 'UCY' in dataset:
            print('WARNING: MASKS/IMAGES ARE NOT CURRENTLY SUPPORTED FOR THIS DATASET:',dataset)
            self.dataset = load_crowds(self.root +paths[2], use_kalman=False,homog_file=self.root +paths[0])

        if self.mode=='by_human':
            self.trajectory=self.dataset.get_trajectories()
            self.groups=self.trajectory.indices

    def __len__(self):
        if self.mode=='by_human':
            return len(self.groups)
        elif self.mode in ['by_frame', 'by_N_frame']:
            return self.dataset.data['frame_id'].nunique()-self.output_window


    def getImages(self,inds):
        frames=[]
        for i in inds:
            self.video.set(1,i) #1 is for CV_CAP_PROP_POS_FRAMES
            ret, f=self.video.read()
            frames.append(f)
        return frames

    def __getitem__(self, item):
        if self.mode=='by_human':
            pos, frame = self.getOneHumanTraj(item)
            return pos, frame
        elif self.mode=='by_frame':
            peopleIDs, pos, pos_1, frame = self.getOneFrame(item)
            return peopleIDs, pos, pos_1, frame
        elif self.mode=='by_N_frame':
            peopleIDs, pos, frame = self.getMultiFrames(item)
            pos, frame = self.filterMissing(peopleIDs, pos, frame)
            #breakpoint()
            return np.array(pos), frame

    def getMasks(self,im,locs):
        # import pdb; pdb.set_trace()
        frames=[]
        for ind, i in enumerate(im):
            fram = np.zeros_like(i)
            for loc in locs[ind]:
                pix_loc=world2image(np.array([loc]),self.H)
                cv2.circle(fram,tuple(pix_loc[0]),5,(255,255,255),-1)
            frames.append(self.transforms(torch.FloatTensor(fram)))
            # cv2.imshow('test',frame)
            # cv2.waitKey()
        return frames

    def filterMissing(self, peopleIDs, pos, frame):
        posDict=defaultdict(list)
        if len(frame)>0:
            frameDict=defaultdict(list)
        else:
            frameDict=None
        for i,people in enumerate(peopleIDs):
            if type(people).__module__ == np.__name__:
                for j, p in enumerate(people):
                    posDict[p].append(pos[i][j])
                    if frameDict:
                        frameDict[p].append(pos[i][j])
            else:
                posDict[people].append(pos[i])
                if frameDict:
                    frameDict[p].append(pos[i])
        positions=[]
        frames=[]

        for k in posDict.keys():
            if len(posDict[k])==self.output_window:
                positions.append(posDict[k])
                if frameDict:
                    frames.append(frameDict[k])
        return positions, frames

    def getMultiFrames(self, item):
        # breakpoint()
        peopleIDs, pos, frames = [], [], []
        for i in range(self.output_window):
            frameID = [self.dataset.data['frame_id'].unique()[item+i]]
            if self.image:
                frame = self.getImages(frameID)
            else:
                frame = []
            people = self.dataset.get_frames(frameID)[0]
            peopleIDs.append(people['agent_id'].to_numpy())
            pos.append(people.filter(['pos_x', 'pos_y']).to_numpy())
            if self.image == 'mask':
                frame = self.getMasks(frame, np.expand_dims(pos[-1], 0))
            frames.extend(frame)
        return peopleIDs, pos, frames

    def getOneFrame(self,item):
        # import pdb; pdb.set_trace()
        # peopleIDs = []
        # locs = []
        # frame = []
        # for window in range(self.input_window + self.output_window):
        #     frameID = [self.dataset.data['frame_id'].unique()[item + window]]
        #     if self.image is not None:
        #         frame.append(self.getImages(frameID))
        #     people = self.dataset.get_frames(frameID)[0]
        #     peopleIDs.append(people['agent_id'].tolist())
        #     locs.append(people.filter(['pos_x', 'pos_y']).to_numpy())
        #     if self.image == 'mask':
        #         frame = self.getMasks(frame[-1], np.expand_dims(locs[-1], 0))
        #
        # targ_locs = locs[-self.output_window:]
        # locs = locs[:self.input_window]

        frameID=[self.dataset.data['frame_id'].unique()[item]]
        if self.image:
            frame=self.getImages(frameID)
        else:
            frame=[]
        people=self.dataset.get_frames(frameID)[0]
        targ_people=[]
        i=1
        while len(targ_people)==0:
            targ_people=self.dataset.get_frames([frameID[0]+i])[0]
            i+=1
        inds = targ_people.agent_id.isin(people.agent_id)
        targ_people=targ_people[inds]
        targ_locs=targ_people.filter(['pos_x','pos_y']).to_numpy()
        inds = people.agent_id.isin(targ_people.agent_id)
        people = people[inds]
        peopleIDs = people['agent_id'].tolist()
        locs = people.filter(['pos_x', 'pos_y']).to_numpy()
        # import pdb; pdb.set_trace()
        if self.image == 'mask':
            frame = self.getMasks(frame, np.expand_dims(locs, 0))
        return [peopleIDs, locs, targ_locs, frame]

    def getOneHumanTraj(self,item):
        group=list(self.groups.keys())[item]
        data=self.trajectory.get_group(group)
        if self.image:
            frames=self.getImages(data['frame_id'].tolist())
        else:
            frames=[]
        positions = data.filter(['pos_x', 'pos_y']).to_numpy()
        if self.image=='mask':
            frames=self.getMasks(frames,positions.reshape(len(positions),1,2))
        # return [torch.FloatTensor(positions),torch.FloatTensor(frames)]
        return [positions, frames]

# x=OpTrajData(dataset='ETH_Hotel',image='mask', mode='by_frame')
# d=DataLoader(x,batch_size=1,shuffle=False)
# people, pos, pos_1, fram=x.__getitem__(3)
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
# for f in fram:
#     import pdb; pdb.set_trace()
#     cv2.imshow('',f)
#     cv2.waitKey(1000)
