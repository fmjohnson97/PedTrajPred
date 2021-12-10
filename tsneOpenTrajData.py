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

class TSNEOpenTrajData(Dataset):
    def __init__(self, dataset, mode, image=False, input_window=8, output_window=12, filter=True):
        # self.root='/Users/faith_johnson/GitRepos/OpenTraj/'
        self.root = '/home/faith/PycharmProjects/PedTrajPredDict/OpenTraj/'
        self.name = dataset
        # H path, Vid Path, load_X arguments
        self.paths = {
            'ETH': ['datasets/ETH/seq_eth/H.txt', 'datasets/ETH/seq_eth/video.avi', '/datasets/ETH/seq_eth/obsmat.txt', (13.8689, 13.8689), (-6.5107, -6.2595)], # (480, 640), (14, 16), (-10,-10),
            'ETH_Hotel': ['datasets/ETH/seq_hotel/H.txt', 'datasets/ETH/seq_hotel/video.avi',
                          '/datasets/ETH/seq_hotel/obsmat.txt', (4.2894, 4.2920), (-10.2537, -10.0058)], #(576, 720), (5, 3), (-6,-9),
            'UCY_Zara1': ['datasets/UCY/zara01/H.txt', 'datasets/UCY/zara01/video.avi',
                          'datasets/UCY/zara01/annotation.vsp', (17.0080, 16.7517), (-1.3615, -1.1406)],
            'UCY_Zara2': ['datasets/UCY/zara02/H.txt', 'datasets/UCY/zara02/video.avi',
                          'datasets/UCY/zara02/annotation.vsp', (17.0984, 16.9911), (-1.4058, -1.2719)]}

        self.mode = mode
        self.image = image
        self.transforms = Compose([GaussianBlur(5)])
        self.input_window = input_window
        self.output_window = output_window
        self.filter=filter

        try:
            paths = self.paths[dataset]
            self.maxs=np.array(paths[3])
            self.mins=np.array(paths[4])
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
        
        self.tsne=TSNE()
        get_embedding()
        
    def get_embedding():
        full_data = []
        for i in range(self.dataset.data['frame_id'].nunique()-self.output_window-self.input_window):
            data = self.getFrames(i,self.output_window+self.input_window)
        

    def __len__(self):
        if self.mode=='by_human':
            return len(self.groups)
        elif self.mode =='by_frame':
            return len(self.embedding)

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
        if self.mode=='by_frame':
            for i in range(len(data['pos'])):
                data['pos'][i]=(data['pos'][i]-self.mins)/(self.maxs-self.mins)
        else:
            data['pos']=(data['pos']-self.mins)/(self.maxs-self.mins)
        data['index']=[item]
        return data

    def unNorm(self,data):
        return data*(self.maxs-self.mins)+self.mins

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
    x = OpenTrajData(dataset='UCY_Zara2', image=False, mode='by_human', filter=False)
    d=DataLoader(x,batch_size=1,shuffle=False)
    #data=x.__getitem__(1) #78 gets none
    max_x=-np.inf
    max_y=-np.inf
    min_x=np.inf
    min_y=np.inf
    for items in d:
        h, _ = torch.max(items['pos'],-1)
        l, _ = torch.min(items['pos'],-1)
        h=h.squeeze()
        l=l.squeeze()
        if h[0]>max_x:
            max_x=h[0]
        if h[1]>max_y:
            max_y=h[1]
        if l[0]<min_x:
            min_x=l[0]
        if l[1]<min_y:
            min_y=l[1]
        
    import pdb; pdb.set_trace()
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
