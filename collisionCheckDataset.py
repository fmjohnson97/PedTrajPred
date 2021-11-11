import numpy as np
from torch.utils.data import Dataset
from utils import getData
import numpy as np
from trajAugmentations import TrajAugs
from finetuneSWAV import set_params

class CollisionCheckDataset(Dataset):
    def __init__(self,args):
        self.data=getData(args)
        if type(self.data)==list:
            self.length=0
            self.partitions=[]
            for d in self.data:
                self.length+=d.__len__()
                self.partitions.append(d.__len__())
        else:
            self.length=self.data.__len__()
        self.augs=TrajAugs()

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        breakpoint()
        if type(self.data) == list:
            inds,=np.where(np.cumsum(self.partitions)>=item)
            batch = self.data[inds[0]].__getitem__(item-np.sum(self.partitions[:inds[0]])-1)
        else:
            batch = self.data.__getitem__(item)

        if len(batch)==4:
            peopleIDs, locs, targ_locs, frame = batch

        breakpoint()

args=set_params()
d=CollisionCheckDataset(args)
print(d.__getitem__(2000))