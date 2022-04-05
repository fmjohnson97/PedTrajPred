from torch.utils.data import Dataset
import torch
import pandas as pd
import numpy as np
from glob import glob

class TSNEGT(Dataset):
    def __init__(self, path, num_clusters, split='train'):
        self.data=pd.read_csv(path)
        if 'newClusters' in self.data.columns:
            print('NewClusters')
            self.data=self.data.filter(['tsne_X','tsne_Y','pos','kmeans','frames','plotPos','newClusters'])
        else:
            self.data=self.data.filter(['tsne_X','tsne_Y','pos','kmeans','frames','plotPos'])

        self.num_clusters=num_clusters
        if split=='train':
            self.data=self.data.iloc[:int(len(self.data)*.7)]
        else:
            self.data = self.data.iloc[int(len(self.data) * .7):]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        # breakpoint()
        try:
            tsneX, tsneY, pos, kmeans, frames, plotPos, newClusters= self.data.iloc[item].tolist()
            originalPos = plotPos[1:-1].strip().split('\n')
            originalPos = [x.strip().split(' ') for x in originalPos]
            originalPos = np.hstack(originalPos)
            originalPos = [float(x.strip()) for x in originalPos if len(x.strip()) > 0]
        except:
            try:
                tsneX, tsneY, pos, kmeans, frames, originalPos = self.data.iloc[item].tolist()
                # breakpoint()
                originalPos = originalPos[1:-1].strip().split('\n')
                originalPos = [x.strip().split(' ') for x in originalPos]
                originalPos = np.hstack(originalPos)
                originalPos = [float(x.strip()) for x in originalPos if len(x.strip()) > 0]
                newClusters=kmeans
            except:
                tsneX, tsneY, pos, kmeans, frames = self.data.iloc[item].tolist()
                originalPos = []
        pos=pos[1:-1].strip().split('\n')
        pos=[x.strip().split(' ') for x in pos]
        pos=np.hstack(pos)
        pos=[float(x.strip()) for x in pos if len(x.strip())>0]
        # target=torch.zeros(self.num_clusters)#, dtype=torch.long)
        # target[kmeans-1]=kmeans
        target = newClusters
        return torch.tensor(pos).float(), torch.tensor([tsneX, tsneY]).float(), target, torch.tensor(originalPos)

if __name__ == '__main__':
    x=TSNEGT('/Users/faith_johnson/GitRepos/PedTrajPred/',50)#trajData_2thresh_8window.csv', 50)
    x.__getitem__(304)
    x.__getitem__(305)