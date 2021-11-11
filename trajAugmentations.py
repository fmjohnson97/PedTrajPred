from torchvision import transforms
import torch
import random
import numpy as np
from scipy.interpolate import interp1d

class TrajAugs():
    def __init__(self, num_augs=1, mask_percent=0.25, args=None, include_aug=False):
        self.augs=[self.resample_up, self.resample_down, self.translate,
                   self.rotate,  self.flip_horiz, self.flip_vert]
        #self.noise, self.translate,
        if args:
            self.include_aug = args.include_aug
            self.num_augs = args.num_augs
            self.mask_percent = args.mask_percent
        else:
            self.include_aug=include_aug
            self.num_augs=num_augs
            self.mask_percent=mask_percent

    def augment(self, traj):
        augs=self.pick_aug()
        for a in augs:
            traj=a(traj)
        if self.include_aug:
            return traj, augs
        else:
            return traj

    def pick_aug(self):
        return random.sample(self.augs, self.num_augs)

    def interpolate(self, traj, new_points):
        func=interp1d(traj[:,0], traj[:,1], 'slinear', fill_value='extrapolate')
        return func(new_points)

    def resample_up(self, traj):
        # make the trajectory seem faster by making each point a little
        # further along the path than it originally was
        self.freq_increment = random.uniform(0.001, 0.03)
        inc=[0]
        inc.extend([self.freq_increment]*(len(traj)-1))
        new_x=traj[:,0]+np.cumsum(inc)
        new_y=self.interpolate(traj,new_x)
        return np.concatenate(([new_x],[new_y]), axis=0).T

    def resample_down(self, traj):
        # make the trajectory seem slower by making each point a little
        # closer to the previous point along the path
        self.freq_increment=random.uniform(0.001, 0.03)
        inc = [0]
        inc.extend([-self.freq_increment] * (len(traj)- 1))
        new_x = traj[:, 0] + np.cumsum(inc)
        new_y = self.interpolate(traj, new_x)
        return np.concatenate(([new_x], [new_y]), axis=0).T

    def masking(self, traj):
        # fill in end with -1 since we normalize to [-1,1]
        mask_len=round(self.mask_percent*len(traj))
        traj_copy=traj.copy()
        traj_copy[-mask_len:] = [[0, 0]] * mask_len
        return traj_copy

    def rotate(self, traj):
        # Rotates by 90, 180, or 270 degrees (to the right?)
        self.rotDegree=random.randint(0,360)#choice([90, 180, 270])
        theta=np.deg2rad(self.rotDegree)
        rot=np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
        traj=np.dot(rot,traj.T)
        return traj.T

    def translate(self, traj):
        self.trans_increment = [random.uniform(-1, 1),random.uniform(-1, 1)]
        return traj+self.trans_increment

    def flip_horiz(self, traj):
        return traj*np.array([[1,-1]])

    def flip_vert(self, traj):
        return traj*np.array([[-1,1]])

if __name__=='__main__':
    traj=np.array([[1,-1],[0.95,-0.95],[0.9,-0.9],[0.85,-0.85],[0.8,-0.8],[0.75,-0.75],[0.7,-0.7]])
    traj2=np.array([[1,-.1],[0.95,-0.095],[0.9,-0.09],[0.85,0.085],[0.8,0.09],[0.75,.01],[0.7,.015]])
    augmentations=TrajAugs()
    from matplotlib import pyplot as plt
    plt.figure()
    # plt.plot(traj[:,0], traj[:,1])
    plt.plot(traj2[:,0], traj2[:,1])
    # temp=augmentations.flip_vert(traj)
    # plt.plot(temp[:,0],temp[:,1])
    # temp=augmentations.flip_horiz(traj)
    # plt.plot(temp[:,0], temp[:,1])
    # temp=augmentations.rotate(traj)
    # plt.plot(temp[:,0],temp[:,1])
    # print(augmentations.rotDegree)
    # temp=augmentations.masking(traj)
    # plt.scatter(temp[:,0], temp[:,1])
    # temp=augmentations.resample_down(traj2)
    # plt.scatter(temp[:,0], temp[:,1])
    # print(augmentations.freq_increment)
    # temp2=augmentations.resample_up(traj2)
    # plt.scatter(temp2[:,0],temp2[:,1])
    # print(augmentations.freq_increment)
    temp=augmentations.translate(traj2)
    plt.plot(temp[:,0],temp[:,1])
    print(augmentations.trans_increment)
    # plt.scatter(np.linspace(1,0.7,50),augmentations.interpolate(traj2,np.linspace(1,0.7,50)))
    # plt.legend(['orig','vert flip', 'horiz flip'])
    plt.show()