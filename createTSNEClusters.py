from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import argparse
from OpenTraj.opTrajData import OpTrajData
from torch.utils.data import DataLoader
import cv2
import numpy as np
from matplotlib import pyplot as plt
from TrajNet.trajnetplusplusbaselines.trajnetbaselines.lstm.data_load_utils import prepare_data
from TrajNet.TrajNetData import TrajNetData

def onclick(event):
    global ix, iy
    ix,iy = event.xdata, event.ydata
    global coords
    coords=[(ix,iy)]
    if len(coords)>0:
        # print(coords)
        fig.canvas.mpl_disconnect(cid)
        plt.close(1)

#finds the nearest value in the array of points to the location of the mouse click
def find_nearest(array,value):
    temp=np.abs(array - value)
    small=np.abs(array - value)
    small.sort()
    small=small[:10]
    diff=1e6
    for s in small:
        p=np.where(temp==s)
        d=np.sum(abs(data[p]-coords))
        if d<diff:
            diff=d
            point=p
    return point

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='opentraj', choices=('trajnet++', 'MOT', 'opentraj'),
                            help='which dataset to tsne cluster')
    parser.add_argument('--data_type', default='position', choices=('position', 'velocity', 'distance'),
                        help='whether to cluster the raw positions or the velocities (diff b/w positions)')
    parser.add_argument('--mode', default='by_human', choices=('by_human', 'by_frame', 'by_N_frame'),
                            help='whether to group by person or by frame for opentraj')
    parser.add_argument('--num_clusters', default=50, type=int)
    parser.add_argument('--num_frames', default=1, type=int,
                        help='number of frames of people to use for opentraj')
    parser.add_argument('--seq_len', default=8, type=int)
    parser.add_argument('--socialN', default=3, type=int)
    parser.add_argument('--collapse_method', default='squeeze', choices=('squeeze','l2','x','y'),
                        help='how to deal with the fact that TSNE only takes 2d arrays (how to get rid \
                                of the 3rd Dim')
    parser.add_argument('--trajnet_path', default='TrajNet/trajnetplusplusbaselines/DATA_BLOCK/trajdata',
                        help='path to trajnet++ data')
    parser.add_argument('--sample', default=1.0, type=float,
                        help='sample ratio when loading train/val scenes')
    parser.add_argument('--goals', action='store_true',
                        help='flag to consider goals of pedestrians') #true makes it false somehow *shrugs*
    args = parser.parse_args()
    return args

def getData(args):
    fullData = []
    velData=[]
    if args.data=='trajnet++':
        train_scenes, train_goals, _ = prepare_data(args.trajnet_path, subset='/train/', sample=args.sample, goals=args.goals)
        data=TrajNetData(train_scenes)
        dloader=DataLoader(data, batch_size=1, shuffle=False, drop_last=False)
        fullData.extend(processData(dloader, args))
    elif args.data=='MOT':
        pass
    elif args.data=='opentraj':
        for name in ['ETH', 'ETH_Hotel', 'UCY_Zara1', 'UCY_Zara2']:
            data = OpTrajData(name, args.mode, None ,input_window=args.num_frames, output_window=args.seq_len)
            dloader=DataLoader(data, batch_size=1, shuffle=False, drop_last=False)
            vel, pos=processData(dloader, args)
            fullData.extend(pos)
            velData.extend(vel)
        # in the format input, target, frames
    else:
        print(args.data,' is an invalid dataset')
        raise NotImplementedError

    return np.array(velData), np.array(fullData)

def processData(dloader, args):
    data=[]
    vels=[]
    mindata=np.inf
    maxdata=-np.inf
    minvels=np.inf
    maxvels=-np.inf
    for i,info in enumerate(dloader):
        if args.mode in ['by_N_frame','by_human']:
            d_Traj=info[0][0].squeeze().numpy()
        elif args.mode=='by_frame':
            d_Traj=info[2][0].numpy()

        if args.data_type=='distance':
            v_Traj=np.sqrt(np.sum(np.diff(d_Traj, axis=0)**2,axis=1))
        elif len(d_Traj.shape)>2:
            v_Traj=np.diff(d_Traj, axis=1)
        else:
            v_Traj=np.diff(d_Traj, axis=0)

        # breakpoint()
        if args.mode=='by_N_frame':
            if len(d_Traj.shape)>2 and d_Traj.shape[0]>=args.socialN:
                d_Traj=d_Traj[:args.socialN]
                v_Traj=v_Traj[:args.socialN]
            else:
                d_Traj=np.array([])
                v_Traj=np.array([])

        if len(d_Traj.shape)>1 and d_Traj.shape[-2]>=args.seq_len:
            if mindata>np.min(d_Traj):
                mindata=np.min(d_Traj)
            if maxdata<np.max(d_Traj):
                maxdata=np.max(d_Traj)
            if minvels>np.min(v_Traj):
                minvels=np.min(v_Traj)
            if maxvels<np.max(v_Traj):
                maxvels=np.max(v_Traj)
            if args.collapse_method=='l2':
                data.append(np.linalg.norm(d_Traj[:args.seq_len],2, axis=1))
                vels.append(np.linalg.norm(v_Traj[:args.seq_len-1], 2, axis=1))
            elif args.collapse_method=='x':
                data.append(d_Traj[:args.seq_len,0])
                vels.append(v_Traj[:args.seq_len-1, 0])
            elif args.collapse_method == 'y':
                data.append(d_Traj[:args.seq_len, 1])
                vels.append(v_Traj[:args.seq_len-1, 1])
            else:
                if len(d_Traj.shape)>2:
                    data.append(d_Traj[:,:args.seq_len,:])
                    vels.append(v_Traj[:,:args.seq_len-1,:])
                else:
                    data.append(d_Traj[:args.seq_len])
                    vels.append(v_Traj[:args.seq_len-1])
    # breakpoint()
    data=np.stack(data)
    data=2. * (data- mindata) / (maxdata-mindata) - 1
    vels = np.stack(vels)
    vels = 2. * (vels - minvels) / (maxvels-minvels) - 1
    return vels, data
    
def filterNumPeople(pos, vel):
    topop=[]
    tosplit=[]
    for i,p in enumerate(pos):
        if len(p)<args.socialN or len(p.shape)<2:
            topop.append(i)
        if len(p.shape)>2 and len(p)>args.socialN:
            tosplit.append(i)
    for i in tosplit:
        # ind=[range(x,x+args.socialN) for x in range(len(pos[i])-len(pos[i])%args.socialN)]
        pos[i]=pos[i][:args.socialN]
    pos=np.delete(pos,topop, axis=0)
    vel=np.delete(vel,topop, axis=0)
    return pos, vel

if __name__=='__main__':
    args=get_args()
    velData, fullData=getData(args)
    # breakpoint()

    if args.data_type==['velocity', 'distance']:
        print('Using',str(len(velData)),'trajectories for TSNE')
    else:
        print('Using',str(len(fullData)),'trajectories for TSNE')
    
    # if args.mode=='by_N_frame':
    #     fullData, velData = filterNumPeople(fullData, velData)
    
    if args.collapse_method == 'squeeze':
        # if len(fullData.shape)<2:
        #     fullData=np.array([d.reshape(len(d),-1) for d in fullData])
        #     if args.data_type!='distance':
        #         velData = np.array([d.reshape(len(d),-1) for d in velData])
        #
        # else:
        fullData=fullData.reshape(len(fullData), -1)
        if args.data_type!='distance':
            velData = velData.reshape(len(velData), -1)
    print(fullData.shape, velData.shape)

    tsneA = TSNE()
    if args.data_type in ['velocity', 'distance']:
        data = tsneA.fit_transform(velData)
    else:
        data = tsneA.fit_transform(fullData)

    kmeans = KMeans(n_clusters=args.num_clusters, random_state=0).fit(data)
    centers = kmeans.cluster_centers_

    plt.scatter(data[:, 0], data[:, 1],  alpha=0.5)#c=kmeans.labels_,
    # plt.scatter(centers[:, 0], centers[:, 1], c='r')
    plt.title(str(args.data).title()+" Raw Data and Centroids, "+str(args.num_clusters)+" Clusters, Len "+str(args.seq_len))
    plt.show()

    print(max(kmeans.labels_))
    # for i in range(max(kmeans.labels_)+1):
    #     clusters=[x for j,x in enumerate(fullData) if kmeans.labels_[j]==i]
    #     for c in clusters:
    #         x = [p for j, p in enumerate(c) if j % 2 == 1]
    #         y = [p for j, p in enumerate(c) if j % 2 == 0]
    #         plt.plot(x,y)
    #         plt.scatter(x[0],y[0],c='r')
    #     plt.title('Cluster '+str(i))
    #     plt.show()

    while (True):
        coords = []
        fig = plt.figure()
        cid = fig.canvas.mpl_connect('button_press_event', onclick)
        plt.scatter(data[:, 0], data[:, 1], c=kmeans.labels_, alpha=0.5)
        plt.scatter(centers[:, 0], centers[:, 1], c='r')
        plt.title(str(args.data).title()+" Raw Data and Centroids, "+str(args.num_clusters)+" Clusters, Len "+str(args.seq_len))
        plt.show()

        point1 = find_nearest(tsneA.embedding_[:, 0], coords[0][0])
        point2 = find_nearest(tsneA.embedding_[:, 1], coords[0][1])

        diff1 = np.abs(data[point1] - coords)
        diff2 = np.abs(data[point2] - coords)
        if np.sum(diff1) <= np.sum(diff2):
            point = point1[0][0]
        else:
            point = point2[0][0]

        plt.scatter(data[:, 0], data[:, 1], c=kmeans.labels_, alpha=0.5)
        plt.scatter(centers[:, 0], centers[:, 1], c='r')
        plt.scatter(data[point][0], data[point][1], c='r', alpha=1)
        plt.title(str(args.data).title() + " Raw Data and Centroids, " + str(args.num_clusters) + " Clusters, Len " + str(args.seq_len))
        plt.show()
        plt.pause(.5)
        plt.close()

        x=[p for i,p in enumerate(fullData[point]) if i%2==1]
        y = [p for i, p in enumerate(fullData[point]) if i % 2 == 0]
        plt.plot(x,y)
        plt.title('Clicked Trajectory')
        plt.show()
