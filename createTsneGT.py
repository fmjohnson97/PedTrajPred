from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import argparse
from torch.utils.data import DataLoader
from sameSizeData import SameSizeData
import cv2
import numpy as np
from matplotlib import pyplot as plt
from collections import defaultdict

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
    parser.add_argument('--num_clusters', default=50, type=int, help='number of clusters for kmeans')
    parser.add_argument('--input_window', default=8, type=int, help='number of frames for the input data')
    parser.add_argument('--output_window', default=0, type=int, help='number of frames for the output data')
    parser.add_argument('--group_size', default=2, type=int, help='number of people to include in each dist group')
    parser.add_argument('--dist_thresh', default=0.01, type=int, help='max avg distance to count people in the same social group')
    parser.add_argument('--traj_thresh', default=2,  type=int, help='number of trajectories in each data point')
    parser.add_argument('--social_thresh', default=2,  type=int, help='number of social groups in each data point')
    args = parser.parse_args()
    return args

def plotTSNE():
    print('creating the tsne embedding')
    trajData = tsne.fit_transform(positions)
    # ToDo: save trajData, especially since positions wont change from run to run

    print('plotting the data')
    kmeans = KMeans(n_clusters=args.num_clusters, random_state=0).fit(trajData)
    centers = kmeans.cluster_centers_

    plt.scatter(trajData[:, 0], trajData[:, 1], c=kmeans.labels_, alpha=0.5)
    plt.scatter(centers[:, 0], centers[:, 1], c='r')
    plt.title(str(args.traj_thresh) + " Traj Data, " + str(args.num_clusters) + " Clusters, Len " + str(
        args.input_window + args.output_window))
    plt.show()

    # for i in range(max(kmeans.labels_)+1):
    #     clusters=[x for j,x in enumerate(trajData) if kmeans.labels_[j]==i]
    #     for c in clusters:
    #         x = [p for j, p in enumerate(c) if j % 2 == 1]
    #         y = [p for j, p in enumerate(c) if j % 2 == 0]
    #         plt.plot(x,y)
    #         plt.scatter(x[0],y[0],c='r')
    #     plt.title('Cluster '+str(i))
    #     plt.show()

    breakpoint()
    for a in range(10):
        coords = []
        fig = plt.figure()
        cid = fig.canvas.mpl_connect('button_press_event', onclick)
        plt.scatter(trajData[:, 0], trajData[:, 1], c=kmeans.labels_, alpha=0.5)
        plt.scatter(centers[:, 0], centers[:, 1], c='r')
        plt.title(str(args.traj_thresh) + " Traj Data, " + str(args.num_clusters) + " Clusters, Len " + str(
            args.input_window + args.output_window))
        plt.show()

        point1 = find_nearest(tsne.embedding_[:, 0], coords[0][0])
        point2 = find_nearest(tsne.embedding_[:, 1], coords[0][1])

        diff1 = np.abs(data[point1] - coords)
        diff2 = np.abs(data[point2] - coords)
        if np.sum(diff1) <= np.sum(diff2):
            point = point1[0][0]
        else:
            point = point2[0][0]

        plt.scatter(trajData[:, 0], trajData[:, 1], c=kmeans.labels_, alpha=0.5)
        plt.scatter(centers[:, 0], centers[:, 1], c='r')
        plt.scatter(trajData[point][0], trajData[point][1], c='r', alpha=1)
        plt.title(str(args.traj_thresh) + " Traj Data, " + str(args.num_clusters) + " Clusters, Len " + str(
            args.input_window + args.output_window))
        plt.show()
        plt.pause(.5)
        plt.close()

        for f in data['frames'][point]:
            # import pdb; pdb.set_trace()
            cv2.imshow('', f.numpy())
            cv2.waitKey(1000)


if __name__=='__main__':
    args=get_args()
    data=defaultdict(list)

    # get the data from each dataset
    for name in ['ETH', 'ETH_Hotel', 'UCY_Zara1', 'UCY_Zara2']:
        dataset=SameSizeData(name, output_window=args.output_window, group_size=args.group_size, distThresh=args.dist_thresh,
                             trajThresh=args.traj_thresh, socialThresh=args.social_thresh)
        for d in dataset:
            if len(d['pos'])>0:
                data['pos'].extend([d['pos']])
                data['peopleIDs'].extend([d['peopleIDs']])
            if len(d['deltas'])>0:
                data['deltas'].extend([d['deltas']])
                data['groupIDs'].extend([d['groupIDs']])
            if len(d['frames'])>0:
                data['frames'].append(d['frames'])

    print("flattening the data")
    #flatten it and keep track of the index for the frames
    if args.traj_thresh is not None:
        positions=[]
        #TODo: keep track of peopleIDs???
        posFrames=[]
        for i,x in enumerate(data['pos']):
            posFrames.extend([data['frames'][i]] * x.shape[0])
            positions.extend(x.reshape(-1,args.traj_thresh*(args.input_window+args.output_window)*2))

    if args.social_thresh is not None:
        groups=[]
        #ToDo: keep track of groupIDs???
        groupFrames=[]
        for i,x in enumerate(data['deltas']):
            groupFrames.extend([data['frames']] * x.shape[0])
            groups.extend(x.reshape(-1,args.social_thresh*(args.input_window+args.output_window)*2))

    print(len(positions))
    tsne = TSNE()
    if args.traj_thresh is not None:
        trajData = tsne.fit_transform(positions)
        np.savetxt('trajData_' + str(args.traj_thresh) + 'thresh_' + str(args.input_window + args.output_window) + 'window.npy', trajData)

    print(len(groups))
    if args.social_thresh is not None:
        socData = tsne.fit_transform(groups)
        np.savetxt('socData_' + str(args.social_thresh) + 'thresh_' + str(args.group_size)+'group_'+ str(args.input_window + args.output_window) + 'window.npy', socData)
    breakpoint()

    if args.traj_thresh is not None:
        plotTSNE()


    if args.social_thresh is not None:
        plotTSNE()