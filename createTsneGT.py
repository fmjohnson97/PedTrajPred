from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import argparse
from torch.utils.data import DataLoader
from sameSizeData import SameSizeData
import cv2
import numpy as np
from matplotlib import pyplot as plt
from collections import defaultdict


# finds the nearest value in the array of points to the location of the mouse click
def find_nearest(data, coord):
    point =np.argmin(np.sum((data - coord)**2, axis=-1))
    return point

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_clusters', default=50, type=int, help='number of clusters for kmeans')
    parser.add_argument('--input_window', default=8, type=int, help='number of frames for the input data')
    parser.add_argument('--output_window', default=0, type=int, help='number of frames for the output data')
    parser.add_argument('--group_size', default=2, type=int, help='number of people to include in each dist group')
    parser.add_argument('--dist_thresh', default=0.0085, type=float, help='max avg distance to count people in the same social group')
    parser.add_argument('--traj_thresh', default=2, type=int, help='number of trajectories in each data point')
    parser.add_argument('--social_thresh', default=2,  type=int, help='number of social groups in each data point')
    args = parser.parse_args()
    return args

def plotTSNE(data, frames, original):
    print('plotting the data')
    kmeans = KMeans(n_clusters=args.num_clusters, random_state=0).fit(data)
    centers = kmeans.cluster_centers_

    # for i in range(max(kmeans.labels_)+1):
    #     clusters=[x for j,x in enumerate(trajData) if kmeans.labels_[j]==i]
    #     for c in clusters:
    #         x = [p for j, p in enumerate(c) if j % 2 == 1]
    #         y = [p for j, p in enumerate(c) if j % 2 == 0]
    #         plt.plot(x,y)
    #         plt.scatter(x[0],y[0],c='r')
    #     plt.title('Cluster '+str(i))
    #     plt.show()

    # def onclick(event):
    #     global ix, iy
    #     ix, iy = event.xdata, event.ydata
    #     global coords
    #     coords = [(ix, iy)]
    #     if len(coords) > 0:
    #         # print(coords)
    #         fig.canvas.mpl_disconnect(cid)
    #         plt.close(1)

    for a in range(30):
        npoints=int(input("How many points will you click?"))
        fig = plt.figure()
        # cid = fig.canvas.mpl_connect('button_press_event', onclick)
        plt.scatter(data[:, 0], data[:, 1], c=kmeans.labels_, alpha=0.5)
        plt.scatter(centers[:, 0], centers[:, 1], c='r')
        plt.title(str(args.traj_thresh) + " Traj Data, " + str(args.num_clusters) + " Clusters, Len " + str(
            args.input_window + args.output_window))
        # plt.title('socData_' + str(args.social_thresh) + 'thresh_' + str(args.group_size) + 'group_' + str(args.input_window + args.output_window)+
        #     "window_Clusters" + str(args.num_clusters))
        plt.waitforbuttonpress()
        points_clicked = plt.ginput(npoints, show_clicks=True)
        plt.show()

        points=[]
        plt.scatter(data[:, 0], data[:, 1], c=kmeans.labels_, alpha=0.5)
        plt.scatter(centers[:, 0], centers[:, 1], c='r')
        plt.title(str(args.traj_thresh) + " Traj Data, " + str(args.num_clusters) + " Clusters, Len " + str(
            args.input_window + args.output_window))
        # plt.title('socData_' + str(args.social_thresh) + 'thresh_' + str(args.group_size) + 'group_' + str(args.input_window + args.output_window)+
        #     "window_Clusters" + str(args.num_clusters))
        for coords in points_clicked:
            points.append(find_nearest(data, coords))
            plt.scatter(data[points[-1]][0], data[points[-1]][1], c='k', alpha=1)

        plt.show()
        plt.pause(.5)
        plt.close()

        # breakpoint()
        # fdataset=SameSizeData(frames[point][0], image='mask', output_window=args.output_window, group_size=args.group_size, distThresh=args.dist_thresh,trajThresh=args.traj_thresh, socialThresh=args.social_thresh)
        for i, point in enumerate(points):
            plt.figure()
            for pos in original[point].reshape(-1, args.input_window+args.output_window, 2):
                plt.plot(pos[:,0], pos[:,1])
                plt.scatter(pos[0][0], pos[0][1])
                plt.title('Point '+str(i))
        plt.figure()
        plt.scatter(data[:, 0], data[:, 1], c=kmeans.labels_, alpha=0.5)
        plt.scatter(centers[:, 0], centers[:, 1], c='r')
        plt.title(str(args.traj_thresh) + " Traj Data, " + str(args.num_clusters) + " Clusters, Len " + str(
            args.input_window + args.output_window))
        # plt.title('socData_' + str(args.social_thresh) + 'thresh_' + str(args.group_size) + 'group_' + str(args.input_window + args.output_window)+
        #     "window_Clusters" + str(args.num_clusters))
        for point in points:
            plt.scatter(data[point][0], data[point][1], c='k', alpha=1)
        plt.show()
        # images = fdataset.getImages({'frames':frames[point][1:], 'pos':original[point].reshape(-1, args.input_window+args.output_window, 2)})
        # for im in images:
        #     cv2.imshow('', im)#.numpy())
        #     cv2.waitKey(1000)

def createManifold(args):
    data = defaultdict(list)
    print("getting and flattening the data")
    # get the data from each dataset
    for name in ['ETH', 'ETH_Hotel', 'UCY_Zara1', 'UCY_Zara2']:
        dataset = SameSizeData(name, input_window=args.input_window, output_window=args.output_window, group_size=args.group_size,
                               distThresh=args.dist_thresh,
                               trajThresh=args.traj_thresh, socialThresh=args.social_thresh)
        for i, d in enumerate(dataset):
            if len(d['pos']) > 0:
                # breakpoint()
                data['pos'].append(d['pos'].flatten())
                data['distTraj'].append(d['distTraj'].flatten())
                data['diffs'].append(d['diffs'].flatten())
                data['spline'].append(d['spline'].flatten())
                data['peopleIDs'].append(d['peopleIDs'])
                data['posFrames'].append(d['frames'])

            if len(d['deltas']) > 0:
                # breakpoint()
                data['deltas'].append(d['deltas'].flatten())
                data['groupIDs'].append(d['groupIDs'])
                data['groupFrames'].append(d['frames'])
                data['plotPos'].append(d['plotPos'].flatten())


    print('creating the tsne embedding')
    # breakpoint()
    tsne = TSNE()
    if args.traj_thresh is not None:
        # print(len(data['pos']))
        # trajData = tsne.fit_transform(data['pos'])
        # np.savetxt('trajData_' + str(args.traj_thresh) + 'thresh_' + str(args.input_window + args.output_window) + 'window.npy', trajData)
        # print(len(data['distTraj']))
        # distTrajData = tsne.fit_transform(data['distTraj'])
        # np.savetxt('distTrajData_' + str(args.traj_thresh) + 'thresh_' + str(args.input_window + args.output_window) + 'window.npy', distTrajData)
        print(len(data['diffs']))
        diffsData = tsne.fit_transform(data['diffs'])
        np.savetxt('diffsData_' + str(args.traj_thresh) + 'thresh_' + str(args.input_window + args.output_window) + 'window.npy', diffsData)
        # print(len(data['spline']))
        # splineData = tsne.fit_transform(data['spline'])
        # np.savetxt('splineData_' + str(args.traj_thresh) + 'thresh_' + str(
        #     args.input_window + args.output_window) + 'window.npy', splineData)

    if args.social_thresh is not None:
        print(len(data['deltas']))
        socData = []#tsne.fit_transform(data['deltas'])
        # np.savetxt('socData_' + str(args.social_thresh) + 'thresh_' + str(args.group_size) + 'group_' + str(
        #     args.input_window + args.output_window) + 'window.npy', socData)

    return diffsData, socData, data


def loadData(args):
    trajData, socData = None, None
    data=defaultdict(list)
    for name in ['ETH', 'ETH_Hotel', 'UCY_Zara1', 'UCY_Zara2']:
        dataset = SameSizeData(name, output_window=args.output_window, group_size=args.group_size,
                               distThresh=args.dist_thresh,
                               trajThresh=args.traj_thresh, socialThresh=args.social_thresh)
        for i, d in enumerate(dataset):
            # if i>len(dataset)/2:
            #     break

            if len(d['pos']) > 0:
                data['pos'].append(d['pos'].flatten())
                # data['distTraj'].append(d['distTraj'].flatten())
                # data['diffs'].append(d['diffs'].flatten())
                # data['spline'].append(d['spline'].flatten())
                data['peopleIDs'].append(d['peopleIDs'])
                data['posFrames'].append(d['frames'])

            if len(d['deltas']) > 0:
                data['deltas'].append(d['deltas'].flatten())
                data['groupIDs'].append(d['groupIDs'])
                data['groupFrames'].append(d['frames'])
                data['plotPos'].append(d['plotPos'].flatten())

    if args.traj_thresh is not None:
        trajData = np.loadtxt('trajData_' + str(args.traj_thresh) + 'thresh_' + str(
            args.input_window + args.output_window) + 'window.npy')
        # distTrajData = np.loadtxt('distTrajData_' + str(args.traj_thresh) + 'thresh_' + str(
        #     args.input_window + args.output_window) + 'window.npy')
        # diffsData = np.loadtxt('diffsData_' + str(args.traj_thresh) + 'thresh_' + str(
        #     args.input_window + args.output_window) + 'window.npy')
        # splineData = np.loadtxt('splineData_' + str(args.traj_thresh) + 'thresh_' + str(
        #     args.input_window + args.output_window) + 'window.npy')


    if args.social_thresh is not None:
        socData = np.loadtxt('socData_' + str(args.social_thresh) + 'thresh_' + str(args.group_size) + 'group_' + str(
            args.input_window + args.output_window) + 'window.npy')

    return trajData, socData, data

if __name__=='__main__':
    args=get_args()
    diffsData, socData, data = createManifold(args)
    # trajData, socData, distTrajData, data = loadData(args)

    import pandas as pd
    kmeans = KMeans(n_clusters=args.num_clusters, random_state=0).fit(diffsData)
    temp=pd.DataFrame()
    temp['tsne_X']=diffsData[:,0]
    temp['tsne_Y'] = diffsData[:, 1]
    temp['pos'] = data['diffs']
    temp['kmeans'] = kmeans.labels_
    temp['frames'] = data['posFrames']
    temp['plotPos']=data['pos']
    temp.to_csv('diffsData_' + str(args.traj_thresh) + 'thresh_'+ str(args.input_window + args.output_window) + 'window.csv') #+ str(args.group_size) + 'group_'

    breakpoint()
    print('Plotting traj data')
    if args.traj_thresh is not None:
        plotTSNE(diffsData, data['posFrames'], data['plotPos'])

    # breakpoint()
    print('Plotting social dist data')
    if args.social_thresh is not None:
        plotTSNE(socData, data['groupFrames'], data['deltas'])