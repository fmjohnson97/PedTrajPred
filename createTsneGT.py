from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import argparse
from torch.utils.data import DataLoader
from sameSizeData import SameSizeData
from firstPOVData import FirstPOVData
import cv2
import numpy as np
from matplotlib import pyplot as plt
from collections import defaultdict
import pandas as pd
from sklearn.decomposition import PCA


# finds the nearest value in the array of points to the location of the mouse click
def find_nearest(data, coord):
    point =np.argmin(np.sum((data - coord)**2, axis=-1))
    return point

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_clusters', default=10, type=int, help='number of clusters for kmeans')
    parser.add_argument('--input_window', default=8, type=int, help='number of frames for the input data')
    parser.add_argument('--output_window', default=0, type=int, help='number of frames for the output data')
    parser.add_argument('--group_size', default=2, type=int, help='number of people to include in each dist group')
    parser.add_argument('--dist_thresh', default=0.0085, type=float, help='max avg distance to count people in the same social group')
    parser.add_argument('--traj_thresh', default=2, type=int, help='number of trajectories in each data point')
    parser.add_argument('--social_thresh', default=2,  type=int, help='number of social groups in each data point')
    args = parser.parse_args()
    return args

def plotTSNE(data, frames, original, color=None):
    print('plotting the data')
    kmeans = KMeans(n_clusters=args.num_clusters, random_state=0).fit(data)
    centers = kmeans.cluster_centers_
    if color is None:
        color=kmeans.labels_

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
        # plt.scatter(data[:, 0], data[:, 1], c=kmeans.labels_, alpha=0.5)
        plt.scatter(data[:, 0], data[:, 1], c=color, alpha=0.5)
        # plt.scatter(centers[:, 0], centers[:, 1], c='r')
        plt.title(str(args.traj_thresh) + " Traj Data, " + str(args.num_clusters) + " Clusters, Len " + str(
            args.input_window + args.output_window))
        # plt.title('socData_' + str(args.social_thresh) + 'thresh_' + str(args.group_size) + 'group_' + str(args.input_window + args.output_window)+
        #     "window_Clusters" + str(args.num_clusters))
        plt.waitforbuttonpress()
        points_clicked = plt.ginput(npoints, show_clicks=True)
        plt.show()

        points=[]
        # plt.scatter(data[:, 0], data[:, 1], c=kmeans.labels_, alpha=0.5)
        plt.scatter(data[:, 0], data[:, 1], c=color, alpha=0.5)
        # plt.scatter(centers[:, 0], centers[:, 1], c='r')
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
        for i, point in enumerate(points):
            plt.figure()
            for pos in original[point].reshape(-1, args.input_window+args.output_window, 2):
                plt.plot(pos[:,0], pos[:,1])
                plt.scatter(pos[0][0], pos[0][1])
                plt.title('Point '+str(i))
        plt.figure()
        plt.scatter(data[:, 0], data[:, 1], c=color, alpha=0.5)
        # plt.scatter(data[:, 0], data[:, 1], c=kmeans.labels_, alpha=0.5)
        # plt.scatter(centers[:, 0], centers[:, 1], c='r')
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
        # dataset = FirstPOVData(name, input_window=args.input_window, output_window=args.output_window,
        #                        trajThresh=args.traj_thresh)
        for i, d in enumerate(dataset):
            if len(d['pos']) > 0:
                # breakpoint()
                data['pos'].append(d['pos'].flatten())
                data['diffs'].append(d['diffs'].flatten())
                data['spline'].append(d['spline'].flatten())
                data['peopleIDs'].append(d['peopleIDs'])
                data['posFrames'].append(d['frames'])
                data['allDiffs'].append(d['allDiffs'].flatten())
                # data['pos'].extend(d['pos'])
                # data['diffs'].extend(d['diffs'])
                # data['allDiffs'].extend(d['allDiffs'])
                # data['angDiffs'].extend(d['angDiffs'])
                # data['diffsPos'].extend(d['diffsPos'])
                # data['diffsFrames'].extend(d['diffsFrames'])

            # if len(d['deltas']) > 0:
            #     # breakpoint()
            #     data['deltas'].append(d['deltas'].flatten())
            #     data['groupIDs'].append(d['groupIDs'])
            #     data['groupFrames'].append(d['frames'])
            #     data['plotPos'].append(d['plotPos'].flatten())


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
        print(len(data['allDiffs']))
        diffsData = tsne.fit_transform(np.array(data['allDiffs']).reshape(len(data['allDiffs']),-1))
        # print(len(data['allDiffs']))
        # diffsData = tsne.fit_transform(np.array(data['allDiffs']))
        # np.savetxt('diffsData_' + str(args.traj_thresh) + 'thresh_' + str(args.input_window + args.output_window) + 'window.npy', diffsData)
        # print(len(data['spline']))
        # splineData = tsne.fit_transform(data['spline'])
        # np.savetxt('splineData_' + str(args.traj_thresh) + 'thresh_' + str(
        #     args.input_window + args.output_window) + 'window.npy', splineData)

    if args.social_thresh is not None:
        # print(len(data['deltas']))
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
        # dataset = FirstPOVData(name, input_window=args.input_window, output_window=args.output_window,
        #                        trajThresh=args.traj_thresh)
        for i, d in enumerate(dataset):
            # if i>len(dataset)/2:
            #     break

            if len(d['pos']) > 0:
                data['pos'].append(d['pos'].flatten())
                data['diffs'].append(d['diffs'].flatten())
                data['spline'].append(d['spline'].flatten())
                data['peopleIDs'].append(d['peopleIDs'])
                data['posFrames'].append(d['frames'])
                data['allDiffs'].append(d['allDiffs'].flatten())
                # data['pos'].extend(d['pos'])
                # data['diffs'].extend(d['diffs'].flatten())
                # data['allDiffs'].extend(d['allDiffs'].flatten())
                # data['angAllDiffs'].extend(d['angAllDiffs'].flatten())
                # data['diffsPos'].extend(d['diffsPos'].flatten())

            # if len(d['deltas']) > 0:
                # data['deltas'].append(d['deltas'].flatten())
                # data['groupIDs'].append(d['groupIDs'])
                # data['groupFrames'].append(d['frames'])
                # data['plotPos'].append(d['plotPos'].flatten())

    if args.traj_thresh is not None:
        print('Loading','diffsData_' + str(args.traj_thresh) + 'thresh_' + str(args.input_window + args.output_window) + 'window.npy')
        trajData = []#np.loadtxt('diffsData_' + str(args.traj_thresh) + 'thresh_' + str(
            # args.input_window + args.output_window) + 'window.npy')
        dataPD = pd.read_csv('allDiffsData_' + str(args.traj_thresh) + 'thresh_' + str(
            args.input_window + args.output_window) + 'window.csv')
        # distTrajData = np.loadtxt('distTrajData_' + str(args.traj_thresh) + 'thresh_' + str(
        #     args.input_window + args.output_window) + 'window.npy')
        # diffsData = np.loadtxt('diffsData_' + str(args.traj_thresh) + 'thresh_' + str(
        #     args.input_window + args.output_window) + 'window.npy')
        # splineData = np.loadtxt('splineData_' + str(args.traj_thresh) + 'thresh_' + str(
        #     args.input_window + args.output_window) + 'window.npy')

    socData=[]
    # if args.social_thresh is not None:
    #     socData = np.loadtxt('socData_' + str(args.social_thresh) + 'thresh_' + str(args.group_size) + 'group_' + str(
    #         args.input_window + args.output_window) + 'window.npy')

    return trajData, socData, dataPD, data

def centroidnp(arr):
    length = arr.shape[0]
    sum_x = np.sum(arr[:, 0])
    sum_y = np.sum(arr[:, 1])
    return sum_x/length, sum_y/length

def custom_clusters(args, diffsData, frames, positions, temp):
    kmeans = KMeans(n_clusters=24, random_state=0).fit(diffsData)
    centers = kmeans.cluster_centers_

    fig = plt.figure()
    plt.scatter(diffsData[:, 0], diffsData[:, 1], alpha=0.5, c=kmeans.labels_)
    # plt.scatter(centers[:, 0], centers[:, 1], c='r')
    plt.title(str(args.traj_thresh) + " Traj Data, " + str(args.num_clusters) + " Clusters, Len " + str(
        args.input_window + args.output_window))
    plt.show()
    npoints = 0#int(input("How many clusters will you click?"))

    clicks=[]
    for n in range(npoints):
        fig = plt.figure()
        plt.scatter(diffsData[:, 0], diffsData[:, 1], alpha=0.5)
        # plt.scatter(centers[:, 0], centers[:, 1], c='r')
        plt.title(str(args.traj_thresh) + " Traj Data, " + str(args.num_clusters) + " Clusters, Len " + str(
            args.input_window + args.output_window))
        plt.waitforbuttonpress()
        points_clicked = plt.ginput(4, show_clicks=True)
        plt.show()
        clicks.append(points_clicked)

    centroids = []
    for points in clicks:
        centroids.append(centroidnp(np.array(points)))
    if centroids ==[]:
        centroids = kmeans.cluster_centers_
    centroids = np.array(centroids)

    empty = np.ones(len(diffsData))
    for i in range(len(diffsData)):
        empty[i]=np.argmin(np.sum((centroids-diffsData[i])**2, axis=1))

    temp['newClusters'] = kmeans.labels_ #empty #

    fig = plt.figure()
    plt.scatter(diffsData[:, 0], diffsData[:, 1], alpha=0.5)
    # plt.scatter(centers[:, 0], centers[:, 1], c='r')
    plt.title(str(args.traj_thresh) + " Traj Data, " + str(args.num_clusters) + " Clusters, Len " + str(
        args.input_window + args.output_window))

    fig2 = plt.figure()
    plt.scatter(diffsData[:,0], diffsData[:,1],c=empty)
    plt.show()

    return temp


if __name__=='__main__':
    args=get_args()
    diffsData, socData, data = createManifold(args)
    # diffsData, socData, dataframe, data = loadData(args)
    # breakpoint()
    import pandas as pd
    kmeans = KMeans(n_clusters=args.num_clusters, random_state=0).fit(diffsData)
    temp=pd.DataFrame()
    temp['tsne_X']=diffsData[:,0]
    temp['tsne_Y'] = diffsData[:, 1]
    temp['pos'] = data['allDiffs'] #data['diffs'] #
    temp['kmeans'] = kmeans.labels_
    temp['frames'] = data['posFrames'] #data['diffsFrames']#
    temp['plotPos']=data['pos'] #data['diffsPos']#
    # temp['newClusters']=kmeans.labels_#dataframe['newClusters']
    temp = custom_clusters(args, diffsData, data['posFrames'], temp['plotPos'], temp)
    temp.to_csv('allDiffsDataTEST3_' + str(args.traj_thresh) + 'thresh_'+ str(args.input_window + args.output_window) + 'window.csv') #+ str(args.group_size) + 'group_'
    # dataframe = custom_clusters(args, dataframe.filter(['tsne_X','tsne_Y']).values, data['posFrames'], data['plotPos'], dataframe)
    # dataframe.to_csv('diffsData_' + str(args.traj_thresh) + 'thresh_' + str(args.input_window + args.output_window) + 'window.csv')

    breakpoint()
    print('Plotting traj data')
    if args.traj_thresh is not None:
        # plotTSNE(diffsData, data['posFrames'], data['plotPos'])
        # plotTSNE(diffsData, data['posFrames'], data['pos'])#, dataframe['newClusters'])
        plotTSNE(diffsData, data['posFrames'], data['diffsPos'])

    # breakpoint()
    print('Plotting social dist data')
    if args.social_thresh is not None:
        plotTSNE(socData, data['groupFrames'], data['deltas'])