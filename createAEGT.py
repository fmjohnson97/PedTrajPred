from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import argparse
from torch.utils.data import DataLoader
import numpy as np
from matplotlib import pyplot as plt
from collections import defaultdict
import pandas as pd
from simplestAE import SimplestAutoEncoder
import torch
from sameSizeData import SameSizeData

# finds the nearest value in the array of points to the location of the mouse click
def find_nearest(data, coord):
    point =np.argmin(np.sum((data - coord)**2, axis=-1))
    return point

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_clusters', default=10, type=int, help='number of clusters for kmeans')
    parser.add_argument('--input_window', default=8, type=int, help='number of frames for the input data')
    parser.add_argument('--output_window', default=0, type=int, help='number of frames for the output data')
    parser.add_argument('--traj_thresh', default=2, type=int, help='number of trajectories in each data point')
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
        # fdataset=SameSizeData(frames[point][0], image='mask', output_window=args.output_window, group_size=args.group_size, distThresh=args.dist_thresh,trajThresh=args.traj_thresh, socialThresh=args.social_thresh)
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

def createManifold(args, net):
    data = defaultdict(list)
    print("getting and flattening the data")
    # get the data from each dataset
    for name in ['ETH', 'ETH_Hotel', 'UCY_Zara1', 'UCY_Zara2']:
        dataset = SameSizeData(name, input_window=args.input_window, output_window=args.output_window,
                               trajThresh=args.traj_thresh)
        loader = DataLoader(dataset, batch_size=1, shuffle=False)
        for d in loader:
            if d['pos'].nelement() > 0:
                # breakpoint()
                scene = d['pos'][0]
                with torch.no_grad():
                    output, latent = net(scene.reshape(-1, (args.input_window) * 2).float().to(device))
                data['latent'].append(latent.flatten().numpy())
                data['pos'].append(scene)
                data['frames'].append(d['frames'])

    print('creating the tsne embedding')
    # breakpoint()
    tsne = TSNE()
    if args.traj_thresh is not None:
        print(len(data['latent']))
        diffsData = tsne.fit_transform(data['latent'])

    return diffsData, data


def loadData(args, net):
    trajData, socData = None, None
    data=defaultdict(list)
    for name in ['ETH', 'ETH_Hotel', 'UCY_Zara1', 'UCY_Zara2']:
        dataset = SameSizeData(name, input_window=args.input_window, output_window=args.output_window,
                               trajThresh=args.traj_thresh)
        loader = DataLoader(dataset, batch_size=1, shuffle=False)
        for d in loader:
            if d['pos'].nelement() > 0:
                # breakpoint()
                scene = d['pos'][0]
                with torch.no_grad():
                    output, latent = net(scene.reshape(-1, (args.input_window) * 2).float().to(device))
                data['latent'].append(latent.flatten().numpy())
                data['pos'].append(scene)
                data['frames'].append(d['frames'])

    if args.traj_thresh is not None:
        print('Loading','diffsData_' + str(args.traj_thresh) + 'thresh_' + str(args.input_window + args.output_window) + 'window.npy')
        trajData = []#np.loadtxt('diffsData_' + str(args.traj_thresh) + 'thresh_' + str(
            # args.input_window + args.output_window) + 'window.npy')
        dataPD = pd.read_csv('diffsData_' + str(args.traj_thresh) + 'thresh_' + str(
            args.input_window + args.output_window) + 'window.csv')

    return trajData, socData, dataPD, data

def centroidnp(arr):
    length = arr.shape[0]
    sum_x = np.sum(arr[:, 0])
    sum_y = np.sum(arr[:, 1])
    return sum_x/length, sum_y/length

def custom_clusters(args, diffsData, frames, positions, temp):
    # kmeans = KMeans(n_clusters=args.num_clusters, random_state=0).fit(diffsData)
    # centers = kmeans.cluster_centers_

    fig = plt.figure()
    plt.scatter(diffsData[:, 0], diffsData[:, 1], alpha=0.5)
    # plt.scatter(centers[:, 0], centers[:, 1], c='r')
    plt.title(str(args.traj_thresh) + " Traj Data, " + str(args.num_clusters) + " Clusters, Len " + str(
        args.input_window + args.output_window))
    plt.show()
    npoints = 10#int(input("How many clusters will you click?"))

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

    temp['newClusters'] = empty

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
    args = get_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = SimplestAutoEncoder(args)
    net.load_state_dict(torch.load('simpleAE.pt'))
    net.eval()
    net = net.to(device)

    diffsData, data = createManifold(args, net)
    # diffsData, socData, dataframe, data = loadData(args)

    import pandas as pd
    kmeans = KMeans(n_clusters=args.num_clusters, random_state=0).fit(diffsData)
    temp=pd.DataFrame()
    temp['tsne_X']=diffsData[:,0]
    temp['tsne_Y'] = diffsData[:, 1]
    temp['pos'] = data['pos']
    temp['latent']=data['latent']
    temp['kmeans'] = kmeans.labels_
    temp['frames'] = data['frames']
    temp['newClusters']=kmeans.labels_#dataframe['newClusters']
    # temp = custom_clusters(args, diffsData, data['posFrames'], data['plotPos'], temp)
    temp.to_csv('aeData_' + str(args.traj_thresh) + 'thresh_'+ str(args.input_window + args.output_window) + 'window.csv') #+ str(args.group_size) + 'group_'
    # dataframe = custom_clusters(args, dataframe.filter(['tsne_X','tsne_Y']).values, data['posFrames'], data['plotPos'], dataframe)
    # dataframe.to_csv('diffsData_' + str(args.traj_thresh) + 'thresh_' + str(args.input_window + args.output_window) + 'window.csv')

    breakpoint()
    print('Plotting traj data')
    if args.traj_thresh is not None:
        # plotTSNE(diffsData, data['posFrames'], data['plotPos'])
        plotTSNE(diffsData, data['frames'], data['pos'])#, dataframe['newClusters'])

