import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

def find_nearest(data, coord):
    point =np.argmin(np.sum((data - coord)**2, axis=-1))
    return point

def plotTSNE(data, original, color):
    print('plotting the data')

    for a in range(30):
        npoints=4#int(input("How many points will you click?"))
        fig = plt.figure()
        plt.scatter(data[:, 0], data[:, 1], c=color, alpha=0.5)
        plt.waitforbuttonpress()
        points_clicked = plt.ginput(npoints, show_clicks=True)
        plt.title('TSNE Embedding for N=3 Trajectories')
        plt.show()

        points=[]
        for coords in points_clicked:
            points.append(find_nearest(data, coords))

        for i, point in enumerate(points):
            plt.figure()
            for pos in original[point].reshape(-1, 8, 2):
                plt.plot(pos[:,0], pos[:,1])
                plt.scatter(pos[0][0], pos[0][1])
                plt.title('Point '+str(i))
        plt.figure()
        plt.scatter(data[:, 0], data[:, 1], c=color, alpha=0.5)
        for point in points:
            plt.scatter(data[point][0], data[point][1], c='k', alpha=1)
        plt.title('TSNE Embedding for N=3 Trajectories')
        plt.show()

def plotClusters(df):
    colors = ['b', 'y', 'g']
    for i in range(5):
        plot = np.array(data[0])[np.where(df['newClusters'].values == i)[0]]
        plt.figure()
        for p in plot:
            # breakpoint()
            p = p.reshape(-1, 16, 2)
            # for j, pos in enumerate(p):
            plt.plot(p[0, :, 0], p[0, :, 1], c=colors[0])
            plt.scatter(p[0, 0, 0], p[0, 0, 1])#, c='r')
        plt.title('Cluster ' + str(i) + ', Person 1')
        plt.axis([0,1,0,1])

        # plt.figure()
        # for p in plot[:10]:
        #     # breakpoint()
        #     p = p.reshape(-1, 8, 2)
        #     # for j, pos in enumerate(p):
        #     plt.plot(p[1, :, 0], p[1, :, 1], c=colors[1])
        #     plt.scatter(p[1, 0, 0], p[1, 0, 1])#, c='r')
        # plt.title('Cluster ' + str(i) + ', Person 2')
        # plt.axis([0, 1, 0, 1])

        # plt.figure()
        # for p in plot[:10]:
        #     # breakpoint()
        #     p = p.reshape(-1, 8, 2)
        #     # for j, pos in enumerate(p):
        #     plt.plot(p[2, :, 0], p[2, :, 1], c=colors[2])
        #     plt.scatter(p[2, 0, 0], p[2, 0, 1])#, c='r')
        # plt.title('Cluster ' + str(i) + ', Person 3')
        # plt.axis([0, 1, 0, 1])
        plt.show()

def makeTSNELabel(maxN, input_window):
    # global GT_TSNE_VALUES
    global TSNE_N_CUTOFFS
    global TSNE_BOUNDS
    # GT_TSNE_VALUES = pd.DataFrame(columns=['tsne_X','tsne_Y','kmeans'])
    TSNE_N_CUTOFFS = {}
    TSNE_BOUNDS = {}
    max_label = 0
    for i in range(1,maxN+1):
        # breakpoint()
        data = pd.read_csv('diffsData_'+str(i)+'thresh_'+str(input_window)+'window.csv')
        temp = data.filter(['tsne_X', 'tsne_Y', 'newClusters'])
        class_bounds =[]
        for b in range(int(temp['newClusters'].max())+1):
            bounds=temp[temp['newClusters']==b]
            coords = np.array([[bounds['tsne_X'].max(),bounds['tsne_Y'].max()],[bounds['tsne_X'].min(),bounds['tsne_Y'].min()]])
            sum_x = np.sum(coords[:, 0])
            sum_y = np.sum(coords[:, 1])
            class_bounds.append([sum_x / 2, sum_y / 2])

        # TSNE_BOUNDS[i]=[[temp['tsne_X'].max(),temp['tsne_Y'].max()],[temp['tsne_X'].min(),temp['tsne_Y'].min()]]
        TSNE_BOUNDS[i]=class_bounds
        temp['newClusters']=temp['newClusters']+max_label
        # GT_TSNE_VALUES = GT_TSNE_VALUES.append(temp)
        max_label = temp['newClusters'].max()+1
        temp = temp['newClusters'].unique()
        temp.sort()
        TSNE_N_CUTOFFS[i] = temp

#[0,5,13,24]
makeTSNELabel(3, 8)
df=pd.read_csv('/Users/faith_johnson/GitRepos/PedTrajPred/diffsData_3thresh_8window.csv')#, index_col=0)
for i in range(24):
    plt.scatter(df['tsne_X'], df['tsne_Y'], c='r')
    temp = df[(df['newClusters']==i)]
    plt.scatter(temp['tsne_X'], temp['tsne_Y'], c='b')
    plt.show()

kmeans = KMeans(n_clusters=24, random_state=0).fit(df.filter(['tsne_X','tsne_Y']).values)
plt.figure()
plt.scatter(df['tsne_X'],df['tsne_Y'],c=df['newClusters'])
plt.figure()
plt.scatter(df['tsne_X'],df['tsne_Y'],c=kmeans.labels_)
plt.title('TSNE Embedding for N=3 Trajectories')
plt.show()

new=np.ones(len(df))*-1
for i in range(5):
    plt.scatter(df['tsne_X'], df['tsne_Y'], c='r')
    temp = df[kmeans.labels_==i]
    plt.scatter(temp['tsne_X'], temp['tsne_Y'], c='b')
    plt.show()
    breakpoint()
    # new[kmeans.labels_==i]=

data = [[],[]]
# breakpoint()
for d in df.iloc:
    _, tsneX, tsneY, pos, kmeans, frames, plotPos, newClusters = d.tolist()
    originalPos = plotPos[1:-1].strip().split('\n')
    originalPos = [x.strip().split(' ') for x in originalPos]
    originalPos = np.hstack(originalPos)
    originalPos = [float(x.strip()) for x in originalPos if len(x.strip()) > 0]
    data[0].append(originalPos)
    data[1].append(int(newClusters))

plotTSNE(df.filter(['tsne_X','tsne_Y']).values, np.array(data[0]), np.array(data[1]))
# plotClusters(df)

