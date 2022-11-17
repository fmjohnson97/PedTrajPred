import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import cv2
from plainTrajData import PlainTrajData
import imageio

def find_nearest(data, coord):
    point =np.argmin(np.sum((data - coord)**2, axis=-1))
    return point

def plotTSNE(data, original, color, frames=None):
    print('plotting the data')
    imgCounter=0
    # if frames:
    #     inds=[i for i in range(len(data)) if frames[i].split('[')[1].split(']')[0].split(',')[0].split("'")[1]=='ETH']
    #     data=data[inds]
    #     original=original[inds]
    #     color = color[inds]
    #     frames =[frames[i] for i in range(len(data)) if frames[i].split('[')[1].split(']')[0].split(',')[0].split("'")[1]=='ETH']


    for a in range(30):
        npoints=4#int(input("How many points will you click?"))
        fig = plt.figure()
        plt.scatter(data[:, 0], data[:, 1], c=color, alpha=0.5)
        plt.title('TSNE Embedding for N=3 Trajectories')
        plt.waitforbuttonpress()
        points_clicked = plt.ginput(npoints, show_clicks=True)
        plt.show()

        points=[]
        for coords in points_clicked:
            points.append(find_nearest(data, coords))

        # for i, point in enumerate(points):
        #     plt.figure()
        #     for pos in original[point].reshape(-1, 16, 2):
        #         plt.plot(pos[:,0], pos[:,1])
        #         plt.scatter(pos[0][0], pos[0][1])
        #         plt.title('Point '+str(i))
        # plt.figure()
        # plt.scatter(data[:, 0], data[:, 1], c=color, alpha=0.5)
        # for point in points:
        #     plt.scatter(data[point][0], data[point][1], c='k', alpha=1)
        # plt.title('TSNE Embedding for N=3 Trajectories')
        # plt.show()
        colors=['b','r','g']
        # images = []
        for i, point in enumerate(points):
            # breakpoint()
            # temp=frames[point].split('[')[1].split(']')[0].split(',')
            # dataset = PlainTrajData(temp[0].split("'")[1])
            # for f in temp[1:]:
            #     dataset.video.set(1, int(f.strip()))  # 1 is for CV_CAP_PROP_POS_FRAMES
            #     ret, im = dataset.video.read()
            #     # im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            #     images.append(im)
            #     cv2.imwrite('frame_'+str(imgCounter)+'.png',im)
            #     imgCounter+=1


            fig = plt.figure()
            ax = fig.add_subplot(111)
            # breakpoint()
            pos = original[point].reshape(-1, 8, 2)
            for p in range(8):
                ax.clear()
                ax.axis([0, 1, 0, 1])
                ax.set_title('Point ' + str(i))
                for j, person in enumerate(pos[:,:p,:]):
                    ax.scatter(person[:,0], person[:,1], c=colors[j])

                    # ax.show()
                plt.pause(0.5)

            # for im in images:
            #     cv2.imshow('im'+str(i),im)
            #     cv2.waitKey(500)
            #     cv2.destroyWindow('im'+str(i))


def plotClusters(df):
    colors = ['b', 'y', 'g']
    for i in range(5):
        plot = np.array(data[0])[np.where(df['newClusters'].values == i)[0]]
        plt.figure()
        for p in plot:
            # breakpoint()
            p = p.reshape(-1, 8, 2)
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
        data = pd.read_csv('allDiffsData_RotAug_'+str(i)+'thresh_'+str(input_window)+'window.csv')
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
# makeTSNELabel(3, 8)
df=pd.read_csv('/Users/faith_johnson/GitRepos/PedTrajPred/allDiffsData_RotAug_3thresh_8window.csv')#, index_col=0)
# df = pd.read_csv('noETH/noETH_sameAug_allDiffsData_RotAug_TSNEGT.csv')#'noETH_allDiffsData_RotAug_TSNEGT.csv'
# df=df[df['N']==1]
# df=df[df['cluster']==2]
# print(len(df))

# kmeans = KMeans(n_clusters=24, random_state=0).fit(df.filter(['tsne_X','tsne_Y']).values)
# plt.figure()
# plt.scatter(df['tsne_X'],df['tsne_Y'],c=df['newClusters'], alpha=.1)
# plt.figure()
# plt.scatter(df['tsne_X'],df['tsne_Y'],c=kmeans.labels_)
# plt.title('TSNE Embedding for N=3 Trajectories')
# plt.show()

# for i in range(34):
#     plt.scatter(df['tsne_X'], df['tsne_Y'], c='r')
#     temp = df[(df['newClusters']==i)]
#     plt.scatter(temp['tsne_X'], temp['tsne_Y'], c='b')
#     plt.show()

# all_heading=[[],[],[]]
# for i in range(10):
#     temp_data = df[(df['newClusters']==i)]
#     temp_heading=[[],[],[]]
#     for t in temp_data.values:
#         _, tsneX, tsneY, pos, kmeans, frames, plotPos, newClusters = t.tolist()  #
#         originalPos = plotPos[1:-1].strip().split('\n')
#         originalPos = [x.strip().split(' ') for x in originalPos]
#         originalPos = np.hstack(originalPos)
#         originalPos = [float(x.strip()) for x in originalPos if len(x.strip()) > 0]
#         originalPos = np.array(originalPos).reshape(-1,8,2)
#         for j,p in enumerate(originalPos):
#             # breakpoint()
#             # hold = np.sum(p ** 2, axis=1)
#             # heading = np.arccos(np.sum(p[:-1, :] * p[1:, :], axis=1) / (hold[:-1] * hold[1:]) ** .5)
#             # heading = np.sum(p[0]*p[-1])/(np.sum(hold[0])*np.sum(hold[-1]))**.5
#             hold = np.diff(p, axis=0)
#             heading = np.arctan2(hold[:,1], hold[:,0])*180/np.pi % 360
#             temp_heading[j].append(heading.tolist())
#     for j, th in enumerate(temp_heading):
#         all_heading[j].append(np.mean(th))
#     plt.figure()
#     plt.hist(temp_heading[0], bins=list(range(360)))
#     # plt.hist(temp_heading[1])
#     # plt.hist(temp_heading[2])
# print(np.array(all_heading).T)
# plt.show()
# breakpoint()

# new=np.ones(len(df))*-1
# for i in range(5):
#     plt.scatter(df['tsne_X'], df['tsne_Y'], c='r')
#     temp = df[kmeans.labels_==i]
#     plt.scatter(temp['tsne_X'], temp['tsne_Y'], c='b')
#     plt.show()
#     breakpoint()
#     # new[kmeans.labels_==i]=
#
data = [[],[],[]]
# breakpoint()
for d in df.iloc:
    _, tsneX, tsneY, pos, kmeans, frames, plotPos, newClusters = d.tolist()#
    # _, tsneX, tsneY, N, newClusters, plotPos, dataset = d.tolist()
    # frames=None
    originalPos = plotPos[1:-1].strip().split('\n')
    originalPos = [x.strip().split(' ') for x in originalPos]
    originalPos = np.hstack(originalPos)
    originalPos = [float(x.strip()) for x in originalPos if len(x.strip()) > 0]
    data[0].append(np.array(originalPos))
    data[1].append(int(newClusters))
    data[2].append(frames)

plotTSNE(df.filter(['tsne_X','tsne_Y']).values, np.array(data[0]), np.array(data[1]), data[2])
# plotClusters(df)

