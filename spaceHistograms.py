import torch
from simpleTSNEPredict import SimpleRegNetwork
import argparse
from plainTrajData import PlainTrajData
from torch.utils.data import DataLoader
import random
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
from collections import defaultdict
import pandas as pd
from PIL import Image, ImageOps
import seaborn as sea

MASTER_COLOR_LIST=[[[255,0,0], [255,143,0], [110,7,7], [125,70,0]],
                   [[255,151,0], [243,255,0], [92,55,0], [98,103,0]],
                   [[255,255,0], [71,255,0], [88,88,0], [17,60,0]],
                   [[0,255,91], [0,247,255], [0,55,20], [0,42,43]],
                   [[0,84,255], [130,0,255], [0,28,85], [32,0,62]]]

HELPFUL_COLORS = [[60,200,0],[255,143,0],
                  [243,255,0],[255,0,0],
                   [130,0,255],[0,247,255],

                  [0,84,255],  [0,28,85], [32,0,62],
                   [125,70,0], [255,151,0], [92,55,0],
                  [98,103,0],[110,7,7],]

def get_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--num_clusters', default=50, type=int, help='number of clusters for kmeans')
    parser.add_argument('--input_window', default=8, type=int, help='number of frames for the input data')
    parser.add_argument('--output_window', default=0, type=int, help='number of frames for the output data')
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--xgrid_num', default=128, type=int) #  right/left axis
    parser.add_argument('--ygrid_num', default=96, type=int) # up/down axis
    parser.add_argument('--maxN', default=3, type=int)
    parser.add_argument('--social_thresh', default=0.9, type=float)#0.9 for trajData
    args = parser.parse_args()
    return args

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


def get_spaced_colors(n):
    max_value = 16581375  # 255**3
    interval = int(max_value / n)
    colors = [hex(I)[2:].zfill(6) for I in range(0, max_value, interval)]
    return [(int(i[:2], 16), int(i[2:4], 16), int(i[4:], 16)) for i in colors]

def get_spaced_inds(n, max_value, row_len):
    interval = int(max_value / n)
    inds = [I for I in range(0, max_value, interval)]
    return [[i//row_len,i%row_len] for i in inds]

def makeColorGrid(min_coords, max_coords, args):
    temp = np.flip(max_coords - min_coords)
    global TSNE_CLUSTER_COLORS
    TSNE_CLUSTER_COLORS = {}
    colors=[]
    from scipy.ndimage.interpolation import zoom
    im=np.array([[[255,0,0],[255,127,0],[255,255,0],[0,255,0],[0,0,255],[127,0,255],[255,0,255]], [[80,0,0],[80,40,0],[80,80,0],[0,80,0],[0,0,80],[40,0,80],[80,0,80]]]).astype(np.uint8)
    zoomed = zoom(im, ((temp[0] // 2 + 1), int((temp[1]) * (args.maxN + 1) // 7), 1), order=1)
    inds=[range(x,x+int(temp[1]+1)) for x in range(0,zoomed.shape[1],int(temp[1]+1+temp[1]//args.maxN))]
    colors=[zoomed[:,i,:] for i in inds]
    # for n in range(2,args.maxN+1):
    #     im = np.array([[MASTER_COLOR_LIST[n-2][0], MASTER_COLOR_LIST[n-2][1]], [MASTER_COLOR_LIST[n-2][2], MASTER_COLOR_LIST[n-2][3]]]).astype(np.uint8)
    #     zoomed = zoom(im, ((temp[0]+1)//2, (temp[1]+1)//2, 1), order=1)
    #     colors.append(zoomed)
    #     plt.figure()
    #     plt.imshow(im, interpolation='nearest')
    #     plt.figure()
    #     plt.imshow(zoomed, interpolation='nearest')
    # plt.show()

    #Solution 2
    temp = get_spaced_colors(sum([len(x) for x in TSNE_BOUNDS.values()]) + 1)
    temp.pop(0)
    random.shuffle(temp)

    for i in range(args.maxN):
        # Solution 1
        # temp=[]
        # for center in TSNE_BOUNDS[i+1]:
        #     # breakpoint()
        #     temp.append(colors[i][int(center[0])][int(center[1])])
        # TSNE_CLUSTER_COLORS[i+1]=temp

        #Solution 2
        TSNE_CLUSTER_COLORS[i + 1] = [np.array(t) for t in temp[:len(TSNE_N_CUTOFFS[i + 1])]]
        for j in range(len(TSNE_N_CUTOFFS[i + 1])):
            temp.pop(0)

        # Solution 3
        # randx=random.sample(list(range(colors[i].shape[0])),k=len(TSNE_N_CUTOFFS[i + 1]))
        # randy = random.sample(list(range(colors[i].shape[1])), k=len(TSNE_N_CUTOFFS[i + 1]))
        # TSNE_CLUSTER_COLORS[i + 1] = colors[i][randx,randy]#np.vsplit(colors[i][randx,randy],len(randx))

        #Solution 4
        # temp = []
        # inds = get_spaced_inds(len(TSNE_N_CUTOFFS[i + 1]),colors[i].shape[0]*colors[i].shape[1],colors[i].shape[1])
        # for j in inds:
        #     temp.append(colors[i][j[0],j[1]])
        # TSNE_CLUSTER_COLORS[i+1]=temp
        # breakpoint()
    return colors


def getGridFill(vals, pos, xgrid, ygrid, color, args):
    # y dim change corresponds to alpha value; higher position = less opaque; lower position = more opaque
    # breakpoint()

    xinds = []
    yinds = []
    for p in pos:
        for point in p:
            xdifs=[point[1]-x for x in xgrid]
            ydifs=[point[0]-y for y in ygrid]
            try:
                xinds.append(np.where(np.diff(np.sign(xdifs))!=0)[0][-1])
                yinds.append(np.where(np.diff(np.sign(ydifs))!=0)[0][-1])
            except Exception as e:
                print(e)
                breakpoint()
    inds=np.unique(np.vstack([xinds,yinds]).T, axis=0)
    for i in inds:
        vals[i[0], i[1]] = [1, 1, 1] * color
        # if sum(vals[i[0],i[1]])==0:
        #     vals[i[0],i[1]]=[1,1,1]*color
        # else:
        #     vals[i[0], i[1]]+=(vals[i[0],i[1]]+[1,1,1]*color)#//2
    return vals

def getNewGroups(pos, diffs, args):
    # hard coding grid to be 3:4 (rows:columns) since that's aspect ratio of the images

    groupDict=defaultdict(int)
    # breakpoint()
    for i,p in enumerate(pos): # blue, orange, green, red, purple, brown
        dists = torch.sum(torch.sum((pos-p)**2,axis=-1)**.5, axis=-1)
        # print(dists)
        # breakpoint()
        inds=np.where(dists<args.social_thresh)
        for ind in inds:
            if len(ind)<=args.maxN:
                groupDict[tuple(ind)]+=1
        # minDists = dists #np.sum(np.sum(dists ** 2, axis=-1), axis=-1)
        # if minDists.shape[0] == args.maxN:
        #     # breakpoint()
        #     closest = [j for j in range(args.maxN)]
        # else:
        #     idx = np.argpartition(minDists, args.maxN)
        #     closest = [ind.item() for ind in idx[:args.maxN]]
        #     closest.sort()
        # groupDict[tuple(closest)]+=1

    groups=list(groupDict.keys())
    if len(groups)<1:
        for i, p in enumerate(pos):
            minDists = dists  # np.sum(np.sum(dists ** 2, axis=-1), axis=-1)
            if minDists.shape[0] == args.maxN:
                # breakpoint()
                closest = [j for j in range(args.maxN)]
            else:
                idx = np.argpartition(minDists, args.maxN)
                closest = [ind.item() for ind in idx[:args.maxN]]
                closest.sort()
            groupDict[tuple(closest)]+=1

    groups = list(groupDict.keys())

    # breakpoint()
    remove=[]
    for i,g in enumerate(groups):
        if sum([all([x in temp for x in g]) for temp in groups])>1:
            remove.append(i)

    # breakpoint()
    remove.reverse()
    for r in remove:
        groups.pop(r)
    if len(groups)<1:
        breakpoint()

    new_pos=[]
    new_diffs=[]
    new_allDiffs=[]
    for g in groups:
        new_pos.append(pos[np.array(list(g))])
        new_diffs.append(diffs[np.array(list(g))])
        allDiffs = []
        for i in range(new_pos[-1].shape[0]):
            temp = np.concatenate((new_pos[-1][:i], new_pos[-1][i + 1:]), axis=0)
            # hold = np.sum(new_pos[-1] ** 2, axis=1)
            # heading = np.arccos(np.sum(x[:-1, :] * x[1:, :], axis=1) / (hold[:-1] * hold[1:]) ** .5)
            if len(temp) > 0:
                temp = new_pos[-1][i][1:] - temp[:, :-1, :]
                # breakpoint()
                allDiffs.append(
                    np.hstack(np.concatenate((np.diff(new_pos[-1][i], axis=0).reshape(1, -1, 2) * 15, temp), axis=0)))
            else:
                # breakpoint()
                allDiffs.append(np.diff(new_pos[-1][i], axis=0))
        new_allDiffs.append(torch.tensor(np.stack(allDiffs)).flatten())

    # breakpoint()
    return new_pos, new_allDiffs, new_diffs, groups


if __name__ == '__main__':
    args = get_args()
    makeTSNELabel(args.maxN, args.input_window)
    nets=[]
    N=np.array(range(1,args.maxN+1))
    for i in N:
        net = SimpleRegNetwork(i*i * (args.input_window-1) * 2).eval()
        net.load_state_dict(torch.load('/Users/faith_johnson/GitRepos/PedTrajPred/simpleRegNet_allDiffsData_RotAug_'+str(i)+'people_'+str(args.input_window)+'window.pt'))
        nets.append(net)

    tsne_preds=[]
    inputs=[]
    people_per_frame=[]
    # colorMap=['maroon','r','tab:orange','y','lime','g','b','indigo','tab:purple','m']

    heatmap_clusters={'ETH':[(1,2),(1,4),(1,7),(2,6),(2,15),
                             (2,23),(3,4),(3,7),(3,24),(1,6),(2,7),(2,5),(2,22),(3,16),],
                      'ETH_Hotel':[(1,2),(1,4),(1,7),(1,8),(2,6),(2,7),(2,15),(2,23),
                                   (2,27),(2,28),(3,4),(3,7),(3,13),(3,16),(3,24),(3,25)],
                      'UCY_Zara1':[(1,2),(1,4),(1,7),(1,8),(2,6),(2,7),(2,15),(2,23),
                                   (2,27),(2,28),(3,4),(3,7),(3,13),(3,16),(3,24),(3,25)],
                      'UCY_Zara2':[(1,2),(1,4),(1,7),(1,8),(2,6),(2,7),(2,15),(2,23),
                                   (2,27),(2,28),(3,4),(3,7),(3,13),(3,16),(3,20),(3,21),
                                   (3,24),(3,25),(3,28)]}

    duplicates={1:[[],[],[]],
                2:[],
                3:[[16,24]]}

    for name in ['ETH', 'ETH_Hotel', 'UCY_Zara1', 'UCY_Zara2']:
        dataset = PlainTrajData(name, input_window=args.input_window, output_window=args.output_window, maxN=args.maxN)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        print("Processing",name)
        for data in tqdm(loader):
            if data['pos'].nelement() > 0:
                # breakpoint()
                if data['diffs'].shape[-3]<args.maxN:
                    # breakpoint()
                    net=nets[data['diffs'].shape[-3]-1]
                    pos = data['pos'][0].float()#.flatten()
                    allDiffs = []
                    for i in range(pos.shape[0]):
                        # breakpoint()
                        temp = np.concatenate((pos[:i], pos[i + 1:]), axis=0)
                        # hold = np.sum(new_pos[-1] ** 2, axis=1)
                        # heading = np.arccos(np.sum(x[:-1, :] * x[1:, :], axis=1) / (hold[:-1] * hold[1:]) ** .5)
                        if len(temp) > 0:
                            temp = pos[i][1:] - temp[:, -1, :]
                            # breakpoint() #.reshape(1, -1, 2)
                            allDiffs.append(np.concatenate((np.diff(pos[i], axis=0) * 15, temp),axis=-1))
                        else:
                            # breakpoint()
                            allDiffs.append(np.diff(pos[i], axis=0))
                    # breakpoint()
                    pos = torch.tensor(np.stack(allDiffs)).flatten()
                    people_per_frame.append(data['diffs'].shape[-3])
                    with torch.no_grad():
                        output = net(pos)
                        tsne_preds.append([output.detach()])
                        inputs.append(data['pos'].flatten())
                    output=output.detach()
                    if len(people_per_frame)!=len(tsne_preds):
                        breakpoint()
                else:
                    # for p in data['diffs'][0]:
                    #     plt.plot(p[:, 0], p[:, 1])
                    #     plt.scatter(p[0][0], p[0][1])
                    # plt.show()
                    # breakpoint()
                    pos,allDiffs, diffs, groups = getNewGroups(data['pos'][0], data['diffs'][0], args)
                    people=[]
                    preds=[]
                    ins=[]
                    for i, p in enumerate(allDiffs):
                        net = nets[pos[i].shape[-3]-1]
                        people.append(pos[i].shape[-3])
                        with torch.no_grad():
                            # breakpoint()
                            output = net(p.flatten().float())
                            ins.append(data['pos'][0][np.array(list(groups[i]))])
                            preds.append(output.detach())
                    output = np.max(np.stack(preds), axis=0)
                    tsne_preds.append(preds)
                    inputs.append(ins)
                    people_per_frame.append(people)
                    if len(people_per_frame)!=len(tsne_preds):
                        breakpoint()

        # behavior_counts=[[0]*len(TSNE_BOUNDS[1]), [0] * len(TSNE_BOUNDS[2]), [0] * len(TSNE_BOUNDS[3])]
        behavior_counts = [[],[],[]]
        color=[[[60,200,0],[255,143,0],[243,255,0],[255,0,0],[130,0,255],[0,247,255],[245, 135, 203],[171, 171, 171]],
               [[60, 200, 0], [255, 143, 0], [243, 255, 0], [255, 0, 0], [130, 0, 255], [0, 247, 255],[245, 135, 203],[171, 171, 171]],
               [[60, 200, 0], [255, 143, 0], [243, 255, 0], [255, 0, 0], [130, 0, 255], [0, 247, 255],[245, 135, 203],[171, 171, 171]]]
        color_choice=[defaultdict(list),defaultdict(list),defaultdict(list)]
        max_plot_coord = [1, 1]  # dataset.max
        min_plot_coord = [0, 0]  # dataset.min
        grids_plotX = np.linspace(min_plot_coord[0], max_plot_coord[0], args.xgrid_num)
        grids_plotY = np.linspace(min_plot_coord[1], max_plot_coord[1], args.ygrid_num)
        vals = [np.zeros((args.xgrid_num, args.ygrid_num, 3)),np.zeros((args.xgrid_num, args.ygrid_num, 3)),np.zeros((args.xgrid_num, args.ygrid_num, 3))]
        for i, tpreds in enumerate(tsne_preds):
            if False:#i>0 and i % (len(tsne_preds)//6) ==0:
                # breakpoint()
                for num in range(args.maxN):
                    # histograms
                    plot = sea.displot(behavior_counts[num], discrete=True, shrink=.9)
                    plot.set_axis_labels('Social Behavior Cluster', 'Count', labelpad=10)
                    plot.set(title='Total Histogram of Behaviors, N=' + str(num + 1))
                    tick_locs = np.array(range(0, len(TSNE_BOUNDS[num + 1]), 2))
                    plt.xticks(tick_locs, range(0, len(TSNE_BOUNDS[num + 1]), 2))
                    plot.fig.savefig('allHist_' + name + '_frame'+str(i)+'_N' + str(num + 1) + '.png', bbox_inches="tight")
                    plot.fig.clf()

                    background = Image.open(name + '.png')
                    plt.figure()
                    # if name=='ETH':
                    # plt.imshow(background, origin='lower')
                    # else:
                    #     breakpoint()
                    #     plt.imshow(background)
                    plotmap = vals[num]
                    plotmap = plotmap.repeat(5, axis=0).repeat(5, axis=1)
                    temp = Image.fromarray(plotmap.astype(np.uint8()))
                    if name == 'ETH':
                        temp = temp.resize((640, 480)).rotate(-90)  # ,expand=True)
                    else:
                        temp = temp.resize((640, 480))
                    temp = np.array(temp)
                    # breakpoint()
                    if name != 'ETH':
                        new_im = np.array(ImageOps.flip(background))[:,:,:3]
                    else:
                        new_im = np.array(background)[:,:,:3]
                    plt.title(name + ' Environment Occupancy Map for N=' + str(num + 1))
                    for x in range(temp.shape[0]):
                        for y in range(temp.shape[1]):
                            if sum(temp[x, y]) != 0:  # .getpixel((i, j))) != 0:
                                # breakpoint()
                                # if temp[x, y][0] > 255 or temp[x, y][1] > 255 or temp[x, y][2] > 255:
                                #     new_im[x, y] = temp[x, y] // len(
                                #         [x for x in heatmap_clusters['ETH'] if x[0] == num + 1])
                                # else:
                                new_im[x, y] = temp[x, y]  # temp.getpixel((i, j))
                    plt.imshow(new_im, origin='lower')
                    plt.savefig(name + '_blendedHeatmap_frame'+str(i)+'_N' + str(num + 1) + '.png')

                behavior_counts = [[], [], []]
                vals = [np.zeros((args.xgrid_num, args.ygrid_num, 3)), np.zeros((args.xgrid_num, args.ygrid_num, 3)),
                        np.zeros((args.xgrid_num, args.ygrid_num, 3))]

            for j, t in enumerate(tpreds):
                pos = inputs[i]
                n = people_per_frame[i]
                if type(n) is list:
                    n = n[j]
                    pos = pos[j]
                cluster = np.argmin(np.sum((TSNE_BOUNDS[n]-t.numpy())**2,axis=-1))
                # behavior_counts[n-1][cluster]+=1
                behavior_counts[n-1].append(cluster)
                if (n,cluster) in heatmap_clusters[name]:
                    if color_choice[n-1][cluster]==[]:
                        if n == 3 and cluster in [16, 24]:
                            ind = set([16, 24]).difference(set([cluster]))
                            if color_choice[n - 1][list(ind)[0]]==[]:
                                color_choice[n - 1][cluster] = color[n - 1][0]
                                color[n - 1].pop(0)
                            else:
                                color_choice[n - 1][cluster] = color_choice[n - 1][list(ind)[0]]
                        else:
                            color_choice[n-1][cluster]=color[n-1][0]
                            color[n-1].pop(0)
                    choice=np.array(color_choice[n-1][cluster])
                    vals[n-1]=getGridFill(vals[n-1], pos.reshape(n, args.input_window, 2), grids_plotX, grids_plotY, choice, args)

            plt.close('all')

        for n in range(args.maxN):
            #histograms
            plot=sea.displot(behavior_counts[n],  discrete=True, shrink=.9, stat='probability')
            plot.set_axis_labels('Social Behavior Cluster', 'Count', labelpad=10)
            plot.set(title='Total Histogram of Behaviors, N='+str(n+1))
            tick_locs=np.array(range(0,len(TSNE_BOUNDS[n+1]),2))
            plt.xticks(tick_locs,range(0,len(TSNE_BOUNDS[n+1]),2))
            # plt.yscale('log')
            plt.yticks([0, 0.05, 0.1, 0.15, 0.2, 0.25,  0.3, 0.35, 0.4, 0.45])

            plot.fig.savefig('allHist_'+name+'_N'+str(n+1)+'.svg', bbox_inches="tight")
            plot.fig.clf()
            '''
            # breakpoint() #a.repeat(2, axis=0).repeat(2, axis=1)
            #heatmaps
            background = Image.open(name + '.png')
            plt.figure()
            # if name=='ETH':
            # plt.imshow(background, origin='lower')
            # else:
            #     breakpoint()
            #     plt.imshow(background)
            plotmap=vals[n]
            plotmap=plotmap.repeat(10,axis=0).repeat(10,axis=1)
            temp = Image.fromarray(plotmap.astype(np.uint8()))
            if name=='ETH':
                temp = temp.resize((640, 480)).rotate(-90)  # ,expand=True)
            else:
                temp = temp.resize((640, 480))
            temp = np.array(temp)
            # breakpoint()
            if name!='ETH':
                new_im = np.array(ImageOps.flip(background))
            else:
                new_im = np.array(background)
            plt.title(name+' Environment Occupancy Map for N=' + str(n+1))
            for x in range(temp.shape[0]):
                for y in range(temp.shape[1]):
                    if sum(temp[x, y]) != 0:  # .getpixel((i, j))) != 0:
                        # breakpoint()
                        if temp[x,y][0]>255 or temp[x,y][1]>255 or temp[x,y][2]>255:
                            new_im[x, y] = temp[x, y]//len([x for x in heatmap_clusters['ETH'] if x[0] == n + 1])
                        else:
                            new_im[x, y] = temp[x, y]  # temp.getpixel((i, j))
            plt.imshow(new_im, origin='lower')
            plt.savefig(name+'_blendedHeatmap_N'+str(n+1)+'.png')
            # plt.figure()
            # plt.hist(behavior_counts[n], bins=len(TSNE_BOUNDS[n+1])+1)#list(range(len(behavior_counts[n])+1)))
            # # tick_locs=np.array(range(0,len(TSNE_BOUNDS[n+1]),2))
            # # plt.xticks(tick_locs,range(0,len(TSNE_BOUNDS[n+1]),2))
            # plt.vlines(range(len(TSNE_BOUNDS[n+1])),0,500)
            # plt.title('Total Histogram of Behaviors, N='+str(n+1))
            # plt.savefig('allHist_'+name+'_N'+str(n+1)+'.png')
            breakpoint()
            '''
    print(color_choice)

'''
N=1:
ETH - walking in/out and standing
Hotel - left/right, standing, walking down
Zara1 - left/right, standing, walking down
Zara2 - left/right, standing, walking down

N=2:
ETH - LF(mostly) in/out, standing/congregating, opposite direction passing
Hotel - right, down+left, left, standing/congregating
Zara1 - right, down+left, left, standing/congregating
Zara2 - right, down+left, left, standing/congregating

N=3:
ETH - lots of 4,7,16,24 which is groups of 3 walking in/out of the building
Hotel - walking left/right in group, standing still
Zara1 - walking left/right in group, standing still
Zara2 - walking left/right in group, standing still, walking passing stationarya
'''