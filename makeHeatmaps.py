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
from PIL import Image
import cv2
from OpenTraj.utils import world2image
import math

MASTER_COLOR_LIST=[[[255,0,0], [255,143,0], [110,7,7], [125,70,0]],
                   [[255,151,0], [243,255,0], [92,55,0], [98,103,0]],
                   [[255,255,0], [71,255,0], [88,88,0], [17,60,0]],
                   [[0,255,91], [0,247,255], [0,55,20], [0,42,43]],
                   [[0,84,255], [130,0,255], [0,28,85], [32,0,62]]]

def get_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--num_clusters', default=50, type=int, help='number of clusters for kmeans')
    parser.add_argument('--input_window', default=8, type=int, help='number of frames for the input data')
    parser.add_argument('--output_window', default=0, type=int, help='number of frames for the output data')
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--xgrid_num', default=24, type=int)
    parser.add_argument('--ygrid_num', default=32, type=int)
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
        TSNE_CLUSTER_COLORS[i + 1] = [tuple(t) for t in temp[:len(TSNE_N_CUTOFFS[i + 1])]]
        for j in range(len(TSNE_N_CUTOFFS[i + 1])):
            temp.pop(0)

        #Solution 3
        # randx=random.sample(list(range(colors[i].shape[0])),k=len(TSNE_N_CUTOFFS[i + 1]))
        # randy = random.sample(list(range(colors[i].shape[1])), k=len(TSNE_N_CUTOFFS[i + 1]))
        # TSNE_CLUSTER_COLORS[i + 1] = np.vsplit(colors[i][randx,randy],len(randx))

        #Solution 4
        # temp = []
        # inds = get_spaced_inds(len(TSNE_N_CUTOFFS[i + 1]),colors[i].shape[0]*colors[i].shape[1],colors[i].shape[1])
        # for j in inds:
        #     temp.append(colors[i][j[0],j[1]])
        # TSNE_CLUSTER_COLORS[i+1]=temp
        # breakpoint()
    return colors


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

def writeCoords(image, coords, color, h_matrix, dmax, dmin):
    coords = coords.reshape(-1,2)
    theta = np.deg2rad(90)
    ox, oy = 0.5, 0.5
    breakpoint()
    for coord in coords:
        coord = coord * (dmax - dmin) + dmin
        coord = world2image(np.array([coord.numpy()]), h_matrix)[0]
        # px, py = coord
        # qx = ox + math.cos(theta) * (px - ox) - math.sin(theta) * (py - oy)
        # qy = oy + math.sin(theta) * (px - ox) + math.cos(theta) * (py - oy)
        # coord=np.hstack([qx, qy])



        image = cv2.circle(image, tuple(coord), 7, tuple(color), -1)

    return image

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

    for name in ['ETH', 'ETH_Hotel', 'UCY_Zara1', 'UCY_Zara2']:
        dataset = PlainTrajData(name, input_window=args.input_window, output_window=args.output_window, maxN=args.maxN)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        print("Processing",name)
        tsne_preds=[]
        max_tsne=[-np.inf, -np.inf]
        min_tsne=[np.inf, np.inf]
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

                if output[0] > max_tsne[0]:
                    max_tsne[0] = output[0]
                if output[1] > max_tsne[1]:
                    max_tsne[1] = output[1]
                if output[0] < min_tsne[0]:
                    min_tsne[0] = output[0]
                if output[1] < min_tsne[1]:
                    min_tsne[1] = output[1]

        max_plot_coord = [1,1]#dataset.max
        min_plot_coord = [0,0]#dataset.min
        grids_plotX=np.linspace(min_plot_coord[0],max_plot_coord[0], args.xgrid_num)
        grids_plotY = np.linspace(min_plot_coord[1], max_plot_coord[1], args.ygrid_num)
        # grids_tsneX = np.linspace(min_tsne[0], max_tsne[0], args.num_grids)
        # grids_tsneY = np.linspace(min_tsne[1], max_tsne[1], args.num_grids)
        print('Computing grids')
        colors=makeColorGrid(np.array(min_tsne), np.array(max_tsne), args)

        print('plotting legend')
        # plot color legend
        fig1 = plt.figure()
        fig1.suptitle('Legend for Graph Colors')
        for i in range(args.maxN):
            ax = plt.subplot((args.maxN+1)//2,2,i+1)
            df = pd.read_csv('/Users/faith_johnson/GitRepos/PedTrajPred/allDiffsData_RotAug_'+str(i+1)+'thresh_'+str(args.input_window)+'window.csv')  # , index_col=0)
            tsneColor=[TSNE_CLUSTER_COLORS[i+1][np.argmin(np.sum((TSNE_BOUNDS[i+1]-point)**2,axis=-1))] for point in df.filter(['tsne_X','tsne_Y']).values]
            ax.scatter(df['tsne_X'].values,df['tsne_Y'].values,c=np.array(tsneColor)/255)
            ax.set_title(str(i+1)+' People')
            ax.set_xticks([])
            ax.set_yticks([])

        print('plotting heatmap')
        maps=[defaultdict(list),defaultdict(list),defaultdict(list)]
        for i, tpred in tqdm(enumerate(tsne_preds)):
            for j, t in enumerate(tpred):
                n=people_per_frame[i]
                pos = inputs[i]
                if type(n) is list:
                    n=n[j]
                    pos = pos[j]
                # breakpoint()
                maps[n-1][np.argmin(np.sum((TSNE_BOUNDS[n]-t.numpy())**2,axis=-1))].append(pos)

        overlays = {1: [[1, 9]],
                    2: [[0, 6, 8]],  # [6,8,22]
                    3: [[5, 27, 8], [23, 15, 20, 26], [12, 24, 16]]}
        duplicates = [[[4, 5]], [[2, 3], [4, 10], [5, 9], [6, 8], [13, 14, 18], [24, 25]],
                      [[20, 26], [24, 16], [5, 27, 31], [23, 15]]]

        for i in range(1, args.maxN+1):
            trajectories=maps[i-1]
            for cluster_key in trajectories.keys():
                traj_values = trajectories[cluster_key]
                background = Image.open(name + '.png').convert('RGB')#.transpose(Image.FLIP_LEFT_RIGHT)
                open_cv_image = np.array(background)
                # Convert RGB to BGR
                background = open_cv_image[:, :, ::-1].copy()
                color = TSNE_CLUSTER_COLORS[i][cluster_key]
                for traj in traj_values:
                    traj=traj.reshape(-1,8,2)
                    image=writeCoords(background, traj, color, dataset.H, dataset.max, dataset.min)

                cv2.imshow(str(i)+" people, cluster "+str(cluster_key),image)
                cv2.waitKey(200)
                breakpoint()

        breakpoint()

        plotm = [m[0] for m in maps]
        avg = np.mean(np.sum(plotm, axis=0) // len(plotm), axis=-1)
        avg = avg / avg.max() * 255
        im = Image.fromarray(avg)
        im = im.resize((640, 480))
        plt.imshow(im.rotate(-90), origin='lower', alpha=.7)
        plt.title('Environment Occupancy Map for ' + name)
        plt.savefig(name+'_heatmap.png')
        plt.show()
        # breakpoint()
        for i in range(1,args.maxN+1):
            overlay_temp=[[],[],[]]
            overlay_color = [[], [], []]
            #plotting the per N map
            plt.figure()
            plt.imshow(background, origin='lower')
            m=[m for m in maps if m[1][0]==i]
            plotm=[m[0] for m in m]
            avg=np.mean(np.sum(plotm,axis=0)//len(plotm),axis=-1)
            avg=avg/avg.max()*255
            im = Image.fromarray(avg)
            im = im.resize((640, 480))
            plt.imshow(im.rotate(-90), origin='lower', alpha=.7)
            plt.title('Environment Occupancy Map for N='+str(i))
            plt.savefig('n'+str(i)+'_heatmap.png')
            #plotting the per cluster/per merged behavior map
            skip=[]
            for j in range(len(TSNE_N_CUTOFFS[i])):
                if j not in skip:
                    mergeInd=[k for k,d in enumerate(duplicates[i-1]) if j in d]
                    if len(mergeInd)>0:
                        plotm = [m[0] for m in m if m[1][1] in duplicates[i-1][mergeInd[0]]]
                        col = [m[1][2] for m in m if m[1][1] in duplicates[i - 1][mergeInd[0]]]
                        skip.extend(duplicates[i-1][mergeInd[0]])
                        incl=','.join([str(dup) for dup in duplicates[i-1][mergeInd[0]]])
                        title = 'Environment Occupancy Map for N=' + str(i)+', Cluster=[' + incl+']'
                    else:
                        plotm=[m[0] for m in m if m[1][1]==j]
                        col = [m[1][2] for m in m if m[1][1] == j]
                        title='Environment Occupancy Map for N=' + str(i)+', Cluster='+str(j)
                    check=[k for k,c in enumerate(overlays[i]) if j in c]
                    if len(check)>0:
                        overlay_temp[check[0]].append(plotm)
                        overlay_color[check[0]].append(col[0].tolist())
                    if len(plotm)>0:
                        color = col[0].tolist()
                        temp = np.sum(plotm, axis=0) // len(plotm)
                        temp = Image.fromarray(temp.astype(np.uint8()))
                        temp = temp.resize((640, 480)).rotate(-90)#,expand=True)
                        temp = np.array(temp)
                        new_im = np.array(background)
                        for x in range(temp.shape[0]):
                            for y in range(temp.shape[1]):
                                if sum(temp[x,y])!=0:#.getpixel((i, j))) != 0:
                                    new_im[x,y] = color #temp.getpixel((i, j))

                        plt.figure()
                        plt.imshow(background, origin='lower')
                        plt.imshow(new_im, origin='lower', alpha=.75)
                        plt.title(title)
                        plt.savefig('n' + str(i) + '_c' + str(j) + '_heatmap.png')
                        # breakpoint()


                        # plt.figure()
                        # plt.imshow(background, origin='lower')
                        # # breakpoint()
                        # # avg = np.sum(plotm, axis=0) #/ len(plotm)
                        # # color_map=(avg/avg.max())*TSNE_CLUSTER_COLORS[i][j]
                        # # breakpoint()
                        # color_map=np.mean(np.sum(plotm,axis=0)//len(plotm),axis=-1)
                        # color_map=color_map/color_map.max()*255
                        # im = Image.fromarray(color_map)
                        # im = im.resize((640, 480))
                        # plt.imshow(im.rotate(-90), origin='lower', alpha=.7)
                        # plt.title(title)
                        # plt.savefig('n'+str(i)+'_c'+str(j)+'_heatmap.png')

            # breakpoint()
            #plotting the overlays
            for o in range(len(overlay_temp)):
                if len(overlay_temp[o])>0:
                    col1=overlay_color[o][0]
                    col2=overlay_color[o][1]
                    map1=overlay_temp[o][0]
                    map1 = np.sum(map1, axis=0) // len(map1)
                    map1 = Image.fromarray(map1.astype(np.uint8()))
                    map1 = map1.resize((640, 480)).rotate(-90)  # ,expand=True)
                    map1 = np.array(map1)
                    map2=overlay_temp[o][1]
                    map2 = np.sum(map2, axis=0) // len(map2)
                    map2 = Image.fromarray(map2.astype(np.uint8()))
                    map2 = map2.resize((640, 480)).rotate(-90)  # ,expand=True)
                    map2 = np.array(map2)
                    new_im = np.array(background)
                    for x in range(map1.shape[0]):
                        for y in range(map1.shape[1]):
                            if sum(map1[x, y]) != 0 and sum(map2[x, y]) == 0:
                                new_im[x, y] = col1
                            elif sum(map2[x, y]) != 0 and sum(map1[x, y]) == 0:
                                new_im[x, y] = col2
                            elif sum(map1[x, y]) != 0 and sum(map2[x, y]) != 0:
                                new_im[x, y] = ((np.array(col1)+np.array(col2))//2).tolist()
                    plt.figure()
                    plt.imshow(background, origin='lower')
                    plt.imshow(new_im, origin='lower', alpha=.75)
                    incl = ','.join([str(dup) for dup in overlays[i][o]])
                    title = 'Overlapped Environment Occupancy Map for N=' + str(i) + ', Cluster=[' + incl + ']'
                    plt.title(title)
                    plt.savefig('_'.join([str(dup) for dup in overlays[i][o]])+ '_overlay_heatmap.png')


            plt.show()
            # breakpoint()



