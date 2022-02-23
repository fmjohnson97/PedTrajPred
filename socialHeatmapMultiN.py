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
    parser.add_argument('--xgrid_num', default=10, type=int)
    parser.add_argument('--ygrid_num', default=10, type=int)
    parser.add_argument('--maxN', default=5, type=int)
    parser.add_argument('--social_thresh', default=0.2, type=float)#0.9 for trajData
    args = parser.parse_args()
    return args


def makeColorGrid(min_coords, max_coords, args):
    temp = np.flip(max_coords - min_coords)
    # breakpoint()
    colors=[]
    from scipy.ndimage.interpolation import zoom
    im=np.array([[[255,0,0],[255,127,0],[255,255,0],[0,255,0],[0,0,255],[127,0,255],[255,0,255]], [[80,0,0],[80,40,0],[80,80,0],[0,80,0],[0,0,80],[40,0,80],[80,0,80]]]).astype(np.uint8)
    factor=args.maxN // 2 + 1
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
    return colors


def getGridFill(pos, xgrid, ygrid, color, args):
    # y dim change corresponds to alpha value; higher position = less opaque; lower position = more opaque
    # breakpoint()
    vals=np.zeros((args.xgrid_num, args.ygrid_num,3))
    xinds = []
    yinds = []
    for p in pos:
        for point in p:
            xdifs=[point[0]-x for x in xgrid]
            ydifs=[point[1]-y for y in ygrid]
            try:
                xinds.append(np.where(np.diff(np.sign(xdifs))!=0)[0][-1])
                yinds.append(np.where(np.diff(np.sign(ydifs))!=0)[0][-1])
            except Exception as e:
                print(e)
                breakpoint()
    inds=np.unique(np.vstack([xinds,yinds]).T, axis=0)
    for i in inds:
        vals[i[0],i[1]]=[1,1,1]*color
    return vals

def getNewGroups(pos, args):
    # hard coding grid to be 3:4 (rows:columns) since that's aspect ratio of the images

    groupDict=defaultdict(int)
    for i,p in enumerate(pos): # blue, orange, green, red, purple, brown
        dists = torch.sum(torch.sum((pos-p)**2,axis=-1)**.5, axis=-1)
        # print(dists)
        # breakpoint()
        inds=np.where(dists<args.social_thresh)
        for ind in inds:
            if len(ind)<=args.maxN:
                groupDict[tuple(ind)]+=1

    groups=list(groupDict.keys())
    if len(groups)<1:
        totals = np.array(list(range(pos.shape[0])))
        inds = [list(range(x,x+args.maxN)) for x in range(len(totals)-args.maxN)]
        for i in inds:
            groups.append(totals[i])

    remove=[]
    for i,g in enumerate(groups):
        if sum([all([x in temp for x in g]) for temp in groups])>1:
            remove.append(i)

    remove.reverse()
    for r in remove:
        groups.pop(r)
    if len(groups)<1:
        breakpoint()
    new_pos=[]
    for g in groups:
        new_pos.append(pos[np.array(list(g))])

    return new_pos, groups

if __name__ == '__main__':
    args = get_args()
    nets=[]
    N=np.array(range(1,args.maxN+1))
    for i in N:
        net = SimpleRegNetwork(i * (args.input_window-1) * 2).eval()
        net.load_state_dict(torch.load('/Users/faith_johnson/GitRepos/PedTrajPred/weights/simpleRegNet_diffsData_'+str(i)+'people_'+str(args.input_window)+'window.pt'))
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
                if data['diffs'].shape[-3]<6:
                    net=nets[data['diffs'].shape[-3]-1]
                    pos = data['diffs'].flatten().float()
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
                    pos, groups = getNewGroups(data['diffs'][0], args)
                    people=[]
                    preds=[]
                    ins=[]
                    for i, p in enumerate(pos):
                        net = nets[p.shape[-3]-1]
                        people.append(p.shape[-3])
                        with torch.no_grad():
                            output = net(p.flatten().float())
                            # breakpoint()
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
            ax.imshow(colors[i])
            ax.set_title(str(i+1)+' People')
            ax.set_xticks([])
            ax.set_yticks([])

        print('plotting heatmap')
        fig2 = plt.figure()
        ax1 = fig2.add_subplot(111)
        fig3 = plt.figure()
        ax2 = fig3.add_subplot(111)
        # ax2.axis([0, 1, 0, 1])#min_plot_coord[0], max_plot_coord[0], min_plot_coord[1], max_plot_coord[1]])
        for i, tpred in tqdm(enumerate(tsne_preds)):
            vals=[]
            ax2.clear()
            for j, t in enumerate(tpred):
                n=people_per_frame[i]
                pos = inputs[i]
                if type(n) is list:
                    n=n[j]
                    pos = pos[j]
                color=colors[n-1][int(t[0]),int(t[1])]
                vals.append(getGridFill(pos.reshape(n,args.input_window,2), grids_plotX, grids_plotY, color, args))
                # breakpoint()
                ax2.set_title(name + ' pedestrian social patterns')
                ax2.axis([0, 1, 0, 1])
                for p in pos.reshape(n,args.input_window,2):
                    # ax2.plot(p, c=[color/255]*p.shape[0])
                    ax2.scatter(p[1:,0],p[1:,1], c=[color/255])
                    ax2.scatter(p[0, 0], p[0, 1], c=[color / 255], marker='x')
                # plt.pause(0.15)
            if len(tpred)>1:
                # breakpoint()
                vals=np.sum(vals, axis=0)
            else:
                vals=vals[0]
            ax1.clear()
            ax1.set_title(name + ' pedestrian social patterns')
            ax1.imshow(vals.astype(np.uint8), interpolation='nearest')
            plt.pause(0.3)



