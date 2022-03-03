import torch.nn as nn
import torchvision.models as models
from tsneGTdataset import TSNEGT
import argparse
from torch.utils.data import DataLoader
import torch
import numpy as np
from matplotlib import pyplot as plt
from trajAugmentations import TrajAugs

# class SimpleCatNetwork(nn.Module):
#     def __init__(self, in_channels, num_clusters):
#         super(SimpleCatNetwork, self).__init__()
#         # self.backbone = models.resnet18(pretrained=True).eval()
#         # num_ftrs = self.backbone.fc.in_features
#         self.projection_head = nn.Linear(in_channels, 64)
#         self.relu = nn.LeakyReLU()
#         self.output_layer = nn.Sequential(nn.Linear(64,128), nn.LeakyReLU(),nn.Linear(128, num_clusters), nn.Sigmoid())
#
#     def forward(self, x):
#         x = self.relu(self.projection_head(x))
#         # x = self.backbone(x)
#         x = self.output_layer(x)
#         return x

class SimpleRegNetwork(nn.Module):
    def __init__(self, in_channels):
        super(SimpleRegNetwork, self).__init__()
        # self.backbone = models.resnext50_32x4d(pretrained=True)
        self.projection_head = nn.Sequential(nn.Linear(in_channels, 64),nn.LeakyReLU(),nn.Linear(64,128))
        self.relu = nn.LeakyReLU()
        self.output_layer = nn.Sequential(nn.Linear(128,64), nn.LeakyReLU(), nn.Linear(64, 16), nn.LeakyReLU(),nn.Linear(16, 2))
        # self.projection_head = nn.Sequential(nn.Linear(in_channels, 64),
        #                                      nn.LeakyReLU(),
        #                                      nn.Linear(64, 64))
        # self.relu = nn.LeakyReLU()
        # self.output_layer = nn.Sequential(nn.Linear(64, 32),
        #                                   nn.LeakyReLU(),
        #                                   nn.Linear(32, 16),
        #                                   nn.LeakyReLU(),
        #                                   nn.Linear(16, 2))

    def forward(self, x):
        x = self.relu(self.projection_head(x))
        # x = self.backbone(x)
        x = self.output_layer(x)
        return x


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_clusters', default=50, type=int, help='number of clusters for kmeans')
    parser.add_argument('--num_people', default=2, type=int, help='number of people to include in each group')
    parser.add_argument('--path', default='/Users/faith_johnson/GitRepos/PedTrajPred/', type=str,
                        help='path to the csv for the dataset data')
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--window', default=8, type=int)
    parser.add_argument('--epochs', default=225, type=int)
    parser.add_argument('--lr', default=1e-2, type=float)
    parser.add_argument('--train', action='store_true')
    args = parser.parse_args()
    return args

def train(net, loader, args):
    loss_func = nn.MSELoss()  # nn.BCELoss() #CrossEntropyLoss() #
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs * len(loader), 1e-16)
    overall_loss = []
    trajAugs=TrajAugs()
    for e in range(args.epochs):
        total_loss = []
        for data in loader:
            # breakpoint()
            pos, tsne, kmeans, originalPos = data
            # pos=trajAugs.augment(pos.reshape(-1, args.window-1, 2)).reshape(-1,(args.window-1)* 2 )
            output = net(pos.float())
            loss = loss_func(output, tsne)  # kmeans)
            opt.zero_grad()
            loss.backward()
            opt.step()
            scheduler.step()
            total_loss.append(loss.item())

        torch.save(net.state_dict(), 'simpleRegNet_diffsData_' + str(args.num_people) + 'people_' + str(args.window)+'window.pt')
        print('Epoch', e, ': Loss =', np.mean(total_loss))
        overall_loss.append(np.mean(total_loss))
    plt.figure()
    plt.plot(overall_loss)
    plt.title('Average Loss per Epoch')
    plt.show()
    return net

def test(net, loader, args):
    loss_func = nn.MSELoss()  # nn.BCELoss() #CrossEntropyLoss() #
    total_loss = []
    for data in loader:
        # breakpoint()
        pos, tsne, kmeans, originalPos = data
        output = net(pos)
        loss = loss_func(output, tsne)  # kmeans)
        total_loss.append(loss.item())

    print('Avg Test Loss:', np.mean(total_loss))
    plt.figure()
    plt.plot(total_loss)
    plt.title('Test Loss')
    plt.show()

    return net

def find_nearest(data, coord):
    point =np.argmin(np.sum((data - coord)**2, axis=-1))
    return point

def graphPreds(net, loader, args):
    preds = []
    real = []
    inputs=[]
    for data in loader:
        pos, tsne, kmeans, originalPos = data
        output = net(pos)
        real.extend(tsne)
        preds.extend(output.detach())
        inputs.extend(originalPos)

    plt.figure()
    preds = np.vstack(preds)
    real = np.vstack(real)
    plt.scatter(real[:, 0], real[:, 1])
    plt.title('Ground Truth TSNE, N='+str(args.num_people))

    plt.figure()
    plt.scatter(preds[:, 0], preds[:, 1])
    plt.title('Predicted TSNE, N='+str(args.num_people))

    plt.figure()
    plt.scatter(real[:, 0], real[:, 1], alpha=0.5)
    plt.scatter(preds[:, 0], preds[:, 1], alpha=0.5)
    plt.legend(['real', 'predicted'])
    plt.title('Real vs Predicted TSNE Embeddings, N='+str(args.num_people))
    plt.show()

    for a in range(30):
        npoints=2#int(input("How many points will you click?"))
        fig = plt.figure()
        plt.scatter(preds[:, 0], preds[:, 1], alpha=.1)
        plt.title(str(args.num_people) + " diffsData TSNE Predictions")
        plt.waitforbuttonpress()
        points_clicked = plt.ginput(npoints, show_clicks=True)
        plt.show()

        points=[]
        gtpoints=[]
        # plt.scatter(preds[:, 0], preds[:, 1])
        # plt.title(str(args.num_people) + " Traj Data TSNE Predictions")
        for coords in points_clicked:
            points.append(find_nearest(preds, coords))
            gtpoints.append(find_nearest(real,coords))
        #     plt.scatter(preds[points[-1]][0], preds[points[-1]][1], c='k')
        #     plt.scatter(real[gtpoints[-1]][0], real[gtpoints[-1]][1], c='r')
        # plt.show()

        # breakpoint()
        colors=['b','r','g','k','m']
        for i, point in enumerate(points):
            plt.figure()
            # breakpoint()
            # for pos in inputs[point].reshape(-1, 8, 2):
            for j, group in enumerate(inputs[point].reshape(args.num_people, -1, args.window, 2)):
                for pos in group:
                    plt.plot(pos[:,0], pos[:,1], c=colors[j])
                    plt.scatter(pos[0][0], pos[0][1], c=colors[j])
                    # plt.title('Point '+str(i))
            plt.title('Point '+str(i))
        for i, point in enumerate(gtpoints):
            plt.figure()
            # breakpoint()
            # for pos in inputs[point].reshape(-1, 8, 2):
            for j, group in enumerate(inputs[point].reshape(args.num_people, -1, args.window, 2)):
                for pos in group:
                    plt.plot(pos[:, 0], pos[:, 1], c=colors[j])
                    plt.scatter(pos[0][0], pos[0][1], c=colors[j])
                    # plt.title('GT Point '+str(i))
            plt.title('GT Point ' + str(i))
        plt.figure()
        plt.scatter(preds[:, 0], preds[:, 1])
        plt.title(str(args.num_people) + " diffsData TSNE Predictions")
        for i in range(len(points)):
            plt.scatter(preds[points[i]][0], preds[points[i]][1], c='k')
            plt.scatter(real[gtpoints[i]][0], real[gtpoints[i]][1], c='r')
        plt.show()


if __name__ == '__main__':
    args=get_args()
    dataset = TSNEGT(args.path + 'diffsData_' + str(args.num_people) + 'thresh_'+str(args.window)+'window.csv', args.num_clusters)
    # dataset = TSNEGT(args.path + 'socData_' + str(args.num_people) + 'thresh_'+ str(args.num_people) +'group_8window.csv', args.num_clusters)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    net = SimpleRegNetwork(args.num_people * (args.window-1) * 2)  # SimpleCatNetwork(32, args.num_clusters) #
    if args.train:
        print('training loop')
        net.load_state_dict(torch.load('simpleRegNet_diffsData_' + str(args.num_people) + 'people_' + str(args.window)+'window.pt'))
        net = train(net, loader, args)
    else:
        net.load_state_dict(torch.load('simpleRegNet_diffsData_' + str(args.num_people) + 'people_' + str(args.window)+'window.pt'))

    net.eval()
    graphPreds(net, loader, args)

    dataset = TSNEGT(args.path + 'diffsData_' + str(args.num_people) + 'thresh_'+str(args.window)+'window.csv', args.num_clusters, split='test')
    # dataset = TSNEGT(args.path + 'socData_' + str(args.num_people) + 'thresh_'+ str(args.num_people) +'group_8window.csv', args.num_clusters, split='test')
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    test(net,loader,args)
    graphPreds(net, loader, args)




