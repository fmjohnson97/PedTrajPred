import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn import Linear, ReLU
from swav.src.multicropdataset import MultiCropDataset
import swav.src.resnet50 as resnet_models
import argparse
import numpy as np
import math
import os
import torch.nn.functional as F
from utils import getData, processData, plotTSNEclusters


class SimpleLinear(nn.Module):
    def __init__(self, args):
        super(SimpleLinear,self).__init__()
        self.fc1=Linear(2,4)
        self.fc2=Linear(4,8)
        self.fc3=Linear(8,16)
        self.relu=ReLU()
        self.prototypes = nn.Linear(args.seq_len*16, args.out_len, bias=False)

    def forward(self,x, y):
        # first image
        x=self.relu(self.fc1(x))
        x=self.relu(self.fc2(x))
        x=self.fc3(x)
        x = torch.flatten(x, 1)
        x = nn.functional.normalize(x, dim=1, p=2)
        # breakpoint()
        #second image --> same process
        y = self.relu(self.fc1(y))
        y = self.relu(self.fc2(y))
        y = self.fc3(y)
        y = torch.flatten(y, 1)
        y = nn.functional.normalize(y, dim=1, p=2)
        return torch.cat((x,y),axis=0), torch.cat((self.prototypes(x),self.prototypes(y)),axis=0)



def distributed_sinkhorn(out):
    Q = torch.exp(out / args.epsilon).t() # Q is K-by-B for consistency with notations from our paper
    B = Q.shape[1] #* args.world_size # number of samples to assign
    K = Q.shape[0] # how many prototypes

    # make the matrix sums to 1
    sum_Q = torch.sum(Q)
    #dist.all_reduce(sum_Q)
    Q /= sum_Q
    if torch.isnan(torch.sum(Q)):
        breakpoint()

    for it in range(args.sinkhorn_iterations):
        if torch.isnan(torch.sum(Q)):
            breakpoint()
        # normalize each row: total weight per prototype must be 1/K
        sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
        #dist.all_reduce(sum_of_rows)
        Q /= sum_of_rows
        Q /= K

        # normalize each column: total weight per sample must be 1/B
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= B

    #ToDO: this line might be wrong??? needs to be Q/sum(Q,axis=0)
    Q *= B # the colomns must sum to 1 so that Q is an assignment
    if torch.isnan(torch.sum(Q)):
        breakpoint()
    return Q.t()


def set_params():
    parser = argparse.ArgumentParser(description="Implementation of SwAV")

    #########################
    #### data parameters ####
    #########################
    parser.add_argument("--data_path", type=str, default="/home/faith/Documents/Udacity/Images",
                        help="path to dataset repository; default is Udacity")
    parser.add_argument("--nmb_crops", type=int, default=[2], nargs="+",
                        help="list of number of crops (example: [2, 6])")
    parser.add_argument("--size_crops", type=int, default=[224], nargs="+",
                        help="crops resolutions (example: [224, 96])")
    parser.add_argument("--min_scale_crops", type=float, default=[0.14], nargs="+",
                        help="argument in RandomResizedCrop (example: [0.14, 0.05])")
    parser.add_argument("--max_scale_crops", type=float, default=[1], nargs="+",
                        help="argument in RandomResizedCrop (example: [1., 0.14])")

    #########################
    ## swav specific params #
    #########################
    parser.add_argument("--crops_for_assign", type=int, nargs="+", default=[0, 1],
                        help="list of crops id used for computing assignments")
    parser.add_argument("--temperature", default=0.1, type=float,
                        help="temperature parameter in training loss")
    parser.add_argument("--epsilon", default=0.05, type=float,
                        help="regularization parameter for Sinkhorn-Knopp algorithm")
    parser.add_argument("--sinkhorn_iterations", default=3, type=int,
                        help="number of iterations in Sinkhorn-Knopp algorithm")
    parser.add_argument("--feat_dim", default=256, type=int,
                        help="feature dimension")
    parser.add_argument("--nmb_prototypes", default=3000, type=int,
                        help="number of prototypes")
    parser.add_argument("--queue_length", type=int, default=128,
                        help="length of the queue (0 for no queue)")
    parser.add_argument("--epoch_queue_starts", type=int, default=0,
                        help="from this epoch, we start using a queue")

    #########################
    #### optim parameters ###
    #########################
    parser.add_argument("--epochs", default=100, type=int,
                        help="number of total epochs to run")
    parser.add_argument("--batch_size", default=1, type=int,
                        help="batch size per gpu, i.e. how many unique instances per gpu")
    parser.add_argument("--base_lr", default=4.8, type=float, help="base learning rate")
    parser.add_argument("--final_lr", type=float, default=0.000001, help="final learning rate")
    parser.add_argument("--freeze_prototypes_niters", default=313, type=int,
                        help="freeze the prototypes during this many iterations from the start")
    parser.add_argument("--wd", default=1e-6, type=float, help="weight decay")
    parser.add_argument("--warmup_epochs", default=10, type=int, help="number of warmup epochs")
    parser.add_argument("--start_warmup", default=0, type=float,
                        help="initial warmup learning rate")

    #########################
    #### dist parameters ###
    #########################
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up distributed
                        training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--world_size", default=1, type=int, help="""
                        number of processes: it is set automatically and
                        should not be passed as argument""")
    parser.add_argument("--rank", default=0, type=int, help="""rank of this process:
                        it is set automatically and should not be passed as argument""")
    parser.add_argument("--local_rank", default=0, type=int,
                        help="this argument is not used and should be ignored")

    #########################
    #### other parameters ###
    #########################
    parser.add_argument("--arch", default="resnet50", type=str, help="convnet architecture")
    parser.add_argument("--hidden_mlp", default=2048, type=int,
                        help="hidden layer dimension in projection head")
    parser.add_argument("--workers", default=1, type=int,
                        help="number of data loading workers")
    parser.add_argument("--checkpoint_freq", type=int, default=25,
                        help="Save the model periodically")
    parser.add_argument("--use_fp16", type=bool, default=True,
                        help="whether to train with mixed precision or not")
    parser.add_argument("--sync_bn", type=str, default="pytorch", help="synchronize bn")
    parser.add_argument("--syncbn_process_group_size", type=int, default=8, help=""" see
                        https://github.com/NVIDIA/apex/blob/master/apex/parallel/__init__.py#L58-L67""")
    parser.add_argument("--dump_path", type=str, default=".",
                        help="experiment dump path for checkpoints and log")
    parser.add_argument("--seed", type=int, default=31, help="seed")

    #######################
    #### custom params ####
    #######################
    parser.add_argument('--data', default='opentraj', choices=('trajnet++', 'MOT', 'opentraj'),
                        help='which dataset to use')
    parser.add_argument("--include_aug", type=bool, default=False,
                        help="whether to return the augmentation functions used on the data")
    parser.add_argument("--num_augs", default=1, type=int,
                        help="number of data augmentations to use (randomly chosen)")
    parser.add_argument("--mask_percent", default=0.25, type=float,
                        help="amount of the data to mask out (set to -1 since it's normed)")
    parser.add_argument('--num_frames', default=1, type=int,
                        help='number of frames of people to use for opentraj')
    parser.add_argument('--sample', default=1.0, type=float,
                        help='sample ratio when loading train/val scenes')
    parser.add_argument('--goals', action='store_true',
                        help='flag to consider goals of pedestrians')  # true makes it false somehow *shrugs*
    parser.add_argument('--mode', default='by_human', choices=('by_human', 'by_frame'),
                        help='whether to group by person or by frame for opentraj')
    parser.add_argument("--image", type=bool, default=False,
                        help="whether to return images or not")
    parser.add_argument('--seq_len', default=8, type=int, help='num steps in each traj input')
    parser.add_argument('--out_len', default=12, type=int, help='dim of each prototype vector')
    parser.add_argument('--num_prototypes', default=128, type=int, help='num prototype vectors')
    parser.add_argument('--num_clusters', default=10, type=int, help='num knn clusters')

    return parser.parse_args()


def train_loop(args, model):
    train_dataset = getData(args)

    print("Building data done with {} datasets loaded.".format(len(train_dataset)))

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.base_lr,
        momentum=0.9,
        weight_decay=args.wd,
    )
    print("Building optimizer done.")

    queue = None
    # queue_path = os.path.join(args.dump_path, "queue" + str(args.rank) + ".pth")
    # if os.path.isfile(queue_path):
    #     queue = torch.load(queue_path)["queue"]
    # the queue needs to be divisible by the batch size
    ''' Ignoring this for now since batch size is technically set to 1'''
    # args.queue_length -= args.queue_length % (args.batch_size * args.world_size)

    iteration=0
    totalLoss = []
    totalAvgEpochLoss=[]

    data_len = 881
    warmup_lr_schedule = np.linspace(args.start_warmup, args.base_lr, data_len * args.warmup_epochs)
    iters = np.arange(data_len * (args.epochs - args.warmup_epochs))
    cosine_lr_schedule = np.array([args.final_lr + 0.5 * (args.base_lr - args.final_lr) * (1 + math.cos(math.pi
                        * t / (data_len * (args.epochs - args.warmup_epochs)))) for t in iters])
    lr_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))

    for epoch in range(args.epochs):
        # train the network for one epoch
        print("============ Starting epoch %i ... ============" % epoch)

        # optionally starts a queue
        if args.queue_length > 0 and epoch >= args.epoch_queue_starts and queue is None:
            # queue = torch.zeros(
            #     len(args.crops_for_assign),
            #     args.queue_length // args.world_size,
            #     args.feat_dim,
            # ).cuda()
            queue=[]

        epochLoss=[]

        for t in train_dataset:
            train_loader = torch.utils.data.DataLoader(
                t,
                batch_size=args.batch_size,
                num_workers=args.workers,
                pin_memory=True,
                drop_last=True
            )
            # TODO: fix this; 881 is supposed to be the length of the train loader

            for it, inputs in enumerate(train_loader):
                ### normalizing the prototypes
                with torch.no_grad():
                    w = model.prototypes.weight.data.clone()
                    w = nn.functional.normalize(w, dim=1, p=2)
                    if torch.isnan(torch.sum(w)):
                        breakpoint()
                    model.prototypes.weight.copy_(w)

                ### setting the learning rate
                for param_group in optimizer.param_groups:
                    try:
                        param_group["lr"] = lr_schedule[iteration]
                    except:
                        breakpoint()

                ### augmenting the input trajectories
                augIn1, augIn2, targets, images =processData(args, inputs)
                if augIn1 is None or len(augIn1)<1:
                    continue
                else:
                    iteration += 1
                    augIn1=augIn1.float().cuda()
                    augIn2=augIn2.float().cuda()

                # ============ multi-res forward passes ... ============
                embedding, output = model(augIn1, augIn2)
                embedding = embedding.detach()
                bs = augIn1.size(0)

                loss = 0
                for i, crop_id in enumerate(args.crops_for_assign):
                    with torch.no_grad():
                        out = output[bs * crop_id: bs * (crop_id + 1)].detach()
                        # time to use the queue
                        if queue is not None:
                            # breakpoint()
                            if len(queue)==args.queue_length:
                                # breakpoint()
                                out = torch.cat((torch.mm(
                                    torch.stack(queue),
                                    model.prototypes.weight.t()
                                ), out))
                                # fill the queue
                                # move the bottom thing in the queue up to the top --> queue[i, bs:] = queue[i, :-bs].clone()
                                queue.pop(0)
                            # add new embeddings to the bottom --> queue[i, :bs] = embedding[crop_id * bs: (crop_id + 1) * bs]
                            queue.extend(embedding[crop_id * bs: (crop_id + 1) * bs])
                        # get assignments
                        q = distributed_sinkhorn(out)[-bs:]

                    # cluster assignment prediction
                    subloss = 0
                    for v in np.delete(np.arange(np.sum(args.nmb_crops)), crop_id):
                        x = output[bs * v: bs * (v + 1)] / args.temperature
                        subloss -= torch.mean(torch.sum(q * F.log_softmax(x, dim=1), dim=1))
                    loss += subloss / (np.sum(args.nmb_crops) - 1)
                loss /= len(args.crops_for_assign)

                #ToDo: either report or backprop loss wrt batch size???
                #loss /= augIn1.shape[0]

                if torch.isnan(loss):
                    breakpoint()
                # ============ backward and optim step ... ============
                optimizer.zero_grad()
                #if args.use_fp16:
                #    with apex.amp.scale_loss(loss, optimizer) as scaled_loss:
                #        scaled_loss.backward()
                #else:
                loss.backward()

                #ToDo: update model params??????

                #ToDo: update prototypes???????

                # cancel gradients for the prototypes
                if iteration < args.freeze_prototypes_niters:
                    for name, p in model.named_parameters():
                        if "prototypes" in name:
                            p.grad = None
                optimizer.step()

                epochLoss.append(loss.item())
                totalLoss.append(loss.item())

        totalAvgEpochLoss.append(np.mean(epochLoss))
        print("Epoch:",epoch,'\t Loss:',np.mean(epochLoss), 'from',str(len(epochLoss)),'samples')
        save_dict = {
            "epoch": epoch + 1,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        torch.save(
            save_dict,
            os.path.join(args.dump_path, "checkpoint.pth.tar"),
        )
        torch.save(model,"trajswav.pt")
        #if epoch % args.checkpoint_freq == 0 or epoch == args.epochs - 1:
        #    shutil.copyfile(
        #        os.path.join(args.dump_path, "checkpoint.pth.tar"),
        #        os.path.join(args.dump_checkpoints, "ckp-" + str(epoch) + ".pth"),
        #    )
        # if queue is not None:
        #     torch.save({"queue": queue}, queue_path)

    fig=plt.figure()
    plt.plot(totalLoss)
    plt.savefig('Traj_Swav_loss.png')
    fig = plt.figure()
    plt.plot(totalAvgEpochLoss)
    plt.savefig('lrsched_Traj_Swav_avgLossPerEpoch_'+ str(args.epochs) + 'epochs_'+str(args.queue_length)+'queue.png')
    fig=plt.figure()
    plt.plot(lr_schedule)
    plt.savefig('learning_rate_traj_swav.png')
    title=" Prototype Vector Centroids, " + str(args.num_clusters) + " Clusters, Len " + str(args.seq_len)
    saveName='lrsched_swav_protos_'+str(args.num_clusters)+'Clusters_len'+str(args.seq_len)+'.png'
    protoData, protoCenters = plotTSNEclusters(model.prototypes.weight.data.clone().cpu().numpy(),args, title, saveName)
    title = " Prototype Vector Centroids T(), " + str(args.num_clusters) + " Clusters, Len " + str(args.seq_len)
    saveName = 'lrsched_swav_trans_protos_' + str(args.epochs) + 'epochs_'+str(args.queue_length)+'queue_seqlen' + str(args.seq_len) + '.png'
    protoDataT, protoCentersT = plotTSNEclusters(model.prototypes.weight.data.t().clone().cpu().numpy(),args, title, saveName)

    data, centers = plotLearnedProtos(model,args)
    # breakpoint()
    return model

@torch.no_grad()
def plotLearnedProtos(model,args):
    predictions=[]
    model.eval()
    train_dataset = getData(args)
    for t in train_dataset:
        train_loader = torch.utils.data.DataLoader(
            t,
            batch_size=args.batch_size,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=True
        )
        for i,inputs in enumerate(train_loader):
            data = processData(args, inputs, doAugs=False)
            if data[0] is not None:
                batch, targets, images = data
                batch=batch.float().cuda()
                embedding, output = model(batch, batch)
                predictions.extend(output[:-len(output)//2].cpu())
    # breakpoint()
    plotTSNEclusters(torch.stack(predictions),args,'Predicted Prototypes from Data','pred_protos_traj_'+str(args.epochs) + 'epochs_'+str(args.queue_length)+'queue_seqlen'+ str(args.seq_len) +'.png')
    return torch.stack(predictions), None




if __name__=='__main__':
    args=set_params()

    # model = resnet_models.__dict__[args.arch](
    #     normalize=True,
    #     hidden_mlp=args.hidden_mlp,
    #     output_dim=args.feat_dim,
    #     nmb_prototypes=args.nmb_prototypes,
    # )
    # model=torch.hub.load('facebookresearch/swav:main','resnet50')
    model=SimpleLinear(args)
    model=model.float()
    model = model.cuda()
    model.train()
    print("Building model done.")

    model=train_loop(args, model)

    # breakpoint()
    # state_dict=torch.load("checkpoint.pth.tar")
    # model.load_state_dict(state_dict['state_dict'])
    # model=torch.load("trajswav.pt")
    # title = " Prototype Vector Centroids, " + str(args.num_clusters) + " Clusters, Len " + str(args.seq_len)
    # saveName = 'swav_protos_' + str(args.num_clusters) + 'Clusters_len' + str(args.seq_len) + '.png'
    # plotTSNEclusters(model.prototypes.weight.data.t().clone().cpu().numpy(), args, title, saveName)
    # plotLearnedProtos(model, args)
    # breakpoint()

