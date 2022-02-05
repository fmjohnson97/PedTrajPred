import torch
import torch.nn as nn
import torch.optim as optim
from utils import getData, processData, plotTSNEclusters, batchify, social_batchify
import argparse
import math
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader

class PositionalEncoding(nn.Module):
    # from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x, posEncInds):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        if True:#posEncInds is None:
            x = x + self.pe[:x.size(0)]
        else:
            #breakpoint()
            # this section does the position encoding relative to the start of the entire sequence
            # of data from the dataset instead of locally within a batch
            times=[self.pe[i] for i in posEncInds]
            x = x+self.pe[posEncInds[0]:posEncInds[0]+x.size(0)]
        return self.dropout(x)


class SimpleTransformer(nn.Module):
    def __init__(self, args):
        super(SimpleTransformer, self).__init__()
        feat_dim, hidden_dim, out_dim = args.feat_dim*2, args.hidden_dim, args.seq_len*2
        nheads, nlayers = args.nheads, args.nlayers
        dropout = args.dropout
        self.embed=nn.Sequential(nn.Linear(feat_dim,feat_dim))#, nn.LeakyReLU(), nn.Linear(feat_dim, feat_dim))
        self.pos_encoder = PositionalEncoding(feat_dim, dropout)
        self.encoder_layers = nn.TransformerEncoderLayer(feat_dim, nheads, hidden_dim)
        self.trans_enc = nn.TransformerEncoder(self.encoder_layers, nlayers)
        # self.enc = nn.Embedding(out_dim, feat_dim)
        self.dec = nn.Linear(feat_dim, out_dim)
        self.feat_dim = feat_dim
        self.init_weights()

    def init_weights(self):
        #initrange = 0.1
        # self.enc.weight.data.uniform_(-initrange, initrange)
        #self.dec.bias.data.zero_()
        #self.dec.weight.data.uniform_(-initrange, initrange)
        for p in self.trans_enc.parameters():
            if p.dim()>1:
                nn.init.xavier_uniform_(p)
        
        for p in self.dec.parameters():
            if p.dim()>1:
                nn.init.xavier_uniform_(p)
                

    def forward(self, x, mask, posEncInds=None):
        # breakpoint()
        # x = self.enc(x) * math.sqrt(x)
        x=self.embed(x)
        x = self.pos_encoder(x, posEncInds)
        emb = self.trans_enc(x, mask)
        output = self.dec(emb)
        return output, emb
        
        
class FullTransformer(nn.Module):
    def __init__(self, args):
        super(FullTransformer, self).__init__()
        feat_dim, hidden_dim, out_dim = args.feat_dim, args.hidden_dim, args.seq_len
        nheads, nlayers = args.nheads, args.nlayers
        
        self.init_weights()
        
    def init_weights(self):
        #initrange = 0.1
        # self.enc.weight.data.uniform_(-initrange, initrange)
        #self.dec.bias.data.zero_()
        #self.dec.weight.data.uniform_(-initrange, initrange)
        for p in self.trans_enc.parameters():
            if p.dim()>1:
                nn.init.xavier_uniform_(p)
        
        for p in self.dec.parameters():
            if p.dim()>1:
                nn.init.xavier_uniform_(p)
        
    def forward(x, tgt, posEncInds, srcMask, tgtMask):
        return None


def set_params():
    parser = argparse.ArgumentParser(description="Social Transformer")

    #####################
    #### Data Params ####
    #####################
    parser.add_argument('--data', default='opentraj', choices=('trajnet++', 'MOT', 'opentraj'),
                        help='which dataset to use')
    parser.add_argument("--include_aug", type=bool, default=False,
                        help="whether to return the augmentation functions used on the data")
    parser.add_argument("--num_augs", default=1, type=int,
                        help="number of data augmentations to use (randomly chosen)")
    parser.add_argument('--num_frames', default=8, type=int,
                        help='number of frames of people to use for target for opentraj')
    parser.add_argument('--mode', default='by_human', choices=('by_human', 'by_frame', 'by_N_frame'),
                        help='whether to group by person or by frame for opentraj')
    parser.add_argument("--image", type=bool, default=False,
                        help="whether to return images or not")
    parser.add_argument('--socialN', default=3, type=int)
    parser.add_argument("--filterMissing", type=bool, default=True,
                        help="whether to return images or not")
    parser.add_argument('--saveName', default='social_simpleTransfPredictions',
                        help='prefix to name all the save files')


    #####################
    #### Arch Params ####
    #####################
    parser.add_argument("--feat_dim", default=8, type=int, help="feature dimension")
    parser.add_argument("--hidden_dim", default=1024, type=int, help="hidden dimension")
    parser.add_argument('--seq_len', default=12, type=int, help='num steps in each traj output')
    parser.add_argument('--num_clusters', default=10, type=int, help='num knn clusters')
    parser.add_argument('--nheads', default=2, type=int, help='num attention heads')
    parser.add_argument('--nlayers', default=2, type=int, help='how deep the transformer will be')
    parser.add_argument("--dropout", default=0.1, type=float, help="percentage for dropout (<1)")

    ######################
    #### Train Params ####
    ######################
    parser.add_argument("--epochs", default=100, type=int,
                        help="number of total epochs to run")
    parser.add_argument("--batch_size", default=1, type=int,
                        help="batch size per gpu, i.e. how many unique instances per gpu")
    parser.add_argument("--seed", type=int, default=31, help="seed")
    parser.add_argument("--warmup_epochs", default=10, type=int, help="number of warmup epochs")
    parser.add_argument("--start_warmup", default=0, type=float,
                        help="initial warmup learning rate")
    parser.add_argument("--base_lr", default=4.8, type=float, help="base learning rate")
    parser.add_argument("--final_lr", type=float, default=1e-5, help="final learning rate")

    return parser.parse_args()

def generate_square_subsequent_mask(sz: int):
    # from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
    #return torch.ones(sz,sz)
    
def train_loop(args, model, train_loader, lr_schedule, iter):
    loss=nn.MSELoss()
    #ToDo: put optimizer outside and keep track of lr schedule
    opt=optim.SGD(model.parameters(),lr=4e-5)
    total_loss=[]
    for i, inputs in enumerate(train_loader):
        #breakpoint()
        trajectory= inputs['pos'].squeeze(0)
        items=None#inputs['index']
        if args.mode=='by_frame' and len(trajectory.shape)>2:
            traj, target, srcMask = social_batchify(trajectory, args)
        elif trajectory.nelement()>2*(args.num_frames+args.seq_len):
            traj, target, srcMask = batchify(trajectory, args)
        else:
            traj, target, srcMask = [], [], []
        if len(traj)>0:
            #breakpoint()
            # for param_group in opt.param_groups:
            #     param_group["lr"] = lr_schedule[iter]

            srcMask = generate_square_subsequent_mask(len(traj)).float()
            #traj = torch.transpose(traj, -1,-2)#np.transpose(traj, (0, -1, -2))
            traj=traj.reshape(-1, 1, args.num_frames*2)
            output, embedding = model(traj.float(), srcMask, items)
            #l=loss(torch.transpose(output,-1,-2),target.float().cuda())
            l=loss(output, target.reshape(-1,1,args.seq_len*2).float())
            opt.zero_grad()
            l.backward()
            #ToDo: clip gradnorm?
            opt.step()
            total_loss.append(l.item())
            iter +=1
    return model, total_loss, iter

@torch.no_grad()
def test_loop(args, model, test_loader):
    #breakpoint()
    loss=nn.MSELoss()
    total_loss=[]
    model.eval()
    for i, inputs in enumerate(test_loader):
        trajectory= inputs['pos'].squeeze(0)
        items=None#inputs['index']
        if args.mode=='by_frame' and len(trajectory.shape)>2:
            traj, target, srcMask = social_batchify(trajectory, args)
        elif trajectory.nelement()>2*(args.num_frames+args.seq_len):
            traj, target, srcMask = batchify(trajectory, args)
        else:
            traj, target, srcMask = [], [], []
        if len(traj)>0:
            #breakpoint()
            srcMask = generate_square_subsequent_mask(len(traj)).float()
            #traj = torch.transpose(traj, -1,-2)#np.transpose(traj, (0, -1, -2))
            traj=traj.reshape(-1, 1, args.num_frames*2)
            output, embedding = model(traj.float(), srcMask, items)
            #l=loss(torch.transpose(output,-1,-2),target.float().cuda())
            l=loss(output, target.reshape(-1,1,args.seq_len*2).float())
            total_loss.append(l.item())
            
        if i%75==0:
            fig=plt.figure()
            #breakpoint()
            traj=traj.reshape(-1, args.num_frames, 2)
            for t in traj:
                plt.scatter(t[:,0], t[:,1], c='g')
            for t in target:
                plt.scatter(t[:,0], t[:,1], c='b')
            for o in output.reshape(-1, args.seq_len, 2).cpu():
                plt.scatter(o[:,0], o[:,1], c='tab:orange')
            plt.title('Input (green) vs GT (blue) vs Pred (orange) Trajectories')
            plt.savefig(args.saveName+str(i)+'_'+str(args.nlayers)+'_'+str(args.nheads)+'_'+str(args.hidden_dim)+'.png')
            
    print('Test Loss:',np.mean(total_loss))
    
    fig=plt.figure()
    plt.plot(total_loss)
    plt.title('Simple Transformer Test Loss, Avg Val= '+str(np.mean(total_loss)))
    plt.savefig(args.saveName+str(args.nlayers)+'_'+str(args.nheads)+'_'+str(args.hidden_dim)+'.png')

def train(model, train_data, args):
    #breakpoint()
    data_len = sum([x.__len__() for x in train_data])
    warmup_lr_schedule = np.linspace(args.start_warmup, args.base_lr, data_len * args.warmup_epochs)
    iters = np.arange(data_len * (args.epochs - args.warmup_epochs))
    cosine_lr_schedule = np.array([args.final_lr + 0.5 * (args.base_lr - args.final_lr) * (1 + math.cos(math.pi
                * t / (data_len * (args.epochs - args.warmup_epochs)))) for t in iters])
    lr_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))
    iter=0
    trackTotLoss = []
    for e in range(args.epochs):
        print('============ Starting Epoch', str(e), '==============')
        epochLoss = []
        if type(train_data) == list:
            for td in train_data:
                loader = DataLoader(td,
                                    shuffle=False,
                                    batch_size=args.batch_size,
                                    pin_memory=True)
                model, totLoss, iter = train_loop(args, model, loader, lr_schedule, iter)
                epochLoss.extend(totLoss)
        else:
            model, epochLoss = train_loop(args, model, train_data)

        trackTotLoss.append(np.mean(epochLoss))
        print('Epoch Loss:', str(trackTotLoss[-1]))
        torch.save(model.state_dict(), args.saveName+ str(args.nlayers) + '_' + str(args.nheads) + '_' + str(
            args.hidden_dim) + '.pt')

    plt.plot(trackTotLoss)
    plt.title('Simple Trans Model Loss')
    plt.savefig(args.saveName + str(args.nlayers) + '_' + str(args.nheads) + '_' + str(
        args.hidden_dim) + '.png')

    return model


if __name__ == '__main__':
    args = set_params()

    model = SimpleTransformer(args)
    model.load_state_dict(torch.load('testOnCPU2_2_1024.pt'))
    model = model.float()
    model = model
    # model.train()

    train_data = getData(args)
    # model = train(model, train_data[:-1], args)

    loader = DataLoader(train_data[-1],
                        shuffle=False,
                        batch_size=args.batch_size,
                        pin_memory=True)
    model.eval()
    test_loop(args, model, loader)
    
    
