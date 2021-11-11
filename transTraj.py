import torch
import torch.nn as nn
import torch.optim as optim
from utils import getData, processData, plotTSNEclusters, batchify
import argparse
import math
import matplotlib.pyplot as plt
import numpy as np

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

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class SimpleTransformer(nn.Module):
    def __init__(self, args):
        super(SimpleTransformer, self).__init__()
        feat_dim, hidden_dim, out_dim = args.feat_dim, args.hidden_dim, args.targ_len
        nheads, nlayers = args.nheads, args.nlayers
        dropout = args.dropout
        self.pos_encoder = PositionalEncoding(feat_dim, dropout)
        self.encoder_layers = nn.TransformerEncoderLayer(feat_dim, nheads, hidden_dim)
        self.trans_enc = nn.TransformerEncoder(self.encoder_layers, nlayers)
        # self.enc = nn.Embedding(out_dim, feat_dim)
        self.dec = nn.Linear(feat_dim, out_dim)
        self.feat_dim = feat_dim
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        # self.enc.weight.data.uniform_(-initrange, initrange)
        self.dec.bias.data.zero_()
        self.dec.weight.data.uniform_(-initrange, initrange)

    def forward(self, x, mask):
        # x = self.enc(x) * math.sqrt(x)
        x = self.pos_encoder(x)
        emb = self.trans_enc(x, mask)
        output = self.dec(emb)
        return output, emb


def set_params():
    parser = argparse.ArgumentParser(description="Implementation of SwAV")

    #####################
    #### Data Params ####
    #####################
    parser.add_argument('--data', default='opentraj', choices=('trajnet++', 'MOT', 'opentraj'),
                        help='which dataset to use')
    parser.add_argument("--include_aug", type=bool, default=False,
                        help="whether to return the augmentation functions used on the data")
    parser.add_argument("--num_augs", default=1, type=int,
                        help="number of data augmentations to use (randomly chosen)")
    parser.add_argument('--num_frames', default=1, type=int,
                        help='number of frames of people to use for target for opentraj')
    parser.add_argument('--mode', default='by_human', choices=('by_human', 'by_frame', 'by_N_frame'),
                        help='whether to group by person or by frame for opentraj')
    parser.add_argument("--image", type=bool, default=False,
                        help="whether to return images or not")

    #####################
    #### Arch Params ####
    #####################
    parser.add_argument("--feat_dim", default=8, type=int, help="feature dimension")
    parser.add_argument("--hidden_dim", default=64, type=int, help="hidden dimension")
    parser.add_argument('--seq_len', default=8, type=int, help='num steps in each traj input')
    parser.add_argument('--targ_len', default=12, type=int, help='num steps in each traj output')
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
    parser.add_argument("--base_lr", default=4.8, type=float, help="base learning rate")
    parser.add_argument("--final_lr", type=float, default=0.000001, help="final learning rate")
    parser.add_argument("--seed", type=int, default=31, help="seed")

    return parser.parse_args()

def generate_square_subsequent_mask(sz: int):
    # from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

def train_loop(args, model, train_loader):
    loss=nn.MSELoss()
    #ToDo: put optimizer outside and keep track of lr schedule
    opt=optim.Adam(model.parameters(),lr=4e-5)
    total_loss=[]
    for i, inputs in enumerate(train_loader):
        trajectory, ims = inputs
        traj, target = batchify(trajectory, args)
        if len(traj)>0:
            # breakpoint()
            mask = generate_square_subsequent_mask(len(traj)).cuda()
            traj = np.transpose(traj, (0, 2, 1))
            output, embedding = model(torch.FloatTensor(traj).cuda(), mask.float())
            l=loss(torch.transpose(output,-1,-2),torch.FloatTensor(target).cuda())
            opt.zero_grad()
            l.backward()
            #ToDo: clip gradnorm?
            opt.step()
            total_loss.append(l.item())
    return model, total_loss
    
@torch.no_grad()
def test_loop(args, model, test_loader):
    loss=nn.MSELoss()
    total_loss=[]
    model.eval()
    for i, inputs in enumerate(test_loader):
        trajectory, ims = inputs
        traj, target = batchify(trajectory, args)
        if len(traj)>0:
            # breakpoint()
            mask = generate_square_subsequent_mask(len(traj)).cuda()
            traj = np.transpose(traj, (0, 2, 1))
            output, embedding = model(torch.FloatTensor(traj).cuda(), mask.float())
            l=loss(torch.transpose(output,-1,-2),torch.FloatTensor(target).cuda())
            total_loss.append(l.item())
            
        if i%75==0:
            fig=plt.figure()
#            breakpoint()
            for t in target[:-1]:
                plt.scatter(t[:,0], t[:,1], c='b')
            for o in torch.transpose(output,-1,-2).cpu()[:-1]:
                plt.scatter(o[:,0], o[:,1], c='tab:orange')
            plt.title('GT (blue) vs Pred (orange) Trajectories')
            plt.savefig('simpleTransfPredictions'+str(i)+'_'+str(args.nlayers)+'_'+str(args.nheads)+'_'+str(args.hidden_dim)+'.png')
            
    print('Test Loss:',np.mean(total_loss))
    
    fig=plt.figure()
    plt.plot(total_loss)
    plt.title('Simple Transformer Test Loss, Avg Val= '+str(np.mean(total_loss[-1])))
    plt.savefig('simpleTransTestLoss_'+str(args.nlayers)+'_'+str(args.nheads)+'_'+str(args.hidden_dim)+'.png')


if __name__ == '__main__':
    args = set_params()

    model = SimpleTransformer(args)
    model = model.float()
    model = model.cuda()
    model.train()

    train_loader = getData(args)
    trackTotLoss=[]
    for e in range(args.epochs):
        print('============ Starting Epoch', str(e), '==============')
        epochLoss=[]
        if type(train_loader) == list:
            for tl in train_loader[:-1]:
                model, totLoss = train_loop(args, model, tl)
                epochLoss.extend(totLoss)
        else:
            model, epochLoss = train_loop(args, model, train_loader)

        trackTotLoss.append(np.mean(epochLoss))
        print('Epoch Loss:',str(trackTotLoss[-1]))
        
    plt.plot(trackTotLoss)
    plt.title('Simple Trans Model Loss')
    plt.savefig('simpleTransformerLoss_'+str(args.nlayers)+'_'+str(args.nheads)+'_'+str(args.hidden_dim)+'.png')
    
    test_loop(args, model, train_loader[-1])
    
    torch.save(model,'simpleTransf_'+str(args.nlayers)+'_'+str(args.nheads)+'_'+str(args.hidden_dim)+'.pt')
