'''

Everything borrowed from here: https://github.com/kleinzcy/Variational-AutoEncoder/blob/master/VAE.ipynb
Then modified a bit to fit my data

'''

import torch
import torch.nn as nn

class CVAE_Net(torch.nn.Module):
    def __init__(self, args):#, device):
        super(CVAE_Net, self).__init__()
        self.lstm = nn.LSTM(2,8,2)#nn.LSTM(2,16,2)
        self.hidden = None
        self.encoder = nn.Linear(16,16)
        # self.encoder = nn.Linear(34, 64)
        #self.encoder = nn.Linear(18,64)
        self.mu_dense = nn.Linear(16, 8)
        self.log_sigma_dense = nn.Linear(16, 8)
        # self.decoder = nn.Linear(209,128)
        self.decoder = nn.Linear(8, 16)
        self.output_fc = nn.Linear(16,args.input_window*2)
        self.relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        # self.lstm2 = nn.LSTM(34,args.output_window*2,2)
        # self.device = device

    def encode(self, x, label=None):
        # breakpoint()
        # onehot = torch.zeros(label.size(0), 145)
        # onehot.scatter_(1, label, 1)
        # label = torch.stack([onehot] * x.size(0)).squeeze(1)
        # x = torch.cat((x, label), dim=-1)
        x = self.relu(self.encoder(x))
        mu = self.mu_dense(x)
        log_sigma = self.log_sigma_dense(x)
        return mu, log_sigma

    def decode(self, x, label=None):
        # breakpoint()
        # onehot = torch.zeros(label.size(0), 145)
        # onehot.scatter_(1, label, 1)
        # label =torch.stack([onehot]*x.size(0)).squeeze(1)
        # x = torch.cat((x, label), dim=-1)
        x = self.relu(self.decoder(x))
        # x, h = self.lstm2(x.unsqueeze(0))
        # output = self.output_fc(x)
        output = self.sigmoid(self.output_fc(x))
        return output

    def sample_z(self, mu, log_sigma):
        # breakpoint()
        eps = torch.randn(mu.shape[0], mu.shape[1])#.to(self.device)
        return mu + torch.exp(log_sigma / 2) * eps

    def forward(self, x, label):
        # breakpoint()
        # x, (h,c) = self.lstm(x)
        mu, log_sigma = self.encode(x.reshape(-1,8*2))#, label)
        # mu, log_sigma = self.encode(x.reshape(-1,8*8), label)
        z = self.sample_z(mu, log_sigma)
        output = self.decode(z)#, label)
        return output, mu, log_sigma, z
