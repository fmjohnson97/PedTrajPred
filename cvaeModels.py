'''

Everything borrowed from here: https://github.com/kleinzcy/Variational-AutoEncoder/blob/master/VAE.ipynb
Then modified a bit to fit my data

'''

import torch
import torch.nn as nn

class CVAE_Net(torch.nn.Module):
    def __init__(self, args):
        super(CVAE_Net, self).__init__()
        self.lstm = nn.LSTM(2,64,2)
        self.hidden = None
        self.encoder = nn.Linear(130,256)
        self.mu_dense = nn.Linear(256, 64)
        self.log_sigma_dense = nn.Linear(256, 64)
        self.decoder = nn.Linear(66,128)
        self.output_fc = nn.Linear(128,args.output_window*2)
        self.relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x, label):
        # breakpoint()
        # c = self.idx2onehot(c)
        x = torch.cat((x, label), dim=-1)
        x = self.relu(self.encoder(x))
        mu = self.mu_dense(x)
        log_sigma = self.log_sigma_dense(x)
        return mu, log_sigma

    def decode(self, x, label):
        # breakpoint()
        # c = self.idx2onehot(c)
        x = torch.cat((x, label), dim=-1)
        x = self.relu(self.decoder(x))
        # output = self.output_fc(x)
        output = self.sigmoid(self.output_fc(x))
        return output

    def sample_z(self, mu, log_sigma):
        # breakpoint()
        eps = torch.randn(mu.shape[0], 64)
        return mu + torch.exp(log_sigma / 2) * eps

    def forward(self, x, label):
        # breakpoint()
        x, (h,c) = self.lstm(x)
        mu, log_sigma = self.encode(h.reshape(-1,64*2), label)
        z = self.sample_z(mu, log_sigma)
        output = self.decode(z, label)
        return output, mu, log_sigma, z