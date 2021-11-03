import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class AnomalyAttention(nn.Module):
    def __init__(self, seq_dim, channels):
        super(AnomalyAttention, self).__init__()
        self.Q = self.K = self.V = self.sigma = torch.zeros((seq_dim, channels))
        self.d_model = channels
        self.n  = seq_dim
        self.P = torch.zeros((seq_dim, seq_dim))
        self.S = torch.zeros((seq_dim, seq_dim))

    def forward(self, x):

        self.initialize(x)
        self.P = self.prior_association()
        self.S = self.series_association()
        print(self.S.shape)
        # assert self.S.shape == (self.n, self.n)
        Z = self.reconstruction()

        return Z

    def initialize(self, x):
        # self.d_model = x.shape[-1]
        self.Q = self.K = self.V = self.sigma = x

    def prior_association(self):
        return torch.ones((self.n, self.n))

    def series_association(self):
        print(self.Q.shape)
        print(self.K.shape)
        return F.softmax((self.Q @ self.K.T) / math.sqrt(self.d_model), dim=0)

    def reconstruction(self):
        return self.S @ self.V

    def association_discrepancy(self):
        return F.kl_div(self.P, self.S) + F.kl_div(self.S, self.P) #not going to be correct dimensions


class AnomalyTransformerBlock(nn.Module):
    def __init__(self, seq_dim, feat_dim):
        super().__init__()
        self.seq_dim, self.feat_dim = seq_dim, feat_dim
       
        self.attention = AnomalyAttention(self.seq_dim, self.feat_dim)
        self.ln1 = nn.LayerNorm(self.feat_dim)
        self.ff = nn.Sequential(
            nn.Linear(self.feat_dim, self.feat_dim),
            nn.ReLU()
        )
        self.ln2 = nn.LayerNorm(self.feat_dim)
        self.association_discrepancy = None

    def forward(self, x):
        x_identity = x 
        x = self.attention(x)
        z = self.ln1(x + x_identity)
        
        z_identity = z
        z = self.ff(z)
        z = self.ln2(z + z_identity)

        self.association_discrepancy = self.attention.association_discrepancy().detach()
        
        return z

class AnomalyTransformer(nn.Module):
    def __init__(self, seqs, in_channels, layers, lambda_):
        super().__init__()
        self.blocks = nn.ModuleList([
            AnomalyTransformerBlock(seqs, in_channels) for _ in range(layers)
        ])
        self.output = None
        self.lambda_ = lambda_
        self.assoc_discrepancy = torch.zeros((seqs, len(self.blocks)))
    
    def forward(self, x):
        for idx, block in enumerate(self.blocks):
            x = block(x)
            self.assoc_discrepancy[:, idx] = block.association_discrepancy
        
        self.assoc_discrepancy = self.assoc_discrepancy.sum(dim=1) #N x 1
        self.output = x
        return x

    def loss(self, x):
        l2_norm = torch.linalg.matrix_norm(self.output - x, ord=2)
        return l2_norm + (lambda_ * self.assoc_discrepancy)

    def anomaly_score(self, x):
        score = F.softmax(-self.assoc_discrepancy, dim=0)