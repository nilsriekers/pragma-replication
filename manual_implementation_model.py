import torch
from torch import nn

import numpy as np

"""
Literal implementation of the models from the paper based in the given formulas.
"""

# Taken from modules.py
from indices import WORD_INDEX
N_PROP_TYPES   = 8
N_PROP_OBJECTS = 35
N_HIDDEN       = 100
N_OUT_SCALAR   = 1

n_features_ref_enc  = N_PROP_TYPES * N_PROP_OBJECTS # Number of features a scene encoding has (280).
n_features_desc_enc = len(WORD_INDEX) # Number of features a description encoding has (1063).

def R(X):
    # Computes the choice ranker, i.e., expression (3) in the paper.
    # INPUT:
    #   X: List with two numbers.
    # OUTPUT:
    #   Ordered list of Softmax of the inputs.
    s_1     = X[0]
    s_2     = X[1]
    exp_s_1 = np.exp(s_1)
    exp_s_2 = np.exp(s_2)
    return [ exp_s_1/(exp_s_1+exp_s_2) , exp_s_2/(exp_s_1+exp_s_2) ]

class LiteralListener_L0(nn.Module):
    def __init__(self):
        super().__init__()
        self.description_encoder = nn.Linear(in_features = n_features_desc_enc, out_features = N_HIDDEN, bias = False)
        self.referent_encoder    = nn.Linear(in_features = n_features_ref_enc, out_features = N_HIDDEN, bias = False)
        self.linear_W4_e         = nn.Linear(in_features = N_HIDDEN, out_features = N_HIDDEN, bias = False)
        self.linear_W5_ed        = nn.Linear(in_features = N_HIDDEN, out_features = N_HIDDEN, bias = False)
        self.linear_w3           = nn.Linear(in_features = N_HIDDEN, out_features = N_OUT_SCALAR, bias = False)
        self.ReLU                = nn.ReLU()
        
    def forward(self, description_rep, referent_rep_1, referent_rep_2):
        # Encodings:
        e_d = self.description_encoder(description_representation) # Description (=caption) encoding.
        e_1 = self.referent_encoder(referent_rep_1)                # Referent (=scene) encoding.
        e_2 = self.referent_encoder(referent_rep_2)                # Referent (=scene) encoding.
        # Choice ranker:
        s_1 = torch.add( self.linear_W4_e(e_1), self.linear_W5_ed(e_d) ) # Elementwise sum.
        s_1 = self.linear_w3( self.ReLU(s_1) )
        s_2 = torch.add( self.linear_W4_e(e_2), self.linear_W5_ed(e_d) )
        s_2 = self.linear_w3( self.ReLU(s_2) )
        pL0 = R([s_1, s_2]) # Expression (4) in the paper.
        return pL0

class LiteralSpeaker_S0(nn.Module):
    def __init__(self):
        super().__init__()
        self.referent_encoder = nn.Linear(in_features = n_features_ref_enc, out_features = N_HIDDEN, bias = False)
        # ...
        
    def forward(self, referent_rep):
        e = self.referent_encoder(referent_rep)
        # ...