import torch
from torch import nn
from torch.nn.functional.normalize

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
        self.referent_encoder = nn.Linear(in_features = n_features_ref_enc, out_features = N_HIDDEN, bias = False) # Referent encoder (i.e., image of abstract scene).
        self.linear_W7        = nn.Linear(in_features = n_features_desc_enc, out_features = N_HIDDEN, bias = False)
        self.ReLU             = nn.ReLU()
        self.linear_W6        = nn.Linear(in_features = N_HIDDEN, out_features = len(WORD_INDEX), bias = False)
        self.SoftMax          = nn.Softmax(dim = 1)
        
    def forward(self, target_referent_rep):
        # Referent encoding:
        e_r = self.referent_encoder(target_referent_rep)
        
        # Extract indicator features on descriptions and single words:
        # (Code taken from modules.py form Jacob Andreas)
        max_words        = max(len(scene.description) for scene in scenes)
        history_features = np.zeros((len(scenes), max_words, len(WORD_INDEX)))
        last_features    = np.zeros((len(scenes), max_words, len(WORD_INDEX)))
        targets          = np.zeros((len(scenes), max_words))
        for i_scene, scene in enumerate(scenes):
            for i_word, word in enumerate(scene.description):
                if word == 0:
                    continue
                for ii_word in range(i_word + 1, len(scene.description)):
                    history_features[i_scene, ii_word, word] += 1
                last_features[i_scene, i_word, word] += 1
                targets[i_scene, i_word] = word
        history_features_tensor = torch.tensor(history_features[:,i_step-1,:])
        last_features_tensor    = torch.tensor(last_features[:,i_step-1,:])
        
        # Compute a vector of scores with one s_i for each vocabulary item:
        probabilities_p_i = []
        for i_step in range(1, max_words):
            input_data                   = torch.cat([history_features_tensor, last_features_tensor], dim = 1) # Horizontal concatenation.
            indicator_features_embedding = self.linear_W7(input_data)
            all_features_embedding       = torch.cat([indicator_features_embedding, e_r]) # Description and target scene embeddings.
            scores                       = self.linear_W6( self.ReLU(all_features_embedding) )
            #probabilities_p_i            = normalize(scores) # Probability distribution over all vocabulary items.
            probabilities_p_i            = self.SoftMax(scores) # ??? Try it out if this or line 92 above works better!
        
        return probabilities_p_i