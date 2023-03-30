import adadelta
import corpus
from indices import WORD_INDEX
from modules_playground import *
import util

import torch
from torch import nn

#import apollocaffe
#from apollocaffe import ApolloNet
#from apollocaffe.layers import Concat
from collections import defaultdict
import itertools
import logging
import numpy as np
import shutil
import sys
import yaml

from MlpScorer_GGG import *

# REMARKS:
#   ``prop_embedding_size´´ is NOT used in functions relevant for our replication.
#   ``word_embedding_size´´ is NOT used in functions relevant for our replication.
CONFIG = """
opt:
    epochs: 10
    batch_size: 100
    alternatives: 1

    rho: 0.95
    eps: 0.000001
    lr: 1
    clip: 10

model:
    prop_embedding_size: 50
    word_embedding_size: 50
    hidden_size: 100
"""

N_TEST_IMAGES      = 100
N_TEST             = N_TEST_IMAGES * 10
N_EXPERIMENT_PAIRS = 100

# literal listener L0: Takes a description and a set of referents, and chooses the referent (i.e., image) most likely to be described.
class Listener0Model(nn.Module):
    def __init__(self, apollo_net, config):
        super().__init__() # PYTORCH
        self.scene_encoder  = LinearSceneEncoder("Listener0", apollo_net, config)  # Referent encoder (i.e., image of abstract scene).
        self.string_encoder = LinearStringEncoder("Listener0", apollo_net, config) # Description encoder (i.e., sentence describing the abstract scene image).
        #self.scorer         = MlpScorer("Listener0", apollo_net, config)           # Choice ranker R.
        self.scorer         = MlpScorer_GGG("Listener0", config)                   # Choice ranker R.

    def forward(self, data, alt_data, dropout):
        """
            data     : Target images (i.e., referents).
            alt_data : Distractor images (i.e., referents) ====> Reason: This model is trained CONTRASTIVELY.
        """
        #self.apollo_net.clear_forward()
        l_true_scene_enc = self.scene_encoder.forward("true", data, dropout)
        #print("l_true_scene_enc:\n", l_true_scene_enc) # DEBUG: Does the PyTorch-call work?
        ll_alt_scene_enc = [self.scene_encoder.forward("alt%d" % i, alt, dropout) for i, alt in enumerate(alt_data)]
        #print("ll_alt_scene_enc:\n", ll_alt_scene_enc) # DEBUG: Does the PyTorch-call work?
        l_string_enc     = self.string_encoder.forward("", data, dropout)
        #print("l_string_enc:\n", l_string_enc)         # DEBUG: Does the PyTorch-call work?
        
        ll_scenes = [l_true_scene_enc] + ll_alt_scene_enc # Concatenate.
        labels    = np.zeros((len(data),), dtype=int)     # MlpScorer needs these as integers.
        
        #print("Shape of l_string_enc:", l_string_enc.shape)         # DEBUG
        #print("ll_scenes:\n", ll_scenes)                            # DEBUG
        #print("Shape of l_true_scene_enc:", l_true_scene_enc.shape) # DEBUG
        #print("Shape of ll_alt_scene_enc:", ll_alt_scene_enc.shape) # DEBUG
        logprobs, accs = self.scorer.forward("", l_string_enc, ll_scenes, labels)
        
        print("logprobs:\n", logprobs) # DEBUG
        print("accs:\n", accs)         # DEBUG
        return logprobs, accs # Result: Distribution over referent choices (i.e., over images).

def train(train_scenes, test_scenes, model, apollo_net, config):
    """
        Trains different types of neuronal network model.
        
        model: L0/L1 or S0/S1 model is passed here.
    """
    n_train = len(train_scenes)
    n_test = len(test_scenes)

    #opt_state = adadelta.State()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    for i_epoch in range(config.epochs):

        with open("vis.html", "w") as vis_f:
            print("<html><body><table>", file=vis_f)

        np.random.shuffle(train_scenes)

        e_train_loss = 0.0
        e_train_acc = 0.0
        e_test_loss = 0.0
        e_test_acc = 0.0

        n_train_batches = (int)(n_train / config.batch_size)
        for i_batch in range(n_train_batches):
            batch_data  = train_scenes[i_batch * config.batch_size : (i_batch + 1) * config.batch_size]
            alt_indices = [np.random.choice(n_train, size=config.batch_size) for i_alt in range(config.alternatives)]
            alt_data    = [[train_scenes[i] for i in alt] for alt in alt_indices]
            
            # Zero the gradients
            optimizer.zero_grad()
            # Perform forward pass
            lls, accs = model(batch_data, alt_data, dropout=True)
            """ COMPUTE THE LOSS HERE!
                    In the ApolloCaffe version, the loss is computed within model.forward() usually as ``SoftmaxWithLoss()´´.
                    The PyTorch implementation requires the loss to be computed outside model.forward()
                    As both, L0 and S0 use SoftMax as last layer, the following loss function is required in the PyTorch implementation: nn.CrossEntropyLoss()
            """
            apollo_net.backward()
            adadelta.update(apollo_net, opt_state, config)

            e_train_loss -= lls.sum()
            e_train_acc  += accs.sum()

        n_test_batches = n_test / config.batch_size
        for i_batch in range(n_test_batches):
            batch_data = test_scenes[i_batch * config.batch_size : (i_batch + 1) * config.batch_size]
            alt_indices = [np.random.choice(n_test, size=config.batch_size) for i_alt in range(config.alternatives)]
            alt_data    = [[test_scenes[i] for i in alt] for alt in alt_indices]
            
            lls, accs = model.forward(batch_data, alt_data, dropout=False)
            """ COMPUTE THE LOSS HERE!
                    In the ApolloCaffe version, the loss is computed within model.forward() usually as ``SoftmaxWithLoss()´´.
                    The PyTorch implementation requires the loss to be computed outside model.forward()
                    As both, L0 and S0 use SoftMax as last layer, the following loss function is required in the PyTorch implementation: nn.CrossEntropyLoss()
            """

            e_test_loss -= lls.sum()
            e_test_acc  += accs.sum()

        with open("vis.html", "a") as vis_f:
            print("</table></body></html>", file=vis_f)

        shutil.copyfile("vis.html", "vis2.html")

        e_train_loss /= n_train_batches
        e_train_acc /= n_train_batches
        e_test_loss /= n_test_batches
        e_test_acc /= n_test_batches

        print("%5.3f  (%5.3f)  :  %5.3f  (%5.3f)" % (
                e_train_loss, e_train_acc, e_test_loss, e_test_acc))

def main():
    #apollocaffe.set_device(0)
    #apollocaffe.set_random_seed(0)
    np.random.seed(0)
    
    job         = "train.base" #sys.argv[1]
    corpus_name = "abstract"   #sys.argv[2]

    config = util.Struct(**yaml.safe_load(CONFIG))
    if corpus_name == "abstract":
        train_scenes, dev_scenes, test_scenes = corpus.load_abstract()
    else:
        assert corpus_name == "birds"
        train_scenes, dev_scenes, test_scenes = corpus.load_birds()
    apollo_net = "" #ApolloNet()
    print("loaded data")
    print("%d training examples" % len(train_scenes))
    
    listener0_model = Listener0Model(apollo_net, config.model)
    #speaker0_model = Speaker0Model(apollo_net, config.model)
    
    if job == "train.base":
        train(train_scenes, dev_scenes, listener0_model, apollo_net, config.opt)
        #train(train_scenes, dev_scenes, speaker0_model, apollo_net, config.opt)
        #apollo_net.save("models/%s.base.caffemodel" % corpus_name)
        exit()


if __name__ == "__main__":
    main()
