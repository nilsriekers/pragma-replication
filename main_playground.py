import adadelta
import corpus
from indices import WORD_INDEX
#from modules import *
import util

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

job         = "train.base" #sys.argv[1]
corpus_name = "abstract"   #sys.argv[2]

config = util.Struct(**yaml.safe_load(CONFIG))
if corpus_name == "abstract":
    train_scenes, dev_scenes, test_scenes = corpus.load_abstract()
else:
    assert corpus_name == "birds"
    train_scenes, dev_scenes, test_scenes = corpus.load_birds()
#apollo_net = ApolloNet()
print("loaded data")
print("%d training examples" % len(train_scenes))