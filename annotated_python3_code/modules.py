#!/usr/bin/env python2

from indices import WORD_INDEX

from apollocaffe.layers import *
from corpus import Scene, Bird # "Scene" and "Bird" refer to namedtuple data, e.g., `` Scene = namedtuple("Scene", ["image_id", "props", "description", "features"]) ´´
                               # Using "Scene" and "Bird" allows to distinguish data without the need for storing it in separate classes.
import numpy as np
from scipy.misc import logsumexp

N_PROP_TYPES = 8
N_PROP_OBJECTS = 35

class EuclideanScorer(object):
    def __init__(self, name, apollo_net, config):
        self.name = name
        self.apollo_net = apollo_net
        self.config = config

    def forward(self, prefix, l_query, ll_targets, labels):
        net = self.apollo_net

        batch_size, n_dims = net.blobs[l_query].shape
        n_targets = len(ll_targets)

        l_cat = "EuclideanScorer_%s_%s_cat" % (self.name, prefix)
        l_inv_query = "EuclideanScorer_%s_%s_inv_query" % (self.name, prefix)
        l_tile_query = "EuclideanScorer_%s_%s_tile_query" % (self.name, prefix)
        l_add_query = "EuclideanScorer_%s_%s_add_query" % (self.name, prefix)
        l_sq = "EuclideanScorer_%s_%s_sq" % (self.name, prefix)
        l_neg = "EuclideanScorer_%s_%s_neg" % (self.name, prefix)
        l_reduce = "EuclideanScorer_%s_%s_reduce" % (self.name, prefix)
        l_label = "EuclideanScorer_%s_%s_target" % (self.name, prefix)
        l_loss = "EuclideanScorer_%s_%s_loss" % (self.name, prefix)

        p_reduce = ["EuclideanScorer_reduce_weight",
                    "EuclideanScorer_reduce_bias"]

        for l_target in ll_targets:
            net.blobs[l_target].reshape((batch_size, n_dims, 1, 1))
        net.f(Concat(l_cat, axis=2, bottoms=ll_targets))
        net.f(Power(l_inv_query, scale=-1, bottoms=[l_query]))
        net.blobs[l_inv_query].reshape((batch_size, n_dims, 1, 1))
        net.f(Tile(l_tile_query, tiles=n_targets, axis=2, bottoms=[l_inv_query]))
        net.f(Eltwise(l_add_query, "SUM", bottoms=[l_tile_query, l_cat]))
        net.f(Power(l_sq, power=2, bottoms=[l_add_query]))
        net.f(Power(l_neg, scale=-1, bottoms=[l_sq]))
        net.f(Convolution(
            l_reduce, (1,1), 1, bottoms=[l_neg], param_names=p_reduce,
            param_lr_mults=[0,0], weight_filler=Filler("constant", 1),
            bias_filler=Filler("constant", 0)))
        net.blobs[l_reduce].reshape((batch_size, n_targets))
        net.f(NumpyData(l_label, labels))
        loss = net.f(SoftmaxWithLoss(l_loss, bottoms=[l_reduce, l_label]))
        denominators = logsumexp(net.blobs[l_reduce].data, axis=1)
        chosen_logprobs = net.blobs[l_reduce].data[range(batch_size), labels.astype(int)]
        chosen_logprobs -= denominators

        predictions = np.argmax(net.blobs[l_reduce].data, axis=1)
        acc = np.mean(predictions==labels)
        accs = predictions == labels

        return chosen_logprobs, accs

# Choice ranker R
#   The choice ranker takes a string encoding and a collection of referent encodings,
#   assigns a score to each (string, referent) pair, and then transforms these scores
#   into a distribution over referents.
class MlpScorer(object):
    def __init__(self, name, apollo_net, config):
        self.name = name
        self.apollo_net = apollo_net
        self.config = config

    #                         v--Corresponds to ``l_string_enc´´ in the caller.
    #                         v        v--Corresponds to ``ll_scenes´´ in the caller.
    #                         v        v           v--Array of zeros.
    def forward(self, prefix, l_query, ll_targets, labels):
        """
            l_query: Embedding of a description (i.e., caption) of shape torch.Size([100, 100])
            ll_targets: Embedding of Scenes (target and distractor). It is a list of tensors 2 tensors each of shape torch.Size([100, 100])
            prefix : Not used --> Passed to forward() only with an empty string.
        """
        net = self.apollo_net

        batch_size, n_dims = net.blobs[l_query].shape
        n_targets = len(ll_targets)

        l_cat = "MlpScorer_%s_%s_cat" % (self.name, prefix)
        l_tile_query = "MlpScorer_%s_%s_tile_query" % (self.name, prefix)
        l_sum = "MlpScorer_%s_%s_sum" % (self.name, prefix)
        l_relu = "MlpScorer_%s_%s_relu" % (self.name, prefix)
        l_ip = "MlpScorer_%s_%s_ip" % (self.name, prefix)
        l_label = "MlpScorer_%s_%s_label" % (self.name, prefix)
        l_loss = "MlpScorer_%s_%s_loss" % (self.name, prefix)

        p_ip = ["MlpScorer_%s_weight", "MlpScorer_%s_bias"]

        for l_target in ll_targets:
            net.blobs[l_target].reshape((batch_size, 1, n_dims))
        net.blobs[l_query].reshape((batch_size, 1, n_dims))
        net.f(Concat(l_cat, axis=1, bottoms=ll_targets)) # Put target and distractor scene ("referent") in same blob. Assumption: Horizontal concatenation.
        net.f(Tile(l_tile_query, tiles=n_targets, axis=1, bottoms=[l_query])) # Make the shape of the description embedding match the shape of the scenes embedding.
        net.f(Eltwise(l_sum, "SUM", bottoms=[l_tile_query, l_cat]))
        net.f(ReLU(l_relu, bottoms=[l_sum]))
        net.f(InnerProduct(l_ip, 1, bottoms=[l_relu], axis=2, param_names=p_ip)) # l_ip -> Inner Product.
        net.blobs[l_ip].reshape((batch_size, n_targets))
        net.f(NumpyData(l_label, labels))    #              v--Row vector with all values 0.
        loss = net.f(SoftmaxWithLoss(l_loss, bottoms=[l_ip, l_label]))

        denominators = logsumexp(net.blobs[l_ip].data, axis=1) # Denominator of equation (3).
        chosen_logprobs = net.blobs[l_ip].data[range(batch_size), labels.astype(int)]
        chosen_logprobs -= denominators

        predictions = np.argmax(net.blobs[l_ip].data, axis=1)
        accs = predictions == labels

        return chosen_logprobs, accs

# Referent encoder (section 3.2)
class LinearSceneEncoder(object):
    def __init__(self, name, apollo_net, config):
        self.name = name
        self.apollo_net = apollo_net
        self.config = config

    def forward(self, prefix, scenes, dropout):
        net = self.apollo_net

        # What type of data do we have? Is it abstract scenes...?
        if isinstance(scenes[0], Scene):
            feature_data = np.zeros((len(scenes), N_PROP_TYPES * N_PROP_OBJECTS)) # For the final evaluation, this was an array of size ``100 x 280´´ (c.f.: section 4.4).
            #                                                                       Because 100 samples were used.
            for i_scene, scene in enumerate(scenes):
                for prop in scene.props:
                    #            v--0..99 v-- 0..7 (sky object s ... toy object t) v--35            v--specifies the exact object (=png image) of the given type.
                    feature_data[i_scene, prop.type_index *                        N_PROP_OBJECTS + prop.object_index] = 1
                    # INTERPRETATION:
                    #   The number of ones ``1´´ corresponds to the number of features used. The position of the ones ``1´´ corresponds to which exact feature (i.e., png-feature image) was used.
                    #   Thus, EACH scene or referent is represented as 1x280 row vector.  This IS the feature representation f(r) --> c.f. section 3.1 in the paper.
        else:
            # ...or rather birds?
            assert isinstance(scenes[0], Bird)
            feature_data = np.zeros((len(scenes), len(scenes[0].features)))
            for i_scene, scene in enumerate(scenes):
                feature_data[i_scene, :] = scenes[i_scene].features # "features" == feature representation f(r) of one referent (i.e., of one image).
                                                                    # --> Set during load_scenes() in corpus.py

        l_data = "LinearSceneEncoder%s_%s_data" % (self.name, prefix)
        l_drop = "LinearSceneEncoder%s_%s_drop" % (self.name, prefix)
        l_ip1 = "LinearSceneEncoder%s_%s_ip1" % (self.name, prefix)
        l_relu1 = "LinearSceneEncoder%s_%s_relu1" % (self.name, prefix)
        l_ip2 = "LinearSceneEncoder%s_%s_ip2" % (self.name, prefix)

        p_ip1 = ["LinearSceneEncoder%s_ip1_weight" % self.name,
                 "LinearSceneEncoder%s_ip1_bias" % self.name]
        p_ip2 = ["LinearSceneEncoder%s_ip2_weight" % self.name,
                 "LinearSceneEncoder%s_ip2_bias" % self.name]

        #               v-- name (passes a string as label)
        #               v       v-- the actual data
        net.f(NumpyData(l_data, feature_data))
        # ``InnerProduct´´ is a fully connected layer. ``nn.Linear´´ is the PyTorch equivalent. The following command creates the referent encoding E_r --> c.f. equation (1) in section 3.2
        net.f(InnerProduct(
                l_ip1, self.config.hidden_size, bottoms=[l_data], # ``bottoms´´ is the input data to this layer.
                param_names=p_ip1))

        return l_ip1

# Description encoder (section 3.2)
class LinearStringEncoder(object):
    def __init__(self, name, apollo_net, config):
        self.name = name
        self.apollo_net = apollo_net
        self.config = config

    def forward(self, prefix, scenes, dropout):
        """
            scenes : Description, i.e., sentence, to be encoded.
        """
        net = self.apollo_net

        # For the final experiment, ``feature_data´´ takes the following format: 100 x 1063.
        #                        v--100     , v--1063
        feature_data = np.zeros((len(scenes), len(WORD_INDEX)))
        for i_scene, scene in enumerate(scenes):
            for word in scene.description:          # "description" attribute comes from the "Scenes" namedtuple from corpus.py
                feature_data[i_scene, word] += 1    # ``feature_data´´ contains integer numbers >= 0 (i.e., 0, 1, 2, ...)

        l_data = "LinearStringEncoder_%s_%s_data" % (self.name, prefix)
        l_ip = "LinearStringEncoder_%s_%s_ip" % (self.name, prefix)

        p_ip = ["LinearStringEncoder_%s_ip_weight" % self.name,
                "LinearStringEncoder_%s_ip_bias" % self.name]

        net.f(NumpyData(l_data, feature_data))
        net.f(InnerProduct(
                l_ip, self.config.hidden_size, bottoms=[l_data],
                param_names=p_ip))

        return l_ip

class BowSceneEncoder(object):
    def __init__(self, name, apollo_net, config):
        self.name = name
        self.apollo_net = apollo_net
        self.config = config

    def forward(self, prefix, scenes, dropout):
        net = self.apollo_net

        max_props = max(len(scene.props) for scene in scenes)
        type_data = np.zeros((len(scenes), max_props))
        object_data = np.zeros((len(scenes), max_props))
        feature_data = np.zeros((len(scenes), max_props, 4))
        for i_scene, scene in enumerate(scenes):
            offset = max_props - len(scene.props)
            for i_prop, prop in enumerate(scene.props):
                type_data[i_scene, offset+i_prop] = prop.type_index
                object_data[i_scene, offset+i_prop] = prop.object_index
                feature_data[i_scene, offset+i_prop, :] = \
                        [prop.x, prop.y, prop.z, prop.flip]

        l_type = "BowSceneEncoder_%s_%s_type_%d"
        l_object = "BowSceneEncoder_%s_%s_object_%d"
        l_feature = "BowSceneEncoder_%s_%s_feature_%d"
        l_type_vec = "BowSceneEncoder_%s_%s_type_vec_%d"
        l_object_vec = "BowSceneEncoder_%s_%s_object_vec_%d"
        l_prop = "BowSceneEncoder_%s_%s_prop_%d"
        l_all_props = "BowSceneEncoder_%s_%s_all_props" % (self.name, prefix)
        l_drop = "BowSceneEncoder_%s_%s_drop" % (self.name, prefix)
        l_ip1 = "BowSceneEncoder_%s_%s_ip1" % (self.name, prefix)
        l_relu1 = "BowSceneEncoder_%s_%s_relu1" % (self.name, prefix)
        l_ip2 = "BowSceneEncoder_%s_%s_ip2" % (self.name, prefix)

        p_type_vec = ["BowSceneEncoder_%s_type_vec" % self.name]
        p_object_vec = ["BowSceneEncoder_%s_object_vec" % self.name]
        p_ip1 = ["BowSceneEncoder_%s_ip1_weight" % self.name, 
                 "BowSceneEncoder_%s_ip1_bias" % self.name]
        p_ip2 = ["BowSceneEncoder_%s_ip2_weight" % self.name, 
                 "BowSceneEncoder_%s_ip2_bias" % self.name]

        for i_step in range(max_props):
            l_type_i = l_type % (self.name, prefix, i_step)
            l_object_i = l_object % (self.name, prefix, i_step)
            l_feature_i = l_feature % (self.name, prefix, i_step)
            l_type_vec_i = l_type_vec % (self.name, prefix, i_step)
            l_object_vec_i = l_object_vec % (self.name, prefix, i_step)
            l_prop_i = l_prop % (self.name, prefix, i_step)

            net.f(NumpyData(l_type_i, type_data[:, i_step]))
            net.f(NumpyData(l_object_i, object_data[:, i_step]))
            net.f(NumpyData(l_feature_i, feature_data[:, i_step]))
            net.f(Wordvec(
                    l_type_vec_i, self.config.prop_embedding_size, N_PROP_TYPES,
                    bottoms=[l_type_i], param_names=p_type_vec))
            net.f(Wordvec(
                    l_object_vec_i, self.config.prop_embedding_size, N_PROP_OBJECTS,
                    bottoms=[l_object_i], param_names=p_object_vec))
            net.f(Concat(
                    l_prop_i, 
                    bottoms=[l_type_vec_i, l_object_vec_i, l_feature_i]))

        net.f(Eltwise(
                l_all_props, "SUM", 
                bottoms=[l_prop % (self.name, prefix, i_step)
                         for i_step in range(max_props)]))
        if dropout:
            net.f(Dropout(l_drop, 0.5, bottoms=[l_all_props]))
            l_p = l_drop
        else:
            l_p = l_all_props

        net.f(InnerProduct(
                l_ip1, self.config.hidden_size, bottoms=[l_p], 
                param_names=p_ip1))

        return l_ip1

class BowStringEncoder(object):
    def __init__(self, name, apollo_net, config):
        self.name = name
        self.apollo_net = apollo_net
        self.config = config

    def forward(self, prefix, scenes, dropout):
        net = self.apollo_net
        
        max_words = max(len(scene.description) for scene in scenes)
        word_data = np.zeros((len(scenes), max_words))
        for i_scene, scene in enumerate(scenes):
            offset = max_words - len(scene.description)
            for i_word, word in enumerate(scene.description):
                word_data[i_scene, offset+i_word] = word

        l_word = "BowStringEncoder_%s_%s_word_%d"
        l_word_vec = "BowStringEncoder_%s_%s_word_vec_%d"
        l_all_words = "BowStringEncoder_%s_%s_all_words" % (self.name, prefix)
        l_drop = "BowStringEncoder_%s_%s_drop" % (self.name, prefix)
        l_ip1 = "BowStringEncoder_%s_%s_ip1" % (self.name, prefix)
        l_relu1 = "BowStringEncoder_%s_%s_relu1" % (self.name, prefix)
        l_ip2 = "BowStringEncoder_%s_%s_ip2" % (self.name, prefix)

        p_word_vec = ["BowStringEncoder_%s_word_vec" % self.name]
        p_ip1 = ["BowStringEncoder_%s_ip1_weight" % self.name, 
                 "BowStringEncoder_%s_ip1_param" % self.name]
        p_ip2 = ["BowStringEncoder_%s_ip2_weight" % self.name, 
                 "BowStringEncoder_%s_ip2_param" % self.name]

        for i_step in range(max_words):
            l_word_i = l_word % (self.name, prefix, i_step)
            l_word_vec_i = l_word_vec % (self.name, prefix, i_step)

            net.f(NumpyData(l_word_i, word_data[:, i_step]))
            net.f(Wordvec(
                    l_word_vec_i, self.config.word_embedding_size, 
                    len(WORD_INDEX), bottoms=[l_word_i], param_names=p_word_vec))

        net.f(Eltwise(
                l_all_words, "MAX", 
                bottoms=[l_word_vec % (self.name, prefix, i_step)
                         for i_step in range(max_words)]))

        if dropout:
            net.f(Dropout(l_drop, 0.5, bottoms=[l_all_words]))
            l_r = l_drop
        else:
            l_r = l_all_words


        net.f(InnerProduct(
                l_ip1, self.config.hidden_size, bottoms=[l_r], 
                param_names=p_ip1))
        net.f(ReLU(l_relu1, bottoms=[l_ip1]))
        net.f(InnerProduct(
                l_ip2, self.config.hidden_size, bottoms=[l_relu1], 
                param_names=p_ip2))

        return l_ip2

# The referent describer D takes an image encoding and outputs a description using a (feedforward) conditional neural language model.
class MlpStringDecoder(object):
    def __init__(self, name, apollo_net, config):
        self.name = name
        self.apollo_net = apollo_net
        self.config = config

    def forward(self, prefix, encoding, scenes, dropout):
        """
        encoding: Is the linear embedding tensor of the targets ``scenes´´ (equation (1)).
        scenes  : Target scenes.
        """
        net = self.apollo_net

        max_words = max(len(scene.description) for scene in scenes)
        history_features = np.zeros((len(scenes), max_words, len(WORD_INDEX)))
        last_features = np.zeros((len(scenes), max_words, len(WORD_INDEX)))
        targets = np.zeros((len(scenes), max_words))
        for i_scene, scene in enumerate(scenes): # For each scene, parse its description and...
            #                             v--Example: description=[1, 2, 248, 11, 295, 14]
            for i_word, word in enumerate(scene.description): # ...for every single word within the description...
                if word == 0:
                    continue #       v--Starting position always changes (by one index ahead).
                for ii_word in range(i_word + 1, len(scene.description)): #...look at vocabulary index of the words starting from the second till last position.
                #                                                          That is, we look at progressively smaller parts of the description as the index ii_word...
                #                                                          keeps moving further and further towards the right boundary of the description.
                    history_features[i_scene, ii_word, word] += 1
                last_features[i_scene, i_word, word] += 1
                targets[i_scene, i_word] = word

        l_history_data = "MlpStringDecoder_%s_%s_history_data_%d"
        l_last_data = "MlpStringDecoder_%s_%s_last_data_%d"
        l_cat_features = "MlpStringDecoder_%s_%s_cat_features_%d"
        l_ip1 = "MlpStringDecoder_%s_%s_ip1_%d"
        l_cat = "MlpStringDecoder_%s_%s_cat_%d"
        l_relu1 = "MlpStringDecoder_%s_%s_relu1_%d"
        l_ip2 = "MlpStringDecoder_%s_%s_ip2_%d"
        l_target = "MlpStringDecoder_%s_%s_target_%d"
        l_loss = "MlpStringDecoder_%s_%s_loss_%d"

        p_ip1 = ["MlpStringDecoder_%s_ip1_weight" % self.name,
                 "MlpStringDecoder_%s_ip1_bias" % self.name] 
        p_ip2 = ["MlpStringDecoder_%s_ip2_weight" % self.name,
                 "MlpStringDecoder_%s_ip2_bias" % self.name] 

        loss = 0
        for i_step in range(1, max_words):
            l_history_data_i = l_history_data % (self.name, prefix, i_step)
            l_last_data_i = l_last_data % (self.name, prefix, i_step)
            l_cat_features_i = l_cat_features % (self.name, prefix, i_step)
            l_ip1_i = l_ip1 % (self.name, prefix, i_step)
            l_cat_i = l_cat % (self.name, prefix, i_step)
            l_relu1_i = l_relu1 % (self.name, prefix, i_step)
            l_ip2_i = l_ip2 % (self.name, prefix, i_step)
            l_target_i = l_target % (self.name, prefix, i_step)
            l_loss_i = l_loss % (self.name, prefix, i_step)

            net.f(NumpyData(l_history_data_i, history_features[:,i_step-1,:]))
            net.f(NumpyData(l_last_data_i, last_features[:,i_step-1,:]))
            net.f(Concat(l_cat_features_i, bottoms=[l_history_data_i, l_last_data_i]))
            net.f(InnerProduct(l_ip1_i, self.config.hidden_size, bottoms=[l_cat_features_i], param_names=p_ip1)) # (W7 @ [dn, d<n, ei])
            net.f(Concat(l_cat_i, bottoms=[l_ip1_i, encoding])) # Taking the referent embedding of the target scene into account.
            net.f(ReLU(l_relu1_i, bottoms=[l_cat_i]))
            net.f(InnerProduct(l_ip2_i, len(WORD_INDEX), bottoms=[l_relu1_i],param_names=p_ip2))                 # W6 @ ReLU(...)
            net.f(NumpyData(l_target_i, targets[:,i_step]))     # This is the target...
            loss += net.f(SoftmaxWithLoss(                      # ...here, the loss of the prediction vs. target is computed!
                l_loss_i, bottoms=[l_ip2_i, l_target_i], 
                ignore_label=0, normalize=False))

        return -np.asarray(loss)

    """
    Draws a sequence of words, i.e., samples d_1, ..., d_n ~ p_S0(.|r_i)
    
    Remark: The paper does not specify, how the sampling actually works.
            This seems to be an "implementational detail".
    """
    def sample(self, prefix, encoding, viterbi):
        """
        encoding: Encoding of abstract scene image, i.e., referent encoding.
        viterbi : Decoding scheme:
                    ``False´´ corresponds to greedy sampling: Randomly sample index of one word from the vocabulary.
                    ``True´´ corresponds to pure sampling (non-deterministic; truly random).
                    Note: Set to False in the pragmatic speaker S1 (SamplingSpeaker1).
        """
        net = self.apollo_net

        max_words = 20
        batch_size = net.blobs[encoding].shape[0]

        out_logprobs = np.zeros((batch_size,))
        samples = np.zeros((batch_size, max_words))
        history_features = np.zeros((batch_size, len(WORD_INDEX)))
        last_features = np.zeros((batch_size, len(WORD_INDEX)))
        samples[:,0] = WORD_INDEX["<s>"]
        last_features[:,WORD_INDEX["<s>"]] += 1

        l_history_data = "MlpStringDecoder_%s_%s_history_data_%d"
        l_last_data = "MlpStringDecoder_%s_%s_last_data_%d"
        l_cat_features = "MlpStringDecoder_%s_%s_cat_features_%d"
        l_ip1 = "MlpStringDecoder_%s_%s_ip1_%d"
        l_cat = "MlpStringDecoder_%s_%s_cat_%d"
        l_relu1 = "MlpStringDecoder_%s_%s_relu1_%d"
        l_ip2 = "MlpStringDecoder_%s_%s_ip2_%d"
        l_softmax = "MlpStringDecoder_%s_%s_softmax_%d"

        p_ip1 = ["MlpStringDecoder_%s_ip1_weight" % self.name,
                 "MlpStringDecoder_%s_ip1_bias" % self.name] 
        p_ip2 = ["MlpStringDecoder_%s_ip2_weight" % self.name,
                 "MlpStringDecoder_%s_ip2_bias" % self.name] 

        # Compute sampling distribution p_S0:
        for i_step in range(1, max_words):
            l_history_data_i = l_history_data % (self.name, prefix, i_step)
            l_last_data_i = l_last_data % (self.name, prefix, i_step)
            l_cat_features_i = l_cat_features % (self.name, prefix, i_step)
            l_ip1_i = l_ip1 % (self.name, prefix, i_step)
            l_cat_i = l_cat % (self.name, prefix, i_step)
            l_relu1_i = l_relu1 % (self.name, prefix, i_step)
            l_ip2_i = l_ip2 % (self.name, prefix, i_step)
            l_softmax_i = l_softmax % (self.name, prefix, i_step)

            net.f(DummyData(l_history_data_i, (1,1,1,1))) # Initialise tensor with random values.
            net.blobs[l_history_data_i].reshape(history_features.shape)
            net.f(DummyData(l_last_data_i, (1,1,1,1)))    # Initialise tensor with random values.
            net.blobs[l_last_data_i].reshape(last_features.shape)
            net.blobs[l_history_data_i].data[...] = history_features
            net.blobs[l_last_data_i].data[...] = last_features
            net.f(Concat(l_cat_features_i, bottoms=[l_history_data_i, l_last_data_i]))
            net.f(InnerProduct(
                l_ip1_i, self.config.hidden_size, bottoms=[l_cat_features_i],
                param_names=p_ip1))
            net.f(Concat(l_cat_i, bottoms=[l_ip1_i, encoding]))
            net.f(ReLU(l_relu1_i, bottoms=[l_cat_i]))
            net.f(InnerProduct(
                l_ip2_i, len(WORD_INDEX), bottoms=[l_relu1_i],
                param_names=p_ip2))
            net.f(Softmax(l_softmax_i, bottoms=[l_ip2_i]))

            # Draw samples (i.e., words d_i) from probability distribution p_S0:
            probs = net.blobs[l_softmax_i].data
            history_features += last_features
            last_features[...] = 0
            for i_datum in range(batch_size):
                d_probs = probs[i_datum,:].astype(float)
                d_probs /= d_probs.sum()
                if viterbi:
                    choice = d_probs.argmax() # Find index of largest word-probability.
                else:
                    choice = np.random.multinomial(1, d_probs).argmax() # Randomly draw one sample, i.e., get the index of one single word from the vocabulary.
                samples[i_datum, i_step] = choice
                last_features[i_datum, choice] += 1
                out_logprobs[i_datum] += np.log(d_probs[choice])

        # This creates a candidate caption / sentence / description of the image e_r.
        out_samples = []
        for i in range(samples.shape[0]):
            this_sample = []
            for j in range(samples.shape[1]):
                word = WORD_INDEX.get(samples[i,j])
                #this_sample.append(word)
                this_sample.append(samples[i,j])
                if word == "</s>":
                    break
            if this_sample[-1] != WORD_INDEX["</s>"]:
                this_sample.append(WORD_INDEX["</s>"])
            out_samples.append(this_sample)
        return out_logprobs, out_samples

class LstmStringDecoder(object):
    def __init__(self, name, apollo_net, config):
        self.name = name
        self.apollo_net = apollo_net
        self.config = config

    def forward(self, prefix, encoding, scenes, dropout):
        net = self.apollo_net

        max_words = max(len(scene.description) for scene in scenes)
        word_data = np.zeros((len(scenes), max_words))
        for i_scene, scene in enumerate(scenes):
            offset = max_words - len(scene.description)
            for i_word, word in enumerate(scene.description):
                #word_data[i_scene, offset+i_word] = word
                word_data[i_scene, i_word] = word

        l_seed = "LstmStringDecoder_%s_%s_seed" % (self.name, prefix)
        l_prev_word = "LstmStringDecoder_%s_%s_word_%d"
        l_word_vec = "LstmStringDecoder_%s_%s_word_vec_%d"
        l_drop_in = "LstmStringDecoder_%s_%s_drop_in_%d"
        l_concat = "LstmStringDecoder_%s_%s_concat_%d"
        l_lstm = "LstmStringDecoder_%s_%s_lstm_%d"
        l_hidden = "LstmStringDecoder_%s_%s_hidden_%d"
        l_mem = "LstmStringDecoder_%s_%s_mem_%d"
        l_drop_out = "LstmStringDecoder_%s_%s_drop_out_%d"
        l_output = "LstmStringDecoder_%s_%s_output_%d"
        l_target = "LstmStringDecoder_%s_%s_target_%d"
        l_loss = "LstmStringDecoder_%s_%s_loss_%d"

        p_word_vec = ["LstmStringDecoder_%s_word_vec" % self.name]
        p_lstm = ["LstmStringDecoder_%s_lstm_iv" % self.name, 
                  "LstmStringDecoder_%s_lstm_ig" % self.name,
                  "LstmStringDecoder_%s_lstm_fg" % self.name, 
                  "LstmStringDecoder_%s_lstm_og" % self.name]
        p_output = ["LstmStringDecoder_%s_output_weight" % self.name, 
                    "LstmStringDecoder_%s_output_bias" % self.name]

        loss = 0
        net.f(NumpyData(
                l_seed, np.zeros((len(scenes), self.config.hidden_size))))
        for i_step in range(1, max_words):
            l_prev_word_i = l_prev_word % (self.name, prefix, i_step)
            l_word_vec_i = l_word_vec % (self.name, prefix, i_step)
            l_drop_in_i = l_drop_in % (self.name, prefix, i_step)
            l_concat_i = l_concat % (self.name, prefix, i_step)
            l_lstm_i = l_lstm % (self.name, prefix, i_step)
            l_hidden_i = l_hidden % (self.name, prefix, i_step)
            l_mem_i = l_mem % (self.name, prefix, i_step)
            l_drop_out_i = l_drop_out % (self.name, prefix, i_step)
            l_output_i = l_output % (self.name, prefix, i_step)
            l_target_i = l_target % (self.name, prefix, i_step)
            l_loss_i = l_loss % (self.name, prefix, i_step)

            if i_step == 1:
                prev_hidden = l_seed
                prev_mem = l_seed
            else:
                prev_hidden = l_hidden % (self.name, prefix, i_step - 1)
                prev_mem = l_mem % (self.name, prefix, i_step - 1)

            net.f(NumpyData(l_prev_word_i, word_data[:, i_step-1]))
            net.f(Wordvec(
                    l_word_vec_i, self.config.word_embedding_size, 
                    len(WORD_INDEX), bottoms=[l_prev_word_i], 
                    param_names=p_word_vec))
            #if dropout:
            #    net.f(Dropout(l_drop_in_i, 0.5, bottoms=[l_word_vec_i]))
            #    l_wv = l_drop_in_i
            #else:
            #    l_wv = l_word_vec_i
            net.f(Concat(l_concat_i, bottoms=[prev_hidden, l_word_vec_i, encoding]))
            net.f(LstmUnit(
                    l_lstm_i, bottoms=[l_concat_i, prev_mem],
                    param_names=p_lstm, tops=[l_hidden_i, l_mem_i],
                    num_cells=self.config.hidden_size))
            net.f(NumpyData(l_target_i, word_data[:, i_step]))
            #if dropout:
            #    net.f(Dropout(l_drop_out_i, 0.5, bottoms=[l_hidden_i]))
            #    l_h = l_drop_out_i
            #else:
            #    l_h = l_hidden_i
            net.f(InnerProduct(
                    l_output_i, len(WORD_INDEX), bottoms=[l_hidden_i],
                    param_names=p_output))
            loss += net.f(SoftmaxWithLoss(
                    #l_loss_i, bottoms=[l_output_i, l_target_i]))
                    l_loss_i, ignore_label=0, bottoms=[l_output_i, l_target_i]))

        return np.asarray(loss)

    # TODO consolidate
    def sample(self, prefix, encoding, viterbi):
        net = self.apollo_net

        batch_size = net.blobs[encoding].shape[0]
        max_words = 20

        l_seed = "LstmStringDecoder_%s_%s_seed"
        l_prev_word = "LstmStringDecoder_%s_%s_word_%d"
        l_word_vec = "LstmStringDecoder_%s_%s_word_vec_%d"
        l_concat = "LstmStringDecoder_%s_%s_concat_%d"
        l_lstm = "LstmStringDecoder_%s_%s_lstm_%d"
        l_hidden = "LstmStringDecoder_%s_%s_hidden_%d"
        l_mem = "LstmStringDecoder_%s_%s_mem_%d"
        l_output = "LstmStringDecoder_%s_%s_output_%d"
        l_softmax = "LstmStringDecoder_%s_%s_softmax_%d"

        p_word_vec = ["LstmStringDecoder_%s_word_vec" % self.name]
        p_lstm = ["LstmStringDecoder_%s_lstm_iv" % self.name, 
                  "LstmStringDecoder_%s_lstm_ig" % self.name,
                  "LstmStringDecoder_%s_lstm_fg" % self.name, 
                  "LstmStringDecoder_%s_lstm_og" % self.name]
        p_output = ["LstmStringDecoder_%s_output_weight" % self.name, 
                    "LstmStringDecoder_%s_output_bias" % self.name]

        samples = np.zeros((batch_size, max_words), dtype=int)
        samples[:,0] = WORD_INDEX["<s>"]

        net.f(NumpyData(
                l_seed, np.zeros((batch_size, self.config.hidden_size))))
        for i_step in range(1, max_words):
            l_prev_word_i = l_prev_word % (self.name, prefix, i_step)
            l_word_vec_i = l_word_vec % (self.name, prefix, i_step)
            l_concat_i = l_concat % (self.name, prefix, i_step)
            l_lstm_i = l_lstm % (self.name, prefix, i_step)
            l_hidden_i = l_hidden % (self.name, prefix, i_step)
            l_mem_i = l_mem % (self.name, prefix, i_step)
            l_output_i = l_output % (self.name, prefix, i_step)
            l_softmax_i = l_softmax % (self.name, prefix, i_step)

            if i_step == 1:
                prev_hidden = l_seed
                prev_mem = l_seed
            else:
                prev_hidden = l_hidden % (self.name, prefix, i_step - 1)
                prev_mem = l_mem % (self.name, prefix, i_step - 1)

            net.f(NumpyData(l_prev_word_i, samples[:, i_step-1]))
            net.f(Wordvec(
                    l_word_vec_i, self.config.word_embedding_size, 
                    len(WORD_INDEX), bottoms=[l_prev_word_i], 
                    param_names=p_word_vec))
            net.f(Concat(
                    l_concat_i, bottoms=[prev_hidden, l_word_vec_i, encoding]))
            net.f(LstmUnit(
                    l_lstm_i, bottoms=[l_concat_i, prev_mem],
                    param_names=p_lstm, tops=[l_hidden_i, l_mem_i],
                    num_cells=self.config.hidden_size))
            net.f(InnerProduct(
                    l_output_i, len(WORD_INDEX), bottoms=[l_hidden_i],
                    param_names=p_output))
            net.f(Softmax(l_softmax_i, bottoms=[l_output_i]))

            choices = []
            for i in range(batch_size):
                probs = net.blobs[l_softmax_i].data[i,:].astype(np.float64)
                probs /= probs.sum()
                if viterbi:
                    choices.append(np.argmax(probs))
                else:
                    choices.append(np.random.choice(len(WORD_INDEX), p=probs))
            samples[:, i_step] = choices

        out_samples = []
        for i in range(samples.shape[0]):
            this_sample = []
            for j in range(samples.shape[1]):
                word = WORD_INDEX.get(samples[i,j])
                this_sample.append(samples[i,j])
                if word == "</s>":
                    break
            out_samples.append(this_sample)
        return out_samples
