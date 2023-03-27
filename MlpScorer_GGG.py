import torch
import torch.nn as nn
import numpy as np

class MlpScorer_GGG(nn.Module):
    def __init__(self, name, config):
        super(MlpScorer_GGG, self).__init__()
        self.name = name
        self.config = config
        
        n_dims = 100 #self.config.embedding_size
        n_targets = 2#self.config.num_scenes
        
        # define layers
        self.cat_layer = nn.Linear(2*n_dims, n_targets*n_dims)
        #self.tile_query_layer = nn.Linear(n_dims, n_targets*n_dims) # WRONG
        #self.sum_layer = nn.Linear(2*n_dims, n_targets*n_dims) # WRONG
        self.relu_layer = nn.ReLU()
        self.ip_layer = nn.Linear(n_targets*n_dims, n_targets)
        self.softmax_layer = nn.Softmax(dim=1)
        
    def forward(self, prefix, l_query, ll_targets, labels):
        """
            l_query: Embedding of a description (i.e., caption) of shape torch.Size([100, 100])
            ll_targets: Embedding of Scenes (target and distractor). It is a list of tensors 2 tensors each of shape torch.Size([100, 100])
            prefix : Not used --> Passed to forward() only with an empty string.
        """
        batch_size, n_dims = l_query.shape
        n_targets = len(ll_targets)

        cat_input = torch.cat(ll_targets, dim=1) # Horizontal concatenation
        #cat_output = self.cat_layer(cat_input) # WRONG
        print("ll_targets:\n", ll_targets) # DEBUG
        print("cat_input.shape: ", cat_input.shape)   # DEBUG
        #print("cat_output:\n", cat_output) # DEBUG
        
        #                                           v-- 2
        tile_query_output = torch.tile(l_query, (1, n_targets)) #self.tile_query_layer(l_query).repeat(1, n_targets)
        print("tile_query_output:\n", tile_query_output)
        print("tile_query_output.shape: ", tile_query_output.shape)
        
        #sum_input = torch.cat((cat_output, tile_query_output), dim=1) # WRONG
        #sum_output = self.sum_layer(sum_input) # WRONG! Should be element-wise sum.
        
        elementwise_sum = torch.add(tile_query_output, cat_input)
        print("elementwise_sum:\n", elementwise_sum)
        
        relu_output = self.relu_layer(elementwise_sum)
        
        ip_output = self.ip_layer(relu_output)
        
        #loss = nn.functional.cross_entropy(ip_output, labels) # Probably not needed.
        
        denominators = torch.logsumexp(ip_output, dim=1) # Denominator of equation (3).
        chosen_logprobs = ip_output[range(batch_size), labels]
        chosen_logprobs -= denominators

        predictions = torch.argmax(ip_output, dim=1)
        accs = predictions == labels

        return chosen_logprobs, accs
