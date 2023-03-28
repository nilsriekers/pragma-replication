import torch
import torch.nn as nn
import numpy as np

class MlpScorer_GGG(nn.Module):
    def __init__(self, name, config):
        super(MlpScorer_GGG, self).__init__()
        self.name = name
        self.config = config
        
        n_dims = 100
        n_targets = 2
        
        # define layers
        self.relu_layer = nn.ReLU()
        self.ip_layer = nn.Linear(in_features = n_targets*n_dims, out_features = n_targets, bias = True)
        self.softmax_layer = nn.Softmax(dim = 1)
        
    def forward(self, prefix, l_query, ll_targets, labels):
        """
            l_query: Embedding of a description (i.e., caption) of shape torch.Size([100, 100])
            ll_targets: Embedding of Scenes (target and distractor). It is a list of tensors 2 tensors each of shape torch.Size([100, 100])
            prefix : Not used --> Passed to forward() only with an empty string.
        """
        batch_size, n_dims = l_query.shape
        n_targets = len(ll_targets)

        cat_output = torch.cat(ll_targets, dim=1) # Horizontal concatenation
        print("ll_targets:\n", ll_targets) # DEBUG
        print("cat_output.shape: ", cat_output.shape)   # DEBUG
        
        #                                           v-- 2
        tile_query_output = torch.tile(l_query, (1, n_targets)) # Horizontally repeat twice (i.e., n_targets times)
        print("tile_query_output:\n", tile_query_output)
        print("tile_query_output.shape: ", tile_query_output.shape)
        
        elementwise_sum = torch.add(tile_query_output, cat_output)
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
