'''
File that contain the models that will be tested
'''

import torch.nn.functional as F
import torch.nn
from torch_geometric.nn import GCNConv

class RGCN_layer(torch.nn.Module):
    '''
    Class to implement the Relational Graph Convolutional Network (RGCN). This Relational module is used for every graph generated via rewiring
    '''
    def __init__(self, input_size, output_size, L):
        super(RGCN_layer, self).__init__()
        self.L = L
        self.input_size = input_size
        self.output_size = output_size
        self.parameter_list = torch.nn.ParameterList()

        # Rewiring layer (relational layer)
        self.rewiring_sublayers = list()
        for i in range(self.L):
            sublayer = GCNConv(self.input_size, self.output_size)
            self.rewiring_sublayers.append(sublayer)
            for param in self.rewiring_sublayers[i].parameters():
                self.parameter_list.append(param)

    def forward(self, x, rewiring_graph_list):
        # Computing the sum for every generated graph
        out = torch.zeros((x.shape[0], self.output_size))
        for i in range(self.L):
            _x = self.rewiring_sublayers[i](x, rewiring_graph_list[i].edge_index)
            out += _x
        return out

class RGCN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, L, output_activation_function = torch.relu):
        super(RGCN, self).__init__()
        self.layer1 = RGCN_layer(input_size, hidden_size, L)
        self.layer2 = RGCN_layer(hidden_size, output_size, L)
        self.output_activation_function = output_activation_function
        self.L = L
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

    def forward(self, x, rewiring_graph_list):
        x = self.layer1(x, rewiring_graph_list)
        x = torch.relu(x)
        x = self.layer2(x, rewiring_graph_list)
        return x
    
class ExtendedRGCN(torch.nn.Module):
    def __init__(self, base_model, add_layer_dim, output_activation_function = torch.softmax):
        super(ExtendedRGCN, self).__init__()
        self.base_model = base_model
        # self.added_layer = RGCN_layer(base_model.output_size, add_layer_dim, base_model.L)
        # self.added_layer = GCNConv(base_model.output_size, add_layer_dim)
        self.added_layer = torch.nn.Linear(base_model.output_size, add_layer_dim)
        self.output_activation_function = output_activation_function

    def forward(self, x, rewiring_graph_list):
        x = self.base_model(x, rewiring_graph_list)
        # x = self.output_activation_function(self.added_layer(x, rewiring_graph_list), dim = 1)
        # x = self.output_activation_function(self.added_layer(x, rewiring_graph_list[0].edge_index), dim = 1)
        x = self.output_activation_function(self.added_layer(x), dim = 1)

        return x