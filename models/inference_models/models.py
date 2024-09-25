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
    def __init__(self, input_size, hidden_size, output_size, L, lr, output_activation_function = torch.relu):
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
    
    def train(self, data, optimizer, epochs, verbose = False):
        for e in range(epochs):
            optimizer.zero_grad()
            H_L = self.encode(data.x.float(), data.graph_list)
            loss = self.recon_loss(H_L, data.graph_list[-1].edge_index)
            if verbose:
                print(f'epoch {e+1} | loss {loss.item()}', end = '\r')
            loss.backward()
            optimizer.step()
        print('\n')
        return