import torch.nn as nn
import torch.nn.functional as F
import torch as th
import numpy as np
import torch.nn.init as init
from utils.th_utils import orthogonal_init_
from torch.nn import LayerNorm

from torch_geometric.nn import GCNConv, GATConv, GATv2Conv 
from torch_geometric.data import Data as gData
from torch_geometric.data import Batch

class GNN(nn.Module):
    def __init__(self, in_channels=1024, out_channels=1024, agent=None, cg_edges="full"):
        super().__init__()
        self.cg_edges=cg_edges
        if agent == "gcn": self.conv1 = GCNConv(in_channels, out_channels) 
        if agent == "gat": self.conv1 = GATConv(in_channels, out_channels) 
        if agent == "gatv2": self.conv1 = GATv2Conv(in_channels, out_channels) 

    def forward(self, x):
        batch_size, num_nodes, fc_dim = x.shape
        edge_index_list = self.build_edge_index(batch_size, num_nodes, type=self.cg_edges)
        graph = self.build_graph_batch(x, edge_index_list)
        x = self.conv1(graph.x, graph.edge_index)
        x = F.relu(x)
        return x

    def build_edge_index(self, batch_size, num_nodes, type):
        if type == "line": 
            edges = [[i, i + 1] for i in range(num_nodes - 1)]  # # arrange agents in a line 
        elif type == "full": 
            edges = [[(j, i + j + 1) for i in range(num_nodes - j - 1)] for j in range(num_nodes - 1)]
            edges = [e for l in edges for e in l] 
        elif type == 'cycle':    # arrange agents in a circle
            edges = [(i, i + 1) for i in range(num_nodes - 1)] + [(num_nodes - 1, 0)] 
        elif type == 'star':     # arrange all agents in a star around agent 0
            edges = [(0, i + 1) for i in range(num_nodes - 1)] 
        edge_index = th.tensor(edges).T.cuda() # # arrange agents in a line 
        edge_index_list = [edge_index for _ in range(batch_size)]
        return edge_index_list

    def build_graph_batch(self, x_list,  edge_index_list):
        data_list = []
        for graph_nodes, ei in zip(x_list, edge_index_list):
            data_list.append(gData(x=graph_nodes, edge_index=ei))
        graph = Batch.from_data_list(data_list)
        return graph

class GNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(GNNAgent, self).__init__()
        self.args = args
        self.cg_edges = args.cg_edges 
        self.agent = args.agent 

        print(f"Using {self.agent} Agent and cg_edges={self.cg_edges}") 

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.gnn = GNN(args.rnn_hidden_dim, args.rnn_hidden_dim, agent=self.agent, cg_edges=self.cg_edges) 
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

        if getattr(args, "use_layer_norm", False):
            self.layer_norm = LayerNorm(args.rnn_hidden_dim)
        
        if getattr(args, "use_orthogonal", False):
            orthogonal_init_(self.fc1)
            orthogonal_init_(self.fc2, gain=args.gain)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        b, a, e = inputs.size()

        # inputs = inputs.view(-1, e)
        x = F.relu(self.fc1(inputs), inplace=True)
        x = self.gnn(x) 
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        hh = self.rnn(x, h_in)

        if getattr(self.args, "use_layer_norm", False):
            q = self.fc2(self.layer_norm(hh))
        else:
            q = self.fc2(hh)

        return q.view(b, a, -1), hh.view(b, a, -1)