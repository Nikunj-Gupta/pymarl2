import torch.nn as nn
import torch.nn.functional as F
import torch as th
import numpy as np
import torch.nn.init as init
from utils.th_utils import orthogonal_init_
from torch.nn import LayerNorm

from modules.agents.gtn import GTN 
from torch_geometric.data import Data as gData
from torch_geometric.data import Batch

class GNN(nn.Module):
    def __init__(self, in_channels=1024, out_channels=1024, num_nodes=None):
        super().__init__()
        self.N = num_nodes 
        self.feature_norm = nn.LayerNorm(in_channels)
        self.gnn = GTN(
                       num_edge=3, 
                       num_channels=2, 
                       w_in=in_channels, 
                       w_out=out_channels, 
                       num_nodes=self.N, 
                       num_layers=1
                    )  

    def get_edge_index(self, type="star"): # need an initial graph construction 
        if type == "line": 
            edges = [[i, i + 1] for i in range(self.N - 1)]  # # arrange agents in a line 
        elif type == "full": 
            edges = [[(j, i + j + 1) for i in range(self.N - j - 1)] for j in range(self.N - 1)]
            edges = [e for l in edges for e in l] 
        elif type == 'cycle':    # arrange agents in a circle
            edges = [(i, i + 1) for i in range(self.N - 1)] + [(self.N - 1, 0)] 
        elif type == 'star':     # arrange all agents in a star around agent 0
            edges = [(0, i + 1) for i in range(self.N - 1)] 
        edge_index = th.tensor(edges).T # # arrange agents in a line     
        return edge_index.cuda() 

    def get_adjacency_matrix(self): 
        A = [] 
        for t in ["line", "cycle", "star"]: 
            edges = self.get_edge_index(type=t) 
            value_tmp = th.ones(edges.shape[1]).type(th.cuda.FloatTensor) 
            A.append((edges, value_tmp)) 
        return A 

    def forward(self, x):        
        x = self.feature_norm(x) 
        A = self.get_adjacency_matrix() 
        eval = not self.training 
        x, H, Ws = self.gnn(A, x, num_nodes=self.N, eval=eval) 
        return x 

class GTNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(GTNAgent, self).__init__()
        self.args = args
        self.agent = args.agent 

        print(f"Using {self.agent} Agent") 

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.gnn = GNN(args.rnn_hidden_dim, args.rnn_hidden_dim, num_nodes=args.n_agents) 
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

        inputs = inputs.view(-1, e)
        x = F.relu(self.fc1(inputs), inplace=True)
        x = self.gnn(x) 
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        hh = self.rnn(x, h_in)

        if getattr(self.args, "use_layer_norm", False):
            q = self.fc2(self.layer_norm(hh))
        else:
            q = self.fc2(hh)

        return q.view(b, a, -1), hh.view(b, a, -1)