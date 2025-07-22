import torch
from torch import nn
from collections import OrderedDict
import math

"""
model architecture
"""

config = {
    "input_length": 64,
    "data_dim": 308,
    "node_dim": 512,
    "layer_count": 2,
    "device": "cuda",
    "weights_path": "results/weights/rnn_weights_epoch_400"
    }

class Reccurent(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, layer_count):
        super(Reccurent, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layer_count = layer_count
        self.output_dim = output_dim
            
        rnn_layers = []
        
        rnn_layers.append(("rnn", nn.LSTM(self.input_dim, self.hidden_dim, self.layer_count, batch_first=True)))
        self.rnn_layers = nn.Sequential(OrderedDict(rnn_layers))
        
        dense_layers = []
        dense_layers.append(("dense", nn.Linear(self.hidden_dim, self.output_dim)))
        self.dense_layers = nn.Sequential(OrderedDict(dense_layers))
    
    def forward(self, x):
        x, (_, _) = self.rnn_layers(x)
        
        x = x[:, -1, :] # only last time step 
        x = self.dense_layers(x)
        
        return x
    
def createModel(config):
    
    rnn = Reccurent(config["data_dim"], config["node_dim"], config["data_dim"], config["layer_count"]).to( config["device"])

    if config["weights_path"] != "":
        
        if config["device"] == 'cuda':
            rnn.load_state_dict(torch.load(config["weights_path"]))
        else:
            rnn.load_state_dict(torch.load(config["weights_path"], map_location=torch.device(config["device"] )))

    rnn.eval()
        
    return rnn
