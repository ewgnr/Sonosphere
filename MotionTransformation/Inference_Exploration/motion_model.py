import torch
from torch import nn
import math
from collections import OrderedDict

"""
model architecture
"""

config = {
    "seq_length": 32,
    "data_dim": 308,
    "latent_dim": 8,
    "rnn_layer_count": 2,
    "rnn_layer_size": 512,
    "dense_layer_sizes": [512],
    "device": "cuda",
    "weights_path": ["results/weights/transformer_encoder_weights_epoch_200", "results/weights/transformer_decoder_weights_epoch_200"]
    }

class Encoder(nn.Module):
    def __init__(self, sequence_length, data_dim, latent_dim, rnn_layer_count, rnn_layer_size, dense_layer_sizes):
        super(Encoder, self).__init__()
        
        self.sequence_length = sequence_length
        self.data_dim = data_dim
        self.latent_dim = latent_dim
        self.rnn_layer_count = rnn_layer_count
        self.rnn_layer_size = rnn_layer_size 
        self.dense_layer_sizes = dense_layer_sizes
    
        # create recurrent layers
        rnn_layers = []
        rnn_layers.append(("encoder_rnn_0", nn.LSTM(self.data_dim, self.rnn_layer_size, self.rnn_layer_count, batch_first=True)))
        
        self.rnn_layers = nn.Sequential(OrderedDict(rnn_layers))
        
        # create dense layers
        
        dense_layers = []
        
        dense_layers.append(("encoder_dense_0", nn.Linear(self.rnn_layer_size, self.dense_layer_sizes[0])))
        dense_layers.append(("encoder_dense_relu_0", nn.ReLU()))
        
        dense_layer_count = len(self.dense_layer_sizes)
        for layer_index in range(1, dense_layer_count):
            dense_layers.append(("encoder_dense_{}".format(layer_index), nn.Linear(self.dense_layer_sizes[layer_index-1], self.dense_layer_sizes[layer_index])))
            dense_layers.append(("encoder_dense_relu_{}".format(layer_index), nn.ReLU()))

        self.dense_layers = nn.Sequential(OrderedDict(dense_layers))
        
        # create final dense layers
            
        self.fc_mu = nn.Linear(self.dense_layer_sizes[-1], self.latent_dim)
        self.fc_std = nn.Linear(self.dense_layer_sizes[-1], self.latent_dim)
        
    def forward(self, x):
        
        #print("x 1 ", x.shape)
        
        x, (_, _) = self.rnn_layers(x)
        
        #print("x 2 ", x.shape)
        
        x = x[:, -1, :] # only last time step 
        
        #print("x 3 ", x.shape)
        
        x = self.dense_layers(x)
        
        #print("x ", x.shape)
        
        mu = self.fc_mu(x)
        std = self.fc_std(x)
 
        #print("mu s ", mu.shape, " lvar s ", log_var.shape)
    
        return mu, std
    
    @staticmethod
    def reparameterize(mu, std):
        z = mu + std*torch.randn_like(std)
        return z
    
class Decoder(nn.Module):
    def __init__(self, sequence_length, data_dim, latent_dim, rnn_layer_count, rnn_layer_size, dense_layer_sizes):
        super(Decoder, self).__init__()
        
        self.sequence_length = sequence_length
        self.data_dim = data_dim
        self.latent_dim = latent_dim
        self.rnn_layer_size = rnn_layer_size
        self.rnn_layer_count = rnn_layer_count
        self.dense_layer_sizes = dense_layer_sizes

        # create dense layers
        dense_layers = []
        
        dense_layers.append(("decoder_dense_0", nn.Linear(latent_dim, self.dense_layer_sizes[0])))
        dense_layers.append(("decoder_relu_0", nn.ReLU()))

        dense_layer_count = len(self.dense_layer_sizes)
        for layer_index in range(1, dense_layer_count):
            dense_layers.append(("decoder_dense_{}".format(layer_index), nn.Linear(self.dense_layer_sizes[layer_index-1], self.dense_layer_sizes[layer_index])))
            dense_layers.append(("decoder_dense_relu_{}".format(layer_index), nn.ReLU()))
 
        self.dense_layers = nn.Sequential(OrderedDict(dense_layers))
        
        # create rnn layers
        rnn_layers = []

        rnn_layers.append(("decoder_rnn_0", nn.LSTM(self.dense_layer_sizes[-1], self.rnn_layer_size, self.rnn_layer_count, batch_first=True)))
        
        self.rnn_layers = nn.Sequential(OrderedDict(rnn_layers))
        
        # final output dense layer
        final_layers = []
        
        final_layers.append(("decoder_dense_{}".format(dense_layer_count), nn.Linear(self.rnn_layer_size, self.data_dim)))
        
        self.final_layers = nn.Sequential(OrderedDict(final_layers))
        
    def forward(self, x):
        #print("x 1 ", x.size())
        
        # dense layers
        x = self.dense_layers(x)
        #print("x 2 ", x.size())
        
        # repeat vector
        x = torch.unsqueeze(x, dim=1)
        x = x.repeat(1, self.sequence_length, 1)
        #print("x 3 ", x.size())
        
        # rnn layers
        x, (_, _) = self.rnn_layers(x)
        #print("x 4 ", x.size())
        
        # final time distributed dense layer
        x_reshaped = x.contiguous().view(-1, self.rnn_layer_size)  # (batch_size * sequence, input_size)
        #print("x 5 ", x_reshaped.size())
        
        yhat = self.final_layers(x_reshaped)
        #print("yhat 1 ", yhat.size())
        
        yhat = yhat.contiguous().view(-1, self.sequence_length, self.data_dim)
        #print("yhat 2 ", yhat.size())

        return yhat
    
    
def createModels(config):
    
    encoder = Encoder(config["seq_length"], config["data_dim"], config["latent_dim"], config["rnn_layer_count"], config["rnn_layer_size"], config["dense_layer_sizes"]).to(config["device"])
    
    ae_dense_layer_sizes_reversed = config["dense_layer_sizes"].copy()
    ae_dense_layer_sizes_reversed.reverse()

    decoder = Decoder(config["seq_length"], config["data_dim"], config["latent_dim"], config["rnn_layer_count"], config["rnn_layer_size"], ae_dense_layer_sizes_reversed).to(config["device"])

    if len(config["weights_path"]) > 0 and config["weights_path"][0] != "":
        
        if config["device"] == "cuda":
            encoder.load_state_dict(torch.load(config["weights_path"][0]))
        else:
            encoder.load_state_dict(torch.load(config["weights_path"][0], map_location=torch.device('cpu')))
    
    if len(config["weights_path"]) > 1 and config["weights_path"][1] != "":

        if config["device"] == "cuda":        
            decoder.load_state_dict(torch.load(config["weights_path"][1]))
        else:
            decoder.load_state_dict(torch.load(config["weights_path"][1], map_location=torch.device('cpu')))

    encoder.eval()
    decoder.eval()
        
    return encoder, decoder