import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .mamba_layers.norm_layers import RMSNorm

class Encoder(nn.Module):

    def __init__(self, mamba_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.encode_layer = nn.ModuleList(mamba_layers)
        self.norm = norm_layer

    def forward(self, x):
        # x [B, L, N]
        for layer in self.encode_layer:
            x = layer(x)
        if self.norm is not None:
            x = self.norm(x)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, mamba_layer, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.mamba_layer = mamba_layer
        self.rmsnorm = RMSNorm(d_model)
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x):
        x = self.rmsnorm(x)
        new_x = self.mamba_layer(x)
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm2(x + y)
    
class InvariantDecoder(nn.Module):

    def __init__(self, decoder_layers, norm_layer=None, projection=None):

        super().__init__()
        self.decode_layers = nn.ModuleList(decoder_layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, query, x_mask=None, cross_mask=None, tau=None, delta=None):

        for layer in self.decode_layers:
            x = layer(x, query, x_mask=x_mask, cross_mask=cross_mask, tau=tau, delta=delta)

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)

        return x
    
class InvariantDecoderLayer(nn.Module):

    def __init__(self, cross_attention, d_model, d_ff=None, dropout=0.1, activation="relu"):

        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, query, x_mask=None, cross_mask=None, tau=None, delta=None):

        B, L, D = query.shape
        x = x + self.dropout(self.cross_attention(query, x, x, attn_mask=x_mask, tau=tau, delta=delta)[0])
        x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(x.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm2(x+y)

class HierarchicalEncoder(nn.Module):

    def __init__(self, mamba_layers, norm_layer=None):
        super(HierarchicalEncoder, self).__init__()
        self.encode_layer = nn.ModuleList(mamba_layers)
        self.norm = norm_layer

    def forward(self, x):
        for layer in self.encode_layer:
            x = layer(x)
        if self.norm is not None:
            x = self.norm(x)
        return x
    
