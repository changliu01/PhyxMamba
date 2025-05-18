import torch
import torch.nn as nn
import numpy as np
from hflayers import Hopfield, HopfieldLayer
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F
import math

class HopfieldMemory(nn.Module):

    def __init__(self, d_model, hidden_size, num_heads, num_patterns, disable_out_projection=True, output_size=None, lookup_weights_as_separated=False):

        super().__init__()
        
        head_dim = hidden_size // num_heads
        assert head_dim * num_heads == hidden_size, "hidden_size must be divisible by num_heads"
        self.hopfield = HopfieldLayer(input_size=d_model, output_size=output_size, hidden_size=head_dim, num_heads=num_heads, quantity=num_patterns, disable_out_projection=disable_out_projection, lookup_weights_as_separated=lookup_weights_as_separated)

    def forward(self, x):

        return self.hopfield(x)
    

class MemoryNetwork(nn.Module):

    def __init__(self, memory_size, d_model, c_out):
        super().__init__()
        self.memory_size = memory_size
        self.memory_dim = d_model
        self.c_out = c_out

        self.memory_key = nn.Parameter(torch.randn(memory_size, d_model))

        self.memory_value = nn.Parameter(torch.randn(memory_size, d_model * c_out))

        self.query_projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        
        self.initialize_weights()

    def forward(self, x):
        '''
        x: (batch_size, pred_len, d_model)
        '''

        B, T, F = x.shape
        x_query = self.query_projection(x)
        attn_weights = torch.matmul(x_query, self.memory_key.transpose(0, 1)) / (self.memory_dim ** 0.5) # (B, T, memory_size)
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1) # (B, T, memory_size)
        residual_weights = torch.matmul(attn_weights, self.memory_value) # (B, T, memory_dim * c_out)
        x = x.repeat(1, 1, self.c_out).reshape(B, T, self.memory_dim * self.c_out)
        output = torch.mul(x, residual_weights)
        output = output.reshape(B, T, self.memory_dim, self.c_out)
        output = torch.sum(output, dim=-2) # (B, T, c_out)

        return output
    
    def initialize_weights(self):
        torch.nn.init.trunc_normal_(self.memory_value, std=0.02)
        torch.nn.init.trunc_normal_(self.memory_key, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


# class TemporalEncoding(nn.Module):

#     '''
#     Reference: https://github.com/ZijieH/CG-ODE/blob/main/lib/gnn_models.py
#     '''

#     def __init__(self, d_hidden):
        
#         super().__init__()
#         self.d_hid = d_hidden
        
#         self.div_term = torch.FloatTensor([1 / np.power(10000, 2 * (hid_j // 2) / self.d_hid) for hid_j in range(self.d_hid)]) #[20]
#         self.div_term = torch.reshape(self.div_term,(1,-1))
#         self.div_term = nn.Parameter(self.div_term, requires_grad=False)

#     def forward(self, t):

#         t = t.view(-1, 1)
#         t = t * 200
#         position_term = torch.matmul(t, self.div_term)
#         position_term[:,0::2] = torch.sin(position_term[:,0::2])
#         position_term[:,1::2] = torch.cos(position_term[:,1::2])

#         return position_term
    
class TemporalEncoding(nn.Module):

    def __init__(self, d_model: int=512, dropout=0.1,  max_seq_len: int=5000, batch_first: bool=True):
        super().__init__()
        self.d_model = d_model
        
        # self.dropout = nn.Dropout(p=dropout)

        self.batch_first = batch_first

        self.x_dim = 1 if batch_first else 0

        # position = torch.arange(max_seq_len).unsqueeze(1) * 200
        position = torch.arange(0, max_seq_len).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe = torch.zeros(max_seq_len, d_model)

        pe[:, 0::2] = torch.sin(position * div_term)
        
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x):
        
        x = x + self.pe[:x.size(self.x_dim)].unsqueeze(0)

        # return self.dropout(x)
        return x
        # 

class TransformerWithCLS(nn.Module):

    def __init__(self, input_dim, d_model, output_dim, n_hdeads=4, num_layers=3, max_seq_len=5000, batch_first=True):
        super().__init__()
        
        self.d_model = d_model
        self.input_proj = nn.Linear(input_dim, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.position_encoding = TemporalEncoding(d_model=d_model, max_seq_len=max_seq_len, batch_first=batch_first)
        encoder_layer = TransformerEncoderLayer(d_model, nhead=n_hdeads, dropout=0.1, batch_first=batch_first)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, output_dim)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.input_proj(x)  # (batch_size, T, d_model)
        cls_token = self.cls_token.expand(batch_size, -1, -1)  # (batch_size, 1, d_model)
        x = torch.cat([cls_token, x], dim=1)  # (batch_size, T+1, d_model)
        x = self.position_encoding(x)  # (batch_size, T+1, d_model)
        x = self.transformer_encoder(x)  # (batch_size, T+1, d_model)
        x = x[:, 0, :]  # (batch_size, d_model)
        # x = x.mean(dim=1)
        x = self.fc(x)  # (batch_size, output_dim)

        return x


class HopfieldContextEncoder(nn.Module):

    def __init__(self, c_in, d_model, context_dim, memory_hidden_dim, num_patterns):
        super().__init__()

        # self.feature_extractor = TransformerWithCLS(input_dim=c_in, d_model=d_model, n_hdeads=4, num_layers=3, output_dim=d_model)
        self.feature_extractor = nn.Linear(c_in, d_model)
        # self.feature_extractor = 
        self.hopfield_memory = MemoryNetwork(memory_size=num_patterns, d_model=d_model, c_out=context_dim)
        # self.hopfield_memory = HopfieldMemory(d_model=d_model, hidden_size=memory_hidden_dim, num_heads=4, num_patterns=num_patterns, output_size=context_dim, disable_out_projection=False, lookup_weights_as_separated=True)

    def forward(self, x):

        '''
        x: (B, T, n_variable)
        '''

        # x = self.test_mapping(x)
        # context_vector = x.mean(dim=1, keepdim=True) # (B, 1, d_model)
        # traj_features = context_vector
        traj_features = self.feature_extractor(x) # (B, context_dim)
        # traj_features = traj_features.mean(dim=1, keepdim=True) # (B, 1, context_dim)
        # traj_features = traj_features.unsqueeze(1) # (B, 1, context_dim)
        context_vector = self.hopfield_memory(traj_features) # (B, 1 context_dim)
        # context_vector = context_vector.squeeze(1) # (B, context_dim)
        context_vector = context_vector.mean(dim=1) # (B, 1, context_dim)

        return context_vector, traj_features.squeeze(1)


class HopfieldMemoryAttention(nn.Module):

    def __init__(self, d_model, hidden_size, num_heads, num_patterns, future_steps):
        super().__init__()
        self.d_model = d_model
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.future_steps = future_steps

        self.hopfieldmemory = HopfieldMemory(d_model=d_model, hidden_size=hidden_size, num_heads=num_heads, num_patterns=num_patterns)
        self.temporalencoding = TemporalEncoding(hidden_size)

        self.wq = nn.Linear(d_model, hidden_size, bias=False)
        self.wk = nn.Linear(hidden_size, hidden_size, bias=False)
        self.wv = nn.Linear(hidden_size, d_model, bias=False)

    def forward(self, x, last_time_embedding, stage):
        
        '''
        x: (B, T, d_model)
        last_time_embedding = (B, 1, d_model)
        '''
        if stage != 1:
            memory_hidden = self.hopfieldmemory(x) # (B, T, hidden_size)
            # positional_encoding = self.temporalencoding(self.t) # (T, hidden_size)
            # memory_hidden_primary = memory_hidden + positional_encoding.unsqueeze(0) # (B, T, hidden_size)
            memory_hidden_primary = self.temporalencoding(memory_hidden)

            memory_hidden = self.wk(memory_hidden_primary) # (B, T, hidden_size)

            last_time_embedding = self.wq(last_time_embedding) # (B, 1, d_model)

            weights = torch.matmul(last_time_embedding, memory_hidden.permute(0, 2, 1)) * (1/torch.sqrt(torch.tensor([self.hidden_size]))).float().to(x.device) # (B, 1, T)

            memory_hidden_v = self.wv(memory_hidden_primary) # (B, T, d_model)
            weights = F.softmax(weights, dim=-1)
            message = torch.bmm(weights, memory_hidden_v) # (B, 1, d_model)
            message = F.gelu(message)

            return message, memory_hidden_primary
        
        else:
            
            memory_hidden = self.hopfieldmemory(x) # (B, T, hidden_size)
            # positional_encoding = self.temporalencoding(self.t) # (T, hidden_size)

            # memory_hidden_primary = memory_hidden + positional_encoding.unsqueeze(0) # (B, T, hidden_size)
            memory_hidden_primary = self.temporalencoding(memory_hidden)

            return None, memory_hidden_primary


