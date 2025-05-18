import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers.mamba_layers.vq_functions import vq, vq_st

class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, kernel_size=3, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        if kernel_size == 0:
            self.value_embedding = nn.Linear(c_in, d_model)
        else:
            self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model, kernel_size=kernel_size)
            
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark=None):
        if x_mark is None:
            x = self.value_embedding(x) + self.position_embedding(x)
        else:
            x = self.value_embedding(
                x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        return self.dropout(x)
    
class DataEmbedding_inverted(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1):
        super(DataEmbedding_inverted, self).__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark=None):
        x = x.permute(0, 2, 1)
        # x: [Batch Variate Time]
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1))
        # x: [Batch Variate d_model]
        return self.dropout(x)
    
class PatchEmbedding(nn.Module):
    """
    ReparamModule, similar to https://arxiv.org/pdf/2211.14730
    """
    def __init__(self, args, embedding_method, patch_model_method, d_model, patch_len, stride, padding, variable_num, token_size, dropout=0.1):
        super(PatchEmbedding, self).__init__()
        # Patching
        self.args = args
        self.patch_len = patch_len
        self.stride = stride
        self.patch_model_method = patch_model_method
        self.embedding_method = embedding_method
        self.delay_emb_dim = args.delay_emb_dim
        self.delay_tau = args.delay_tau
        

        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))

        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space

        # self.value_embedding = nn.Linear(patch_len, d_model, bias=False)

        # Positional embedding
        self.position_embedding = PositionalEmbedding(d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

        self.patch_modeling = PatchModeling(embedding_method, patch_model_method, d_model, variable_num, token_size, delay_emb_dim=None if embedding_method == 'default' else self.delay_emb_dim)

    def forward(self, x):
        # do patching x: [B, T, V]
        batch_size, seq_len, n_vars = x.shape
        
        if self.embedding_method == 'default':
            x = x.permute(0, 2, 1) # x: [B, V, T]
            x = self.padding_patch_layer(x)
            x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride) # x: [B, V, L, patch_len]
        
        elif self.embedding_method == 'psr':
            
            pad_len = (self.delay_emb_dim - 1) * self.delay_tau
            pad_tuple = (0, 0, pad_len, 0, 0, 0)
            x = F.pad(x, pad_tuple, "constant", 0) # x: [B*V, L, m]
            x = PSR(x, self.delay_emb_dim, self.delay_tau, mode='indep') # x: [B*V, len_embedded, m]
            x = x.unfold(dimension=-2, size=self.patch_len, step=self.stride) # x: [B*V, L, m, patch_len]
            x = x.reshape(batch_size, n_vars, -1, self.delay_emb_dim*self.patch_len) # x: [B, V, L, m * patch_len]
        else:
            raise NotImplementedError(f"Embedding method {self.embedding_method} not implemented.")
        
        x = self.patch_modeling(x) 
        x = x + self.position_embedding(x) # [B*V, L, d_model] if self.patch_model_method == 'default' else [B, L, d_model]

        # if self.patch_model_method == 'default':
        #     x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3])) # x: [B * V, patch_len, patch_dim]
        # elif self.patch_model_method == 'flatten_linear':
            
        # Input encoding
        # x = self.value_embedding(x) + self.position_embedding(x)

        return self.dropout(x), n_vars
    

# class PSREmbedding(nn.Module):

#     def __init__(self, embedding_dim, tau):

def PSR(input_data, embedding_dim, delay, mode):

    '''
    input_data: [B, T, N]
    embedding_dim: m
    delay: tau

    '''

    batch_size, seq_length, input_channels = input_data.shape

    device = input_data.device
    len_embedded = seq_length - (embedding_dim - 1) * delay
    embedded_data = torch.zeros(
        batch_size, len_embedded, embedding_dim, input_channels, device=device
    )

    for i in range(embedding_dim):
        start_idx = i * delay
        end_idx = start_idx + len_embedded
        embedded_data[:, :, i, :] = input_data[:, start_idx:end_idx, :]

    if mode == "merged_seq":
        embedded_data = embedded_data.permute(0, 1, 3, 2).reshape(
            batch_size, len_embedded, -1
        ) # [B, L, V * m]
    elif mode == "merged":
        embedded_data = embedded_data.reshape(batch_size, len_embedded, -1)
        
    else:  # independent
        embedded_data = embedded_data.permute(0, 3, 1, 2).reshape(
            batch_size * input_channels, len_embedded, embedding_dim
        )  # [BC, T, D]

    return embedded_data


class PatchModeling(nn.Module):

    def __init__(self, embedding_method, patch_model_method, d_model, variable_num, token_size, delay_emb_dim=None):

        super().__init__()

        self.embedding_method = embedding_method
        self.patch_model_method = patch_model_method
        self.variable_num = variable_num
        self.token_size = token_size
        self.d_model = d_model
        self.delay_emb_dim = delay_emb_dim

        if self.embedding_method == 'default':
            self.expand = 1
        elif self.embedding_method == 'psr':
            self.expand = self.delay_emb_dim
        else:
            raise NotImplementedError(f"Embedding method {self.embedding_method} not implemented.")



        if self.patch_model_method == 'default':
            self.linear_mapping = nn.Linear(self.expand * token_size, d_model)
        elif self.patch_model_method == 'flatten_linear':
            self.linear_mapping = nn.Linear(self.expand * variable_num * token_size, d_model)
        else:
            raise NotImplementedError(f"Patch model method {self.patch_model_method} not implemented.")

    def forward(self, x):

        '''
        x: (B, V, L, D)
        '''

        if self.patch_model_method == 'default':
            x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3])) # x: [B * V, patch_len, patch_dim]
            x = self.linear_mapping(x) # (B * V, L, D)
        elif self.patch_model_method == 'flatten_linear':
            x = x.permute(0, 2, 1, 3) # (B, L, V, D)
            x = x.reshape(x.shape[0], x.shape[1], -1) # (B, L, V * D)
            x = self.linear_mapping(x) # (B, L, D)
        else:
            raise NotImplementedError(f"Patch model method {self.patch_model_method} not implemented.")

        # x = x.permute(0, 2, 1, 3) # (B, L, V, D)
        # x = x.reshape(x.shape[0], x.shape[1], -1) # (B, L, V * D)
        # x = self.linear_mapping(x) # (B, L, D)

        return x


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model, kernel_size=3):
        super(TokenEmbedding, self).__init__()
        
        assert kernel_size % 2 == 1
        padding = int((kernel_size-1)/2)
        # padding = 1 if torch.__version__ >= '1.5.0' else 2
        # self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
        #                            kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=kernel_size, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        '''
        x: B, T, N
        # B: batch_size;    
        # T: seq_len;       
        # N: number of features/variate
        '''
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x
    
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()
        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(
            self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x
    
class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()
    
class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h': 4, 't': 5, 's': 6,
                    'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)
    
class EnvEmbedding(nn.Module):
    def __init__(self, K, D):
        super().__init__()
        # codebook
        self.embedding = nn.Embedding(K, D)
        self.embedding.weight.data.uniform_(-1./K, 1./K)

    def forward(self, x): # x [b, l, h_d]
        x_ = x.contiguous()
        latents = vq(x_, self.embedding.weight)
        return latents
    
    def straight_through(self, z_e_x):# x [b, h_d]
        '''
        z_e_x: the latent vectors for environments
        '''
        z_e_x_ = z_e_x.contiguous()
        # get the feature from the codebook and its index
        z_q_x_, indices = vq_st(z_e_x_, self.embedding.weight.detach()) # z_q_x_: [b, h_d]    indices:[b]
        z_q_x = z_q_x_.contiguous()

        z_q_x_bar_flatten = torch.index_select(self.embedding.weight, dim=0, index=indices)
        z_q_x_bar_ = z_q_x_bar_flatten.view_as(z_e_x_)
        z_q_x_bar = z_q_x_bar_.contiguous()

        return z_q_x, z_q_x_bar, indices

    def straight_through_test(self, z_e_x):# the index is soft
        inputs = z_e_x.contiguous()
        codebook = self.embedding.weight.detach()

        with torch.no_grad():
            embedding_size = codebook.size(1)
            inputs_flatten = inputs.view(-1, embedding_size)
            codebook_sqr = torch.sum(codebook ** 2, dim=1)
            inputs_sqr = torch.sum(inputs_flatten ** 2, dim=1, keepdim=True)
            # Compute the distances to the codebook
            distances = torch.addmm(codebook_sqr + inputs_sqr, inputs_flatten, codebook.t(), alpha=-2.0, beta=1.0) 
            # get the index
            indices = torch.softmax(distances, dim=1)    
            # compute the env vector
            codes_flatten = torch.mm(indices, codebook)
            codes = codes_flatten.view_as(inputs)

            return codes.contiguous(), None, indices
        
class AveragePoolingLayer(nn.Module):

    def __init__(self, d_model, activation='tanh'):

        super().__init__()

        self.d_model = d_model
        self.coef_map = nn.Linear(self.d_model, self.d_model, bias=True)
        self.ps = PositionalEmbedding(d_model=self.d_model)
        self.activation = nn.Tanh() if activation == 'tanh' else nn.GELU()

    def forward(self, x):

        '''
        x: (B, L, D)
        '''
        x = x + self.ps(x) # (B,L,D)

        coef_vec = self.activation(self.coef_map(x.mean(dim=1))).unsqueeze(1) # (B,1,D)

        coef = torch.matmul(coef_vec, x.transpose(1, 2)).transpose(1, 2) # (B,L,1)

        output = self.activation(coef * x) # (B,L,D)

        return output.mean(dim=1) # (B,D)

