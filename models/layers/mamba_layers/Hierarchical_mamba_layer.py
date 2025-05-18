import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba_ssm import Mamba2


class MambaLayer(nn.Module):

    def __init__(self, args, decoder_current, decoder_future):

        super().__init__()

        self.args = args
        self.variable_num = args.variable_num
        self.mamba = Mamba2(d_model=args.d_model, d_state=args.mamba_d_state, expand=args.mamba_expand, headdim=args.mamba_headdim)
        self.decoder_current = decoder_current
        self.decoder_future = decoder_future


    def forward(self, embed_data):

        '''
        embed_data: [batch_size, seq_len (len embedding), d_model]
        '''

        embed_data = self.mamba(embed_data) # (batch_size, seq_len, d_model)

        pred_current = self.decoder_current(embed_data)
        pred_future = self.decoder_future(embed_data, self.variable_num)

        return pred_current, pred_future
    

class MambaLayer4MTP(nn.Module):

    def __init__(self, args, decoder_current):

        super().__init__()

        self.args = args
        self.variable_num = args.variable_num
        self.mamba = Mamba2(d_model=args.d_model, d_state=args.mamba_d_state, expand=args.mamba_expand, headdim=args.mamba_headdim)
        self.decoder_current = decoder_current


    def forward(self, embed_data):

        '''
        embed_data: [batch_size, seq_len (len embedding), d_model]
        '''

        embed_data = self.mamba(embed_data) # (batch_size, seq_len, d_model)

        pred_current = self.decoder_current(embed_data)
        output_hidden = embed_data

        return pred_current, output_hidden


class BareMambaLayer(nn.Module):

    def __init__(self, d_model, d_state, expand, headdim):

        super().__init__()

        self.mamba = Mamba2(d_model=d_model, d_state=d_state, expand=expand, headdim=headdim)

    def forward(self, embed_data):

        '''
        embed_data: [batch_size, seq_len (len embedding), d_model]
        '''
        embed_data = self.mamba(embed_data) # (batch_size, seq_len, d_model)
        return embed_data