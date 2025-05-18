import os
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba_ssm import Mamba2
from .layers.mamba_layers.Embedding_layers import DataEmbedding, PatchEmbedding
from .layers.mamba_layers.revin_layers import RevIN
from .layers.mamba_layers.hopfield_layers import HopfieldMemoryAttention
from .layers.mamba_layers.Decoder_layers import Decoder
from .layers.mamba_layers.Hierarchical_mamba_layer import MambaLayer4MTP, BareMambaLayer
from .layers.mamba_layers.MTP import MTPModule



class Model(nn.Module):

    def __init__(self, args):

        super().__init__()
        self.args = args
        self.max_context_length = args.lookback

        if not self.args.CD:
            c_in = 1
        else:
            c_in = args.variable_num

        if self.args.use_revin:
            self.revin = RevIN(num_features=c_in)
        
        if self.args.token_size == 1:
            self.embedding = DataEmbedding(c_in=c_in, d_model=args.d_model, kernel_size=args.data_emb_conv_kernel)
            self.lm_head = nn.Linear(args.d_model, c_in, bias=True)
            
        elif self.args.token_size > 1:
            self.embedding = PatchEmbedding(args=self.args, embedding_method=args.embedding_method, patch_model_method=args.patch_model_method, d_model=args.d_model, patch_len=args.token_size, stride=args.token_size, padding=0, variable_num=args.variable_num, token_size=args.token_size)
            self.decoder = Decoder(args)
           


        if self.args.hier_layers > 0:
            self.decoder_current = nn.Linear(args.d_model, args.d_model, bias=True)
            self.hier_mamba = nn.ModuleList(
                [MambaLayer4MTP(args, decoder_current=self.decoder_current) for _ in range(self.args.hier_layers)]
            )

        else:

            self.mamba = nn.ModuleList([BareMambaLayer(d_model=args.d_model, d_state=args.mamba_d_state, expand=args.mamba_expand, headdim=args.mamba_headdim) for _ in range(args.mamba_layers)])

            # self.mamba = Mamba2(d_model=args.d_model, d_state=args.mamba_d_state, expand=args.mamba_expand, headdim=args.mamba_headdim)

        self.mtp_modules = nn.ModuleList([
             MTPModule(args, self.embedding, self.decoder) for _ in range(self.args.mtp_steps)
        ])


    def forward(self, x):

        '''
        input_time_series: (B, T, n_variable)
        '''
        B, T, N = x.shape
        if self.args.token_size == 1:
            if not self.args.CD:
                x = x.permute(0, 2, 1) # (B, n_variable, T)
                x = x.reshape(B*N, T, 1) # (B*n_variable, T)

        if self.args.use_revin:
            x = self.revin(x, mode='norm') # (B, T, n_variable)
        elif self.args.use_norm:
            means = x.mean(dim=1, keepdim=True)
            x = x - means
            stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
            x = x / stdev

        if self.args.token_size == 1:
            embed_data = self.embedding(x) # (B, T, d_model) or (B*V, L, d_model)
        else:
            embed_data, _ = self.embedding(x) # (B, T, d_model) or (B*V, L, d_model)

        embed_data = embed_data.contiguous() # (B+, L, d_model)

        if self.args.hier_layers > 0:
            # output_hidden should go through the decoder for further outputs
            residual, output_hidden = embed_data, None
            for i in range(self.args.hier_layers):
                pred_current, pred_future = self.hier_mamba[i](residual)
                residual = residual - pred_current
                if output_hidden is None:
                    output_hidden = pred_future
                else:
                    output_hidden = output_hidden + pred_future

            x = self.decoder(output_hidden, N)

            #### used in training stage 1 only!
            if self.args.mtp_steps > 0 and self.training and self.stage == 1:
                seq_len = output_hidden.shape[1]
                mtp_output_list = []
                h_prev = output_hidden # (B, token_seq_len, d_model)
                for k, mtp_module in enumerate(self.mtp_modules):
                    
                    if seq_len - k - 1 <= 0:
                        break
                    h_prev_k = h_prev[:, :seq_len - k - 1, :]
                    emb_next = embed_data[:, k+1:, :]
                    mtp_output, h_k = mtp_module(h_prev_k, emb_next)
                    mtp_output_list.append(mtp_output)
                    h_prev = h_k

        else:
            # mamba_embedding = self.mamba(embed_data) # (B, T, d_model) or (B*V, L, d_model)
            for i in range(self.args.mamba_layers):
                if i == 0:
                    mamba_embedding = self.mamba[i](embed_data)
                else:
                    mamba_embedding = self.mamba[i](mamba_embedding)

            if self.args.token_size == 1:
                x = self.lm_head(mamba_embedding) # (B, T, n_variable) or (B*V, L, patch_len)
            else:
                x = self.decoder(mamba_embedding, N)

            if self.args.mtp_steps > 0 and self.training and self.stage == 1:
                seq_len = mamba_embedding.shape[1]
                mtp_output_list = []
                h_prev = mamba_embedding # (B, token_seq_len, d_model)
                for k, mtp_module in enumerate(self.mtp_modules):
                    if seq_len - k - 1 <= 0:
                        break
                    h_prev_k = h_prev[:, :seq_len - k - 1, :]
                    emb_next = embed_data[:, k+1:, :]
                    mtp_output, h_k = mtp_module(h_prev_k, emb_next)
                    mtp_output_list.append(mtp_output)
                    h_prev = h_k
                    

        if self.args.token_size == 1:
            if not self.args.CD:
                x = x.reshape(B*N, T).reshape(B, N, T) # (B, n_variable, T)
                x = x.permute(0, 2, 1)

        if self.args.use_revin:
            x = self.revin(x, mode='denorm')
        elif self.args.use_norm:
            x = x * (stdev[:, 0, :].unsqueeze(1).repeat(1, x.shape[1], 1))
            x = x + (means[:, 0, :].unsqueeze(1).repeat(1, x.shape[1], 1))

        if self.args.mtp_steps > 0 and self.training and self.stage == 1:
            if self.args.use_revin:
                mtp_output_list = [self.revin(mtp_output, mode='denorm') for mtp_output in mtp_output_list]
            elif self.args.use_norm:
                mtp_output_list = [mtp_output * (stdev[:, 0, :].unsqueeze(1).repeat(1, mtp_output.shape[1], 1)) + (means[:, 0, :].unsqueeze(1).repeat(1, mtp_output.shape[1], 1)) for mtp_output in mtp_output_list]
        
        if self.args.mtp_steps > 0 and self.training and self.stage == 1:

            return x, mtp_output_list
        else:
            return x
    
    def autoregression(self, source, future=1, token_size=1, long_term=False): ### not finish

        inference_steps = future // token_size
        dis = future - inference_steps * token_size
        if dis != 0:
            inference_steps += 1
        pred_y = []
        for j in range(inference_steps):
            if len(pred_y) != 0:
                if source.shape[1] < self.max_context_length:
                    source = torch.cat([source, pred_y[-1]], dim=1)
                else:
                    source = torch.cat([source[:, token_size:, :], pred_y[-1]], dim=1)
                
            outputs = self.forward(source)
        
            pred_y.append(outputs[:, -token_size:, :])
                
        pred_y = torch.cat(pred_y, dim=1)

        if dis != 0:
            pred_y = pred_y[:, :dis-token_size, :]

        return pred_y
        
        
        

