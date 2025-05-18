import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba_ssm import Mamba2

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        """
        RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)
    
class MTPModule(nn.Module):

    def __init__(self, args, embedding_model, decoder_model):

        super().__init__()
        self.args = args
        self.embedding = embedding_model
        self.decoder = decoder_model
        self.d_model = args.d_model
        self.mamba = Mamba2(d_model=args.d_model, d_state=args.mamba_d_state, expand=args.mamba_expand, headdim=args.mamba_headdim)
        self.proj = nn.Linear(2 * self.d_model, self.d_model)
        self.norm = RMSNorm(self.d_model)

    def forward(self, h_prev, emb_next):

        # h_prev: [batch_size, seq_len - depth, d_model]
        # next_tokens: [batch_size, seq_len - depth]
        
        # emb_next = self.embedding(next_tokens)[:, :h_prev.shape[1], :]  # [batch_size, seq_len - depth, d_model]

        h_prev_norm = self.norm(h_prev)
        emb_next_norm = self.norm(emb_next)
        combined = torch.cat([h_prev_norm, emb_next_norm], dim=-1)  # [batch_size, seq_len - depth, 2*d_model]
        h_prime = self.proj(combined)  # [batch_size, seq_len - depth, d_model]
        h_k = self.mamba(h_prime)

        output = self.decoder(h_k, self.args.variable_num)

        return output, h_k
    



