import torch
import torch.nn as nn

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
