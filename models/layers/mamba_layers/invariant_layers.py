import numpy as np
import torch
import torch.nn as nn
from models.layers.mamba_layers.resnet import resnet34
import torch.nn.functional as F
from einops import rearrange, reduce

class skip_embed_final_shallow(torch.nn.Module):
    def __init__(self, input_dim, latent_dim, output_dim, norm_layer):
        super().__init__()
        self.embed = nn.Sequential(
                               nn.Linear(input_dim, output_dim),
                               norm_layer(output_dim),
                               nn.ReLU(inplace = True),
                               )
        self.skip = nn.Linear(input_dim, output_dim)

    def forward(self, input):
        return self.embed(input) + self.skip(input)

class InvariantNet(torch.nn.Module):
    def __init__(self, embed_dim, args = None):
        super().__init__()
            
        self.encoder = resnet34(num_classes = embed_dim, norm_layer = nn.BatchNorm2d)
        self.proj_head = skip_embed_final_shallow(512, 512, embed_dim, nn.BatchNorm1d)

    def forward(self, traj):

        # traj: (B, T, variables)
        traj = traj.permute(0, 2, 1)[:, None, :, :]
        x0, x1, x2, x3, x4, x = self.encoder(traj)
        embed = self.proj_head(x)
        return F.normalize(embed, dim = -1) # (B, embed_dim)

class InvariantNetV2(nn.Module):

    def __init__(self, input_dims, output_dims, kernels):

        super().__init__()
        self.kernels = kernels
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.EnvEncoder = nn.ModuleList([nn.Conv1d(input_dims, output_dims, k, padding=k-1) for k in kernels])

    def forward(self, x):

        '''
        x: (B, D, T) --> (B, T, D)
        '''

        env_rep = []
        for idx, mod in enumerate(self.EnvEncoder):
            out = mod(x)
            if self.kernels[idx] != 1:
                out = out[..., :-(self.kernels[idx] - 1)]
            env_rep.append(out.transpose(1, 2))  # b t d
        env_rep = reduce(
            rearrange(env_rep, 'list b t d -> list b t d'),
            'list b t d -> b t d', 'mean'
            )
        
        return env_rep



