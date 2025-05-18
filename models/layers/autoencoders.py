import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers.SelfAttention_Family_i import FullAttention, AttentionLayer
import math

class PositionalEncoder(nn.Module):
    def __init__(self, dropout=0.1,  max_seq_len: int=5000, d_model: int=512, batch_first: bool=True):
        super().__init__()
        self.d_model = d_model
        
        self.dropout = nn.Dropout(p=dropout)

        self.batch_first = batch_first

        self.x_dim = 1 if batch_first else 0

        position = torch.arange(max_seq_len).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe = torch.zeros(max_seq_len, d_model)

        pe[:, 0::2] = torch.sin(position * div_term)
        
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # print(x.shape)
        # print(self.pe.shape)
        x = x + self.pe.squeeze(1)[:x.size(self.x_dim)]

        return self.dropout(x)

class Variational_Encoder(nn.Module):

    def __init__(self, args):

        super().__init__()

        self.args = args
        self.input_embedding = nn.Linear(self.args.embed_dim, self.args.chronos_hidden_size)
        self.positional_encoding = PositionalEncoder(d_model=self.args.chronos_hidden_size, dropout=self.args.dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model = self.args.chronos_hidden_size, nhead=self.args.n_heads, batch_first = True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers = self.args.n_layers)

        self.mean_net = nn.Linear(self.args.chronos_hidden_size, self.args.chronos_hidden_size)
        self.std_net = nn.Linear(self.args.chronos_hidden_size, self.args.chronos_hidden_size)
    
    def forward(self, x):

        '''
        x: (batch_size aka. batch_size * variable_num, seq_len, embed_dim)
        '''

        x = self.input_embedding(x)
        x = self.positional_encoding(x)

        x = self.encoder(x)

        mean = self.mean_net(x)
        std = self.std_net(x)

        return mean, std

class Variational_Decoder(nn.Module):

    def __init__(self, args):

        super().__init__()

        self.args = args
        self.input_embedding = nn.Linear(self.args.chronos_hidden_size, self.args.embed_dim)
        self.positional_encoding = PositionalEncoder(d_model=self.args.embed_dim, dropout=self.args.dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model = self.args.embed_dim, nhead=self.args.n_heads, batch_first = True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers = self.args.n_layers)
        
        # self.mean_net = nn.Linear(self.args.embed_dim, self.args.embed_dim)
        # self.std_net = nn.Linear(self.args.embed_dim, self.args.embed_dim)
        
        self.output_layer = nn.Linear(self.args.embed_dim, self.args.embed_dim)
    
    def forward(self, x):

        '''
        x: (batch_size aka. batch_size * variable_num, seq_len, chronos_hidden_size)
        '''

        x = self.input_embedding(x)
        x = self.positional_encoding(x)

        x = self.encoder(x)

        x = self.output_layer(x)

        return x


def loss_false_parallel(code_batch: torch.Tensor) -> torch.Tensor:
    """
    并行处理多个批次的 False-Nearest-Neighbor 正则化损失函数。

    Parameters
    ----------
    code_batch : torch.Tensor
        一个形状为 (Batch, B, L) 的张量，表示多个批次的编码输入。
    k : int, optional
        用于计算邻域的最近邻数量，默认为 1。

    Returns
    -------
    torch.Tensor
        一个标量张量，表示平均后的正则化损失。
    """
    
    # Ensure code_batch is a 3D tensor
    if code_batch.dim() != 3:
        raise ValueError(f"code_batch must be a 3D tensor, got {code_batch.dim()}D tensor instead.")
    
    Batch, B, L = code_batch.size()

    k = max(1, torch.ceil(torch.tensor([0.01 * B])).int().item())

    # Regularization parameters (fixed as per Kennel et al. 1992)
    rtol = 20.0
    atol = 2.0

    # Create a lower triangular mask including the diagonal
    # Shape: (1, L, L) to broadcast over Batch
    tri_mask = torch.tril(torch.ones((L, L), dtype=torch.float32, device=code_batch.device)).unsqueeze(0)  # (1, L, L)

    # Expand tri_mask to (Batch, L, L)
    tri_mask = tri_mask.expand(Batch, L, L)  # (Batch, L, L)

    # Apply the mask to the code_batch
    # code_batch: (Batch, B, L) -> (Batch, 1, B, L)
    # tri_mask: (Batch, L, L) -> (Batch, L, 1, L)
    # batch_masked: (Batch, L, B, L)
    batch_masked = tri_mask.unsqueeze(2) * code_batch.unsqueeze(1)  # (Batch, L, B, L)

    # Compute squared norms along the last dimension
    # X_sq: (Batch, L, B, 1)
    X_sq = torch.sum(batch_masked ** 2, dim=-1, keepdim=True)  # (Batch, L, B, 1)

    # Compute pairwise distance matrix
    # pdist_vector: (Batch, L, B, B)
    pdist_vector = (
        X_sq
        + X_sq.transpose(-2, -1)
        - 2 * torch.matmul(batch_masked, batch_masked.transpose(-2, -1))
    )  # (Batch, L, B, B)

    all_dists = pdist_vector  # (Batch, L, B, B)

    # Compute all_ra as per Kennel et al.
    # range_tensor: (1, L)
    range_tensor = 1.0 / torch.arange(1, L + 1, dtype=torch.float32, device=code_batch.device).view(1, L)  # (1, L)

    # Compute standard deviation along B dimension
    # std_batch_masked: (Batch, L, 1, L)
    std_batch_masked = torch.std(batch_masked, dim=2, keepdim=True, unbiased=False)  # (Batch, L, 1, L)

    # Compute sum of squared standard deviations
    # sum_sq_std: (Batch, L, 1)
    sum_sq_std = torch.sum(std_batch_masked ** 2, dim=-1)  # (Batch, L, 1)

    # Squeeze to remove the singleton dimension
    sum_sq_std = sum_sq_std.squeeze(-1)  # (Batch, L)

    # Compute all_ra
    # all_ra: (Batch, L)
    all_ra = torch.sqrt(range_tensor * sum_sq_std)  # (Batch, L)

    # Avoid singularity by clipping distances
    # Clamp all_dists to [1e-14, max(all_dists)]
    max_dists = torch.max(all_dists)
    all_dists = torch.clamp(all_dists, min=torch.tensor(1e-14).to(all_dists.device), max=max_dists)

    # Find the indices of the k+1 smallest distances (including self)
    # Using torch.topk on the negated distances to get smallest k+1
    # inds: (Batch, L, B, k+1)
    _, inds = torch.topk(-all_dists, k=k + 1, dim=-1, largest=True, sorted=True)  # (Batch, L, B, k+1)

    # Gather the corresponding distances
    # neighbor_dists_d: (Batch, L, B, k+1)
    neighbor_dists_d = torch.gather(all_dists, dim=-1, index=inds)  # (Batch, L, B, k+1)

    # Shift all_dists and inds by one to compute neighbor_new_dists
    # all_dists_shifted: (Batch, L-1, B, B)
    all_dists_shifted = all_dists[:, 1:, :, :]  # (Batch, L-1, B, B)
    # inds_shifted: (Batch, L-1, B, k+1)
    inds_shifted = inds[:, :-1, :, :]          # (Batch, L-1, B, k+1)

    # Gather the new neighbor distances
    # neighbor_new_dists: (Batch, L-1, B, k+1)
    neighbor_new_dists = torch.gather(all_dists_shifted, dim=-1, index=inds_shifted)  # (Batch, L-1, B, k+1)

    # Compute scaled distances as per Equation 4 of Kennel et al.
    # scaled_dist: (Batch, L-1, B, k+1)
    scaled_dist = torch.sqrt(
        (neighbor_new_dists - neighbor_dists_d[:, :-1, :, :]) / neighbor_dists_d[:, :-1, :, :]
    )  # (Batch, L-1, B, k+1)

    # Apply Kennel conditions
    # Condition 1: scaled_dist > rtol
    is_false_change = scaled_dist > rtol  # (Batch, L-1, B, k+1)

    # Condition 2: neighbor_new_dists > atol * all_ra[:-1]
    # all_ra_shifted: (Batch, L-1, 1, 1) for broadcasting
    all_ra_shifted = all_ra[:, :-1].unsqueeze(-1).unsqueeze(-1)  # (Batch, L-1, 1, 1)
    is_large_jump = neighbor_new_dists > (atol * all_ra_shifted)  # (Batch, L-1, B, k+1)

    # Combine both conditions
    is_false_neighbor = is_false_change | is_large_jump  # (Batch, L-1, B, k+1)

    # Exclude the first neighbor (typically the point itself) and convert to integer
    # total_false_neighbors: (Batch, L-1, B, k)
    total_false_neighbors = is_false_neighbor[..., 1:k + 1].int()  # (Batch, L-1, B, k)

    # Compute regularization weights
    # reg_weights: 1 - mean over (B, k), shape: (Batch, L-1)
    reg_weights = 1.0 - torch.mean(total_false_neighbors.float(), dim=(2, 3))  # (Batch, L-1)

    # Pad with zero at the beginning to match the latent dimension
    # reg_weights: (Batch, L)
    reg_weights = F.pad(reg_weights, (1, 0), "constant", 0)  # (Batch, L)

    # Compute the average batch activity using L2 norm
    activations_batch_averaged = torch.sqrt(torch.mean(code_batch ** 2, dim=1))  # (Batch, L)

    # Compute the final loss for each Batch
    loss = torch.sum(reg_weights * activations_batch_averaged, dim=1)  # (Batch,)

    # Aggregate the loss over all Batches (e.g., mean)
    final_loss = torch.mean(loss)  # Scalar

    return final_loss.float()


        

