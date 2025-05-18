import torch
import torch.nn as nn
import torch.nn.functional as F

class Memory(nn.Module):
    """ Memory prompt
    """
    def __init__(self, num_memory, memory_dim, args=None):
        super().__init__()

        self.args = args

        self.num_memory = num_memory
        self.memory_dim = memory_dim

        self.memMatrix = nn.Parameter(torch.randn(num_memory, memory_dim))  # M,C
        self.keyMatrix = nn.Parameter(torch.randn(num_memory, memory_dim))  # M,C
        
        self.x_proj = nn.Linear(memory_dim, memory_dim)

        # self.k_proj = nn.Linear(memory_dim, memory_dim)

        # self.v_proj = nn.Linear(memory_dim, memory_dim)
        
        self.initialize_weights()

        print("model initialized memory")


    def initialize_weights(self):
        # torch.nn.init.kaiming_normal_(self.memMatrix, mode='fan_in', nonlinearity='relu')
        # torch.nn.init.kaiming_normal_(self.keyMatrix, mode='fan_in', nonlinearity='relu')
        # torch.nn.init.trunc_normal_(self.memMatrix, std=0.02)
        # torch.nn.init.trunc_normal_(self.keyMatrix, std=0.02)
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

    def forward(self, x, Type='',shape=None):
        """
        :param x: query features with size [N,C], where N is the number of query items,
                  C is same as dimension of memory slot

        :return: query output retrieved from memory, with the same size as x.
        """
        # dot product

        assert x.shape[-1]==self.memMatrix.shape[-1]==self.keyMatrix.shape[-1], "dimension mismatch"
        # assert x.shape[-1]==self.memMatrix.shape[-1], "dimension mismatch"

        x_query = torch.tanh(self.x_proj(x))
        # x_query = self.x_proj(x)

        # att_weight = F.linear(input=x_query, weight=self.k_proj(self.memMatrix))  # [N,C] by [M,C]^T --> [N,M]
        att_weight = F.linear(input=x_query, weight=self.keyMatrix)  # [N,C] by [M,C]^T --> [N,M]

        att_weight = F.softmax(att_weight, dim=-1)  # NxM

        # out = F.linear(att_weight, self.v_proj(self.memMatrix).permute(1, 0))  # [N,M] by [M,C]  --> [N,C]
        out = F.linear(input=att_weight, weight=self.memMatrix.permute(1, 0))
        
        return dict(out=out, att_weight=att_weight)