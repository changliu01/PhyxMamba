import torch
import torch.nn as nn

class MultiScaleDilatedConv(nn.Module):
    def __init__(self, 
                 in_channels=1, 
                 out_channels=64, 
                 kernel_size=6, 
                 dilation_rates=[1, 2, 4]  # 示例：[2^0+1=2, 2^1+1=3, 2^2+1=5]
                ):
        
        super().__init__()
        self.convs = nn.ModuleList()
        
        # 为每个膨胀率创建独立的空洞卷积层
        for d in dilation_rates:
            # 计算保持输入输出长度一致的padding值
            padding = (kernel_size - 1) * d // 2
            conv = nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                dilation=d,
                padding=padding  # 自动填充以保证 T_out = T_in
            )
            self.convs.append(conv)
        
    def forward(self, x):
        
        """
        输入形状: (B, T, 1)
        输出形状: (B, T, C_out)
        """
        # 调整输入形状适配PyTorch的Conv1d (B, C, T)
        x = x.permute(0, 2, 1)  # (B, 1, T)
        
        # 多尺度空洞卷积
        multi_scale_features = []
        for conv in self.convs:
            out = conv(x)          # (B, C_out, T)
            out = out.permute(0, 2, 1)  # (B, T, C_out)
            multi_scale_features.append(out)
        
        # 堆叠多尺度特征 (F, B, T, C_out)
        stacked = torch.stack(multi_scale_features, dim=0)
        
        # 沿第0维（多尺度维度）平均
        final_embedding = torch.mean(stacked, dim=0)  # (B, T, C_out)
        
        return final_embedding