import torch
import torch.nn as nn
import torch.nn.functional as F

# class FrequencyModel(nn.Module):

#     def __init__(self, args):

#     def forward(self, x):

class FrequencyModel(nn.Module):

    def __init__(self, args):

        super().__init__()

        self.args = args
    
    def forward(self, x):

        '''
        x: (B*N, T, 1)
        '''
        
        _, T, _ = x.shape

        if self.args.fft_mode == 0:
            
            raise NotImplementedError
            # x_f = torch.fft.rfft(x.permute(0, 2, 1), n=T, dim=-1) # (B*N, 1, T//2+1)
            # x_f = torch.cat([x_f.real, x_f.imag], dim=1) # (B*N, 2, T//2+1)

        elif self.args.fft_mode == 1:
            x_f = torch.fft.rfft(x.permute(0, 2, 1), dim=-1).squeeze(dim=1) # (B*N, T//2+1)

            amplitude = torch.abs(x_f) # (B*N, T//2+1)

            if self.args.fft_thred == 0:
                threshold = amplitude.mean(dim=-1, keepdim=True).to(x)
                mask_temp = amplitude > threshold

            elif self.args.fft_thred == 1:
                threshold = torch.quantile(amplitude, 0.8, dim=-1, keepdim=True).to(x)
                mask_temp = amplitude > threshold

            elif self.args.fft_thred == 2:
                max_freq_idx = 3
                mask_temp = torch.zeros_like(x_f, dtype=torch.bool).to(x)
                mask_temp[:, :max_freq_idx] = True
            
            X_f_filtered = x_f * mask_temp
            x_f = torch.fft.irfft(X_f_filtered, dim=-1, n=T).unsqueeze(dim=-1) # (B*N, T, 1)
        
        else:
            raise NotImplementedError
    
        return x_f # (B*N, T, 1)
        


