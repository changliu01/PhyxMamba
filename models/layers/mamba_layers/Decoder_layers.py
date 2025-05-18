import torch
import torch.nn as nn

class Decoder(nn.Module):

    def __init__(self, args, dropout_rate=0.1):
        super().__init__()
        self.args = args
        self.d_model = args.d_model
        self.variable_num = args.variable_num
        self.token_size = args.token_size
        self.patch_model_method = args.patch_model_method

        if self.patch_model_method == 'default':
            self.lm_head = nn.Linear(args.d_model, args.token_size, bias=True)
        elif self.patch_model_method == 'flatten_linear':
            self.lm_head = nn.Linear(args.d_model, args.token_size * args.variable_num, bias=True)
        else:
            raise ValueError(f"Unknown patch model method: {self.args.patch_model_method}")
        
        if self.args.decoder_dropout > 0:

            self.dropout_network = nn.Sequential(
                nn.GELU(),
                nn.Dropout(self.args.decoder_dropout),
                nn.Linear(self.lm_head.weight.shape[0], self.lm_head.weight.shape[0]),
            )
        
    def forward(self, x, n_var):

        '''
        x: [B / B*V, L, d_model]
        output: [B, T, V]
        '''

        patch_len = x.shape[1]
        
        if self.patch_model_method == 'default':
            output = self.lm_head(x) # (B*V, L, token_size)

            if self.args.decoder_dropout > 0:
                output = self.dropout_network(output)
            output = output.reshape(-1, n_var, self.token_size * patch_len).transpose(1, 2) # (B*V, token_size, L)
        
        elif self.patch_model_method == 'flatten_linear':
            output = self.lm_head(x) # (B, patch_len, token_size * variable_num)
            if self.args.decoder_dropout > 0:
                output = self.dropout_network(output)
            output = output.reshape(-1, patch_len, self.variable_num, self.token_size).transpose(1, 2).reshape(-1, self.variable_num, self.token_size * patch_len).transpose(1, 2)  # (B, T, N)
        else:
            raise ValueError(f"Unknown patch model method: {self.args.patch_model_method}")
        
        return output