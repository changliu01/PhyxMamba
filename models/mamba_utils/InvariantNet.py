import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from functools import partial
from mamba_ssm.modules.mamba2 import Mamba2
from mamba_ssm.modules.mlp import GatedMLP
from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
from models.mamba_utils.DilatedConv import MultiScaleDilatedConv
from models.mamba_utils.Frequency import FrequencyModel
from models.mamba_utils.Memorynet import Memory
from models.layers.mamba_layers.invariant_layers import InvariantNet
from mamba_ssm.modules.block import Block
from models.mamba_utils.MambaBlockBackbone import MixerModel


class InvariantMemoryNet(nn.Module):

    def __init__(self, args, c_in, d_model):

        super().__init__()
        # self.dilatedconv = InvariantNet(d_model)
        self.args = args

        if self.args.freq_memory:
            self.frequency_model = FrequencyModel(args)
            self.freq_dilatedconv = MultiScaleDilatedConv(in_channels=c_in, out_channels=d_model, kernel_size=6, dilation_rates=[2, 4, 8, 16])
            self.freq_memory = Memory(num_memory=args.num_memory, memory_dim=d_model)

        if self.args.temporal_memory:
            
            self.dilatedconv = MultiScaleDilatedConv(in_channels=c_in, out_channels=d_model, kernel_size=6, dilation_rates=[2, 4, 8, 16])
            self.temporal_memory = Memory(num_memory=args.num_memory, memory_dim=d_model)
        
        # if self.args.temporal_prompt:
        #     self.mamba = Mamba2(d_model=d_model)

        
        

    def forward(self, input_time_series, norm):

        # print(input_time_series.shape)
        # B, T, N = input_time_series.shape
        # if norm:
        #     means = input_time_series.mean(dim=1, keepdim=True)
        #     input_time_series = input_time_series - means
        #     stdev = torch.sqrt(torch.var(input_time_series, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        #     input_time_series = input_time_series / stdev

        # input_time_series = input_time_series.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)

        # if self.args.flip_conv:
        #     input_time_series = torch.flip(input_time_series, dims=[1])

        if self.args.temporal_memory:

            time_prompt = self.dilatedconv(input_time_series) # (B*N, T, C_out)
            time_prompt = self.temporal_memory(time_prompt)['out'] # (B*N, T, C_out)
        else:
            time_prompt = None
        
        if self.args.freq_memory:
            freq_reconstructed = self.frequency_model(input_time_series) # (B*N, T, 1)

            freq_prompt = self.freq_dilatedconv(freq_reconstructed) # (B*N, T, C_out)
            freq_prompt = self.freq_memory(freq_prompt)['out'] # (B*N, T, C_out)
        else:
            freq_prompt = None

        # if self.args.temporal_prompt:
        #     time_prompt = self.mamba(time_prompt) # (B*N, T, C_out)

        

           

        # if self.args.flip_conv:
        #     time_prompt = torch.flip(time_prompt, dims=[1])

        # time_prompt = self.dilatedconv(input_time_series)
        return dict(time_prompt=time_prompt, freq_prompt=freq_prompt)
    

def create_block(
        d_model,
        d_intermediate,
        ssm_cfg=None,
        attn_layer_idx=None,
        attn_cfg=None,
        norm_epsilon=1e-5,
        rms_norm=False,
        residual_in_fp32=False,
        fused_add_norm=False,
        layer_idx=None,
        device=None,
        dtype=None,
    ):

    if ssm_cfg is None:
        ssm_cfg = {}
    if attn_layer_idx is None:
        attn_layer_idx = []
    if attn_cfg is None:
        attn_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    if layer_idx not in attn_layer_idx:
        # Create a copy of the config to modify
        ssm_cfg = copy.deepcopy(ssm_cfg) if ssm_cfg is not None else {}
        mixer_cls = partial(
            Mamba2,
            layer_idx=layer_idx,
            **ssm_cfg,
            **factory_kwargs
        )
    else:
        raise NotImplementedError
    
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )

    if d_intermediate == 0:
        mlp_cls = nn.Identity
    else:
        mlp_cls = partial(
            GatedMLP, hidden_features=d_intermediate, out_features=d_model, **factory_kwargs
        )
    
    block = Block(
        d_model, 
        mixer_cls,
        mlp_cls,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )

    block.layer_idx = layer_idx

    return block

# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,  # Now only used for embedding layer.
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)