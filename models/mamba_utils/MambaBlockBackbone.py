import math
from functools import partial
import json
import os
import copy
from collections import namedtuple
import torch
import torch.nn as nn
from mamba_ssm.modules.mamba2 import Mamba2
from mamba_ssm.modules.mlp import GatedMLP
from mamba_ssm.modules.block import Block
from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
from models.layers.mamba_layers.Embedding_layers import DataEmbedding
from models.mamba_utils.DilatedConv import MultiScaleDilatedConv
from models.mamba_utils.Memorynet import Memory

class MixerModel(nn.Module):

    def __init__(
        self, 
        d_model,
        n_layer,
        d_intermediate,
        c_in,
        args,
        invariant_memory_net,
        ssm_cfg=None,
        attn_layer_idx=None,
        attn_cfg=None,
        norm_epsilon: float = 1e-5,
        rms_norm: bool = False,
        initializer_cfg=None,
        fused_add_norm=False,
        residual_in_fp32=False,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.args = args
        self.input_embedding = DataEmbedding(c_in=c_in, d_model=d_model)
        self.invariant_memory_net = invariant_memory_net
        
        # if self.args.temporal_memory:
        #     self.dilatedconv = MultiScaleDilatedConv(in_channels=c_in, out_channels=d_model, kernel_size=6, dilation_rates=[4, 8, 16, 32])
        #     self.temporal_memory = Memory(num_memory=args.num_memory, memory_dim=d_model)

        self.fused_add_norm = fused_add_norm
        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")        
        self.layers = nn.ModuleList(
            [
                create_block(
                    d_model,
                    d_intermediate=d_intermediate,
                    ssm_cfg=ssm_cfg,
                    attn_layer_idx=attn_layer_idx,
                    attn_cfg=attn_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    **factory_kwargs,
                )
                for i in range(n_layer)
            ]
        )
        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            d_model, eps=norm_epsilon, **factory_kwargs
        )
        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
                n_residuals_per_layer=1 if d_intermediate == 0 else 2,  # 2 if we have MLP
            )
        )

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }
    
    def forward(self, input_series, inference_params=None, **mixer_kwargs):

        hidden_states = self.input_embedding(input_series)
        
        if self.args.temporal_memory or self.args.freq_memory:
            # time_prompt = self.invariant_memory_net(input_series, norm=False)
            prompt_dict = self.invariant_memory_net(input_series, norm=False)
            time_prompt, freq_prompt = prompt_dict['time_prompt'], prompt_dict['freq_prompt']
            
            # time_prompt_input = self.dilatedconv(input_series) # (B, T, C_out)
            # time_prompt = self.temporal_memory(time_prompt_input)['out']

        if self.args.temporal_memory and self.args.use_in_encoding:
            hidden_states = hidden_states + time_prompt
        
        if self.args.freq_memory and self.args.use_in_encoding:
            hidden_states = hidden_states + freq_prompt
        
            
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=inference_params, **mixer_kwargs
            )
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            hidden_states = layer_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
                is_rms_norm=isinstance(self.norm_f, RMSNorm)
            )

        if self.args.temporal_memory and self.args.use_in_decoding:
            hidden_states = hidden_states + time_prompt

        if self.args.freq_memory and self.args.use_in_decoding:
            hidden_states = hidden_states + freq_prompt

        # return hidden_states
    
        if self.args.temporal_memory or self.args.freq_memory:
            return hidden_states, prompt_dict
        else:
            return hidden_states




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
