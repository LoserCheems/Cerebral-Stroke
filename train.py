import math
import copy
import warnings
import re
import sys

import torch
import torch.utils.checkpoint
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss, LayerNorm, MSELoss, BCEWithLogitsLoss
from torch.nn.utils import skip_init
from typing import Optional, Tuple, Union, List, Callable, Dict, Any
from copy import deepcopy

from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    SequenceClassifierOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from transformers.generation.logits_process import LogitsProcessor
from transformers.generation.utils import LogitsProcessorList, StoppingCriteriaList, GenerationConfig, ModelOutput


if sys.platform != 'darwin':
    torch._C._jit_set_profiling_mode(False)
    torch._C._jit_set_profiling_executor(False)
    torch._C._jit_override_can_fuse_on_cpu(True)
    torch._C._jit_override_can_fuse_on_gpu(True)

logger = logging.get_logger(__name__)

def default_init(cls, *args, **kwargs):
    return cls(*args, **kwargs)

def _config_to_kwargs(args):
    common_kwargs = {
        "dtype": args.torch_dtype,
    }
    return common_kwargs

# 无效分数预测处理
class InvalidScoreLogitsProcessor(LogitsProcessor):
    """
    :class：`transformers.LogitsProcessor`，确保 logits 是有效的浮点数。
    """
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # 检查scores是否存在NaN或inf
        if torch.isnan(scores).any() or torch.isinf(scores).any():
            scores.zero_()
            scores[..., 5] = 5e4
        return scores
    
# 前缀编码器
class PrefixEncoder(torch.nn.Module):
    """
    torch.nn 模型，用于编码前缀
    输入形状：(batch-size, prefix-length)
    输出形状：(batch-size, prefix-length, 2*layers*hidden)
    """

    def __init__(self, config):
        super().__init__()
        self.prefix_projection = config.prefix_projection
        # 如果 prefix_projection 为 True，则使用两层 MLP 编码前缀
        if self.prefix_projection:
            # Use a two-layer MLP to encode the prefix
            # 使用两层 MLP 编码前缀
            kv_size = config.num_layers * config.kv_channels * config.multi_query_group_num * 2
            self.embedding = torch.nn.Embedding(config.pre_seq_len, kv_size)
            self.trans = torch.nn.Sequential(
                torch.nn.Linear(kv_size, config.hidden_size),
                torch.nn.Tanh(),
                torch.nn.Linear(config.hidden_size, kv_size)
            )
        # 如果 prefix_projection 为 False，则直接使用 Embedding 编码前缀
        else:
            self.embedding = torch.nn.Embedding(
                config.pre_seq_len,
                config.num_layers * config.kv_channels * config.multi_query_group_num * 2
                )

    def forward(self, prefix: torch.Tensor):
        if self.prefix_projection:
            prefix_tokens = self.embedding(prefix)
            past_key_values = self.trans(prefix_tokens)
        else:
            past_key_values = self.embedding(prefix)
        return past_key_values
    
# 分割张量
def split_tensor_along_last_dim(
        tensor: torch.Tensor,
        num_partitions: int,
        contiguous_split_chunks: bool = False,
) -> List[torch.Tensor]:
    """
    将张量沿其最后一个维度分割。

    参数：
        tensor：输入张量。
        num_partitions：要分割张量的分区数
        contiguous_split_chunks：如果为 True，则使每个块在内存中连续。

    返回：
        张量列表
    """
    # 获取大小和维度。
    last_dim = tensor.dim() - 1
    last_dim_size = tensor.size()[last_dim] // num_partitions
    # 分割。
    tensor_list = torch.split(tensor, last_dim_size, dim=last_dim)
    # 注意：torch.split 默认不创建连续的张量。
    if contiguous_split_chunks:
        return tuple(chunk.contiguous() for chunk in tensor_list)

    return tensor_list

# 旋转位置嵌入
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, original_impl=False, device=None, dtype=None):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=device).to(dtype=dtype) / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.dim = dim
        self.original_impl = original_impl

    def forward_impl(
            self, seq_len: int, n_elem: int, dtype: torch.dtype, device: torch.device, base: int = 10000
    ):
        # $\Theta = {\theta_i = 10000^{\frac{2(i-1)}{d}}, i \in [1, 2, ..., \frac{d}{2}]}$
        theta = 1.0 / (base ** (torch.arange(0, n_elem, 2, dtype=torch.float, device=device) / n_elem))

        # 创建位置索引 `[0, 1, ..., seq_len - 1]`
        seq_idx = torch.arange(seq_len, dtype=torch.float, device=device)

        # 计算位置索引和 $\theta_i$ 的乘积
        idx_theta = torch.outer(seq_idx, theta).float()

        cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)

        # 这是为了模仿 complex32 的行为，否则我们将得到不同的结果
        if dtype in (torch.float16, torch.bfloat16, torch.int8):
            cache = cache.bfloat16() if dtype == torch.bfloat16 else cache.half()
        return cache

    def forward(self, max_seq_len, offset=0):
        return self.forward_impl(
            max_seq_len, self.dim, dtype=self.inv_freq.dtype, device=self.inv_freq.device
        )

# 应用旋转位置嵌入
@torch.jit.script
def apply_rotary_pos_emb(x: torch.Tensor, rope_cache: torch.Tensor) -> torch.Tensor:
    # x: [sq, b, np, hn]
    sq, b, np, hn = x.size(0), x.size(1), x.size(2), x.size(3)
    rot_dim = rope_cache.shape[-2] * 2
    x, x_pass = x[..., :rot_dim], x[..., rot_dim:]
    # 截断以支持可变大小
    rope_cache = rope_cache[:sq]
    xshaped = x.reshape(sq, -1, np, rot_dim // 2, 2)
    rope_cache = rope_cache.view(sq, -1, 1, xshaped.size(3), 2)
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * rope_cache[..., 0] - xshaped[..., 1] * rope_cache[..., 1],
            xshaped[..., 1] * rope_cache[..., 0] + xshaped[..., 0] * rope_cache[..., 1],
        ],
        -1,
    )
    x_out2 = x_out2.flatten(3)
    return torch.cat((x_out2, x_pass), dim=-1)

# 均方根归一化
class RMSNorm(torch.nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, device=None, dtype=None, **kwargs):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(normalized_shape, device=device, dtype=dtype))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor):
        input_dtype = hidden_states.dtype
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)

        return (self.weight * hidden_states).to(input_dtype)
    
# 自注意力核心
class CoreAttention(torch.nn.Module):
    def __init__(self, config, layer_number):
        super(CoreAttention, self).__init__()

        self.apply_query_key_layer_scaling = config.apply_query_key_layer_scaling
        self.attention_softmax_in_fp32 = config.attention_softmax_in_fp32
        if self.apply_query_key_layer_scaling:
            self.attention_softmax_in_fp32 = True
        self.layer_number = max(1, layer_number)

        projection_size = config.kv_channels * config.num_attention_heads

        # 每个注意头和每个分区值。
        self.hidden_size_per_partition = projection_size
        self.hidden_size_per_attention_head = projection_size // config.num_attention_heads
        self.num_attention_heads_per_partition = config.num_attention_heads

        coeff = None
        self.norm_factor = math.sqrt(self.hidden_size_per_attention_head)
        if self.apply_query_key_layer_scaling:
            coeff = self.layer_number
            self.norm_factor *= coeff
        self.coeff = coeff

        self.attention_dropout = torch.nn.Dropout(config.attention_dropout)

    def forward(self, query_layer, key_layer, value_layer, attention_mask=None):
        pytorch_major_version = int(torch.__version__.split('.')[0])
        if pytorch_major_version >= 2:
            # 对query_layer, key_layer, value_layer进行转置
            # [sq, b, np, hn] -> [b, np, sq, hn]
            query_layer, key_layer, value_layer = [k.permute(1, 2, 0, 3) for k in [query_layer, key_layer, value_layer]]
            if attention_mask is None and query_layer.shape[2] == key_layer.shape[2]:
                context_layer = torch.nn.functional.scaled_dot_product_attention(query_layer, key_layer, value_layer,
                                                                                 is_causal=True)
            else:
                if attention_mask is not None:
                    attention_mask = ~attention_mask
                context_layer = torch.nn.functional.scaled_dot_product_attention(query_layer, key_layer, value_layer,
                                                                                 attention_mask)
            context_layer = context_layer.permute(2, 0, 1, 3)
            new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size_per_partition,)
            context_layer = context_layer.reshape(*new_context_layer_shape)
        else:
            # 原始注意力分数
            # [b, np, sq, sk]
            output_size = (query_layer.size(1), query_layer.size(2), query_layer.size(0), key_layer.size(0))

            # [sq, b, np, hn] -> [sq, b * np, hn]
            query_layer = query_layer.view(output_size[2], output_size[0] * output_size[1], -1)
            # [sk, b, np, hn] -> [sk, b * np, hn]
            key_layer = key_layer.view(output_size[3], output_size[0] * output_size[1], -1)
            # 预分配输入张量：[b * np, sq, sk]
            matmul_input_buffer = torch.empty(
                output_size[0] * output_size[1], output_size[2], output_size[3], dtype=query_layer.dtype,
                device=query_layer.device
            )

            # 原始注意力分数。[b * np, sq, sk]
            matmul_result = torch.baddbmm(
                matmul_input_buffer,
                query_layer.transpose(0, 1),  # [b * np, sq, hn]
                key_layer.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk]
                beta=0.0,
                alpha=(1.0 / self.norm_factor),
            )

            # 更改视图为 [b, np, sq, sk]
            attention_scores = matmul_result.view(*output_size)

            # ===========================
            # 注意概率和辍学
            # ===========================

            # 注意分数和注意掩码 [b, np, sq, sk]
            if self.attention_softmax_in_fp32:
                attention_scores = attention_scores.float()
            if self.coeff is not None:
                attention_scores = attention_scores * self.coeff
            if attention_mask is None and attention_scores.shape[2] == attention_scores.shape[3]:
                attention_mask = torch.ones(output_size[0], 1, output_size[2], output_size[3],
                                            device=attention_scores.device, dtype=torch.bool)
                attention_mask.tril_()
                attention_mask = ~attention_mask
            if attention_mask is not None:
                attention_scores = attention_scores.masked_fill(attention_mask, float("-inf"))
            attention_probs = F.softmax(attention_scores, dim=-1)
            attention_probs = attention_probs.type_as(value_layer)
            attention_probs = self.attention_dropout(attention_probs)
            # =========================
            # 上下文层。[sq, b, hp]
            # =========================

            # 值层 -> 上下文层。
            # [sk, b, np, hn] --> [b, np, sq, hn]

            # 上下文层形状：[b, np, sq, hn]
            output_size = (value_layer.size(1), value_layer.size(2), query_layer.size(0), value_layer.size(3))

            # 更改视图 [sk, b * np, hn]
            value_layer = value_layer.view(value_layer.size(0), output_size[0] * output_size[1], -1)
        
            # 更改视图 [b * np, sq, sk]
            attention_probs = attention_probs.view(output_size[0] * output_size[1], output_size[2], -1)
         
            # 矩阵乘法：[b * np, sq, hn]
            context_layer = torch.bmm(attention_probs, value_layer.transpose(0, 1))
          
            # 更改视图 [b, np, sq, hn]
            context_layer = context_layer.view(*output_size)
            # [b, np, sq, hn] --> [sq, b, np, hn]
            context_layer = context_layer.permute(2, 0, 1, 3).contiguous()
            # [sq, b, np, hn] --> [sq, b, hp]
            new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size_per_partition,)
            context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer
    
# 自注意力
class SelfAttention(torch.nn.Module):
    """
    自注意层抽象类。

    自注意层接受大小为 [s, b, h] 的输入，并返回相同大小的输出。
    """

    def __init__(self, config, layer_number, device=None):
        super(SelfAttention, self).__init__()
        self.layer_number = max(1, layer_number)

        self.projection_size = config.kv_channels * config.num_attention_heads

        # 每个注意头和每个分区值。
        self.hidden_size_per_attention_head = self.projection_size // config.num_attention_heads
        self.num_attention_heads_per_partition = config.num_attention_heads

        self.multi_query_attention = config.multi_query_attention
        self.qkv_hidden_size = 3 * self.projection_size
        if self.multi_query_attention:
            self.num_multi_query_groups_per_partition = config.multi_query_group_num
            self.qkv_hidden_size = (
                    self.projection_size + 2 * self.hidden_size_per_attention_head * config.multi_query_group_num
            )
        self.query_key_value = nn.Linear(
            config.hidden_size, 
            self.qkv_hidden_size,
            bias=config.add_bias_linear or config.add_qkv_bias,
            device=device, 
            **_config_to_kwargs(config)
        )

        self.core_attention = CoreAttention(config, self.layer_number)

        # 输出。
        self.dense = nn.Linear(
            self.projection_size, 
            config.hidden_size, 
            bias=config.add_bias_linear,
            device=device, 
            **_config_to_kwargs(config)
        )

    # 为推理预先分配内存
    def _allocate_memory(self, inference_max_sequence_len, batch_size, device=None, dtype=None):
        if self.multi_query_attention:
            num_attention_heads = self.num_multi_query_groups_per_partition
        else:
            num_attention_heads = self.num_attention_heads_per_partition
        return torch.empty(
            inference_max_sequence_len,
            batch_size,
            num_attention_heads,
            self.hidden_size_per_attention_head,
            dtype=dtype,
            device=device,
        )

    def forward(
            self, hidden_states, attention_mask, rotary_pos_emb, kv_cache=None, use_cache=True
    ):
        # 隐状态：[sq, b, h]
        # =================================================
        # 为推理预先分配密钥值的内存。
        # =================================================
        # =====================
        # 查询、键和值
        # =====================

        # 注意头 [sq, b, h] --> [sq, b, (np * 3 * hn)]
        mixed_x_layer = self.query_key_value(hidden_states)

        if self.multi_query_attention:
            (query_layer, key_layer, value_layer) = mixed_x_layer.split(
                [
                    self.num_attention_heads_per_partition * self.hidden_size_per_attention_head,
                    self.num_multi_query_groups_per_partition * self.hidden_size_per_attention_head,
                    self.num_multi_query_groups_per_partition * self.hidden_size_per_attention_head,
                ],
                dim=-1,
            )
            query_layer = query_layer.view(
                query_layer.size()[:-1] + (self.num_attention_heads_per_partition, self.hidden_size_per_attention_head)
            )
            key_layer = key_layer.view(
                key_layer.size()[:-1] + (self.num_multi_query_groups_per_partition, self.hidden_size_per_attention_head)
            )
            value_layer = value_layer.view(
                value_layer.size()[:-1]
                + (self.num_multi_query_groups_per_partition, self.hidden_size_per_attention_head)
            )
        else:
            new_tensor_shape = mixed_x_layer.size()[:-1] + \
                               (self.num_attention_heads_per_partition,
                                3 * self.hidden_size_per_attention_head)
            mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)

            # [sq, b, np, 3 * hn] --> 3 [sq, b, np, hn]
            (query_layer, key_layer, value_layer) = split_tensor_along_last_dim(mixed_x_layer, 3)

        # 应用相对位置编码（旋转嵌入）
        if rotary_pos_emb is not None:
            query_layer = apply_rotary_pos_emb(query_layer, rotary_pos_emb)
            key_layer = apply_rotary_pos_emb(key_layer, rotary_pos_emb)

        # 调整推理的键和值
        if kv_cache is not None:
            cache_k, cache_v = kv_cache
            key_layer = torch.cat((cache_k, key_layer), dim=0)
            value_layer = torch.cat((cache_v, value_layer), dim=0)
        if use_cache:
            kv_cache = (key_layer, value_layer)
        else:
            kv_cache = None

        if self.multi_query_attention:
            key_layer = key_layer.unsqueeze(-2)
            key_layer = key_layer.expand(
                -1, -1, -1, self.num_attention_heads_per_partition // self.num_multi_query_groups_per_partition, -1
            )
            key_layer = key_layer.contiguous().view(
                key_layer.size()[:2] + (self.num_attention_heads_per_partition, self.hidden_size_per_attention_head)
            )
            value_layer = value_layer.unsqueeze(-2)
            value_layer = value_layer.expand(
                -1, -1, -1, self.num_attention_heads_per_partition // self.num_multi_query_groups_per_partition, -1
            )
            value_layer = value_layer.contiguous().view(
                value_layer.size()[:2] + (self.num_attention_heads_per_partition, self.hidden_size_per_attention_head)
            )

        # ==================================
        # 核心注意力计算
        # ==================================

        context_layer = self.core_attention(query_layer, key_layer, value_layer, attention_mask)

        # =================
        # 输出。[sq, b, h]
        # =================

        output = self.dense(context_layer)

        return output, kv_cache
    
# 多层感知机
class MLP(torch.nn.Module):
    """
    MLP 将使用 h 隐藏状态输入，将其投影到 4*h 隐藏维度，执行非线性变换，然后将状态投影回 h 隐藏维度。
    """

    def __init__(self, config, device=None):
        super(MLP, self).__init__()

        self.add_bias = config.add_bias_linear
        

        if config.activation_function == "swiglu":
            def swiglu(x):
                x = torch.chunk(x, 2, dim=-1)
                return F.silu(x[0]) * x[1]
            self.activation_func = swiglu

            self.dense_h_to_4h = nn.Linear(
            config.hidden_size,
            config.ffn_hidden_size * 2,
            bias=self.add_bias,
            device=device,
            **_config_to_kwargs(config)
        )
        else:
            self.activation_func = torch.nn.functional.gelu
            self.dense_h_to_4h = nn.Linear(
            config.hidden_size,
            config.ffn_hidden_size,
            bias=self.add_bias,
            device=device,
            **_config_to_kwargs(config)
        )
        
        # 投影回 h。
        self.dense_4h_to_h = nn.Linear(
            config.ffn_hidden_size,
            config.hidden_size,
            bias=self.add_bias,
            device=device,
            **_config_to_kwargs(config)
        )

    def forward(self, hidden_states):
        # [s, b, 4hp]
        intermediate_parallel = self.dense_h_to_4h(hidden_states)
        intermediate_parallel = self.activation_func(intermediate_parallel)
        # [s, b, h]
        output = self.dense_4h_to_h(intermediate_parallel)
        return output

class Block(torch.nn.Module):
    """
    单个 transformer 层。

    Transformer 层接受大小为 [s, b, h] 的输入，并返回相同大小的输出。
    """

    def __init__(self, config, layer_number, device=None):
        super(Block, self).__init__()
        self.layer_number = layer_number

        self.apply_residual_connection_post_layernorm = config.apply_residual_connection_post_layernorm

        self.fp32_residual_connection = config.fp32_residual_connection

        LayerNormFunc = RMSNorm if config.rmsnorm else LayerNorm
        # 输入数据的层归一化。
        self.input_layernorm = LayerNormFunc(
            config.hidden_size, 
            eps=config.layernorm_epsilon, 
            device=device,
            dtype=config.torch_dtype
        )

        # 自注意力。
        self.self_attention = SelfAttention(config, layer_number, device=device)
        self.hidden_dropout = config.hidden_dropout

        # 注意输出的层归一化
        self.post_attention_layernorm = LayerNormFunc(
            config.hidden_size, 
            eps=config.layernorm_epsilon, 
            device=device,
            dtype=config.torch_dtype
        )

        # 多层感知机
        self.mlp = MLP(config, device=device)

    def forward(
            self, 
            hidden_states, 
            attention_mask, 
            rotary_pos_emb, 
            kv_cache=None, 
            use_cache=True,
    ):
        # 隐状态：[s, b, h]
        # transformer 层开始的层归一化。
        layernorm_output = self.input_layernorm(hidden_states)
        # 自注意力。
        attention_output, kv_cache = self.self_attention(
            layernorm_output,
            attention_mask,
            rotary_pos_emb,
            kv_cache=kv_cache,
            use_cache=use_cache
        )

        # 残差连接。
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = hidden_states

        layernorm_input = torch.nn.functional.dropout(attention_output, p=self.hidden_dropout, training=self.training)
        layernorm_input = residual + layernorm_input

        # 自注意力后的层归一化。
        layernorm_output = self.post_attention_layernorm(layernorm_input)

        # 多层感知机。
        mlp_output = self.mlp(layernorm_output)

        # 第二个残差连接。
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = layernorm_input

        output = torch.nn.functional.dropout(mlp_output, p=self.hidden_dropout, training=self.training)
        output = residual + output

        return output, kv_cache
    
class Transformer(torch.nn.Module):

    def __init__(self, config, device=None):
        super(Transformer, self).__init__()

        self.fp32_residual_connection = config.fp32_residual_connection
        self.post_layer_norm = config.post_layer_norm

        # 层数。
        self.num_layers = config.num_layers

        # Transformer 层。
        def build_layer(layer_number):
            return Block(config, layer_number, device=device)

        self.layers = torch.nn.ModuleList([build_layer(i + 1) for i in range(self.num_layers)])

        if self.post_layer_norm:
            LayerNormFunc = RMSNorm if config.rmsnorm else LayerNorm
            # 输出前的最终层归一化。
            self.final_layernorm = LayerNormFunc(
                config.hidden_size, 
                eps=config.layernorm_epsilon, 
                device=device,
                dtype=config.torch_dtype
            )

        self.gradient_checkpointing = False

    def _get_layer(self, layer_number):
        return self.layers[layer_number]

    def forward(
            self, hidden_states, attention_mask, rotary_pos_emb, kv_caches=None,
            use_cache: Optional[bool] = True,
            output_hidden_states: Optional[bool] = False,
    ):
        if not kv_caches:
            kv_caches = [None for _ in range(self.num_layers)]
        presents = () if use_cache else None
        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` 与梯度检查点不兼容。设置 `use_cache=False`..."
                )
                use_cache = False

        all_self_attentions = None
        all_hidden_states = () if output_hidden_states else None
        for index in range(self.num_layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer = self._get_layer(index)
            if self.gradient_checkpointing and self.training:
                layer_ret = torch.utils.checkpoint.checkpoint(
                    layer,
                    hidden_states,
                    attention_mask,
                    rotary_pos_emb,
                    kv_caches[index],
                    use_cache
                )
            else:
                layer_ret = layer(
                    hidden_states,
                    attention_mask,
                    rotary_pos_emb,
                    kv_cache=kv_caches[index],
                    use_cache=use_cache
                )
            hidden_states, kv_cache = layer_ret
            if use_cache:
                presents = presents + (kv_cache,)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 最终层归一化。
        if self.post_layer_norm:
            hidden_states = self.final_layernorm(hidden_states)

        return hidden_states, presents, all_hidden_states, all_self_attentions
    
class PreTrainedModel(PreTrainedModel):
    """
    一个抽象类来处理权重初始化和一个简单的接口来下载和加载预训练模型。
    """

    is_parallelizable = False
    supports_gradient_checkpointing = True
    config_class = None
    base_model_prefix = "transformer"
    _no_split_modules = ["Block"]

    def _init_weights(self, module: nn.Module):
        return

    def get_masks(self, input_ids, past_key_values, padding_mask=None):
        batch_size, seq_length = input_ids.shape
        full_attention_mask = torch.ones(batch_size, seq_length, seq_length, device=input_ids.device)
        full_attention_mask.tril_()
        past_length = 0
        if past_key_values:
            past_length = past_key_values[0][0].shape[0]
        if past_length:
            full_attention_mask = torch.cat((torch.ones(batch_size, seq_length, past_length,
                                                        device=input_ids.device), full_attention_mask), dim=-1)
        if padding_mask is not None:
            full_attention_mask = full_attention_mask * padding_mask.unsqueeze(1)
        if not past_length and padding_mask is not None:
            full_attention_mask -= padding_mask.unsqueeze(-1) - 1
        full_attention_mask = (full_attention_mask < 0.5).bool()
        full_attention_mask.unsqueeze_(1)
        return full_attention_mask

    def get_position_ids(self, input_ids, device):
        batch_size, seq_length = input_ids.shape
        position_ids = torch.arange(seq_length, dtype=torch.long, device=device).unsqueeze(0).repeat(batch_size, 1)
        return position_ids

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, Transformer):
            module.gradient_checkpointing = value

# 语言模型嵌入
class Embedding(torch.nn.Module):
    def __init__(self, config, device=None):
        super(Embedding, self).__init__()

        self.hidden_size = config.hidden_size
        # 单词嵌入（并行）。
        self.word_embeddings = nn.Embedding(
            config.padded_vocab_size,
            self.hidden_size,
            dtype=config.torch_dtype,
            device=device
        )
        self.fp32_residual_connection = config.fp32_residual_connection

    def forward(self, input_ids):
        # 嵌入。
        words_embeddings = self.word_embeddings(input_ids)
        embeddings = words_embeddings
        # 数据格式更改以避免显式转置：[b s h] --> [s b h]。
        embeddings = embeddings.transpose(0, 1).contiguous()
        # 如果设置了 fp32 残差连接的输入标志，则转换为 float。
        if self.fp32_residual_connection:
            embeddings = embeddings.float()
        return embeddings
    
# 模型
class Model(PreTrainedModel):
    def __init__(self, config, device=None, empty_init=False):
        super().__init__(config)
        if empty_init:
            init_method = skip_init
        else:
            init_method = default_init
        init_kwargs = {}
        if device is not None:
            init_kwargs["device"] = device
        self.embedding = init_method(Embedding, config, **init_kwargs)
        self.num_layers = config.num_layers
        self.multi_query_group_num = config.multi_query_group_num
        self.kv_channels = config.kv_channels

        # 旋转位置嵌入
        self.seq_length = config.seq_length
        rotary_dim = (
            config.hidden_size // config.num_attention_heads if config.kv_channels is None else config.kv_channels
        )

        self.rotary_pos_emb = RotaryEmbedding(rotary_dim // 2, original_impl=config.original_rope, device=device,
                                              dtype=config.torch_dtype)
        self.encoder = init_method(Transformer, config, **init_kwargs)
        self.output_layer = init_method(nn.Linear, config.hidden_size, config.padded_vocab_size, bias=False,
                                        dtype=config.torch_dtype, **init_kwargs)
        self.pre_seq_len = config.pre_seq_len
        self.prefix_projection = config.prefix_projection
        if self.pre_seq_len is not None:
            for param in self.parameters():
                param.requires_grad = False
            self.prefix_tokens = torch.arange(self.pre_seq_len).long()
            self.prefix_encoder = PrefixEncoder(config)
            self.dropout = torch.nn.Dropout(0.1)

    def get_input_embeddings(self):
        return self.embedding.word_embeddings

    def get_prompt(self, batch_size, device, dtype=torch.half):
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(device)
        past_key_values = self.prefix_encoder(prefix_tokens).type(dtype)
        past_key_values = past_key_values.view(
            batch_size,
            self.pre_seq_len,
            self.num_layers * 2,
            self.multi_query_group_num,
            self.kv_channels
        )
       
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 1, 0, 3, 4]).split(2)
        return past_key_values

    def forward(
            self,
            input_ids,
            position_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.BoolTensor] = None,
            full_attention_mask: Optional[torch.BoolTensor] = None,
            past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            use_cache: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ):
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size, seq_length = input_ids.shape

        if inputs_embeds is None:
            inputs_embeds = self.embedding(input_ids)

        if self.pre_seq_len is not None:
            if past_key_values is None:
                past_key_values = self.get_prompt(
                    batch_size=batch_size, device=input_ids.device,
                    dtype=inputs_embeds.dtype
                    )
            if attention_mask is not None:
                attention_mask = torch.cat(
                    [attention_mask.new_ones((
                            batch_size, 
                            self.pre_seq_len
                            )),
                            attention_mask], 
                            dim=-1
                            )

        if full_attention_mask is None:
            if (attention_mask is not None and not attention_mask.all()) or (past_key_values and seq_length != 1):
                full_attention_mask = self.get_masks(input_ids, past_key_values, padding_mask=attention_mask)

        # 旋转位置嵌入
        rotary_pos_emb = self.rotary_pos_emb(self.seq_length)
        if position_ids is not None:
            rotary_pos_emb = rotary_pos_emb[position_ids]
        else:
            rotary_pos_emb = rotary_pos_emb[None, :seq_length]
        rotary_pos_emb = rotary_pos_emb.transpose(0, 1).contiguous()

        # 运行编码器。
        hidden_states, presents, all_hidden_states, all_self_attentions = self.encoder(
            inputs_embeds, full_attention_mask, rotary_pos_emb=rotary_pos_emb,
            kv_caches=past_key_values, use_cache=use_cache, output_hidden_states=output_hidden_states
        )

        if not return_dict:
            return tuple(v for v in [hidden_states, presents, all_hidden_states, all_self_attentions] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )
    
# 序列分类
class SequenceClassification(PreTrainedModel):
    def __init__(self, config, empty_init=False, device=None):
        super().__init__(config)

        self.num_labels = config.num_labels
        self.transformer = Model(config, empty_init=empty_init, device=device)

        self.classifier_head = nn.Linear(config.hidden_size, config.num_labels, bias=True)
        if config.classifier_dropout is not None:
            self.dropout = nn.Dropout(config.classifier_dropout)
        else:
            self.dropout = None
        self.config = config

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            full_attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
            inputs_embeds: Optional[torch.LongTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor, ...], SequenceClassifierOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 检查输入数据是否有nan或其他异常值
        if input_ids is not None:
            if torch.isnan(input_ids).any() or torch.isinf(input_ids).any():
                raise ValueError("输入数据包含nan或inf值")
            if input_ids.max() >= self.config.padded_vocab_size:
                raise ValueError("输入数据包含超出词汇表大小的值")
            if input_ids.min() < 0:
                raise ValueError("输入数据包含负值")
            if input_ids.dtype != torch.long:
                raise ValueError("输入数据的数据类型不是torch.long")
            if input_ids.dim() != 2:
                raise ValueError("输入数据的维度不是2")
            if input_ids.size(1) > self.config.seq_length:
                raise ValueError("输入数据的长度超出了模型的最大长度")

        transformer_outputs = self.transformer(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            full_attention_mask=full_attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = transformer_outputs[0]
        pooled_hidden_states = hidden_states[-1]
        if self.dropout is not None:
            pooled_hidden_states = self.dropout(pooled_hidden_states)
        logits = self.classifier_head(pooled_hidden_states)
        # 无效分数处理为负无穷
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            logits[torch.isnan(logits)] = float("-inf")
            logits[torch.isinf(logits)] = float("-inf")


        

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze().float(), labels.squeeze())
                else:
                    loss = loss_fct(logits.float(), labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels).float(), labels.view(-1))
                # 如果是单标签分类的话, logits的维度是[batch_size, num_labels], labels的维度是[batch_size]
                
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits.float(), labels.view(-1, self.num_labels))

        if not return_dict:
            output = (logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

# 如果序列分类效果不佳, 尝试条件生成
class ConditionalGeneration(PreTrainedModel):
    def __init__(self, config, empty_init=False, device=None):
        super().__init__(config)

        self.max_sequence_length = config.max_length
        self.transformer = Model(config, empty_init=empty_init, device=device)
        self.config = config

    def _update_model_kwargs_for_generation(
            self,
            outputs: ModelOutput,
            model_kwargs: Dict[str, Any],
            is_encoder_decoder: bool = False,
            standardize_cache_format: bool = False,
    ) -> Dict[str, Any]:
        # update past_key_values
        # 更新 past_key_values
        model_kwargs["past_key_values"] = self._extract_past_from_model_output(
            outputs, standardize_cache_format=standardize_cache_format
        )

        # update attention mask
        # 更新注意掩码
        if "attention_mask" in model_kwargs:
            attention_mask = model_kwargs["attention_mask"]
            model_kwargs["attention_mask"] = torch.cat(
                [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
            )

        # update position ids
        # 更新位置 id
        if "position_ids" in model_kwargs:
            position_ids = model_kwargs["position_ids"]
            new_position_id = position_ids[..., -1:].clone()
            new_position_id += 1
            model_kwargs["position_ids"] = torch.cat(
                [position_ids, new_position_id], dim=-1
            )

        model_kwargs["is_first_forward"] = False
        return model_kwargs
    
    def prepare_inputs_for_generation(
            self,
            input_ids: torch.LongTensor,
            past_key_values: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            use_cache: Optional[bool] = None,
            is_first_forward: bool = True,
            **kwargs
    ) -> dict:
        # only last token for input_ids if past is not None
        # 如果过去不是 None，则只有最后一个令牌用于 input_ids
        if position_ids is None:
            position_ids = self.get_position_ids(input_ids, device=input_ids.device)
        if not is_first_forward:
            if past_key_values is not None:
                position_ids = position_ids[..., -1:]
                input_ids = input_ids[:, -1:]
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "return_last_logit": True,
            "use_cache": use_cache
        }
    
    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[Tuple[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            return_last_logit: Optional[bool] = False,
    ):
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        transformer_outputs = self.transformer(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = transformer_outputs[0]
        if return_last_logit:
            hidden_states = hidden_states[-1:]
        lm_logits = self.transformer.output_layer(hidden_states)
        lm_logits = lm_logits.transpose(0, 1).contiguous()

        loss = None
        if labels is not None:
            lm_logits = lm_logits.to(torch.float32)
            # 移位，使得 tokens < n 预测 n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # 扁平化 tokens
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            lm_logits = lm_logits.to(hidden_states.dtype)
            loss = loss.to(hidden_states.dtype)

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
    
    @staticmethod
    def _reorder_cache(
            past: Tuple[Tuple[torch.Tensor, torch.Tensor], ...], beam_idx: torch.LongTensor
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
        """
        此函数用于重新排序 `past_key_values` 缓存，如果调用 [`~PreTrainedModel.beam_search`] 或
        [`~PreTrainedModel.beam_sample`]。这是为了在每一代步骤中将 `past_key_values` 与正确的 beam_idx 匹配。

        输出与 `past` 共享相同的内存存储。
        """
        return tuple(
            (
                layer_past[0].index_select(1, beam_idx.to(layer_past[0].device)),
                layer_past[1].index_select(1, beam_idx.to(layer_past[1].device)),
            )
            for layer_past in past
        )
    
    def process_response(self, output, history):
        content = ""
        history = deepcopy(history)
        for response in output.split("<|assistant|>"):
            if "\n" in response:
                metadata, content = response.split("\n", maxsplit=1)
            else:
                metadata, content = "", response
            if not metadata.strip():
                content = content.strip()
                history.append({"role": "assistant", "metadata": metadata, "content": content})
            else:
                history.append({"role": "assistant", "metadata": metadata, "content": content})
                if history[0]["role"] == "system" and "tools" in history[0]:
                    content = "\n".join(content.split("\n")[1:-1])
                    parameters = eval(content)
                    content = {"name": metadata.strip(), "parameters": parameters}
                else:
                    content = {"name": metadata.strip(), "content": content}
        return content, history

    @torch.inference_mode()
    def chat(self, tokenizer, query: str, history: List[Dict] = None, role: str = "user",
             max_length: int = 8192, num_beams=1, do_sample=True, top_p=0.8, temperature=0.8, logits_processor=None,
             **kwargs):
        if history is None:
            history = []
        if logits_processor is None:
            logits_processor = LogitsProcessorList()
        logits_processor.append(InvalidScoreLogitsProcessor())
        gen_kwargs = {"max_length": max_length, "num_beams": num_beams, "do_sample": do_sample, "top_p": top_p,
                      "temperature": temperature, "logits_processor": logits_processor, **kwargs}
        inputs = tokenizer.build_chat_input(query, history=history, role=role)
        inputs = inputs.to(self.device)
        eos_token_id = [tokenizer.eos_token_id, tokenizer.get_command("<|user|>")]
        outputs = self.generate(**inputs, **gen_kwargs, eos_token_id=eos_token_id)
        outputs = outputs.tolist()[0][len(inputs["input_ids"][0]):-1]
        response = tokenizer.decode(outputs)
        history.append({"role": role, "content": query})
        response, history = self.process_response(response, history)
        return response, history
    
# 读取数据
import json
from sentencepiece import SentencePieceTrainer, SentencePieceProcessor

with open('./data/z_medical_stroke_all.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
print(data[0])

# 将数据的instruction替换
for i in range(len(data)):
    data[i]['instruction'] = "现在要做一个根据脑卒中病人的主诉、现病史、既往史和分级表进行严重程度分级的任务。案例会给出主诉、现病史、既往史。\n\t分级如下:\n\t正常\n\t轻度\n\t中度\n\t重度\n\t重度以上\n请根据病人的主诉、现病史、既往史进行分级。\n"

# 现在的数据的output字段是例如 2分 5分, 对其进行转换 0-1:正常 2-4:轻度 5-15:中度 16-20:重度 21-42:重度以上
one = 0
two = 0
three = 0
four = 0
five = 0
for i in range(len(data)):
    if data[i]['output'] == '0分' or data[i]['output'] == '1分':
        data[i]['output'] = "正常"
        one += 1
    elif data[i]['output'] == '2分' or data[i]['output'] == '3分' or data[i]['output'] == '4分':
        data[i]['output'] = "轻度"
        two += 1
    elif data[i]['output'] == '5分' or data[i]['output'] == '6分' or data[i]['output'] == '7分' or data[i]['output'] == '8分' or data[i]['output'] == '9分' or data[i]['output'] == '10分' or data[i]['output'] == '11分' or data[i]['output'] == '12分' or data[i]['output'] == '13分' or data[i]['output'] == '14分' or data[i]['output'] == '15分':
        data[i]['output'] = "中度"
        three += 1
    elif data[i]['output'] == '16分' or data[i]['output'] == '17分' or data[i]['output'] == '18分' or data[i]['output'] == '19分' or data[i]['output'] == '20分':
        data[i]['output'] = "重度"
        four += 1
    elif data[i]['output'] == '21分' or data[i]['output'] == '22分' or data[i]['output'] == '23分' or data[i]['output'] == '24分' or data[i]['output'] == '25分' or data[i]['output'] == '26分' or data[i]['output'] == '27分' or data[i]['output'] == '28分' or data[i]['output'] == '29分' or data[i]['output'] == '30分' or data[i]['output'] == '31分' or data[i]['output'] == '32分' or data[i]['output'] == '33分' or data[i]['output'] == '34分' or data[i]['output'] == '35分' or data[i]['output'] == '36分' or data[i]['output'] == '37分' or data[i]['output'] == '38分' or data[i]['output'] == '39分' or data[i]['output'] == '40分' or data[i]['output'] == '41分' or data[i]['output'] == '42分':
        data[i]['output'] = "重度以上"
        five += 1
print(f"正常:{one} 轻度:{two} 中度:{three} 重度:{four} 重度以上:{five}")



# 将替换后的数据保存为jsonl格式 instuction + input 作为 input字段 output作为output字段
input_len = []
with open('./data/train.jsonl', 'w', encoding='utf-8') as f:
    for i in range(len(data)):
    # for i in range(64):
        f.write(json.dumps({'input': data[i]['instruction'] + data[i]['input'], 'output': data[i]['output']}, ensure_ascii=False) + '\n')
        lens = len(data[i]['instruction'] + data[i]['input'])
        input_len.append(lens)
max_len = max(input_len)
print(f"max_len: {max_len}")

# 正常;轻度;中度各8个作为eval
one = 0; two = 0; three = 0; four = 0; five = 0
eval_data = []
with open('./data/eval.jsonl', 'w', encoding='utf-8') as f:
    for i in range(len(data)):
        if data[i]['output'] == '正常' and one < 8:
            f.write(json.dumps({'input': data[i]['instruction'] + data[i]['input'], 'output': data[i]['output']}, ensure_ascii=False) + '\n')
            one += 1
        elif data[i]['output'] == '轻度' and two < 8:
            f.write(json.dumps({'input': data[i]['instruction'] + data[i]['input'], 'output': data[i]['output']}, ensure_ascii=False) + '\n')
            two += 1
        elif data[i]['output'] == '中度' and three < 8:
            f.write(json.dumps({'input': data[i]['instruction'] + data[i]['input'], 'output': data[i]['output']}, ensure_ascii=False) + '\n')
            three += 1
        else:
            eval_data.append(data[i])
print(f"eval: 正常:{one} 轻度:{two} 中度:{three} 重度:{four} 重度以上:{five}")

# 词表
with open('./data/train.jsonl', 'r', encoding='utf-8') as f:
    train_data = f.readlines()

vocab = sorted(set(''.join([json.loads(i)['input'] for i in train_data])))
print(len(vocab), vocab)

with open('./data/vocab.txt', 'w', encoding='utf-8') as f:
    for word in vocab:
        f.write(repr(word)[1:-1] + '\n')

# 训练sentencepiece
SentencePieceTrainer.Train(
    input='./data/vocab.txt', # 词表路径
    model_prefix='./model/CS_tokenizer', # 保存模型的前缀
    vocab_size=2406, # 词表大小
    model_type='BPE', # 模型类型
)

# 加载sentencepiece模型
sp = SentencePieceProcessor()
sp.Load('./model/CS_tokenizer.model')

# 测试sentencepiece
print(sp.Decode(sp.Encode('现在要做一个根据脑卒中病人的主诉、现病史、既往史和分级表进行严重程度分级的任务。案例会给出主诉、现病史、既往史。\n\t分级如下:\n\t正常\n\t轻度\n\t中度\n\t重度\n\t重度以上\n请根据病人的主诉、现病史、既往史进行分级。\n')))

import json
import os
import re
from typing import List, Optional, Union, Dict
from sentencepiece import SentencePieceProcessor
from transformers import PreTrainedTokenizer
from transformers.utils import logging, PaddingStrategy
from transformers.tokenization_utils_base import EncodedInput, BatchEncoding

class SPTokenizer:
    def __init__(self, model_path: str):
        # 重新加载分词器
        assert os.path.isfile(model_path), model_path
        self.sp_model = SentencePieceProcessor(model_file=model_path)
        # 开始/结束 token 的 id
        self.n_words: int = self.sp_model.vocab_size()
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()
        self.pad_id: int = self.sp_model.unk_id()
        assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()

        role_special_tokens = ["<|system|>", "<|user|>", "<|assistant|>"]
        special_tokens = ["[MASK]", "[gMASK]", "[sMASK]", "sop", "eop"] + role_special_tokens
        self.special_tokens = {}
        self.index_special_tokens = {}
        for token in special_tokens:
            self.special_tokens[token] = self.n_words
            self.index_special_tokens[self.n_words] = token
            self.n_words += 1
        self.role_special_token_expression = "|".join([re.escape(token) for token in role_special_tokens])

    def tokenize(self, s: str, encode_special_tokens=False):
        if encode_special_tokens:
            last_index = 0
            t = []
            for match in re.finditer(self.role_special_token_expression, s):
                if last_index < match.start():
                    t.extend(self.sp_model.EncodeAsPieces(s[last_index:match.start()]))
                t.append(s[match.start():match.end()])
                last_index = match.end()
            if last_index < len(s):
                t.extend(self.sp_model.EncodeAsPieces(s[last_index:]))
            return t
        else:
            return self.sp_model.EncodeAsPieces(s)

    def encode(self, s: str, bos: bool = False, eos: bool = False) -> List[int]:
        assert type(s) is str
        t = self.sp_model.Encode(s)
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t

    def decode(self, t: List[int]) -> str:
        text, buffer = "", []
        for token in t:
            if token in self.index_special_tokens:
                if buffer:
                    text += self.sp_model.Decode(buffer)
                    buffer = []
                text += self.index_special_tokens[token]
            else:
                buffer.append(token)
        if buffer:
            text += self.sp_model.Decode(buffer)
        return text

    def decode_tokens(self, tokens: List[str]) -> str:
        text = self.sp_model.DecodePieces(tokens)
        return text

    def convert_token_to_id(self, token):
        """ 使用词汇表将单词（str）转换为id。"""
        if token in self.special_tokens:
            return self.special_tokens[token]
        return self.sp_model.PieceToId(token)

    def convert_id_to_token(self, index):
        """使用词汇表将索引（整数）转换为单词（str）。"""
        if index in self.index_special_tokens:
            return self.index_special_tokens[index]
        if index in [self.eos_id, self.bos_id, self.pad_id] or index < 0 or index > self.sp_model.vocab_size():
            return ""
        return self.sp_model.IdToPiece(index)


class Tokenizer(PreTrainedTokenizer):
    vocab_files_names = {"vocab_file": "tokenizer.model"}

    model_input_names = ["input_ids", "attention_mask", "position_ids"]

    def __init__(self, vocab_file, padding_side="left", clean_up_tokenization_spaces=False, encode_special_tokens=False,
                 **kwargs):
        self.name = "GLMTokenizer"

        self.vocab_file = vocab_file
        self.tokenizer = SPTokenizer(vocab_file)
        self.special_tokens = {
            "<bos>": self.tokenizer.bos_id,
            "<eos>": self.tokenizer.eos_id,
            "<pad>": self.tokenizer.pad_id
        }
        self.encode_special_tokens = encode_special_tokens
        super().__init__(padding_side=padding_side, clean_up_tokenization_spaces=clean_up_tokenization_spaces,
                         encode_special_tokens=encode_special_tokens,
                         **kwargs)

    def get_command(self, token):
        if token in self.special_tokens:
            return self.special_tokens[token]
        assert token in self.tokenizer.special_tokens, f"{token} is not a special token for {self.name}"
        return self.tokenizer.special_tokens[token]

    @property
    def unk_token(self) -> str:
        return "<unk>"

    @property
    def pad_token(self) -> str:
        return "<unk>"

    @property
    def pad_token_id(self):
        return self.get_command("<pad>")

    @property
    def eos_token(self) -> str:
        return "</s>"

    @property
    def eos_token_id(self):
        return self.get_command("<eos>")

    @property
    def vocab_size(self):
        return self.tokenizer.n_words

    def get_vocab(self):
        """ 将词汇表作为字典返回 """
        vocab = {self._convert_id_to_token(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    def _tokenize(self, text, **kwargs):
        return self.tokenizer.tokenize(text, encode_special_tokens=self.encode_special_tokens)

    def _convert_token_to_id(self, token):
        """ 使用词汇表将单词（str）转换为id。"""
        return self.tokenizer.convert_token_to_id(token)

    def _convert_id_to_token(self, index):
        """使用词汇表将索引（整数）转换为单词（str）。"""
        return self.tokenizer.convert_id_to_token(index)

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        return self.tokenizer.decode_tokens(tokens)

    def get_prefix_tokens(self):
        prefix_tokens = [self.get_command("[gMASK]"), self.get_command("sop")]
        return prefix_tokens
    
    def build_single_message(self, role, metadata, message):
        assert role in ["system", "user", "assistant"], role
        role_tokens = [self.get_command(f"<|{role}|>")] + self.tokenizer.encode(f"{metadata}\n")
        message_tokens = self.tokenizer.encode(message)
        tokens = role_tokens + message_tokens
        return tokens
    
    def build_chat_input(self, query, history=None, role="user"):
        if history is None:
            history = []
        input_ids = []
        for item in history:
            content = item["content"]
            if item["role"] == "system" and "tools" in item:
                content = content + "\n" + json.dumps(item["tools"], indent=4, ensure_ascii=False)
            input_ids.extend(self.build_single_message(item["role"], item.get("metadata", ""), content))
        input_ids.extend(self.build_single_message(role, "", query))
        input_ids.extend([self.get_command("<|assistant|>")])
        return self.batch_encode_plus([input_ids], return_tensors="pt", is_split_into_words=True)


    def build_inputs_with_special_tokens(
            self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        通过连接和添加特殊标记，从序列或序列对构建用于序列分类任务的模型输入。BERT序列具有以下格式:

        - 单个序列: `[CLS] X [SEP]`
        - 一对序列: `[CLS] A [SEP] B [SEP]`

        参数:
            token_ids_0 (`List[int]`):
                要添加特殊标记的ID列表。
            token_ids_1 (`List[int]`, *optional*):
                序列对的可选第二个ID列表。

        返回:
            `List[int]`: 带有适当的特殊标记的[input IDs](../glossary#input-ids)列表。
        """
        prefix_tokens = self.get_prefix_tokens()
        token_ids_0 = prefix_tokens + token_ids_0
        if token_ids_1 is not None:
            token_ids_0 = token_ids_0 + token_ids_1 + [self.get_command("<eos>")]
        return token_ids_0

    def _pad(
            self,
            encoded_inputs: Union[Dict[str, EncodedInput], BatchEncoding],
            max_length: Optional[int] = None,
            padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
            pad_to_multiple_of: Optional[int] = None,
            return_attention_mask: Optional[bool] = None,
    ) -> dict:
        """
        填充编码的输入（在左/右和预定义长度或批次中的最大长度）

        参数:
            encoded_inputs:
                标记化输入（`List[int]`）的字典或标记化输入（`List[List[int]]`）的批处理。
            max_length: 返回的列表的最大长度和可选的填充长度（见下文）。
                将通过考虑特殊标记来截断。
            padding_strategy: 用于填充的PaddingStrategy。

                - PaddingStrategy.LONGEST 填充到批处理中最长的序列
                - PaddingStrategy.MAX_LENGTH: 填充到最大长度（默认）
                - PaddingStrategy.DO_NOT_PAD: 不填充
                分词器填充边在self.padding_side中定义:

                    - 'left': 在序列的左侧填充
                    - 'right': 在序列的右侧填充
            pad_to_multiple_of: (optional) 如果设置，将序列填充到提供的值的倍数。
                这对于启用NVIDIA硬件上的Tensor Core特别有用，计算能力为`>= 7.5`（Volta）。
            return_attention_mask:
                (optional) 设置为False以避免返回注意力掩码（默认值：设置为模型特定值）

        返回:
            带有填充的标记化输入的字典。
        """
        # 从模型默认加载
        assert self.padding_side == "left"

        required_input = encoded_inputs[self.model_input_names[0]]
        seq_length = len(required_input)

        if padding_strategy == PaddingStrategy.LONGEST:
            max_length = len(required_input)

        if max_length is not None and pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
            max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of

        needs_to_be_padded = padding_strategy != PaddingStrategy.DO_NOT_PAD and len(required_input) != max_length

        # 如果不存在，则初始化注意力掩码。
        if "attention_mask" not in encoded_inputs:
            encoded_inputs["attention_mask"] = [1] * seq_length

        if "position_ids" not in encoded_inputs:
            encoded_inputs["position_ids"] = list(range(seq_length))

        if needs_to_be_padded:
            difference = max_length - len(required_input)

            if "attention_mask" in encoded_inputs:
                encoded_inputs["attention_mask"] = [0] * difference + encoded_inputs["attention_mask"]
            if "position_ids" in encoded_inputs:
                encoded_inputs["position_ids"] = [0] * difference + encoded_inputs["position_ids"]
            encoded_inputs[self.model_input_names[0]] = [self.pad_token_id] * difference + required_input

        return encoded_inputs

tokenizer = Tokenizer(vocab_file='./model/CS_tokenizer.model')

from transformers import PretrainedConfig


class Config(PretrainedConfig):
    def __init__(
        self,
        num_layers=12,
        padded_vocab_size=tokenizer.vocab_size,
        hidden_size=768,
        ffn_hidden_size=768*4,
        kv_channels=48,
        num_attention_heads=12,
        seq_length=2048,
        hidden_dropout=0.0,
        classifier_dropout=0.1,
        attention_dropout=0.0,
        layernorm_epsilon=1e-5,
        rmsnorm=True,
        apply_residual_connection_post_layernorm=False,
        post_layer_norm=True,
        add_bias_linear=False,
        add_qkv_bias=True,
        bias_dropout_fusion=True,
        multi_query_attention=False,
        multi_query_group_num=2,
        apply_query_key_layer_scaling=True,
        attention_softmax_in_fp32=True,
        fp32_residual_connection=False,
        quantization_bit=0,
        pre_seq_len=None,
        prefix_projection=False,
        original_rope = True,
        use_cache=True,
        activation_function="swiglu",
        torch_dtype="float16",
        tie_word_embeddings=False,
        eos_token_id=2,
        pad_token_id=0,
        num_labels=0,
        problem_type=None,
        **kwargs
    ):
        self.num_layers = num_layers
        self.vocab_size = padded_vocab_size
        self.padded_vocab_size = padded_vocab_size
        self.hidden_size = hidden_size
        self.ffn_hidden_size = ffn_hidden_size
        self.kv_channels = kv_channels
        self.num_attention_heads = num_attention_heads
        self.seq_length = seq_length
        self.hidden_dropout = hidden_dropout
        self.classifier_dropout = classifier_dropout
        self.attention_dropout = attention_dropout
        self.layernorm_epsilon = layernorm_epsilon
        self.rmsnorm = rmsnorm
        self.apply_residual_connection_post_layernorm = apply_residual_connection_post_layernorm
        self.post_layer_norm = post_layer_norm
        self.add_bias_linear = add_bias_linear
        self.add_qkv_bias = add_qkv_bias
        self.bias_dropout_fusion = bias_dropout_fusion
        self.multi_query_attention = multi_query_attention
        self.multi_query_group_num = multi_query_group_num
        self.apply_query_key_layer_scaling = apply_query_key_layer_scaling
        self.attention_softmax_in_fp32 = attention_softmax_in_fp32
        self.fp32_residual_connection = fp32_residual_connection
        self.quantization_bit = quantization_bit
        self.pre_seq_len = pre_seq_len
        self.prefix_projection = prefix_projection
        self.original_rope = original_rope
        self.use_cache = use_cache
        self.activation_function = activation_function
        self.torch_dtype = torch_dtype
        self.tie_word_embeddings = tie_word_embeddings
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.num_labels = num_labels
        self.problem_type = problem_type
        
        super().__init__(**kwargs)

import os
import torch


config = Config()
config.vocab_size = tokenizer.vocab_size
config.num_layers = 12
config.problem_type = "multi_label_classification"
config.num_labels = 5


model = SequenceClassification(config) # 用于序列分类
# model = ConditionalGeneration(config) # 用于条件生成
# cuda可用且是Linux系统时才使用torch.compile
if torch.cuda.is_available() and os.name == 'posix':
    model = torch.compile(model)

print(model, model.num_parameters())
# print(model.chat(tokenizer, "现在要做一个根据脑卒中病人的主诉、现病史、既往史和分级表进行严重程度分级的任务。案例会给出主诉、现病史、既往史。\n\t分级如下:\n\t正常\n\t轻度\n\t中度\n\t重度\n\t重度以上\n请根据病人的主诉、现病史、既往史进行分级。\n", history=[], role="user", max_length=128)) if model.__class__.__name__ == "ConditionalGeneration" else None
# output = model.forward(input_ids=torch.tensor([tokenizer.encode("现在要做一个根据脑卒中病人的主诉、现病史、既往史和分级表进行严重程度分级的任务。案例会给出主诉、现病史、既往史。\n\t分级如下:\n\t正常\n\t轻度\n\t中度\n\t重度\n\t重度以上\n请根据病人的主诉、现病史、既往史进行分级。\n")]), labels=torch.tensor([tokenizer.encode("现在要做一个根据脑卒中病人的主诉、现病史、既往史和分级表进行严重程度分级的任务。案例会给出主诉、现病史、既往史。\n\t分级如下:\n\t正常\n\t轻度\n\t中度\n\t重度\n\t重度以上\n请根据病人的主诉、现病史、既往史进行分级。\n")]), return_dict=True)

# 构建训练数据
import json
import torch
from torch.utils.data import Dataset, DataLoader

# input字段是训练数据的输入, output字段是训练数据的目标输出 正常编码为0 轻度编码为1 中度编码为2 重度编码为3 重度以上编码为4
class MyDataset(Dataset):
    def __init__(self, file_path, tokenizer: Tokenizer, model_type, max_length=2048, padding='max_length', truncation=True, problem_type="single_label_classification"):
        self.data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = json.loads(line)
                self.data.append((line['input'], line['output']))
        self.tokenizer = tokenizer
        self.model_type = model_type
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation
        self.problem_type = problem_type
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        input_text = self.data[idx][0]
        output_text = self.data[idx][1]
        if self.model_type == "SequenceClassification": # 序列分类 文本 + 标签
            input_ids = self.tokenizer.encode(input_text, padding=self.padding, max_length=self.max_length, truncation=self.truncation)
            
            if output_text == '正常':
                label = 0
            elif output_text == '轻度':
                label = 1
            elif output_text == '中度':
                label = 2
            elif output_text == '重度':
                label = 3
            elif output_text == '重度以上':
                label = 4

            if self.problem_type == "single_label_classification":
                return torch.tensor(input_ids, dtype=torch.long), torch.tensor(label, dtype=torch.long)
            elif self.problem_type == "multi_label_classification":
                return torch.tensor(input_ids, dtype=torch.long), torch.nn.functional.one_hot(torch.tensor(label), num_classes=config.num_labels).float()
        elif self.model_type == "ConditionalGeneration": # 条件生成 文本
            input_ids = self.tokenizer.encode(input_text+output_text, padding=self.padding, max_length=self.max_length, truncation=self.truncation)
            return torch.tensor(input_ids, dtype=torch.long), torch.tensor(input_ids, dtype=torch.long)

    
# 构建训练数据集
train_data = MyDataset('./data/train.jsonl', tokenizer, model.__class__.__name__, problem_type=config.problem_type)
train_loader = DataLoader(train_data, batch_size=8, shuffle=True) # NOTE 这样刚好可以在3090上跑起来
# 构建验证数据集
eval_data = MyDataset('./data/eval.jsonl', tokenizer, model.__class__.__name__, problem_type=config.problem_type)
eval_loader = DataLoader(eval_data, batch_size=8, shuffle=True)

for i, (input_ids, label) in enumerate(train_loader):
    print(tokenizer.decode(input_ids[0].tolist()))
    print(label[0])
    break

# 学习率
def lrate(
    step: int,
    warmup_steps: int = len(train_loader),
    d_model: int = config.hidden_size,
) -> float:
    """学习率计算"""
    if step == 0:
        return 0.0
    return (d_model ** -0.5) * min(step ** -0.5, step * warmup_steps ** -1.5)

# 优化器, 学习率调度器
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
optimizer = AdamW(model.parameters(), lr=1, betas=(0.9, 0.98), eps=1e-9)
scheduler = LambdaLR(optimizer, lr_lambda=lrate)
scaler = torch.cuda.amp.GradScaler(enabled=True) if device.type == 'cuda' else None

# 训练函数
import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


torch.manual_seed(233)
# torch.autograd.set_detect_anomaly(True)  # 启用异常检测

def trainer(
    model: torch.nn.Module,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    tokenizer: Tokenizer = tokenizer,
    device: torch.device = device,
    optimizer: torch.optim.Optimizer = optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR = scheduler,
    scaler: torch.cuda.amp.GradScaler = scaler,
    epochs: int = 10,
):
    """训练函数"""
    # 分布式训练 命令 python -m torch.distributed.launch --nproc_per_node=4 train.py
    
    if torch.cuda.device_count() > 1 and os.name == 'posix':
        rank = int(os.environ['RANK'])
        torch.cuda.set_device(rank % torch.cuda.device_count())
        dist.init_process_group(backend="nccl")
        model.to(device)
        model = DDP(model, device_ids=[torch.cuda.current_device()])
    else:
        model.to(device)
    global_step = 0
    train_loss = []
    acc = 0.0
    ppl = 0.0

    for epoch in range(epochs):
        model.train()
        for step, batch in enumerate(train_loader):
            input_ids, labels = batch
            input_ids, labels = input_ids.to(device), labels.to(device)
            
            # 梯度清零
            optimizer.zero_grad(set_to_none=True)
            # 前向传播
            if scaler is not None:
                with torch.cuda.amp.autocast(enabled=True):
                    outputs = model(input_ids, labels=labels)
                    loss = outputs.loss
                # 反向传播
                scaler.scale(loss).backward()

                # 梯度裁剪
                scaler.unscale_(optimizer) # 去除放大的梯度
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # 梯度裁剪
                # 更新参数
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(input_ids, labels=labels)
                print(outputs.logits, labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
            # 更新学习率
            scheduler.step(global_step)
            global_step += 1
            train_loss.append(loss.item())

            # 打印训练信息
            if step % (len(train_loader) // 10) == 0:
            # if (step+1) % 3 == 0:
                train_loss = torch.tensor(train_loss).mean().item()
                print(f"Epoch: [{epoch+1}/{epochs}] Step: [{step}/{len(train_loader)}] Learning rate: {optimizer.param_groups[0]['lr']:.8f} Loss: {train_loss:.8f}")
                train_loss = []

        # 验证模型
        # if epoch % (epochs // 10) == 0:
        if epoch % 1 == 0:
            # 验证模型
            model.eval()
            if model.__class__.__name__ == "SequenceClassification":
                for step, batch in enumerate(eval_loader):
                    input_ids, labels = batch
                    input_ids, labels = input_ids.to(device), labels.to(device)
                    with torch.no_grad():
                        outputs = model(input_ids, labels=labels)
                        if config.problem_type == "single_label_classification":
                            acc += (outputs.logits.argmax(dim=-1) == labels).sum().item()
                            print(f"\n===\nlabel: {labels.tolist()} \npredict: {outputs.logits.argmax(dim=-1).tolist()}\n===\n")
                        elif config.problem_type == "multi_label_classification":
                            acc += torch.sum((outputs.logits.argmax(dim=-1) == labels.argmax(dim=-1)))
                            print(f"\n===\nlabel:\t\t {labels.argmax(dim=-1).tolist()} \npredict:\t {outputs.logits.argmax(dim=-1).tolist()}\n===\n")
                    ppl += torch.exp(outputs.loss)
                acc /= len(eval_loader.dataset)
                ppl /= len(eval_loader.dataset)
                print(f"Epoch: [{epoch+1}/{epochs}] Accuracy: {acc:.4f} Perplexity: {ppl:.4f}")
                acc = 0.0
                ppl = 0.0
                
            elif model.__class__.__name__ == "ConditionalGeneration":
                # 用chat方法进行验证
                with open('./data/eval.jsonl', 'r', encoding='utf-8') as f:
                    eval_data = f.readlines()
                # input作为问题, output作为答案
                eval_questions = [json.loads(i)['input'] for i in eval_data]
                eval_answers = [json.loads(i)['output'] for i in eval_data]
                model_answers = []
                for i in range(len(eval_questions)):
                    model_answer, _ = model.chat(tokenizer, eval_questions[i], history=[], role="user", max_length=2048)
                    model_answers.append(model_answer)
                    print(f"answer: {eval_answers[i]} \nmodel_answer: {model_answer}\n")
                # 计算准确率
                acc = sum([1 for i in range(len(eval_answers)) if eval_answers[i] == model_answers[i]]) / len(eval_answers)
                print(f"Epoch: [{epoch+1}/{epochs}] Accuracy: {acc:.4f}")
                acc = 0.0

trainer(model, train_loader, eval_loader, tokenizer, device, optimizer, scheduler, scaler, epochs=100)
model.save_pretrained('./model/脑卒中严重程度分级')