from dataclasses import dataclass
from typing import Optional
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from torch.nn import functional as F

from visualize import visualize_patch


def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """
    Drop paths（随机深度）按样本应用（在残差块的主路径中）。

    参数:
        x (torch.Tensor): 输入张量。
        drop_prob (float): 丢弃概率，范围在[0, 1]之间。默认值为0.0。
        training (bool): 是否处于训练模式。默认值为False。
        scale_by_keep (bool): 是否根据保留概率进行缩放。默认值为True。
    
    返回:
        torch.Tensor: 应用drop path后的张量。
    """
    if drop_prob == 0. or not training:
        # 如果丢弃概率为0或不在训练模式下，直接返回输入张量
        return x
    # 计算保留概率
    keep_prob = 1 - drop_prob
    # 创建一个与x形状相同但除了第一个维度外其余维度为1的形状，用于广播
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    # 根据保留概率生成一个与x形状相同的随机张量
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        # 如果需要，根据保留概率缩放随机张量
        random_tensor.div_(keep_prob)
    # 将输入张量与随机张量相乘，实现drop path
    return x * random_tensor


class DropPath(torch.nn.Module):
    """
    Drop paths（随机深度）按样本应用（在残差块的主路径中）。

    参数:
        drop_prob (float): 丢弃概率，范围在[0, 1]之间。默认值为0.0。
        scale_by_keep (bool): 是否根据保留概率进行缩放。默认值为True。
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        """
        前向传播函数。

        参数:
            x (torch.Tensor): 输入张量。
        
        返回:
            torch.Tensor: 应用drop path后的张量。
        """
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        """
        返回对象的额外表示信息。

        返回:
            str: 包含丢弃概率的字符串表示。
        """
        return f'drop_prob={round(self.drop_prob,3):0.3f}'


def find_multiple(n: int, k: int):
    """
    找到大于或等于n的最小k的倍数。

    参数:
        n (int): 目标整数。
        k (int): 倍数因子。
    
    返回:
        int: 大于或等于n的最小k的倍数。
    """
    if n % k == 0:
        # 如果n已经是k的倍数，直接返回n
        return n
    # 否则，返回n加上k减去n除以k的余数
    return n + k - (n % k)


@dataclass
class ModelArgs:
    """
    模型参数配置类。

    参数:
        dim (int): 模型维度。默认值为4096。
        n_layer (int): 层数。默认值为32。
        n_head (int): 多头注意力机制中的头数。默认值为32。
        n_kv_head (Optional[int]): KV多头注意力机制中的头数。如果为None，则默认为n_head。默认值为None。
        multiple_of (int): SwiGLU隐藏层大小的倍数，确保其为2的幂次方。默认值为256。
        ffn_dim_multiplier (Optional[float]): FFN维度乘数。如果为None，则不应用。默认值为None。
        rope_base (float): ROPE（旋转位置编码）的基值。默认值为10000。
        norm_eps (float): 归一化层的epsilon值。默认值为1e-5。
        initializer_range (float): 初始化范围。默认值为0.02。
        token_dropout_p (float): Token dropout概率。默认值为0.1。
        attn_dropout_p (float): 注意力机制dropout概率。默认值为0.0。
        resid_dropout_p (float): 残差连接dropout概率。默认值为0.1。
        ffn_dropout_p (float): FFN dropout概率。默认值为0.1。
        drop_path_rate (float): Drop path率。默认值为0.0。
        num_classes (int): 类别数。默认值为1000。
        caption_dim (int): 字幕维度。默认值为2048。
        class_dropout_prob (float): 分类器dropout概率。默认值为0.1。
        model_type (str): 模型类型。默认值为'c2i'。
        vocab_size (int): 词汇表大小。默认值为16384。
        cls_token_num (int): CLS token数量。默认值为1。
        block_size (int): 块大小。默认值为256。
        max_batch_size (int): 最大批量大小。默认值为32。
        max_seq_len (int): 最大序列长度。默认值为2048。
    """
    dim: int = 4096
    n_layer: int = 32
    n_head: int = 32
    n_kv_head: Optional[int] = None
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    rope_base: float = 10000
    norm_eps: float = 1e-5
    initializer_range: float = 0.02

    token_dropout_p: float = 0.1
    attn_dropout_p: float = 0.0
    resid_dropout_p: float = 0.1
    ffn_dropout_p: float = 0.1
    drop_path_rate: float = 0.0

    num_classes: int = 1000
    caption_dim: int = 2048
    class_dropout_prob: float = 0.1
    model_type: str = 'c2i'

    vocab_size: int = 16384
    cls_token_num: int = 1
    block_size: int = 256
    max_batch_size: int = 32
    max_seq_len: int = 2048


class LabelEmbedder(nn.Module):
    """
    标签嵌入器。将类标签嵌入到向量表示中。同时处理用于无分类器指导的标签dropout。

    参数:
        num_classes (int): 类别数。
        hidden_size (int): 嵌入向量的维度。
        dropout_prob (float): Dropout概率。用于无分类器指导的标签dropout。
    """

    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        # 判断是否使用CFG嵌入
        use_cfg_embedding = dropout_prob > 0
        # 初始化嵌入表
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        # 类别数
        self.num_classes = num_classes
        # Dropout概率
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        丢弃标签以启用无分类器指导。

        参数:
            labels (torch.Tensor): 标签张量。
            force_drop_ids (torch.Tensor, optional): 强制丢弃的ID。如果为None，则随机丢弃。默认值为None。
        
        返回:
            torch.Tensor: 丢弃后的标签张量。
        """
        if force_drop_ids is None:
            # 如果没有强制丢弃ID，则随机生成drop_ids
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            # 否则，根据force_drop_ids确定drop_ids
            drop_ids = force_drop_ids == 1
        # 将需要丢弃的标签替换为num_classes
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        """
        前向传播函数。

        参数:
            labels (torch.Tensor): 标签张量。
            train (bool): 是否处于训练模式。
            force_drop_ids (torch.Tensor, optional): 强制丢弃的ID。如果为None，则不强制。默认值为None。
        
        返回:
            torch.Tensor: 嵌入后的张量。
        """
        # 判断是否使用dropout
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            # 如果在训练模式下且使用dropout，或者有强制丢弃ID，则进行标签dropout
            labels = self.token_drop(labels, force_drop_ids)
        # 将标签嵌入到向量中，并在第二个维度上增加一个维度
        embeddings = self.embedding_table(labels).unsqueeze(1)
        return embeddings


class RMSNorm(torch.nn.Module):
    """
    RMSNorm（均方根归一化）层。

    参数:
        dim (int): 输入特征的维度。
        eps (float): 用于数值稳定的极小常数，默认为1e-5。
    """
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        # 初始化极小常数
        self.eps = eps
        # 初始化可学习的权重参数，形状为(dim,)
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        """
        计算输入张量的RMS归一化。

        参数:
            x (torch.Tensor): 输入张量，形状为 (..., dim)。

        返回:
            torch.Tensor: 归一化后的张量。
        """
        # 计算x的均方根（RMS），即 sqrt(mean(x^2) + eps)
        # 对输入张量进行归一化
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        """
        前向传播函数。

        参数:
            x (torch.Tensor): 输入张量，形状为 (..., dim)。

        返回:
            torch.Tensor: 应用RMS归一化后的张量。
        """
        # 将输入张量转换为float类型，计算归一化后，再转换回原始类型
        output = self._norm(x.float()).type_as(x)
        # 将归一化后的张量乘以权重参数
        return output * self.weight


class FeedForward(nn.Module):
    """
    前馈神经网络层（Feed-Forward Network, FFN），通常用于Transformer架构中。

    参数:
        config (ModelArgs): 模型配置参数，包含模型维度等设置。
    """
    def __init__(self, config: ModelArgs):
        super().__init__()
        # 隐藏层维度，默认为模型维度的4倍
        hidden_dim = 4 * config.dim
        # 调整隐藏层维度为原来的2/3
        hidden_dim = int(2 * hidden_dim / 3)
        # 自定义维度因子乘数
        if config.ffn_dim_multiplier is not None:
            hidden_dim = int(config.ffn_dim_multiplier * hidden_dim)
        # 确保隐藏层维度是multiple_of的倍数
        hidden_dim = find_multiple(hidden_dim, config.multiple_of)

        # 定义三个线性层，w1和w3用于输入的变换，w2用于输出的变换
        self.w1 = nn.Linear(config.dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(config.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, config.dim, bias=False)
        # 定义dropout层，防止过拟合并增强泛化能力
        self.ffn_dropout = nn.Dropout(config.ffn_dropout_p)

    def forward(self, x):
        """
        前向传播函数。

        参数:
            x (torch.Tensor): 输入张量，形状为 (batch_size, seq_length, dim)。

        返回:
            torch.Tensor: FFN层的输出，形状为 (batch_size, seq_length, dim)。
        """
        # 计算w1(x)并应用SiLU激活函数，然后与w3(x)相乘
        # 应用dropout
        # 通过w2层得到最终输出
        return self.ffn_dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class KVCache(nn.Module):
    """
    KV缓存，用于存储键（Key）和值（Value），以加速自注意力机制的计算。

    参数:
        max_batch_size (int): 最大批量大小。
        max_seq_length (int): 最大序列长度。
        n_head (int): 多头注意力机制中的头数。
        head_dim (int): 每个注意力头的维度。
    """
    def __init__(self, max_batch_size, max_seq_length, n_head, head_dim):
        super().__init__()
        # 定义缓存的形状
        cache_shape = (max_batch_size, n_head, max_seq_length, head_dim)
        # 初始化键和值的缓存张量，初始值为零
        self.register_buffer('k_cache', torch.zeros(cache_shape))
        self.register_buffer('v_cache', torch.zeros(cache_shape))

    def update(self, input_pos, k_val, v_val):
        """
        更新缓存中的键和值。

        参数:
            input_pos (torch.Tensor): 输入位置索引，形状为 (S,)。
            k_val (torch.Tensor): 键值张量，形状为 (B, H, S, D)。
            v_val (torch.Tensor): 值张量，形状为 (B, H, S, D)。

        返回:
            tuple: 更新后的键和值缓存。
        """
        # 获取当前键和值缓存
        k_out = self.k_cache
        v_out = self.v_cache
        # 将新的键和值写入缓存中指定的位置
        k_out[:, :, input_pos] = k_val.to(k_out.dtype)
        v_out[:, :, input_pos] = v_val.to(k_out.dtype)

        return k_out, v_out


def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None) -> torch.Tensor:
    """
    计算缩放点积注意力（Scaled Dot-Product Attention）。

    参数:
        query (torch.Tensor): 查询张量，形状为 (L, B, H, D)。
        key (torch.Tensor): 键张量，形状为 (S, B, H, D)。
        value (torch.Tensor): 值张量，形状为 (S, B, H, D)。
        attn_mask (torch.Tensor, optional): 注意力掩码。默认为None。
        dropout_p (float): Dropout概率，默认为0.0。
        is_causal (bool): 是否使用因果掩码。默认为False。
        scale (float, optional): 缩放因子。如果为None，则使用1 / sqrt(D)。默认为None。

    返回:
        torch.Tensor: 注意力输出，形状为 (L, B, H, D)。
    """
    # 获取查询和键的序列长度
    L, S = query.size(-2), key.size(-2)
    # 计算缩放因子
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    # 初始化注意力偏差张量
    attn_bias = torch.zeros(L, S, dtype=query.dtype).cuda()
    if is_causal:
        # 如果使用因果掩码，则不应有其他掩码
        assert attn_mask is None
        # 生成一个上三角为1，下三角为0的布尔掩码
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0).cuda()
        # 将非上三角位置填充为负无穷
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            # 如果掩码是布尔类型，则将非掩码位置填充为负无穷
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            # 否则，将掩码直接加到注意力偏差中
            attn_bias += attn_mask
    with torch.cuda.amp.autocast(enabled=False):
        # 计算缩放点积注意力权重
        attn_weight = query.float() @ key.float().transpose(-2, -1) * scale_factor
    # 添加注意力偏差
    attn_weight += attn_bias
    # 应用softmax激活函数
    attn_weight = torch.softmax(attn_weight, dim=-1)
    # 应用dropout
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    # 返回注意力输出
    return attn_weight @ value


class Attention(nn.Module):
    """
    自注意力机制层。

    参数:
        config (ModelArgs): 模型配置参数，包含模型维度、多头数等设置。
    """
    def __init__(self, config: ModelArgs):
        super().__init__()
        # 确保模型维度可以被多头数整除
        assert config.dim % config.n_head == 0
        # 模型维度
        self.dim = config.dim
        # 每个注意力头的维度
        self.head_dim = config.dim // config.n_head
        # 多头数
        self.n_head = config.n_head
        # KV多头数
        self.n_kv_head = config.n_kv_head if config.n_kv_head is not None else config.n_head
        # 总KV维度
        total_kv_dim = (self.n_head + 2 * self.n_kv_head) * self.head_dim

        # 为所有注意力头初始化键、查询、值的投影矩阵
        self.wqkv = nn.Linear(config.dim, total_kv_dim, bias=False)
        self.wo = nn.Linear(config.dim, config.dim, bias=False)
        # KV缓存初始化为None
        self.kv_cache = None

        # 正则化参数
        # 注意力dropout概率
        self.attn_dropout_p = config.attn_dropout_p
        # 残差dropout
        self.resid_dropout = nn.Dropout(config.resid_dropout_p)

    def forward(
            self, x: torch.Tensor, freqs_cis=None, input_pos=None, mask=None
    ):
        """
        前向传播函数。

        参数:
            x (torch.Tensor): 输入张量，形状为 (batch_size, seq_length, dim)。
            freqs_cis (torch.Tensor, optional): 旋转位置编码，默认为None。
            input_pos (torch.Tensor, optional): 输入位置索引，默认为None。
            mask (torch.Tensor, optional): 注意力掩码，默认为None。

        返回:
            torch.Tensor: 自注意力机制的输出，形状为 (batch_size, seq_length, dim)。
        """
        # 获取批量大小和序列长度
        bsz, seqlen, _ = x.shape
        # KV头的总维度
        kv_size = self.n_kv_head * self.head_dim
        # 将输入张量分割为查询、键、值张量
        xq, xk, xv = self.wqkv(x).split([self.dim, kv_size, kv_size], dim=-1)

        # 重塑查询、键、值张量的形状以适应多头结构
        xq = xq.view(bsz, seqlen, self.n_head, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_head, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_head, self.head_dim)

        # 应用旋转位置编码
        xq = apply_rotary_emb(xq, freqs_cis)
        xk = apply_rotary_emb(xk, freqs_cis)

        # 转置张量以适应后续的注意力计算
        xq, xk, xv = map(lambda x: x.transpose(1, 2), (xq, xk, xv))

        if self.kv_cache is not None:
            # 如果有KV缓存，则更新缓存
            keys, values = self.kv_cache.update(input_pos, xk, xv)
        else:
            # 否则，直接使用键和值
            keys, values = xk, xv
        # 重复键和值以适应多头结构
        keys = keys.repeat_interleave(self.n_head // self.n_kv_head, dim=1)
        values = values.repeat_interleave(self.n_head // self.n_kv_head, dim=1)

        # 计算缩放点积注意力
        output = scaled_dot_product_attention(
            xq, keys, values,
            attn_mask=mask,
            is_causal=True if mask is None else False,  # is_causal=False is for KV cache
            dropout_p=self.attn_dropout_p if self.training else 0)
        
        # 转置张量并重塑形状以适应输出维度
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim)

        # 应用残差dropout
        output = self.resid_dropout(self.wo(output))
        return output


class TransformerBlock(nn.Module):
    """
    Transformer 块，包含自注意力机制、前馈神经网络、归一化和残差连接。

    参数:
        config (ModelArgs): 模型配置参数，包含模型维度、多头数等设置。
        drop_path (float): DropPath的概率，用于随机深度正则化。
    """
    def __init__(self, config: ModelArgs, drop_path: float):
        super().__init__()
        # 初始化自注意力机制层
        self.attention = Attention(config)
        # 初始化前馈神经网络层
        self.feed_forward = FeedForward(config)
        # 初始化注意力层的RMS归一化
        self.attention_norm = RMSNorm(config.dim, eps=config.norm_eps)
        # 初始化前馈层的RMS归一化
        self.ffn_norm = RMSNorm(config.dim, eps=config.norm_eps)
        # 如果drop_path概率大于0，则使用DropPath层，否则使用恒等层
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(
            self, x: torch.Tensor, freqs_cis: torch.Tensor, start_pos: int, mask: Optional[torch.Tensor] = None):
        """
        前向传播函数。

        参数:
            x (torch.Tensor): 输入张量，形状为 (batch_size, seq_length, dim)。
            freqs_cis (torch.Tensor): 旋转位置编码，形状为 (seq_length, head_dim // 2, 2)。
            start_pos (int): 起始位置索引。
            mask (Optional[torch.Tensor]): 注意力掩码，默认为None。

        返回:
            torch.Tensor: Transformer块的输出，形状为 (batch_size, seq_length, dim)。
        """
        # 自注意力机制 + DropPath + 残差连接
        h = x + self.drop_path(self.attention(self.attention_norm(x), freqs_cis, start_pos, mask))
        # 前馈神经网络 + DropPath + 残差连接
        out = h + self.drop_path(self.feed_forward(self.ffn_norm(h)))
        return out


def precompute_freqs_cis(seq_len: int, n_elem: int, base: int = 10000, cls_token_num=120):
    """
    预计算旋转位置编码的复数表示。

    参数:
        seq_len (int): 序列长度。
        n_elem (int): 元素数量。
        base (int): 基数，默认为10000。
        cls_token_num (int): CLS token的数量，默认为120。

    返回:
        torch.Tensor: 预计算的旋转位置编码，形状为 (cls_token_num + seq_len, head_dim // 2, 2)。
    """
    # 计算频率
    freqs = 1.0 / (base ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem))
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs)  # (seq_len, head_dim // 2)
    # 计算复数表示
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    # 堆叠实部和虚部
    cache = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1)  # (cls_token_num+seq_len, head_dim // 2, 2)
    # 添加CLS token的编码
    cond_cache = torch.cat(
        [torch.zeros(cls_token_num, n_elem // 2, 2), cache])  # (cls_token_num+seq_len, head_dim // 2, 2)
    return cond_cache


def precompute_freqs_cis_2d(grid_size: int, n_elem: int, base: int = 10000, cls_token_num=120):
    """
    预计算二维旋转位置编码的复数表示。

    参数:
        grid_size (int): 网格大小。
        n_elem (int): 元素数量。
        base (int): 基数，默认为10000。
        cls_token_num (int): CLS token的数量，默认为120。

    返回:
        torch.Tensor: 预计算的二维旋转位置编码，形状为 (cls_token_num + grid_size ** 2, head_dim // 2, 2)。
    """
    # split the dimension into half, one for x and one for y
    # 计算一半的维度
    half_dim = n_elem // 2
    freqs = 1.0 / (base ** (torch.arange(0, half_dim, 2)[: (half_dim // 2)].float() / half_dim))
    t = torch.arange(grid_size, device=freqs.device)
    freqs = torch.outer(t, freqs)  # (grid_size, head_dim // 2)
    # 计算二维频率网格
    freqs_grid = torch.concat([
        freqs[:, None, :].expand(-1, grid_size, -1),
        freqs[None, :, :].expand(grid_size, -1, -1),
    ], dim=-1)  # (grid_size, grid_size, head_dim // 2)
    # 计算复数表示
    cache_grid = torch.stack([torch.cos(freqs_grid), torch.sin(freqs_grid)],
                             dim=-1)  # (grid_size, grid_size, head_dim // 2, 2)
    cache = cache_grid.flatten(0, 1)
    # 添加CLS token的编码
    cond_cache = torch.cat(
        [torch.zeros(cls_token_num, n_elem // 2, 2), cache])  # (cls_token_num+grid_size**2, head_dim // 2, 2)
    return cond_cache


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor):
    """
    应用旋转位置编码到输入张量。

    参数:
        x (torch.Tensor): 输入张量，形状为 (batch_size, seq_length, n_head, head_dim)。
        freqs_cis (torch.Tensor): 旋转位置编码，形状为 (seq_length, head_dim // 2, 2)。

    返回:
        torch.Tensor: 应用旋转位置编码后的张量。
    """
    # 重塑x以适应旋转位置编码
    # x: (bs, seq_len, n_head, head_dim)
    # freqs_cis (seq_len, head_dim // 2, 2)
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2)  # (bs, seq_len, n_head, head_dim//2, 2)
    # 重塑freqs_cis以匹配x的形状
    freqs_cis = freqs_cis.view(1, xshaped.size(1), 1, xshaped.size(3), 2)  # (1, seq_len, 1, head_dim//2, 2)
    # 应用旋转位置编码
    x_out2 = torch.stack([
        xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
        xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1],
    ], dim=-1)
    x_out2 = x_out2.flatten(3)
    return x_out2.type_as(x)


class AR(nn.Module):
    """
    自回归模型（Autoregressive Model），用于生成序列数据。

    参数:
        seq_len (int): 序列长度。
        patch_size (int): 补丁大小。
        cond_embed_dim (int): 条件嵌入的维度。
        embed_dim (int): 嵌入的维度。
        num_blocks (int): Transformer块的数量。
        num_heads (int): 多头注意力机制中的头数。
        grad_checkpointing (bool): 是否启用梯度检查点，默认为False。
        **kwargs: 其他关键字参数。
    """
    def __init__(self, seq_len, patch_size, cond_embed_dim, embed_dim, num_blocks, num_heads,
                 grad_checkpointing=False, **kwargs):
        super().__init__()

        # 序列长度
        self.seq_len = seq_len
        # patch大小
        self.patch_size = patch_size

        # 梯度检查点标志
        self.grad_checkpointing = grad_checkpointing

        # 网络结构
        # patch嵌入层
        self.patch_emb = nn.Linear(3 * patch_size ** 2, embed_dim, bias=True)
        # patch嵌入层的LayerNorm
        self.patch_emb_ln = nn.LayerNorm(embed_dim, eps=1e-6)
        # 可学习的位置嵌入
        self.pos_embed_learned = nn.Parameter(torch.zeros(1, seq_len+1, embed_dim))
        # 条件嵌入层
        self.cond_emb = nn.Linear(cond_embed_dim, embed_dim, bias=True)

        # 模型配置
        self.config = model_args = ModelArgs(dim=embed_dim, n_head=num_heads)
        # 初始化Transformer块列表
        self.blocks = nn.ModuleList([TransformerBlock(config=model_args, drop_path=0.0) for _ in range(num_blocks)])

        # 2d rotary pos embedding
        # 二维旋转位置编码
        grid_size = int(seq_len ** 0.5)
        assert grid_size * grid_size == seq_len
        # 计算旋转位置编码
        self.freqs_cis = precompute_freqs_cis_2d(grid_size, model_args.dim // model_args.n_head,
                                                 model_args.rope_base, cls_token_num=1).cuda()

        # KVCache
        # KV缓存
        self.max_batch_size = -1
        self.max_seq_length = -1

        # LayerNorm层
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

        # 初始化权重
        self.initialize_weights()

    def initialize_weights(self):
        """
        初始化模型权重。
        """
        # 初始化位置嵌入参数
        torch.nn.init.normal_(self.pos_embed_learned, std=.02)

        # 初始化nn.Linear和nn.LayerNorm的权重
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """
        初始化nn.Linear和nn.LayerNorm的权重。

        参数:
            module (nn.Module): 要初始化的模块。
        """
        if isinstance(m, nn.Linear):
            # 使用xavier_uniform初始化线性层的权重
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                # 如果有偏置，则初始化为常数0
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                # 如果有偏置，则初始化为常数0
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                # 如果有权重，则初始化为常数1.0
                nn.init.constant_(m.weight, 1.0)

    def setup_caches(self, max_batch_size, max_seq_length):
        """
        设置KV缓存。

        参数:
            max_batch_size (int): 最大批量大小。
            max_seq_length (int): 最大序列长度。
        """
        # 计算每个注意力头的维度
        head_dim = self.config.dim // self.config.n_head
        # 调整序列长度，使其为8的倍数
        max_seq_length = find_multiple(max_seq_length, 8)
        self.max_seq_length = max_seq_length
        self.max_batch_size = max_batch_size
        # 为每个Transformer块初始化KV缓存
        for b in self.blocks:
            b.attention.kv_cache = KVCache(max_batch_size, max_seq_length, self.config.n_head, head_dim)

        # 生成因果掩码：用于防止模型在生成时看到未来的信息
        causal_mask = torch.tril(torch.ones(self.max_seq_length, self.max_seq_length, dtype=torch.bool))
        self.causal_mask = causal_mask
        # 计算旋转位置编码
        grid_size = int(self.seq_len ** 0.5)
        assert grid_size * grid_size == self.seq_len
        self.freqs_cis = precompute_freqs_cis_2d(grid_size, self.config.dim // self.config.n_head,
                                                 self.config.rope_base, 1)

    def patchify(self, x):
        """
        将输入图像分割成patch。

        参数:
            x (torch.Tensor): 输入图像，形状为 (batch_size, channels, height, width)。

        返回:
            torch.Tensor: 分割后的patch，形状为 (batch_size, patches, patch_dim)。
        """
        # 获取批量大小、通道数、高度和宽度
        bsz, c, h, w = x.shape
        # 获取patch大小
        p = self.patch_size
        # 计算patch的数量
        h_, w_ = h // p, w // p

        # 重塑张量以分割成patch
        x = x.reshape(bsz, c, h_, p, w_, p)
        x = torch.einsum('nchpwq->nhwcpq', x)
        x = x.reshape(bsz, h_ * w_, c * p ** 2)
        # 返回分割后的patch
        return x  # [n, l, d]

    def unpatchify(self, x):
        """
        将patch重新组合成图像。

        参数:
            x (torch.Tensor): patch张量，形状为 (batch_size, patches, patch_dim)。

        返回:
            torch.Tensor: 重新组合后的图像，形状为 (batch_size, channels, height, width)。
        """
        bsz = x.shape[0]
        p = self.patch_size
        # 计算高度和宽度
        h_, w_ = int(np.sqrt(self.seq_len)), int(np.sqrt(self.seq_len))

        # 重塑张量以重新组合patch
        x = x.reshape(bsz, h_, w_, 3, p, p)
        x = torch.einsum('nhwcpq->nchpwq', x)
        x = x.reshape(bsz, 3, h_ * p, w_ * p)
        # 返回重新组合后的图像
        return x  # [n, 3, h, w]

    def predict(self, x, cond_list, input_pos=None):
        """
        前向预测函数。

        参数:
            x (torch.Tensor): 输入张量，形状为 (batch_size, patches, patch_dim)。
            cond_list (List[torch.Tensor]): 条件输入列表。
            input_pos (torch.Tensor, optional): 输入位置索引，默认为None。

        返回:
            List[torch.Tensor]: 预测结果列表。
        """
        # 将输入patch嵌入到隐藏空间
        x = self.patch_emb(x)
        # 拼接条件嵌入
        x = torch.cat([self.cond_emb(cond_list[0]).unsqueeze(1).repeat(1, 1, 1), x], dim=1)

        # 添加位置嵌入
        x = x + self.pos_embed_learned[:, :x.shape[1]]
        # 应用LayerNorm
        x = self.patch_emb_ln(x)

        if input_pos is not None:
            # 使用KV缓存
            freqs_cis = self.freqs_cis[input_pos]
            mask = self.causal_mask[input_pos]
            x = x[:, input_pos]
        else:
            # 训练模式
            freqs_cis = self.freqs_cis[:x.shape[1]]
            mask = None

        # 应用Transformer块
        if self.grad_checkpointing and not torch.jit.is_scripting() and self.training:
            # 如果启用梯度检查点且在训练模式下，使用checkpoint函数以节省显存
            for block in self.blocks:
                x = checkpoint(block, x, freqs_cis, input_pos, mask)
        else:
            # 否则，正常应用每个Transformer块
            for block in self.blocks:
                x = block(x, freqs_cis, input_pos, mask)
        # 应用LayerNorm以稳定输出
        x = self.norm(x)

        # 返回中间条件
        if input_pos is not None:
            # 如果指定了输入位置，则返回第一个位置的中间条件
            middle_cond = x[:, 0]
        else:
            # 否则，返回除最后一个位置外的所有中间条件
            middle_cond = x[:, :-1]

        return [middle_cond]

    def forward(self, imgs, cond_list):
        """
        前向传播函数，用于训练阶段。

        参数:
            imgs (torch.Tensor): 输入图像，形状为 (batch_size, channels, height, width)。
                                 其中，batch_size 是批量大小，channels 是通道数，height 和 width 分别是图像的高度和宽度。
            cond_list (List[torch.Tensor]): 条件输入列表，包含用于生成的条件信息。

        返回:
            Tuple[torch.Tensor, List[torch.Tensor], int]: 
                - patches (torch.Tensor): 分割后的补丁，形状为 (batch_size * patches_per_image, 3, patch_size, patch_size)。
                - cond_list_next (List[torch.Tensor]): 下一级的条件列表。
                - 0 (int): 一个常量整数0，作为占位符。
        """
        # 将输入图像分割成patch以获取标签
        patches = self.patchify(imgs)
        # 创建一个与patch数量相同的全1掩码，形状为 (batch_size, patches_per_image)
        mask = torch.ones(patches.size(0), patches.size(1)).to(patches.device)

        # 获取下一级的条件，通过调用predict方法
        # 使用当前patch和条件列表生成下一级的条件
        cond_list_next = self.predict(patches, cond_list)

        # 重塑条件和patch以适应下一级的输入格式
        for cond_idx in range(len(cond_list_next)):
            # 将每个条件张量重塑为 (batch_size * patches_per_image, -1)
            cond_list_next[cond_idx] = cond_list_next[cond_idx].reshape(cond_list_next[cond_idx].size(0) * cond_list_next[cond_idx].size(1), -1)

        # 重塑patch张量，首先将其展平为 (batch_size * patches_per_image, -1)
        patches = patches.reshape(patches.size(0) * patches.size(1), -1)
        # 然后将其重塑为 (batch_size, 3, patch_size, patch_size)
        patches = patches.reshape(patches.size(0), 3, self.patch_size, self.patch_size)

        # 返回分割后的patch、下一级的条件和常量0
        return patches, cond_list_next, 0

    def sample(self, cond_list, num_iter, cfg, cfg_schedule, temperature, filter_threshold, next_level_sample_function,
               visualize=False):
        """
        生成函数，用于从条件输入中采样生成序列数据。

        参数:
            cond_list (List[torch.Tensor]): 条件输入列表，包含用于生成的条件信息。
            num_iter (int): 采样迭代次数，通常等于序列长度。
            cfg (float): 无分类器指导的权重，用于控制生成的多样性。
            cfg_schedule (str): cfg调度策略，可以是"linear"或其他自定义策略。
            temperature (float): 温度参数，控制生成的多样性。
            filter_threshold (float): 过滤阈值，用于筛选生成的样本。
            next_level_sample_function (callable): 下一级的采样函数，用于生成下一个位置的样本。
            visualize (bool): 是否可视化生成过程，默认为False。

        返回:
            torch.Tensor: 生成的结果，形状为 (batch_size, channels, height, width)。
        """
        if cfg == 1.0:
            # 如果cfg为1.0，则批量大小等于条件列表中第一个张量的批量大小
            bsz = cond_list[0].size(0)
        else:
            # 否则，批量大小减半，用于实现无分类器指导
            bsz = cond_list[0].size(0) // 2

        # 初始化patch张量，形状为 (batch_size, seq_len, 3 * patch_size**2)
        patches = torch.zeros(bsz, self.seq_len, 3 * self.patch_size**2).cuda()
        # 采样迭代次数等于序列长度
        num_iter = self.seq_len

        device = cond_list[0].device
        # 设置KV缓存，最大批量大小和最大序列长度
        with torch.device(device):
            self.setup_caches(max_batch_size=cond_list[0].size(0), max_seq_length=num_iter)

        # 采样过程
        for step in range(num_iter):
            # 克隆当前patch
            cur_patches = patches.clone()

            if not cfg == 1.0:
                # 如果cfg不为1.0，则将patch拼接以实现无分类器指导
                patches = torch.cat([patches, patches], dim=0)

            # 获取下一级的条件，通过调用predict方法，并指定当前步的位置索引
            cond_list_next = self.predict(patches, cond_list, input_pos=torch.Tensor([step]).int())
            # cfg调度
            if cfg_schedule == "linear":
                # 如果调度策略为"linear"，则cfg_iter线性增加
                cfg_iter = 1 + (cfg - 1) * (step + 1) / self.seq_len
            else:
                # 否则，cfg_iter等于cfg
                cfg_iter = cfg

            # 调用下一级的采样函数生成当前步的patch
            sampled_patches = next_level_sample_function(cond_list=cond_list_next, cfg=cfg_iter,
                                                         temperature=temperature, filter_threshold=filter_threshold)
            # 重塑采样patch
            sampled_patches = sampled_patches.reshape(sampled_patches.size(0), -1)
            # 将采样补丁赋值给当前patch
            cur_patches[:, step] = sampled_patches.to(cur_patches.dtype)
            # 更新当前patch
            patches = cur_patches.clone()

            # 可视化生成过程（用于Colab等环境）
            if visualize:
                visualize_patch(self.unpatchify(patches))

        # 清理KV缓存
        for b in self.blocks:
            b.attention.kv_cache = None
        # 将生成的patch重新组合成图像
        patches = self.unpatchify(patches)
        
        return patches
