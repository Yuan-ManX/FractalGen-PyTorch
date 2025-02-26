from functools import partial
import math
import torch
import torch.nn as nn
from timm.models.vision_transformer import DropPath, Mlp


def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None) -> torch.Tensor:
    """
    计算缩放点积注意力（Scaled Dot-Product Attention）。

    参数:
        query (torch.Tensor): 查询张量，形状为 (..., L, D)。
                              其中，L 是查询序列长度，D 是查询的维度。
        key (torch.Tensor): 键张量，形状为 (..., S, D)。
                            其中，S 是键序列长度，D 是键的维度。
        value (torch.Tensor): 值张量，形状为 (..., S, D)。
                              其中，S 是值序列长度，D 是值的维度。
        attn_mask (torch.Tensor, optional): 注意力掩码，形状为 (..., L, S)。
                                             用于屏蔽注意力计算中的某些位置。默认为None。
        dropout_p (float): Dropout概率，范围在[0, 1]之间，用于在训练时随机丢弃部分注意力权重。默认为0.0。
        is_causal (bool): 是否使用因果掩码（因果掩码用于防止模型在生成时看到未来的信息）。默认为False。
        scale (float, optional): 缩放因子。如果为None，则使用 1 / sqrt(D)。默认为None。

    返回:
        torch.Tensor: 注意力输出，形状为 (..., L, D)。
    """
    # 获取查询和键的序列长度
    L, S = query.size(-2), key.size(-2)
    # 计算缩放因子
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    # 初始化注意力偏差张量，形状为 (L, S)
    attn_bias = torch.zeros(L, S, dtype=query.dtype).cuda()
    if is_causal:
        assert attn_mask is None
        # 生成一个上三角为1，下三角为0的布尔掩码，用于实现因果掩码
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0).cuda()
        # 将非上三角位置填充为负无穷，实现因果掩码
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        # 确保数据类型与查询张量一致
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            # 如果注意力掩码是布尔类型，则将非掩码位置填充为负无穷
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            # 如果注意力掩码是其他类型，则将其直接加到注意力偏差中
            attn_bias += attn_mask
    # 使用自动混合精度（如果可用），以加速计算
    with torch.cuda.amp.autocast(enabled=False):
        # 计算缩放点积注意力权重
        attn_weight = query @ key.transpose(-2, -1) * scale_factor
    # 添加注意力偏差
    attn_weight += attn_bias
    # 应用softmax激活函数
    attn_weight = torch.softmax(attn_weight, dim=-1)
    # 应用dropout
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    # 返回注意力输出
    return attn_weight @ value


class CausalAttention(nn.Module):
    """
    因果注意力机制层（Casual Attention Layer），用于处理自回归模型中的注意力计算。

    参数:
        dim (int): 输入和输出的维度。
        num_heads (int): 多头注意力机制中的头数，默认为8。
        qkv_bias (bool): 查询、键和值线性变换是否使用偏置，默认为False。
        qk_norm (bool): 是否对查询和键进行归一化，默认为False。
        attn_drop (float): 注意力层的dropout概率，默认为0.0。
        proj_drop (float): 投影层的dropout概率，默认为0.0。
        norm_layer (nn.Module): 归一化层，默认为nn.LayerNorm。
    """
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.0,
            proj_drop: float = 0.0,
            norm_layer: nn.Module = nn.LayerNorm
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        # 多头数
        self.num_heads = num_heads
        # 每个注意力头的维度
        self.head_dim = dim // num_heads
        # 缩放因子
        self.scale = self.head_dim ** -0.5

        # 查询、键和值线性变换层
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        # 查询和键的归一化层
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        # 注意力层的dropout
        self.attn_drop = nn.Dropout(attn_drop)
        # 输出投影层
        self.proj = nn.Linear(dim, dim)
        # 输出投影层的dropout
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数。

        参数:
            x (torch.Tensor): 输入张量，形状为 (batch_size, sequence_length, dim)。

        返回:
            torch.Tensor: 注意力机制的输出，形状为 (batch_size, sequence_length, dim)。
        """
        # 获取批量大小、序列长度和维度
        B, N, C = x.shape
        # 计算查询、键和值张量
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        # 分离查询、键和值
        q, k, v = qkv.unbind(0)
        # 对查询和键进行归一化
        q, k = self.q_norm(q), self.k_norm(k)

        # 计算缩放点积注意力
        x = scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attn_drop.p if self.training else 0.0, # 如果在训练模式下，应用dropout
            is_causal=True 
        )

        # 重塑张量以适应输出维度
        x = x.transpose(1, 2).reshape(B, N, C)
        # 应用输出投影层
        x = self.proj(x)
        # 应用输出投影层的dropout
        x = self.proj_drop(x)
        return x


class CausalBlock(nn.Module):
    """
    因果块（Causal Block），用于构建因果Transformer模型。

    参数:
        dim (int): 输入和输出的维度。
        num_heads (int): 多头注意力机制中的头数。
        mlp_ratio (float): MLP（多层感知机）隐藏层维度与输入维度的比率，默认为4.0。
        qkv_bias (bool): 查询、键和值线性变换是否使用偏置，默认为False。
        proj_drop (float): 投影层的dropout概率，默认为0.0。
        attn_drop (float): 注意力层的dropout概率，默认为0.0。
        drop_path (float): DropPath的概率，用于随机深度正则化，默认为0.0。
        act_layer (nn.Module): 激活函数层，默认为nn.GELU（高斯误差线性单元）。
        norm_layer (nn.Module): 归一化层，默认为nn.LayerNorm。
    """
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, proj_drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        # 第一个归一化层
        self.norm1 = norm_layer(dim)
        # 因果注意力机制层
        self.attn = CausalAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=proj_drop)
        # DropPath用于随机深度正则化，如果drop_path > 0，则使用DropPath，否则使用恒等层
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # 第二个归一化层
        self.norm2 = norm_layer(dim)
        # MLP隐藏层维度
        mlp_hidden_dim = int(dim * mlp_ratio)
        # MLP层
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=proj_drop)

    def forward(self, x):
        """
        前向传播函数。

        参数:
            x (torch.Tensor): 输入张量，形状为 (batch_size, seq_length, dim)。

        返回:
            torch.Tensor: 因果块的输出，形状为 (batch_size, seq_length, dim)。
        """
        # 自注意力机制 + DropPath + 残差连接
        x = x + self.drop_path(self.attn(self.norm1(x)))
        # MLP + DropPath + 残差连接
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class MlmLayer(nn.Module):
    """
    MLM（Masked Language Modeling）层，用于处理语言模型的输出。

    参数:
        vocab_size (int): 词汇表大小。
    """
    def __init__(self, vocab_size):
        super().__init__()
        # 偏置参数，形状为 (1, vocab_size)
        self.bias = nn.Parameter(torch.zeros(1, vocab_size))

    def forward(self, x, word_embeddings):
        """
        前向传播函数。

        参数:
            x (torch.Tensor): 输入张量，形状为 (batch_size, seq_length, hidden_dim)。
            word_embeddings (torch.Tensor): 词嵌入矩阵，形状为 (vocab_size, hidden_dim)。

        返回:
            torch.Tensor: MLM层的输出，形状为 (batch_size, seq_length, vocab_size)。
        """
        # 转置词嵌入矩阵为 (hidden_dim, vocab_size)
        word_embeddings = word_embeddings.transpose(0, 1)
        # 计算 logits
        logits = torch.matmul(x, word_embeddings)
        # 添加偏置
        logits = logits + self.bias
        return logits


class PixelLoss(nn.Module):
    """
    像素损失层，用于计算像素级别的重建损失。

    参数:
        c_channels (int): 条件嵌入的通道数。
        width (int): 隐藏层的宽度。
        depth (int): Transformer块的深度。
        num_heads (int): 多头注意力机制中的头数。
        r_weight (float): R通道的权重，默认为1.0。
    """
    def __init__(self, c_channels, width, depth, num_heads, r_weight=1.0):
        super().__init__()

        # 像素均值和标准差，用于归一化
        self.pix_mean = torch.Tensor([0.485, 0.456, 0.406])
        self.pix_std = torch.Tensor([0.229, 0.224, 0.225])

        # 条件投影层，将条件嵌入映射到隐藏空间
        self.cond_proj = nn.Linear(c_channels, width)
        # R、G、B通道的代码本，用于量化像素值
        self.r_codebook = nn.Embedding(256, width)
        self.g_codebook = nn.Embedding(256, width)
        self.b_codebook = nn.Embedding(256, width)

        # LayerNorm层
        self.ln = nn.LayerNorm(width, eps=1e-6)
        # Transformer块列表
        self.blocks = nn.ModuleList([
            CausalBlock(width, num_heads=num_heads, mlp_ratio=4.0,
                        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                        proj_drop=0, attn_drop=0)
            for _ in range(depth)
        ])
        # 归一化层
        self.norm = nn.LayerNorm(width, eps=1e-6)

        # R、G、B通道的MLM层
        self.r_weight = r_weight
        self.r_mlm = MlmLayer(256)
        self.g_mlm = MlmLayer(256)
        self.b_mlm = MlmLayer(256)

        # 损失函数
        self.criterion = torch.nn.CrossEntropyLoss(reduction="none")

        # 初始化权重
        self.initialize_weights()

    def initialize_weights(self):
        # 初始化代码本权重
        torch.nn.init.normal_(self.r_codebook.weight, std=.02)
        torch.nn.init.normal_(self.g_codebook.weight, std=.02)
        torch.nn.init.normal_(self.b_codebook.weight, std=.02)

        # 初始化nn.Linear和nn.LayerNorm的权重
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # 使用xavier_uniform初始化线性层的权重
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    def predict(self, target, cond_list):
        """
        预测函数，用于计算像素值的预测。

        参数:
            target (torch.Tensor): 目标像素值，形状为 (batch_size, 3, H, W)。
            cond_list (List[torch.Tensor]): 条件输入列表。

        返回:
            Tuple[torch.Tensor, torch.Tensor]: 预测的logits和目标像素值。
        """
        # 重塑张量为 (batch_size, 3)
        target = target.reshape(target.size(0), target.size(1))
        # 将像素值归一化到 [0, 255]
        mean = self.pix_mean.cuda().unsqueeze(0) # 像素均值，形状为 (1, 3)
        std = self.pix_std.cuda().unsqueeze(0) # 像素标准差，形状为 (1, 3)
        # 反归一化
        target = target * std + mean
        # 添加小噪声以避免银行家舍入导致的像素分布不一致
        target = (target * 255 + 1e-2 * torch.randn_like(target)).round().long() # 四舍五入并转换为整数

        # 取中间条件
        cond = cond_list[0]
        # 将条件嵌入投影到隐藏空间，并与代码本嵌入拼接
        x = torch.cat(
            [self.cond_proj(cond).unsqueeze(1), self.r_codebook(target[:, 0:1]), self.g_codebook(target[:, 1:2]),
             self.b_codebook(target[:, 2:3])], dim=1)
        # 应用LayerNorm
        x = self.ln(x)

        # 应用Transformer块
        for block in self.blocks:
            x = block(x)

        # 应用归一化层
        x = self.norm(x)
        with torch.cuda.amp.autocast(enabled=False):
            # 计算R、G、B通道的logits
            r_logits = self.r_mlm(x[:, 0], self.r_codebook.weight)
            g_logits = self.g_mlm(x[:, 1], self.g_codebook.weight)
            b_logits = self.b_mlm(x[:, 2], self.b_codebook.weight)

        # 拼接logits
        logits = torch.cat([r_logits.unsqueeze(1), g_logits.unsqueeze(1), b_logits.unsqueeze(1)], dim=1)
        # 返回logits和目标像素值
        return logits, target

    def forward(self, target, cond_list):
        """
        前向传播函数，用于计算像素级别的重建损失。

        参数:
            target (torch.Tensor): 目标像素值张量，形状为 (batch_size, 3, H, W)。
                                   其中，batch_size 是批量大小，3 是RGB通道数，H 和 W 分别是图像的高度和宽度。
            cond_list (List[torch.Tensor]): 条件输入列表，包含用于生成的条件信息。

        返回:
            torch.Tensor: 计算得到的平均损失。
        """
        # 计算预测的logits和目标像素值
        logits, target = self.predict(target, cond_list)
        # 计算R通道的损失
        loss_r = self.criterion(logits[:, 0], target[:, 0])
        # 计算G通道的损失
        loss_g = self.criterion(logits[:, 1], target[:, 1])
        # 计算B通道的损失
        loss_b = self.criterion(logits[:, 2], target[:, 2])

        if self.training:
            # 如果在训练阶段，计算加权总损失，R通道权重更高
            loss = (self.r_weight * loss_r + loss_g + loss_b) / (self.r_weight + 2)
        else:
            # 如果不在训练阶段（例如推理阶段），计算无约束的负对数似然损失（NLL）
            loss = (loss_r + loss_g + loss_b) / 3
        
        # 返回平均损失
        return loss.mean()

    def sample(self, cond_list, temperature, cfg, filter_threshold=0):
        """
        生成函数，用于从条件输入中采样生成像素值。

        参数:
            cond_list (List[torch.Tensor]): 条件输入列表，包含用于生成的条件信息。
            temperature (float): 温度参数，控制生成的多样性。温度越高，生成的多样性越高。
            cfg (float): 无分类器指导的权重，用于控制生成的多样性。
            filter_threshold (float): 过滤阈值，用于筛选生成的样本，默认为0。

        返回:
            torch.Tensor: 生成的结果，形状为 (batch_size, 3)。
                          其中，batch_size 是批量大小，3 是RGB通道数。
        """
        if cfg == 1.0:
            # 如果cfg为1.0，则批量大小等于条件列表中第一个张量的批量大小
            bsz = cond_list[0].size(0)
        else:
            # 否则，批量大小减半，用于实现无分类器指导
            bsz = cond_list[0].size(0) // 2
        # 初始化像素值张量，形状为 (batch_size, 3)
        pixel_values = torch.zeros(bsz, 3).cuda()

        for i in range(3):
            if cfg == 1.0:
                # 如果cfg为1.0，则直接预测像素值
                logits, _ = self.predict(pixel_values, cond_list)
            else:
                # 否则，将输入拼接以实现无分类器指导
                logits, _ = self.predict(torch.cat([pixel_values, pixel_values], dim=0), cond_list)
            # 选择当前通道的logits
            logits = logits[:, i]
            # 应用温度参数
            logits = logits * temperature

            if not cfg == 1.0:
                # 如果cfg不为1.0，则进行无分类器指导处理
                cond_logits = logits[:bsz] # 条件logits
                uncond_logits = logits[bsz:] # 无条件logits

                # 计算条件概率
                cond_probs = torch.softmax(cond_logits, dim=-1)
                # 创建掩码，筛选出非常不可能的条件logits
                mask = cond_probs < filter_threshold
                # 对非常不可能的条件logits进行替换
                uncond_logits[mask] = torch.max(
                    uncond_logits,
                    cond_logits - torch.max(cond_logits, dim=-1, keepdim=True)[0] + torch.max(uncond_logits, dim=-1, keepdim=True)[0]
                )[mask]

                # 应用cfg权重
                logits = uncond_logits + cfg * (cond_logits - uncond_logits)

            # 计算像素值预测的概率分布
            probs = torch.softmax(logits, dim=-1)
            # 使用多项式分布进行采样
            sampled_ids = torch.multinomial(probs, num_samples=1).reshape(-1)
            # 将采样结果转换为像素值，并应用归一化
            pixel_values[:, i] = (sampled_ids.float() / 255 - self.pix_mean[i]) / self.pix_std[i]

        # 将像素值转换回 [0, 1]
        return pixel_values
