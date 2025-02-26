from functools import partial
import torch
import torch.nn as nn

from ar import AR
from mar import MAR
from pixelloss import PixelLoss


class FractalGen(nn.Module):
    """
    分形生成模型（Fractal Generative Model）

    该模型采用分形层次结构，通过递归生成不同尺度的图像，逐步细化生成过程。

    参数:
        img_size_list (List[int]): 图像尺寸列表，定义了每个分形层次的图像大小。
                                    例如，[256, 128, 64] 表示有三个层次，图像大小分别为256x256, 128x128, 64x64。
        embed_dim_list (List[int]): 嵌入维度列表，定义了每个分形层次的嵌入维度。
        num_blocks_list (List[int]): Transformer块数量列表，定义了每个分形层次中使用的Transformer块数量。
        num_heads_list (List[int]): 多头注意力头数列表，定义了每个分形层次中多头注意力的头数。
        generator_type_list (List[str]): 生成器类型列表，定义了每个分形层次使用的生成器类型。
                                         例如，"ar" 表示自回归生成器，"mar" 表示多尺度自回归生成器。
        label_drop_prob (float): 标签丢弃概率，用于无分类器指导，默认为0.1。
        class_num (int): 类别数量，默认为1000。
        attn_dropout (float): 注意力层的dropout概率，默认为0.1。
        proj_dropout (float): 投影层的dropout概率，默认为0.1。
        guiding_pixel (bool): 是否使用引导像素，默认为False。
        num_conds (int): 条件数量，默认为1。
        r_weight (float): 残差权重，默认为1.0。
        grad_checkpointing (bool): 是否启用梯度检查点，默认为False。
        fractal_level (int): 当前分形层次，默认为0。
    """
    def __init__(self,
                 img_size_list,
                 embed_dim_list,
                 num_blocks_list,
                 num_heads_list,
                 generator_type_list,
                 label_drop_prob=0.1,
                 class_num=1000,
                 attn_dropout=0.1,
                 proj_dropout=0.1,
                 guiding_pixel=False,
                 num_conds=1,
                 r_weight=1.0,
                 grad_checkpointing=False,
                 fractal_level=0):
        super().__init__()

        # 分形特定参数
        # 当前分形层次
        self.fractal_level = fractal_level
        # 分形层次总数
        self.num_fractal_levels = len(img_size_list)

        # 第一层分形级别的类别嵌入
        if self.fractal_level == 0:
            # 类别数量
            self.num_classes = class_num
            # 类别嵌入层
            self.class_emb = nn.Embedding(class_num, embed_dim_list[0])
            # 标签丢弃概率
            self.label_drop_prob = label_drop_prob
            # 伪潜在向量参数
            self.fake_latent = nn.Parameter(torch.zeros(1, embed_dim_list[0]))
            # 初始化类别嵌入权重和伪潜在向量
            torch.nn.init.normal_(self.class_emb.weight, std=0.02)
            torch.nn.init.normal_(self.fake_latent, std=0.02)

        # 当前层次的生成器
        if generator_type_list[fractal_level] == "ar":
            # 如果生成器类型为"ar"，则使用AR生成器
            generator = AR
        elif generator_type_list[fractal_level] == "mar":
            # 如果生成器类型为"mar"，则使用MAR生成器
            generator = MAR
        else:
            # 如果生成器类型未实现，则抛出异常
            raise NotImplementedError
        
        # 初始化当前层次的生成器
        self.generator = generator(
            seq_len=(img_size_list[fractal_level] // img_size_list[fractal_level+1]) ** 2, # 计算序列长度
            patch_size=img_size_list[fractal_level+1],
            cond_embed_dim=embed_dim_list[fractal_level-1] if fractal_level > 0 else embed_dim_list[0], # 条件嵌入维度
            embed_dim=embed_dim_list[fractal_level], # 当前层次的嵌入维度
            num_blocks=num_blocks_list[fractal_level], # 当前层次的Transformer块数量
            num_heads=num_heads_list[fractal_level], # 当前层次的多头注意力头数
            attn_dropout=attn_dropout, # 注意力层的dropout概率
            proj_dropout=proj_dropout, # 投影层的dropout概率
            guiding_pixel=guiding_pixel if fractal_level > 0 else False, # 是否使用引导像素
            num_conds=num_conds, # 条件数量
            grad_checkpointing=grad_checkpointing,  # 是否启用梯度检查点
        )

        # 递归构建下一个分形层次
        if self.fractal_level < self.num_fractal_levels - 2:
            self.next_fractal = FractalGen(
                img_size_list=img_size_list,
                embed_dim_list=embed_dim_list,
                num_blocks_list=num_blocks_list,
                num_heads_list=num_heads_list,
                generator_type_list=generator_type_list,
                label_drop_prob=label_drop_prob,
                class_num=class_num,
                attn_dropout=attn_dropout,
                proj_dropout=proj_dropout,
                guiding_pixel=guiding_pixel,
                num_conds=num_conds,
                r_weight=r_weight,
                grad_checkpointing=grad_checkpointing,
                fractal_level=fractal_level+1
            )
        else:
            # 最后一个分形层次使用PixelLoss
            self.next_fractal = PixelLoss(
                c_channels=embed_dim_list[fractal_level],
                depth=num_blocks_list[fractal_level+1],
                width=embed_dim_list[fractal_level+1],
                num_heads=num_heads_list[fractal_level+1],
                r_weight=r_weight,
            )

    def forward(self, imgs, cond_list):
        """
        前向传播函数，用于递归计算损失。

        参数:
            imgs (torch.Tensor): 输入图像张量。
            cond_list (List[torch.Tensor]): 条件输入列表。

        返回:
            torch.Tensor: 总损失。
        """
        if self.fractal_level == 0:
            # 计算类别嵌入条件
            class_embedding = self.class_emb(cond_list)
            if self.training:
                # 根据label_drop_prob随机丢弃标签
                drop_latent_mask = (torch.rand(cond_list.size(0)) < self.label_drop_prob).unsqueeze(-1).cuda().to(class_embedding.dtype)
                class_embedding = drop_latent_mask * self.fake_latent + (1 - drop_latent_mask) * class_embedding
            else:
                # 对于评估（无条件NLL），使用恒定的掩码
                drop_latent_mask = torch.ones(cond_list.size(0)).unsqueeze(-1).cuda().to(class_embedding.dtype)
                class_embedding = drop_latent_mask * self.fake_latent + (1 - drop_latent_mask) * class_embedding
            cond_list = [class_embedding for _ in range(5)]

        # 获取图像patch和下一级的条件
        imgs, cond_list, guiding_pixel_loss = self.generator(imgs, cond_list)
        # 从下一分形层次递归计算损失
        loss = self.next_fractal(imgs, cond_list)
        # 返回总损失加上引导像素损失
        return loss + guiding_pixel_loss

    def sample(self, cond_list, num_iter_list, cfg, cfg_schedule, temperature, filter_threshold, fractal_level,
               visualize=False):
        """
        递归生成样本。

        参数:
            cond_list (List[torch.Tensor]): 条件输入列表。
            num_iter_list (List[int]): 每个分形层次的迭代次数列表。
            cfg (float): 无分类器指导的权重。
            cfg_schedule (str): cfg调度策略，默认为"constant"。
            temperature (float): 温度参数。
            filter_threshold (float): 过滤阈值。
            fractal_level (int): 当前分形层次。
            visualize (bool): 是否可视化生成过程，默认为False。

        返回:
            torch.Tensor: 生成的结果。
        """
        if fractal_level < self.num_fractal_levels - 2:
            # 如果不是最后一层，则递归调用下一层的sample方法
            next_level_sample_function = partial(
                self.next_fractal.sample,
                num_iter_list=num_iter_list,
                cfg_schedule="constant",
                fractal_level=fractal_level + 1
            )
        else:
            # 如果是最后一层，则直接调用PixelLoss的sample方法
            next_level_sample_function = self.next_fractal.sample

        # 使用当前生成器递归生成样本
        return self.generator.sample(
            cond_list, num_iter_list[fractal_level], cfg, cfg_schedule,
            temperature, filter_threshold, next_level_sample_function, visualize
        )


def fractalar_in64(**kwargs):
    """
    创建一个基于自回归生成器的分形生成模型，适用于64x64分辨率的图像。

    参数:
        **kwargs: 其他关键字参数，可以传递给FractalGen类的构造函数。
    
    返回:
        FractalGen: 配置好的分形生成模型实例。
    """
    # 创建FractalGen实例，指定以下参数：
    model = FractalGen(
        img_size_list=(64, 4, 1), # 图像尺寸列表，从64x64开始，逐层缩小到4x4，最后到1x1
        embed_dim_list=(1024, 512, 128), # 嵌入维度列表，每个层次的嵌入维度分别为1024, 512, 128
        num_blocks_list=(32, 8, 3), # Transformer块数量列表，每个层次分别使用32, 8, 3个块
        num_heads_list=(16, 8, 4), # 多头注意力头数列表，每个层次分别使用16, 8, 4个头
        generator_type_list=("ar", "ar", "ar"), # 生成器类型列表，每个层次都使用自回归生成器（"ar"）
        fractal_level=0, # 当前分形层次，0表示从最高层次开始
        **kwargs) 
    return model


def fractalmar_in64(**kwargs):
    """
    创建一个基于多尺度自回归生成器的分形生成模型，适用于64x64分辨率的图像。

    参数:
        **kwargs: 其他关键字参数，可以传递给FractalGen类的构造函数。
    
    返回:
        FractalGen: 配置好的分形生成模型实例。
    """
    # 创建FractalGen实例，指定以下参数：
    model = FractalGen(
        img_size_list=(64, 4, 1), # 图像尺寸列表，从64x64开始，逐层缩小到4x4，最后到1x1
        embed_dim_list=(1024, 512, 128), # 嵌入维度列表，每个层次的嵌入维度分别为1024, 512, 128
        num_blocks_list=(32, 8, 3), # Transformer块数量列表，每个层次分别使用32, 8, 3个块
        num_heads_list=(16, 8, 4), # 多头注意力头数列表，每个层次分别使用16, 8, 4个头
        generator_type_list=("mar", "mar", "ar"), # 生成器类型列表，前两个层次使用多尺度自回归生成器（"mar"），最后一个层次使用自回归生成器（"ar"）
        fractal_level=0, # 当前分形层次，0表示从最高层次开始
        **kwargs)
    return model


def fractalmar_base_in256(**kwargs):
    """
    创建一个基于多尺度自回归生成器的分形生成模型，适用于256x256分辨率的基础版本图像。

    参数:
        **kwargs: 其他关键字参数，可以传递给FractalGen类的构造函数。
    
    返回:
        FractalGen: 配置好的分形生成模型实例。
    """
    # 创建FractalGen实例，指定以下参数：
    model = FractalGen(
        img_size_list=(256, 16, 4, 1), # 图像尺寸列表，从256x256开始，逐层缩小到16x16, 4x4，最后到1x1
        embed_dim_list=(768, 384, 192, 64), # 嵌入维度列表，每个层次的嵌入维度分别为768, 384, 192, 64
        num_blocks_list=(24, 6, 3, 1), # Transformer块数量列表，每个层次分别使用24, 6, 3, 1个块
        num_heads_list=(12, 6, 3, 4), # 多头注意力头数列表，每个层次分别使用12, 6, 3, 4个头
        generator_type_list=("mar", "mar", "mar", "ar"), # 生成器类型列表，前三个层次使用多尺度自回归生成器（"mar"），最后一个层次使用自回归生成器（"ar"）
        fractal_level=0, # 当前分形层次，0表示从最高层次开始
        **kwargs)
    return model


def fractalmar_large_in256(**kwargs):
    """
    创建一个基于多尺度自回归生成器的分形生成模型，适用于256x256分辨率的大型版本图像。

    参数:
        **kwargs: 其他关键字参数，可以传递给FractalGen类的构造函数。
    
    返回:
        FractalGen: 配置好的分形生成模型实例。
    """
    # 创建FractalGen实例，指定以下参数：
    model = FractalGen(
        img_size_list=(256, 16, 4, 1), # 图像尺寸列表，从256x256开始，逐层缩小到16x16, 4x4，最后到1x1
        embed_dim_list=(1024, 512, 256, 64), # 嵌入维度列表，每个层次的嵌入维度分别为1024, 512, 256, 64
        num_blocks_list=(32, 8, 4, 1), # Transformer块数量列表，每个层次分别使用32, 8, 4, 1个块
        num_heads_list=(16, 8, 4, 4), # 多头注意力头数列表，每个层次分别使用16, 8, 4, 4个头
        generator_type_list=("mar", "mar", "mar", "ar"), # 生成器类型列表，前三个层次使用多尺度自回归生成器（"mar"），最后一个层次使用自回归生成器（"ar"）
        fractal_level=0, # 当前分形层次，0表示从最高层次开始
        **kwargs)
    return model


def fractalmar_huge_in256(**kwargs):
    """
    创建一个基于多尺度自回归生成器的分形生成模型，适用于256x256分辨率的超大型版本图像。

    参数:
        **kwargs: 其他关键字参数，可以传递给FractalGen类的构造函数。
    
    返回:
        FractalGen: 配置好的分形生成模型实例。
    """
    # 创建FractalGen实例，指定以下参数：
    model = FractalGen(
        img_size_list=(256, 16, 4, 1), # 图像尺寸列表，从256x256开始，逐层缩小到16x16, 4x4，最后到1x1
        embed_dim_list=(1280, 640, 320, 64), # 嵌入维度列表，每个层次的嵌入维度分别为1280, 640, 320, 64
        num_blocks_list=(40, 10, 5, 1), # Transformer块数量列表，每个层次分别使用40, 10, 5, 1个块
        num_heads_list=(16, 8, 4, 4), # 多头注意力头数列表，每个层次分别使用16, 8, 4, 4个头
        generator_type_list=("mar", "mar", "mar", "ar"), # 生成器类型列表，前三个层次使用多尺度自回归生成器（"mar"），最后一个层次使用自回归生成器（"ar"）
        fractal_level=0, # 当前分形层次，0表示从最高层次开始
        **kwargs)
    return model
