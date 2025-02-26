import numpy as np
from PIL import Image


def center_crop_arr(pil_image, image_size):
    """
    对输入的PIL图像进行中心裁剪，使其边长为指定的image_size。

    该实现基于ADM（Adversarial Dreaming Model）的中心裁剪方法。
    首先，通过不断将图像尺寸减半，直到最小边长小于2倍的image_size。
    然后，将图像放大到最小边长等于image_size的整数倍。
    最后，从放大后的图像中裁剪出中心部分。

    参数:
        pil_image (PIL.Image.Image): 输入的PIL图像对象。
        image_size (int): 目标裁剪尺寸，图像的最终边长。

    返回:
        PIL.Image.Image: 中心裁剪后的图像对象，边长为image_size。
    """
    # 循环判断，如果图像的最小边长大于等于2倍的image_size，则将图像尺寸减半
    while min(*pil_image.size) >= 2 * image_size:
        # 使用最近邻插值法（Image.BOX）将图像尺寸减半
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    # 计算缩放比例，使得最小边长等于image_size
    scale = image_size / min(*pil_image.size)
    # 使用双三次插值法（Image.BICUBIC）将图像放大到新的尺寸
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    # 将PIL图像转换为NumPy数组，形状为 (高度, 宽度, 通道数)
    arr = np.array(pil_image)
    # 计算裁剪的起始y坐标
    crop_y = (arr.shape[0] - image_size) // 2
    # 计算裁剪的起始x坐标
    crop_x = (arr.shape[1] - image_size) // 2
    # 从数组中裁剪出中心部分，形状为 (image_size, image_size, 通道数)
    # 将裁剪后的数组转换回PIL图像对象
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])
