[Assignment3 | Style Transfer](https://github.com/FortiLeiZhang/cs231n/blob/master/code/cs231n/assignment3/StyleTransfer-PyTorch.ipynb)
---
所谓 style transfer，就是找一张原图，一张样式图，固定一个 CNN 模型，设计一种 loss 的计算方法，然后在 CNN 中 backprop 出 loss 对原图中每一个像素点的梯度，根据此梯度来更新原图中的每一像素的值。最后得到的图片既有原图的内容，又有样式图的风格。这里的核心问题是 loss 计算方法的设计。

### Computing loss
这里 loss 的计算对象是 CNN 中每一层所产生的特征图。

#### Content loss
Content loss 的计算方法为：将原图输入进 CNN，每一层会产生一组特征图；将生成图输入 CNN，产生另一组特征图p，两组特征图逐层的欧式距离就是 content loss。

#### Style loss
计算 style loss 首先要计算每一个特征图的互相关矩阵 Gram matrix。对于形如 (C, H*W) 的特征图，其 gram matrix 形如 (C, C) ：
$$G_{ij}  = \sum_k F_{ik} F_{jk}$$

有了 gram matrix，style loss 就是生成图与特征图逐层 gram 的欧式距离。

#### Total variation regularization
为了使得图片内容更加平滑，加入一个正则项，计算方法是图片相邻点的欧式距离之和。

### Feature Inversion
如果将一张图片初始化为白噪声，并且关闭掉 style loss，只用 content loss 和 reg，那么用上述的方法学习，可以生成一张同原图近似的图片。
