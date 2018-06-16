[Assignment 2 | ConvolutionalNetworks](https://github.com/FortiLeiZhang/cs231n/blob/master/code/cs231n/assignment2/ConvolutionalNetworks.ipynb)
---
## Convolution
首先说说这个aside。仅仅将 filter 设置为特定的数字，就可以对图像进行某种处理，这里是将图像灰度化和在一个图层取其水平边缘，这直观的给出了一个 filter 的作用。第一层的 filter 可以直观的看出其作用，以后各层 filter 是第一层 filter 的各种线性、非线性组合，可能不会有直观的表示。
而训练的目的就是找出各种表示不同特征的 filter，可能有些 filter 表征图片中的边，有些 filter 表征角，这些 filter 不同的组合就可以区分出各种物体。

#### Forward
Naive 的实现没什么可说的，这里要注意的是输入输出的维度，和np.pad函数的使用。

#### Backward



































end
