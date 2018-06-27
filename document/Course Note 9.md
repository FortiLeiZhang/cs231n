[Convolutional Neural Networks: Architectures, Convolution / Pooling Layers](http://cs231n.github.io/convolutional-networks/)
---
# Convolutional Neural Networks
CNN 默认要处理的对象是图片，所以输入都是形如 (H, W, 3)，这里的3表示RGB三个颜色通道。同时注意，由于是图片，所以输入中的每一个像素在整个图片中的相对位置也有了意义。CNN 的典型结构是 [INPUT - CONV - RELU - POOL - FC]。其中 CONV 和 FC 是带参数的层，而 ReLU 和 POOL 不引入额外的参数。

## Convolutional Layer
每一层 Convolutional Layer 都是用一组 filter 沿着输入矩阵进行滑动点乘。这里要注意的是，每一个 filter 都要对输入的所有 channel 进行矩阵点乘。
> Every filter is small spatially (along width and height), but extends through the full depth of the input volume.

#### Filter
Convolutinal layer 最基本的操作单元是 filter，通常用3×3或者5×5来表示一个filter大小。但是，这个表示方法常常会给初学者带来误解，认为一个 filter 仅有一层。这里再次强调，每一个 filter 都要对输入的所有 channel 进行矩阵点乘，所以每一个 filter 有 in-channel 层。

例如，如果输入是 32×32×3 的，那么每一个 filter 都是5×5×3的，每一次点乘，此 filter 所有3个 channel 上的 weight 都与图片相应位置的数字相乘，然后将结果相加，最后将3个 channel 上的3个结果再相加，得到 output 矩阵中的一个数字。将 filter 沿着图片滑动一遍，就得到一个 output 矩阵。需要注意的是，这里的 output 矩阵是一层的，也就是说用一个 5×5×3 的 filter，沿着一幅 32×32×3 的图片进行滑动点乘，得到一个 n×n×1的输出矩阵，这里 n 的大小与滑动方式和补0方式有关，但是得到的 output 肯定是一层的。如果有512个这样的 filter，那么就得到512个 n×n×1 的输出矩阵，再将这512个矩阵沿着 depth 方向叠在一起，就得到最后的 n×n×512 的输出。

所以，在讲 filter 大小的时候，如果知道输入 channel 数，就尽量带出来，比如说5×5×3的 filter 就要比只说 5×5 的 filter 要清楚；即使输入 channel 数未知，最好也说5×5×in_channel 的filter。最差的情况，如果看到一个5×5 的 filter，最起码心里要知道其实它是一个5×5×in_channel 的filter。这样想的好处可以在今后理解1×1的filter时体现出来。

Filter 的 stride 和 padding 很好理解。stride 是滑动步长，padding 是前后补0，牢记下面这个公式就好了：
> out_weight = (in_weight - filter_size + 2 * padding) / stride + 1

对于给定的 input 和 filter，output 各项计算公式如下：

Input: $W_1 \times H_1 \times D_1$

Filter: $F \times F \times D_1$， Stride: S，Padding: P， Number of
Filter: K

Output: $W_2 \times H_2 \times K$

$W_2 = (W_1 - F + 2 * P) / S + 1$

$H_2 = (H_1 - F + 2 * P) / S + 1$

这里要插一段，在 Resnet 中，输入图片的大小是 $224 \times 224$ 的，但是它的第一层 filter 是
```python
self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
```
这里输入与 filter 并不匹配，而且后面紧跟的 max pool 也不匹配
```python
self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
```
实际上这里在计算的时候把最右侧补的一列 0 给忽略掉了，所以这一层的输出大小为：
>(224 - 7 + 2 * 3) / 2 + 1 = 112
>
> (112 - 3 + 2 * 1) / 2 + 1 = 56



#### Convolution 的计算
Naive 的计算方法就是沿着输入矩阵滑动点乘；计算速度更快的方法是将输入矩阵和filter矩阵都展成 vector 就行矩阵相乘，这样的缺点是内存消耗大。

#### Backpropagation
CNN 的 Backpropagation 作业里有，到那里再说。

#### 1x1 convolution
第一次简单 1x1 convolutio layer 时也是很难理解，这玩意有什么用，但是如果像上面提到的，如果记作 1x1xin_channel convolutio layer，那就茅塞顿开。它的作用就是将输入的所有层，通过不同 weight 的叠加，整合出新的一层，比如输入层是32x32x512的，那么通过256个1x1x512个 filter，就得到了32x32x256的输出，其中输出的每一层都是输入的512层的不同线性组合方式的叠加。这样的好处显而易见的是可以降低输出的depth，另外的说法是可以使CNN更深。

#### Dilated convolutions
打孔卷积，filter 并不是与输入矩阵的每一个元素都点积，而是有选择性的打孔。尚处研究阶段，不用太深究。

## Pooling Layer
通常用2x2，stride=2的 max pooling。另外，目前有用 stride=2 的 filter来代替 pooling 的趋势。

#### Normalization Layer
现在不常用了，但是作业里还有几种实现。

#### Fully Connected Layer
同NN，这里要注意的是 convolution layer 和 FC 是可以相互转化的。想想也是，都是做点乘，如果 filter 设计一下，同样可以达到 FC 的效果。具体方法找个例子用代码实现一下就全明白了，这里不赘述了。

# ConvNet Architectures
常见的结构为：INPUT -> CONV -> RELU -> POOL -> FC -> RELU -> FC。
Notes 里给出的建议是不要自己研究 CNN 的结构了，因为算力要求太高，不是个人能够完成的。相反，找几个主流的结构研究清楚，然后用这些来好好调教自己的数据。

#### Input Layer
常取2的倍数

#### Conv Layer
3x3或者5x5。目前先进的做法是用多层小的 filter 叠加来取代一层大的 filter，两者有相同的视野。再用1x1的 filter 来降低输出 depth，从而减少计算量。

#### Pooling Layer
2x2 stride = 2

# Case Studies
这里面每一个 case 都值得下大力气好好研究一番，放在另外一个 project 里把相应论文好好写写，这里就不再简述了。


















end
