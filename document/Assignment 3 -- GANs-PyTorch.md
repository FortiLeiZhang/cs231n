[Assignment3 | GANs](https://github.com/FortiLeiZhang/cs231n/blob/master/code/cs231n/assignment3/GANs-PyTorch.ipynb)
---

生成对抗模型 ( Generative Adversarial Network ) 的基本思想是用一个生成模型 ( Generative Model ) 来生成图片，与原来 training set 中的真实图片放在一起，给判决模型 ( Discrimitive Model ) 判定。经过长时间训练后，生成模型生成的图片如果能以假乱真骗过判决模型，那么就可以将这个生成模型单独拿出来用作图片生成器。下面大概介绍一下原理和代码实现。

### GAN 理论推导
生成模型 G 的输入是白噪声，经过一组 NN ，输出的是一组生成图片 $G_{data}$，并将其中的所有图片 label 记为 0，即 $p\{ \tilde{x} \in G_{data}\} = 0$，反之 $p\{ \tilde{x} \notin G_{data}\} = 1$。
另有一组真实的图片 $R_{data}$，并将其中的所有图片 label 记为 1，即 $p\{ x \in R_{data}\} = 1$，反之 $p\{ x \notin R_{data}\} = 0$。
将 $G_{data}$ 和 $R_{data}$ 和在一起，就组成了图片集 $P_{data} = \{ G_{data}, R_{data}\}$。

判决模型 D 的输入是上述的图片集，经过一组 NN，输出的是一组 (0, 1) 之间的数，小于 0.5 表明 D 认为此图片来自于生成图片集 $G_{data}$；反之，大于 0.5 表明 D 认为此图片来自于真实图片集 $R_{data}$。

#### 判决模型 D
从数学的角度来看，判决模型 D 的作用就是试图用一个分布 $q$ 来拟合图片集 $P_{data}$ 的二项分布 $p$。如何衡量分布 $q$ 和 分布 $p$ 的拟合程度，自然的想到了用 cross entropy。关于 cross entropy 的解释，请参考 [Course Note 2](https://github.com/FortiLeiZhang/cs231n/blob/master/document/Course%20Note%202.md) 中相应的部分。cross entropy 的定义式如下：
$$
L(x) = - \sum_{i} p_i (x) \cdot \log \bigg( q_i (x) \bigg)
$$

当 $x$ 来自于真实图片集 $R_{data}$ 时，判决模型 D 的输出是 $q(x) = D(x)$，图片的 label $p(x) = 1$，所以此时的 cross entropy 为：
$$
\begin{aligned}
L(x) &= - \sum_{i} p_i (x) \cdot \log \bigg(q_i (x)\bigg) \newline
&= - p\{ x \in R_{data}\} \cdot \log \bigg(q(x)\bigg) - p\{ x \notin R_{data}\} \cdot \log \bigg(1 - q(x)\bigg) \newline
&= - 1 \cdot  \log \bigg( D(x) \bigg) - 0 \cdot  \log \bigg( 1 - D(x) \bigg) \newline
&= - \log \bigg( D(x) \bigg)
\end{aligned}
$$

同样，当 $\tilde{x}$ 来自于生成图片集 $G_{data}$ 时，判决模型 D 的输出是 $q(\tilde{x}) = D(\tilde{x})$，图片的 label $p(\tilde{x}) = 0$，所以此时的 cross entropy 为：
$$
\begin{aligned}
L(\tilde{x}) &= - \sum_{i} p(\tilde{x}) \cdot \log \bigg(q(\tilde{x})\bigg) \newline
&= - p\{ \tilde{x} \in G_{data}\} \cdot \log \bigg(q(\tilde{x})\bigg) - p\{ \tilde{x} \notin G_{data}\} \cdot \log \bigg(1 - q(\tilde{x})\bigg) \newline
&= 0 \cdot  \log \bigg( D(\tilde{x}) \bigg) - 1 \cdot  \log \bigg( 1 - D(\tilde{x}) \bigg) \newline
&= - \log \bigg( 1 - D(\tilde{x}) \bigg)
\end{aligned}
$$

综上，总的 cross entropy 为：
$$
L(x) = - \log \bigg( D(x) \bigg) - \log \bigg( 1 - D(\tilde{x}) \bigg)
$$
所以，判决模型 D 的目标就是最小化这个 cross entropy，等价于最大化其相反数，即
$$
\underset{D}{\text{maximize}}\; \mathbb{E}_{x \sim R_\text{data}} \bigg( \log D(x) \bigg) + \mathbb{E}_{\tilde{x} \sim G_\text{data}} \bigg( \log \Big(1-D(\tilde{x}) \Big) \bigg)
$$

#### 生成模型 G
生成模型 G 的作用是将白噪声转换成图片 $G_{data}$，即 $\tilde{x} = G(z)$。 同时它的学习目标是尽可能的欺骗判决模型 D，使得从 $G_{data}$ 里取出的，本该标 0 的图片 $\tilde{x}$，被判决模型 D 标 1。从数学的角度来讲，就是最大化 $- 1 \cdot  \log \bigg( 1 - D(\tilde{x}) \bigg)$，等价于最小化其相反数，即:
$$
\underset{G}{\text{minimize}}\;  \mathbb{E}_{\tilde{x} \sim G_\text{data}} \bigg( \log \Big(1-D(\tilde{x}) \Big) \bigg)
$$

综上所述，整个 GAN 的学习目标就是论文中所说的一个 two-player minimax game，即：
$$
\underset{G}{\text{minimize}} \; \underset{D}{\text{maximize}}\; \mathbb{E}_{x \sim R_\text{data}} \bigg( \log D(x) \bigg) + \mathbb{E}_{\tilde{x} \sim G_\text{data}} \bigg( \log \Big(1-D(\tilde{x}) \Big) \bigg)
$$

上面一段解释了论文找那个突然蹦出来的 (1) 式是怎么来的。

但是，通常来说，我们不去最小化
$$
\underset{G}{\text{minimize}}\;  \mathbb{E}_{\tilde{x} \sim G_\text{data}} \bigg( \log \Big(1-D(\tilde{x}) \Big) \bigg)
$$

相反，我们通常最大化
$$
\underset{G}{\text{maxmize}}\;  \mathbb{E}_{\tilde{x} \sim G_\text{data}} \bigg( \log \Big(D(\tilde{x}) \Big) \bigg)
$$

这是因为，log(x) 在 0 附近的斜率比在 1 附近大的多，也就是说在 0 附近收敛的更快，学习的更快。如果我们以 $\log(1 - D(x))$ 作为学习的目标函数，那么在 GAN 刚刚开始学习的时候，应为此时生成模型 G 很弱，所以生成的图片很容易被判决模型 D 识别出来，所以 $D(x)$ 此时的值很小趋近于 0。因此，此时 GAN 其实是在 log(1) 附近学习，收敛的很慢；反而随着生成模型 G 不断学习，后来生成的图片很容易欺骗判决模型 D，使其输出一个趋近于 1 的数，此时 GAN 是在 log(0) 附近学习，反而收敛的很快。这一点与我们想要的学习过程恰恰相反：我们想在生成模型 G 很差时快点学习，而在生成模型 G 比较好的时候慢点学习。所以实际中优化的目标函数要用其等价的形式。

### GAN 代码实现
在实际的代码实现中，我们还是习惯于最小某个函数，并且用 batch 的平均值来代替式子中的数学期望，即
$$
\begin{aligned}
\mathrm{generator:} \; &\ell_G  =  - \frac{1}{N} \sum_{i}\log D(\tilde{x}_i) \newline
\mathrm{discriminator:} \; &\ell_D  =  - \frac{1}{N} \sum_{i}\log D(x_i) - \frac{1}{N} \sum_{i}\log \Big( 1 - D(\tilde{x}_i) \Big)
\end{aligned}
$$

训练过程分两步，先训练判决模型 D，然后再训练生成模型 G。在一次训练循环中，判决模型 D 可以进行多次训练。要注意的是，训练判决模型 D 时需要生成图片，但此时并不对生成模型 D 的参数进行学习，因此仅针对真实图片进行 backprop 而不对生成图片进行 backprop。训练生成模型 G 时，在一次训练循环中，仅训练一次，此时仅需要生成图片，不需要真实图片的参与。

#### Conv2d and ConvTranspose2d
ConvTranspose2d 实现的是 Conv2d 的逆过程，也就是将一张 $m \times m$ 的图片，upsampling 到 $n \times n$，这里 $n > m$。 ConvTranspose2d 的实现方法，与 [Assignment 2 | ConvolutionalNetworks](https://github.com/FortiLeiZhang/cs231n/blob/master/document/Assignment%202%20--%20ConvolutionalNetworks.md) 计算 dx 的方法完全相同。实际上，不论在 PyTorch 还是在 TensorFlow 里面，ConvTranspose2d 的实现和计算 dx 的梯度的实现，使用的是同一段代码。在 [PyTorch 的文档](https://pytorch.org/docs/stable/nn.html)里明确说明了这一点：
> This module can be seen as the gradient of Conv2d with respect to its input.

ConvTranspose2d 中的参数设置，特别是 padding 和 函数的输出形状破费一些思量。我的建议是首先找出其对偶的 Conv2d，此 Conv2d 中的参数就是 ConvTranspose2d 中的参数，除了输入输出形状互换以外。通过下面这个例子详细说一下。

作业中在实现 CNN GAN 时，给出的 Generator 实现为：
> Reshape into Image Tensor of shape 7, 7, 128
>
>Conv2D^T (Transpose): 64 filters of 4x4, stride 2, 'same' padding
>
>ReLU
>BatchNorm
>Conv2D^T (Transpose): 1 filter of 4x4, stride 2, 'same' padding

第一个 ConvTranspose2d 的输入形状显然是 (7, 7)，有 128 个 channel，ConvTranspose2d 的 filter 形状是 (4, 4)，stride = 2，使用 'SAME' padding。那么，ConvTranspose2d 的输出形状到底是多少呢？这里的 'SAME' padding 到底是 pad 多少 0 呢？

首先，'SAME' padding 的说法显然是来自 TensorFlow 而非 PyTorch，所以，我们看 TensorFlow 的 [tf.nn.conv2d_transpose](https://www.tensorflow.org/api_docs/python/tf/nn/conv2d_transpose) 手册。里面在讲 padding 的时候，直接链接到了 [
tf.nn.convolution](
https://www.tensorflow.org/api_docs/python/tf/nn/convolution) 中关于 padding 的说明：
> If padding == "SAME": output_spatial_shape[i] = ceil(input_spatial_shape[i] / strides[i])
>
> If padding == "VALID": output_spatial_shape[i] = ceil((input_spatial_shape[i] - (spatial_filter_shape[i]-1) * dilation_rate[i]) / strides[i]).

注意，这里的 input 和 output 是针对 Conv2d 而言的，它正好和 ConvTranspose2d 是相反的！所以，如果使用 'SAME' 的话，ConvTranspose2d 的 output 应该是 output = input * stride。因为 Conv2d 在计算时有一个 ceil 在算式里，所以，ConvTranspose2d 的 output 大小不是唯一的。以作业为例，ConvTranspose2d 的输入形如 (7, 7)，stride = 2，那么其对偶的 Conv2d 的输入大小可以是 7 * 2 = 14，也可以是 7 * 2 + 1 = 15。因为 Conv2d 计算输出大小的公式是 ceil ( input / stride )。这也是为什么在 PyTorch 版本的 ConvTranspose2d 中还要额外给一个 output_padding 的参数，而且还有一个 [Note](https://pytorch.org/docs/stable/nn.html) 说：

> However, when :attr`stride` >1, Conv2d maps multiple input shapes to the same output shape. output_padding is provided to resolve this ambiguity by effectively increasing the calculated output shape on one side. Note that output_padding is only used to find output shape, but does not actually add zero-padding to output.

到这里，我们得到了对偶的 Conv2d 的输入是 (14, 14) 或者 (15, 15)，输出是 (7, 7)，kernel 是 (4, 4)，stride = 2，那么 padding 就可以计算出来了：
  * 如果输入是取 (14, 14) 的话，(14 - 4 + 2 * padding) / 2 + 1 = 7，此时的 padding 是 1.
  * 如果输入是取 (15, 15) 的话，(15 - 4 + 2 * padding) / 2 + 1 = 7，此时的 padding 是 1. 那么 padding 是 0.5。

这两个结果没有对错之分，只不过我们取一个偶数的值是比较好的，所以这里取 14。到此为止，Conv2d 的所有参数都已经确定，即
```python
torch.nn.Conv2d(128, 64, (4, 4), stride=2, padding=1)
```
它的输入是形如 (batch_size, 128, 14, 14)，输出是形如 (batch_size, 64, 7, 7) 的。

因此，其对偶的 ConvTranspose2d 的参数与其完全相同，即为：
```python
torch.nn.ConvTranspose2d(128, 64, (4, 4), stride=2, padding=1)
```
它的输入是形如 (batch_size, 128, 7, 7)，输出是形如 (batch_size, 64, 14, 14) 的。从而将一张 7 * 7 的图片 upsampling 到了 14 * 14。

第二个 ConvTranspose2d 层用同样方法计算

>Conv2D^T (Transpose): 1 filter of 4x4, stride 2, 'same' padding

Conv2d 的输出为 14 * 14 的图片，采用 stride 2，'same' padding，所以 Conv2d 的输入为28或者29，这里取28。 需要 padding = 1，所以
```python
torch.nn.Conv2d(64, 1, (4, 4), stride=2, padding=1)
```
那么其对偶的 ConvTranspose2d 为：
```python
torch.nn.ConvTranspose2d(64, 1, (4, 4), stride=2, padding=1)
```

> Inline Question 1:
>
We will look at an example to see why alternating minimization of the same objective (like in a GAN) can be tricky business.
>
>Consider $f(x,y)=xy$. What does $\min_x\max_y f(x,y)$ evaluate to? (Hint: minmax tries to minimize the maximum value achievable.)
>
> Now try to evaluate this function numerically for 6 steps, starting at the point $(1,1)$,
by using alternating gradient (first updating y, then updating x) with step size $1$.
You'll find that writing out the update step in terms of $x_t,y_t,x_{t+1},y_{t+1}$ will be useful.
>
> Record the six pairs of explicit values for $(x_t,y_t)$ in the table below.

具体数据见作业中的代码输出。

> Inline Question 2:
>
> Using this method, will we ever reach the optimal value? Why or why not?

不会到达最优解，因为学习曲线在震荡。

> Inline Question 3:
>
> If the generator loss decreases during training while the discriminator loss stays at a constant high value from the start, is this a good sign? Why or why not? A qualitative answer is sufficient

不是好的现象。discriminator 的 loss 由两部分组成，第一部分说明它辨识真图片的能力，与 generator 无关；第二部分说明它辨识生成图片的能力。如果 generator 的 loss 下降，说明它欺骗 discriminator 的能力提高了，而 discriminator 的 loss 始终保持很大，很可能是因为 discriminator 太差，甚至连辨识真实图片的能力都很差，极端情况就是 discriminator 依 50% 的概率在乱猜。所以说，generator loss 的下降也很可能是因为 discriminator 太差，很容易被欺骗。
