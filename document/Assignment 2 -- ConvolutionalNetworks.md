[Assignment 2 | ConvolutionalNetworks](https://github.com/FortiLeiZhang/cs231n/blob/master/code/cs231n/assignment2/ConvolutionalNetworks.ipynb)
---
## Convolution
首先说说这个aside。仅仅将 filter 设置为特定的数字，就可以对图像进行某种处理，这里是将图像灰度化和在一个图层取其水平边缘，这直观的给出了一个 filter 的作用。第一层的 filter 可以直观的看出其作用，以后各层 filter 是第一层 filter 的各种线性、非线性组合，可能不会有直观的表示。
而训练的目的就是找出各种表示不同特征的 filter，可能有些 filter 表征图片中的边，有些 filter 表征角，这些 filter 不同的组合就可以区分出各种物体。

#### Forward
Naive 的实现没什么可说的，这里要注意的是输入输出的维度，和np.pad函数的使用。

#### Backward
这里 x 形如 (N, C, in_H, in_W)，w 形如 (K, C, f_H, f_W)，y 形如 (N, K, out_H, out_W)，b 形如(K, )，dx, dw, dout分别与 x, w, y 形状相同。

##### db
db 仅与 dout 有关，且 b 形如(K, )，dout 形如 (N, K, out_H, out_W)，所以根据维度分析，可以得到:
```python
db = np.sum(dout, axis=(0, 2, 3))
```

##### dx
这里写出 dx 和 dw 的闭式解很复杂，而且不容易写出代码，所以这里用一个例子来推出 dx 和 dw 的计算过程，根据此计算过程可以将代码写出。这里 stride = 1，pad = 0，x，w，y为：
$$
x = \begin{bmatrix}
 x_{11}&   x_{12}&  x_{13}\newline
 x_{21}&   x_{22}&  x_{23}\newline
 x_{31}&   x_{32}&  x_{33}
\end{bmatrix}, \quad
w = \begin{bmatrix}
 w_{11}&   w_{12}\newline
 w_{21}&   w_{22}
\end{bmatrix}, \quad
y = \begin{bmatrix}
 y_{11}&   y_{12}\newline
 y_{21}&   y_{22}
\end{bmatrix}
$$
将 $y = x \times w$ 展开：
$$
\begin{aligned}
y_{11} &= w_{11}x_{11} + w_{12}x_{12} + w_{21}x_{21} + w_{22}x_{22} \newline
y_{12} &= w_{11}x_{12} + w_{12}x_{13} + w_{21}x_{22} + w_{22}x_{23} \newline
y_{21} &= w_{11}x_{21} + w_{12}x_{22} + w_{21}x_{31} + w_{22}x_{32} \newline
y_{22} &= w_{11}x_{22} + w_{12}x_{23} + w_{21}x_{32} + w_{22}x_{33} \newline
\end{aligned}
$$
所以：
$$
\mathrm{d} x = \begin{bmatrix}
 \frac{\partial L}{\partial y}\frac{\partial y}{\partial x_{11}}&   \frac{\partial L}{\partial y}\frac{\partial y}{\partial x_{12}}&  \frac{\partial L}{\partial y}\frac{\partial y}{\partial x_{13}}\newline
 \frac{\partial L}{\partial y}\frac{\partial y}{\partial x_{21}}&   \frac{\partial L}{\partial y}\frac{\partial y}{\partial x_{22}}&  \frac{\partial L}{\partial y}\frac{\partial y}{\partial x_{23}}\newline
 \frac{\partial L}{\partial y}\frac{\partial y}{\partial x_{31}}&   \frac{\partial L}{\partial y}\frac{\partial y}{\partial x_{32}}&  \frac{\partial L}{\partial y}\frac{\partial y}{\partial x_{33}}
\end{bmatrix}
$$
与 $x_{11}$ 相关的仅有 $y_{11}$，所以第一项 $\frac{\partial L}{\partial y}\frac{\partial y}{\partial x_{11}} = \partial y_{11} \cdot w_{11}$，与 $x_{12}$ 相关的有两项 $y_{11}$ 和 $y_{12}$，所以第二项 $\frac{\partial L}{\partial y}\frac{\partial y}{\partial x_{12}} = \partial y_{11} \cdot w_{12} + \partial y_{12} \cdot w_{11}$，依次类推，将最后结果写成如下形式就一目了然了：
$$
\mathrm{d} x = \begin{bmatrix}
\partial y_{11} \cdot w_{11}&   \partial y_{11} \cdot w_{12} + &  \newline
&\partial y_{12} \cdot w_{11} & \partial y_{12} \cdot w_{12} \newline
& & \newline
 \partial y_{11} \cdot w_{21} + &  \partial y_{11} \cdot w_{22} + &   \newline
  &  \partial y_{12} \cdot w_{21} + & \partial y_{12} \cdot w_{22} \newline
 \partial y_{21} \cdot w_{11} &  \partial y_{21} \cdot w_{12} + &  \newline
  &  \partial y_{22} \cdot w_{11} & \partial y_{22} \cdot w_{12} \newline
 & & \newline
 \partial y_{21} \cdot w_{21} &  \partial y_{21} \cdot w_{22} + &  \newline
  &  \partial y_{22} \cdot w_{21} & \partial y_{22} \cdot w_{22} \newline
\end{bmatrix}
$$
显然，dx的计算方法是在一个形如 x 的矩阵上滑动，先计算
$$
\partial y_{11} \cdot
\begin{bmatrix}
 w_{11}&   w_{12}\newline
 w_{21}&   w_{22}
\end{bmatrix}
$$
并将结果放在 dx 的第一个形如 w 的块上，然后计算
$$
\partial y_{12} \cdot
\begin{bmatrix}
 w_{11}&   w_{12}\newline
 w_{21}&   w_{22}
\end{bmatrix}
$$
滑动 stride，并将结果放在 dx 的第二个形如 w 的块上，依次类推。

这里需要注意的是：
1. 例子里的 pad = 0。如果 pad 不为0的话，所有对 x 的计算都要针对扩充后的 x_pad，得到的结果也是 dx_pad，最后返回的结果 dx 要将 dx_pad 去掉 pad。
2. 滑动的次数由 dout 的形状决定，滑动的步长由 stride 决定。

以上仅是针对 x 的最后两个维度的计算，前两个维度加循环即可
```python
for i in range(N):
    for oc in range(K):
        for ww in range(out_w):
            for hh in range(out_h):
                dpad_x[i, :, (s*hh):(s*hh+f_h), (s*ww):(s*ww+f_w)] += dout[i, oc, hh, ww] * w[oc, ...]

dx = dpad_x[:, :, p:(in_h+p), p:(in_w+p)]
```

**回来补充内容**

后面在实现 GAN 的时候，要用到 ConvTranspose2d 来讲实现 Conv2d 的逆过程，也就是将一张 $m \times m$ 的图片，upsampling 到 $n \times n$，这里 $n > m$。 ConvTranspose2d 的实现方法，与上面计算 dx 的方法完全相同。实际上，不论在 PyTorch 还是在 TensorFlow 里面，ConvTranspose2d 的实现和计算 dx 的梯度的实现，使用的是同一段代码。在 [PyTorch 的文档](https://pytorch.org/docs/stable/nn.html)里明确说明了这一点：
> This module can be seen as the gradient of Conv2d with respect to its input.

实际上，我用 PyTorch 的 torch.nn.functional.conv_transpose2d 实现了一下计算 dx 的梯度，得到的结果是一样的：

```python
dout_tensor = torch.from_numpy(dout)
w_tensor = torch.from_numpy(w)
dx = torch.nn.functional.conv_transpose2d(dout_tensor, w_tensor, bias=None, stride=s, padding=p)
dx = dx.numpy()
```

##### dw
根据如上同样的方法：
$$
\mathrm{d} w = \begin{bmatrix}
 \frac{\partial L}{\partial y}\frac{\partial y}{\partial w_{11}}&   \frac{\partial L}{\partial y}\frac{\partial y}{\partial w_{12}} \newline
 \frac{\partial L}{\partial y}\frac{\partial y}{\partial w_{21}}&   \frac{\partial L}{\partial y}\frac{\partial y}{\partial w_{22}} \newline
\end{bmatrix}
$$
根据$y = x \times w$ 展开式：
$$
\begin{aligned}
y_{11} &= w_{11}x_{11} + w_{12}x_{12} + w_{21}x_{21} + w_{22}x_{22} \newline
y_{12} &= w_{11}x_{12} + w_{12}x_{13} + w_{21}x_{22} + w_{22}x_{23} \newline
y_{21} &= w_{11}x_{21} + w_{12}x_{22} + w_{21}x_{31} + w_{22}x_{32} \newline
y_{22} &= w_{11}x_{22} + w_{12}x_{23} + w_{21}x_{32} + w_{22}x_{33} \newline
\end{aligned}
$$
可以得到：
$$
\begin{aligned}
\partial w_{11} &= \partial y_{11}x_{11} + \partial y_{12}x_{12} + \partial y_{21}x_{21} + \partial y_{22}x_{22} \newline
\partial w_{12} &= \partial y_{11}x_{12} + \partial y_{12}x_{13} + \partial y_{21}x_{22} + \partial y_{22}x_{23} \newline
\partial w_{21} &= \partial y_{11}x_{21} + \partial y_{12}x_{22} + \partial y_{21}x_{31} + \partial y_{22}x_{32} \newline
\partial w_{22} &= \partial y_{11}x_{22} + \partial y_{12}x_{23} + \partial y_{21}x_{32} + \partial y_{22}x_{33} \newline
\end{aligned}
$$
将结果写为如下形式：
$$
\mathrm{d} w = \partial y_{11} \cdot
\begin{bmatrix}
 x_{11}&   x_{12}\newline
 x_{21}&   x_{22}
\end{bmatrix} + \partial y_{12} \cdot
\begin{bmatrix}
 x_{12}&   x_{13}\newline
 x_{22}&   x_{23}
\end{bmatrix} + \partial y_{21} \cdot
\begin{bmatrix}
 x_{21}&   x_{22}\newline
 x_{31}&   x_{32}
\end{bmatrix} + \partial y_{22} \cdot
\begin{bmatrix}
 x_{22}&   x_{23}\newline
 x_{32}&   x_{33}
\end{bmatrix}
$$
显然，dw 的计算方法是首先用 $\partial y_{11}$ 与 x 的第一个形如 w 的矩阵相乘，然后移动 stride 步，再用 $\partial y_{12}$ 与 x 的第二个形如 w 的矩阵相乘，将所得结果按位相加。

同样，这里也要注意的是：
1. 如果 pad 不为0的话，要针对扩展后的 x_pad 进行计算，但是得到的结果即为 dw，不需要strip。
2. 滑动的次数依旧由 dout 决定，步长由 stride 决定。
```python
for i in range(N):
    for oc in range(K):
        for ww in range(out_w):
            for hh in range(out_h):
                dw[oc, ...] += dout[i, oc, hh, ww] * x_pad[i, :, (s*hh):(s*hh+f_h), (s*ww):(s*ww+f_w)]
```

## Max-Pooling
forward 和 backward 都很简单，写代码的时候注意一下就可以了。

## Fast Layers
这是用矩阵相乘的方法实现 convolution，需要很精巧的设计将矩阵展开成向量。具体实现太精巧了，没仔细研究。

## Three Layer ConvNet
作业里的这个三层 CNN 设计的不是很好，可能是为了简单一些。这里只把作业中新建网络时检查的几个点再强调一下。
1. Sanity check loss：要做的是随机初始化一个网络，softmax 的 loss 应该为 $\log(N)$，svm 的 loss 应该为 $N-1$。
2. Gradient check：检查梯度的公式计算结果和数据计算结果是不是一致。
3. Overfit small data：找个小的数据集，然后 overfit 此数据集。

















end
