[Assignment 2 | Batch Normalization](https://github.com/FortiLeiZhang/cs231n/blob/master/code/cs231n/assignment2/BatchNormalization.ipynb)
---

上文吐槽BN部分讲的太烂，2018年果然更新了这一部分，slides里加了好多内容，详见[Lecture 6的slides](http://cs231n.stanford.edu/slides/2018/cs231n_2018_lecture06.pdf)第54到61页，以及[Lecture 7的slides](http://cs231n.stanford.edu/slides/2018/cs231n_2018_lecture07.pdf)第11到18页，这里结合着[原始论文](https://arxiv.org/abs/1502.03167)和作业，把BN及其几个变种好好总结一下。

# Batch Normalization
## Train
前面的作业中已经见识到了，weight初始化时方差的调校真的是很麻烦，小了梯度消失不学习，大了梯度爆炸没法学习。
即使开始初始化的很好，随着学习的深入，网络的加深，每一层的方差已经不再受控；另外，特别是对于刚开始的几层，方差上稍微的变化，都会在forward prop时逐级放大的传递下去。
作业中只是三五层的小网络，要是几十上百层的网络，可以想象学习几乎是不可能的。

既然每一层输入的方差会产生如此多的问题，这就产生了第一个想法，何不将每一层的输入直接标准化为0均值单位方差。由于NN的train多是基于mini-batch的，所以这里标准化也是基于mini-batch。

输入x是包含N个sample的mini-batch，每个sample有D个feature。对每个feature进行标准化，即：
$$
\begin{aligned}
\mu_j &= \frac{1}{N} \sum_{i = 1}^{N} x_{i,j} \newline
\sigma_{j}^{2} &= \frac{1}{N} \sum_{i = 1}^{N} \left ( x_{i,j} - \mu_j\right)^2
\end{aligned}
$$
标准化后的输出为:
$$
\hat{x} = \frac{x - \mu_j}{\sqrt{\sigma_j^2 + \epsilon}}
$$

但是但是但是，这里武断的使输入均值为0，方差为1真的是最好的选择么？不一定。如果不是最好的选择，
设为多少是最好的选择呢？不知道。不知道的话怎么办呢?
那就让NN自己去学习一个最好的去呗。所以才有了下一步：

$$
y = \gamma \cdot \hat{x} + \beta
$$

其中，$\gamma$和 $\beta$是要学习的参数，将输入的均值和方差从(0,1)又拉到了 $(\gamma, \beta)$。

所以，通常说起来BN是一层，但是我认为，BN是两层：Normalization Layer和Shift Layer，这两层是紧密相连，不可分割的。其中，Normalization Layer将输入的均值和方差标准化为(0,1)，Shift Layer又将其拉到 $(\gamma, \beta)$。这里，$(\gamma, \beta)$ 和其他的weight、bias一样，都是通过backprop算梯度，然后再用SGD等方法更新学习得到。

好，这里强调两个问题，也是我第一遍看paper时的疑惑，也是2017年视频中那位小姑娘讲课时犯的错误:

1. 一提到BN层的作用，马上想到的是：将输入映射为0均值单位方差的高斯分布。错！首先它不一定是高斯分布，可以是任意的分布，BN仅仅改变均值方差，不改变分布。其次，均值方差不是(0,1)，而是 $(\gamma, \beta)$。说(0, 1)的是忘记了shift这一层。
2. 原文中有一句，还打了斜体:
> To address this, we make sure that the transformation inserted in the network can represent the identity transform.

当时看的时候就不明白，既然费半天劲减均值除方差，怎么这里又要 "represent the identity transform"? 而且加上后边的 $(\gamma, \beta)$操作，就更看不懂了。其实这里漏看了一个 “can” 。既然 $(\gamma, \beta)$ 是学习来的，它们当然可以是原始输入的均值方差了，所以BN有表达一个identity transform的能力，而不是必须要表达一个identity transform。

总结一下：
```python
input:
      x: (N, D)
intermediates:
      mean: (1, D)  
          mean = np.mean(x, axis=0)
      var: (1, D)
          var = np.var(x, axis=0)
      xhat: (N, D)
          xhat = (x - mean) / (np.sqrt(var + eps))
learnable params:
      gamma: (1, D)
      beta: (1, D)
输出：
      y = gamma * xhat + beta
```
## Test
在test时，就没有mini-batch可用来算 $\mu$和 $\sigma^2$了，此时常用的方法是在train的过程中记录一个 $\mu$和 $\sigma^2$的滑动均值在test的时候使用。

BN通常放在FC/Conv之后，ReLU之前。

### Backprop
BN的backprop是这次作业的难点，还要用两种方法做，这里一步一步尽量详细地把推导过程写出来。
#### $\mathrm{d} \beta$
$\mathrm{d} \beta$ 用维度分析法：
$$
y = \gamma \cdot \hat{x} + \beta
$$
其中 $y$ 形如(N, D)，$\gamma$ 和 $\beta$ 形如(D,)，$\hat{x}$ 形如(N, D)，所以 $\mathrm{d} \beta$必然为：
```python
dbeta = np.sum(dout, axis=0)
```
这里就不赘述了。

#### $\mathrm{d} \gamma$
其实 $\mathrm{d} \gamma$ 也可以用维度分析法得到，$\mathrm{d} y$ 和 $\mathrm{d} \hat{x}$ 都形如(N, D)，而 $\mathrm{d} \gamma$ 形如(D,)，显然 $\mathrm{d} \gamma$ 应为：
```python
dgamma = np.sum(xhat * dout, axis=0)
```
这里还是把过程写一下吧
$$
\begin{bmatrix}
 y_{11}&   y_{12}&  ... &  y_{1D}\newline
 y_{21}&   y_{22}&  ... &  y_{2D}\newline
      & ...      &  ... & \newline
  y_{N1}&   y_{N2}& ... &  y_{ND}
\end{bmatrix} =
\begin{bmatrix}
\gamma_1& \gamma_2& ... & \gamma_D
\end{bmatrix} \cdot
\begin{bmatrix}
 x_{11}&   x_{12}&  ... &  x_{1D}\newline
 x_{21}&   x_{22}&  ... &  x_{2D}\newline
      & ...      &  ... & \newline
  x_{N1}&   x_{N2}& ... &  x_{ND}
\end{bmatrix}
$$
展开可得：
$$
\begin{aligned}
y_{11} = \gamma_1 \cdot x_{11}, \quad \quad & y_{12} = \gamma_2 \cdot x_{12}, & ... \newline
y_{21} = \gamma_1 \cdot x_{21}, \quad \quad & y_{22} = \gamma_1 \cdot x_{22}, & ...
\end{aligned}
$$
由此可得：
$$
\frac{\partial \mathrm{L}}{\partial \gamma_q} = \frac{\partial \mathrm{L}}{\partial y} \cdot \frac{\partial y}{\partial \gamma_q} = \sum_{i,j} \frac{\partial \mathrm{L}}{\partial y_{ij}} \cdot \frac{\partial y_{ij}}{\partial \gamma_q}
$$
而仅当 $j = q$ 时有
$$
\frac{\partial y_{ij}}{\partial \gamma_q} = x_{iq}
$$
其余均为0，故：
$$
\frac{\partial \mathrm{L}}{\partial \gamma_q} = \sum_{i=1}^{N} \frac{\partial \mathrm{L}}{\partial y_{iq}} \cdot \frac{\partial y_{iq}}{\partial \gamma_q} = \sum_{i=1}^{N}x_{iq} \cdot \mathrm{d} y_{iq}
$$

#### $\mathrm{d} x$：第一种方法
![计算图](https://kratzert.github.io/images/bn_backpass/BNcircuit.png)
先画出forward和backward的计算图，如图所示。forward的代码如下：
```python
x_mean = 1 / N * np.sum(x, axis=0)
x_mean_0 = x - x_mean
x_mean_0_sqr = x_mean_0 ** 2
x_var = 1 / N * np.sum(x_mean_0_sqr, axis=0)
x_std = np.sqrt(x_var + eps)
inv_x_std = 1 / x_std
x_hat = x_mean_0 * inv_x_std

out = gamma * x_hat + beta
cache = (x_mean, x_mean_0, x_mean_0_sqr, x_var, x_std, inv_x_std, x_hat, gamma, eps)
```
这里需要注意的是
1. 尽量将每一步化成最简单的加、乘操作，并且将每一步等号左边的项全部cache起来。这样做的目的是减少backprop时的计算量，但是相应的存贮量就会增加。所以说NN的内存需求要远远大于weights和bias的数目。
2. 计算mean是，用 1/N * np.sum()，不要用np.mean()，否则在backprop的时候会把 1/N 漏掉。

如果forward的每一步计算分解的足够细的话，backprop可以很清楚：
```python
# out = gamma * x_hat + beta
# (N,D) (D,)    (N,D)   (D,)
Dx_hat = dout * gamma

# x_hat = x_mean_0 * inv_x_std
# (N,D)   (N,D)      (D,)
Dx_mean_0 = Dx_hat * (inv_x_std)
Dinv_x_std = np.sum(Dx_hat * (x_mean_0), axis=0)

# inv_x_std = 1 / x_std
# (D,)            (D,)
Dx_std = Dinv_x_std * (- x_std ** (-2))

# x_std = np.sqrt(x_var + eps)
# (D,)           (D,)
Dx_var = Dx_std * (0.5 * (x_var + eps) ** (-0.5))

# x_var = 1 / N * np.sum(x_mean_0_sqr, axis=0)
# (D,)                   (N,D)
Dx_mean_0_sqr = Dx_var * (1 / N * np.ones_like(x_mean_0_sqr))

# x_mean_0_sqr = x_mean_0 ** 2
# (N,D)          (N,D)
Dx_mean_0 += Dx_mean_0_sqr * (2 * x_mean_0)

# x_mean_0 = x - x_mean
# (N,D)     (N,D) (D,)
Dx = Dx_mean_0 * (1)
Dx_mean = - np.sum(Dx_mean_0, axis=0)

# x_mean = 1 / N * np.sum(x, axis=0)
# (D,)                   (N,D)
Dx += Dx_mean * (1 / N * np.ones_like(x_hat))

dx = Dx
```
这里要注意的是：
1. 一定要把每一步计算中每一项的维度搞清楚写下来。注意这一步：
```python
# x_hat = x_mean_0 * inv_x_std
# (N,D)   (N,D)      (D,)
Dx_mean_0 = Dx_hat * (inv_x_std)
Dinv_x_std = np.sum(Dx_hat * (x_mean_0), axis=0)
```
因为numpy在进行矩阵运算的时候会进行自动的broadcast，所以这里 inv_x_std 实际是形如 (D,)，但是计算是会broadcast成为(N, D)。仅从式子看的话，很容易误写为：
```python
Dinv_x_std = Dx_hat * (x_mean_0)
```
这时如果进行一下维度分析，会发现 Dinv_x_std 显然要形如 (D,)，但是右侧点积的结果形如 (N, D)，显然要对 axis=0 进行 sum。同理还有这一行：
```python
# x_mean_0 = x - x_mean
# (N,D)     (N,D) (D,)
Dx = Dx_mean_0 * (1)
Dx_mean = np.sum(Dx_mean_0 * (-1), axis=0)
```

2. 对 $y = \sum_{i} x_i$ 的求导，这里
$$
\begin{aligned}
y = &\ [y_1,  y_2, ... , y_D] \newline \newline
x = &\begin{bmatrix}
 x_{11}&   x_{12}&  ... &  x_{1D}\newline
 x_{21}&   x_{22}&  ... &  x_{2D}\newline
      & ...      &  ... & \newline
  x_{N1}&   x_{N2}& ... &  x_{ND}
\end{bmatrix}
\end{aligned}
$$
其中
$$
\begin{aligned}
y_1 &= \frac{1}{N} \left( x_{11} + x_{21} + ... + x_{N1}\right) \newline
y_2 &= \frac{1}{N} \left( x_{12} + x_{22} + ... + x_{N2}\right) \newline
&...
\end{aligned}
$$
所以
$$
\mathrm{d} x_{11} = \frac{\partial \mathrm{L}}{\partial y} \cdot \frac{\partial y}{\partial x_{11}} = \sum_{i} \frac{\partial \mathrm{L}}{\partial y_{i}} \cdot \frac{\partial y_{i}}{\partial x_{11}} = \frac{\partial \mathrm{L}}{\partial y_{1}} \cdot \frac{\partial y_{1}}{\partial x_{11}} = \mathrm{d} y_1 \cdot \frac{1}{N}
$$
综上：
$$
\begin{aligned}
\mathrm{d} x &= \frac{1}{N} \cdot \begin{bmatrix}
 \mathrm{d} y_1&   \mathrm{d} y_2&  ... &  \mathrm{d} y_D\newline
 \mathrm{d} y_1&   \mathrm{d} y_2&  ... &  \mathrm{d} y_D\newline
      & ...      &  ... & \newline
  \mathrm{d} y_1&   \mathrm{d} y_2& ... &  \mathrm{d} y_D
\end{bmatrix} \newline
&= \frac{1}{N} \cdot \mathrm{d} y \cdot \begin{bmatrix}
 1&   1&  ... &  1\newline
 1&   1&  ... &  1\newline
      & ...      &  ... & \newline
  1&   1& ... &  1
\end{bmatrix}_{N \times D}
\end{aligned}
$$
```python
# x_mean = 1 / N * np.sum(x, axis=0)
# (D,)                   (N,D)
Dx += Dx_mean * (1 / N * np.ones_like(x_hat))
```

3. 注意到backprop时 Dx_mean_0 两次出现在等式左边，这说明在计算图中有两条路径通向 Dx_mean_0，这两条路径的结果要相加，所以第二次出现时要用 +=:
```python
Dx_mean_0 = Dx_hat * (inv_x_std)
Dx_mean_0 += Dx_mean_0_sqr * (2 * x_mean_0)
```
#### $\mathrm{d} x$：第二种方法
第二种方法的公式推导实在是太繁了，我再也不想写第二遍了。先来个计算图：
$$
x \rightarrow \hat{x} \rightarrow y \rightarrow L
$$
中间参数分别为：
$$
\begin{aligned}
\mathrm{d} out &= \frac{\partial L}{\partial y} \newline
y &= \gamma \cdot \hat{x} + \beta \newline
\hat{x} &= \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \newline
\mu &= \frac{1}{N} \sum_{n=1}^{N} x_n \newline
\sigma^2 &= \frac{1}{N} \sum_{n=1}^{N} \left(x_n - \mu\right)^2
\end{aligned}
$$
计算对 $x_{ij}$ 的导数：
$$
\begin{aligned}
\frac{\partial L}{\partial x_{ij}} &= \sum_{n,d} \frac{\partial L}{\partial y_{nd}} \cdot \frac{\partial y_{nd}}{\partial x_{ij}} \newline
&= \sum_{n,d} \frac{\partial L}{\partial y_{nd}} \cdot \frac{\partial y_{nd}}{\partial \hat{x_{nd}}} \cdot \frac{\partial \hat{x_{nd}}}{\partial x_{ij}}
\end{aligned}
$$
其中：
$$
\begin{aligned}
y_{nd} &= \gamma_d \cdot \hat{x_{nd}} + \beta_d \newline
\hat{x_{nd}} &= \frac{x_{nd} - \mu_d}{\sqrt{\sigma_d^2 + \epsilon}} \newline
\mu_d &= \frac{1}{N} \sum_{n=1}^{N} x_{nd} \newline
\sigma_d^2 &= \frac{1}{N} \sum_{n=1}^{N} \left(x_{nd} - \mu_d \right)^2 \newline
\frac{\partial y_{nd}}{\partial \hat{x_{nd}}} &= \gamma_d
\end{aligned}
$$
下面的工作就是要计算 $\frac{\partial \hat{x_{nd}}}{\partial x_{ij}}$:
$$
\begin{aligned}
\frac{\partial \hat{x_{nd}}}{\partial x_{ij}} &= \frac{\partial}{\partial x_{ij}} \left( \frac{x_{nd} - \mu_d}{\sqrt{\sigma_d^2 + \epsilon}}\right) \newline
&= \left( \sigma_d^2 + \epsilon \right)^{-\frac{1}{2}} \cdot \frac{\partial}{\partial x_{ij}} \left( x_{nd} - \mu_d \right) + \left( x_{nd} - \mu_d \right) \cdot \frac{\partial}{\partial x_{ij}} \left( \sigma_d^2 + \epsilon \right)^{-\frac{1}{2}} \newline
&= \left( \sigma_d^2 + \epsilon \right)^{-\frac{1}{2}} \cdot \frac{\partial}{\partial x_{ij}} \left( x_{nd} - \mu_d \right) - \frac{1}{2} \left( \sigma_d^2 + \epsilon \right)^{-\frac{3}{2}}\left( x_{nd} - \mu_d \right) \cdot \frac{\partial \sigma_d^2}{\partial x_{ij}}
\end{aligned}
$$
下面分别计算，首先：
$$
\begin{aligned}
\frac{\partial}{\partial x_{ij}} \left( x_{nd} - \mu_d \right) &= \frac{\partial}{\partial x_{ij}} \left( x_{nd} - \frac{1}{N} \sum_{t=1}^{N} x_{td} \right) \newline
&= \frac{\partial x_{nd}}{\partial x_{ij}} - \frac{1}{N} \frac{\partial}{\partial x_{ij}} \left( \sum_{t=1}^{N} x_{td} \right)
\end{aligned}
$$
第一项，当且仅当 $n=i$, $d=j$ 时不为0，第二项中仅有 $d=j$ 项不为0，故：
$$
\frac{\partial}{\partial x_{ij}} \left( x_{nd} - \mu_d \right) = \delta_{n, i} \cdot \delta_{d, j} - \frac{1}{N} \delta_{d, j}
$$
接着计算：
$$
\begin{aligned}
\frac{\partial \sigma_d^2}{\partial x_{ij}} &= \frac{\partial}{\partial x_{ij}} \left( \frac{1}{N} \sum_{n=1}^{N} \left(x_{nd} - \mu_d \right)^2 \right) \newline
&= \frac{1}{N} \sum_{n=1}^{N} \frac{\partial}{\partial x_{ij}} \left( \left( x_{nd} - \mu_d \right)^2  \right) \newline
&= \frac{2}{N} \sum_{n=1}^{N} \left( x_{nd} - \mu_d \right) \frac{\partial}{\partial x_{ij}} \left( x_{nd} - \mu_d \right) \newline
&= \frac{2}{N} \sum_{n=1}^{N} \left( x_{nd} - \mu_d \right) \cdot \left( \delta_{n, i} \cdot \delta_{d, j} - \frac{1}{N} \delta_{d, j} \right) \newline
&= \frac{2}{N} \sum_{n=1}^{N} \left( x_{nd} - \mu_d \right) \cdot \delta_{n, i} \cdot \delta_{d, j} - \frac{2}{N^2} \sum_{n=1}^{N} \left( x_{nd} - \mu_d \right) \cdot \delta_{d, j}
\end{aligned}
$$
第一项中，仅有 $n=i$ 一项不为0：
$$
\sum_{n=1}^{N} \left( x_{nd} - \mu_d \right) \cdot \delta_{n, i} \cdot \delta_{d, j} = \left( x_{id} - \mu_d \right) \cdot \delta_{d, j}
$$
第二项：
$$
\sum_{n=1}^{N} \left( x_{nd} - \mu_d \right) = \sum_{n=1}^{N} x_{nd} - N \cdot \mu_d
$$
而 $\mu_d = \frac{1}{N} \sum_{n=1}^{N} x_{nd}$，因此上式为0。
所以：
$$
\frac{\partial \sigma_d^2}{\partial x_{ij}} = \frac{2}{N} \left( x_{id} - \mu_d \right) \cdot \delta_{d, j}
$$
综上:
$$
\begin{aligned}
\frac{\partial \hat{x_{nd}}}{\partial x_{ij}} &= \left( \sigma_d^2 + \epsilon \right)^{-\frac{1}{2}} \cdot \frac{\partial}{\partial x_{ij}} \left( x_{nd} - \mu_d \right) - \frac{1}{2} \left( \sigma_d^2 + \epsilon \right)^{-\frac{3}{2}}\left( x_{nd} - \mu_d \right) \cdot \frac{\partial \sigma_d^2}{\partial x_{ij}} \newline
&= \left( \sigma_d^2 + \epsilon \right)^{-\frac{1}{2}} \cdot \left( \delta_{n, i} \cdot \delta_{d, j} - \frac{1}{N} \delta_{d, j} \right) - \frac{1}{N} \left( \sigma_d^2 + \epsilon \right)^{-\frac{3}{2}}\left( x_{nd} - \mu_d \right) \left( x_{id} - \mu_d \right) \cdot \delta_{d, j}
\end{aligned}
$$







end
