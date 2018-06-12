[Assignment 2 | Batch Normalization](https://github.com/FortiLeiZhang/cs231n/blob/master/code/cs231n/assignment2/FullyConnectedNets.ipynb)
---

上文吐槽BN部分讲的太烂，2018年果然更新了这一部分，slides里加了好多内容，详见[Lecture 6的slides](http://cs231n.stanford.edu/slides/2018/cs231n_2018_lecture06.pdf)第54到61页，以及[Lecture 7的slides](http://cs231n.stanford.edu/slides/2018/cs231n_2018_lecture07.pdf)第11到18页，这里结合着[原始论文](https://arxiv.org/abs/1502.03167)和作业，把BN及其几个变种好好总结一下。

### Batch Normalization
#### Train
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
标准化后的输出为：
$$
\hat{x}_{i, j} = \frac{x_{i, j} - \mu_j}{\sqrt{\sigma_{j}^{2} + \epsilon}}
$$
但是但是但是，这里武断的使输入均值为0，方差为1真的是最好的选择么？不一定。如果不是最好的选择，设为多少是最好的选择呢？不知道。不知道的话怎么办呢？那就让NN自己去学习一个最好的去呗。所以才有了下一步：
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
#### Test
在test时，就没有mini-batch可用来算 $\mu$和 $\sigma^2$了，此时常用的方法是在train的过程中记录一个 $\mu$和 $\sigma^2$的滑动均值在test的时候使用。

BN通常放在FC/Conv之后，ReLU之前。































end
