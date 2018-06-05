[Neural Networks Part 1: Setting up the Architecture](http://cs231n.github.io/neural-networks-1/)
---

### 常用的激活函数
#### Sigmoid
sigmoid函数的为 $\sigma(x) = \frac{1}{1+\exp(-x)}$, 值域为 $(0, 1)$， $x \rightarrow -\infty$时取0，$x \rightarrow +\infty$时取1，
 grad为 $(1 - \sigma(x)) \cdot \sigma(x)$。
 sigmoid函数用作激活函数时的主要缺点在于：
 1. 当 $\sigma(x)$ 为0或者1时，由 $(1 - \sigma(x)) \cdot \sigma(x)$ 可以看出，它的梯度都是0。也就是当x取值很大时，对x的梯度就消失了。这就要求在初始化weight是要格外小心，以防 $\sigma(x)$ 函数进入饱和区。
 2. sigmoid函数的值域是  $(0, 1)$，恒为正且不以0为中心。所以经过sigmoid以后的输出全部变成了正值，导致 $\mathrm{d} w$ 的符号完全取决于local grad。

 #### Tanh
 也是sigmoid函数的一个变体：$\tanh (x) = 2\sigma(2x) - 1$。值域为$(-1, 1)$，但是依然有饱和区。

 #### ReLU
 ReLU函数的优点在于没有饱和区，并且计算简单不需要额外参数。ReLU的缺点在于：ReLU函数在<0的区域函数值和grad都为0，如果在relu前的节点输出全部为负值，那么该节点从此之后再也不会被update到，对learning再无贡献，而且不会有任何机会被重新激活。这种情况通常发生在relu backprop一个很大的grad，或者是learning rate选取的太大。

  #### Leaky ReLU
  Leaky relu的出现就是为解决上述dead unit 的问题。

  #### Maxout
  输出两组函数中取值较大的一个

  总结：用ReLU，但是要注意learning rate的选取并且关注dead unit的比例。Leaky ReLU和Maxout可以试试，不要用sigmoid。

### NN层数和复杂度的选择
对于同一个data set，小的NN会有表达能力不足的问题，大的NN会有overfit的问题。小的NN会有比较少的局部极小值点(local minima)，网络会很快地收敛到局部极小值，但是这些局部极小值与全局最小值相比，仍然会有很大的loss；而大的NN会有更多的局部极小值点，这些点与全局最小值有着相类似的loss。从训练的结果来看，小NN得到的最终loss有很大的方差：可能会有比较好的结果，但也可能得到非常差的结果；而大NN得到的最终loss方差比较小。

所以，这里的结论是，不要因为担心overfitting而去选择小的NN；相反，在算力允许的情况下用大的NN，并且采用 L2 reg，dropout，增加输入噪声等方法来控制overfitting。





























  end
