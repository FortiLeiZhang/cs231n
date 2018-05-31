[Optimization: Stochastic Gradient Descent](http://cs231n.github.io/optimization-1/)

这一节主要讲optimization的相关内容。重点在于各种grads的实现，特别是与矩阵相关的grads的实现，包括公式推导和代码实现。note 3中先给出了svm的grads，以后还会遇到softmax，conv，relu，BN等等各种grads。将会结合作业详细的给出各种grads的公式推导和代码实现。

#### [Assignment 1: SVM grads的计算](https://github.com/FortiLeiZhang/cs231n/blob/master/code/cs231n/assignment1/svm.ipynb)

##### 公式推导

对一个sample $x_i$，svm的loss为：

$$
\begin{aligned}
L_i = & \sum_{j \neq y_i}^C \max\left( 0, w_j x_i - w_{y_i} x_i + \Delta \right) \newline
= & \max\left( 0, w_0 x_i - w_{y_i} x_i + \Delta \right) + \max\left( 0, w_1 x_i - w_{y_i} x_i + \Delta \right)  + ... + \max\left( 0, w_j x_j - w_{y_i} x_i + \Delta \right) + ...
\end{aligned}
$$

$L_i$ 对 $w_j$ 求导：

$$
\mathrm{d}w_j =  \frac{\partial L_i}{\partial w_j} = 0 + 0 + ... +
 \mathbb{1} \left( w_j x_i - w_{y_i} x_i + \Delta > 0\right) \cdot x_i
$$

$L_i$ 对 $w_{y_i}$ 求导：

$$
\begin{aligned}
\mathrm{d}w_{y_i} = \frac{\partial L_i}{\partial w_{y_i}} =&
\mathbb{1} \left( w_0 x_i - w_{y_i} x_i + \Delta > 0\right) \cdot (-x_i) +
 \mathbb{1} \left( w_1 x_i - w_{y_i} x_i + \Delta > 0\right) \cdot (-x_i) + ... + \mathbb{1} \left( w_j x_i - w_{y_i} x_i + \Delta > 0\right) \cdot (-x_i) + ... \newline
 =& - \left(  \sum_{j \neq y_i}^C  \mathbb{1} \left( w_j x_i - w_{y_i} x_i + \Delta > 0\right) \right) \cdot x_i
 \end{aligned}
$$

##### [代码实现](https://github.com/FortiLeiZhang/cs231n/blob/master/code/cs231n/assignment1/cs231n/classifiers/linear_svm.py)

###### svm_naive

$\mathrm{d}W$ 必定与 $W$ 有同样的shape，这一点是今后计算grad必须要首先确定的。在这里，$\mathrm{d}W$的shape是(3073, 10)。接下来看 $L$ 的下标是 $i \in [0, N)$，即是N个sample之一，$w$ 的下标是 $j \in [0, C)$，即10个class之一。如果此列对应的不是true class，并且score大于0，就把这个sample的$x_i$加到 $\mathrm{d}W$ 的此列；如果此列对应的是true class，要计算其余9个class中，有几个的score大于0，然后与这个sample的$x_i$相乘，放到 $\mathrm{d}W$ 对应列。如此遍历N个sample结束。

###### svm_vectorize
这里介绍非常重要的维数分析法，该方法可大大简化vectorize的分析过程，而且不易出错。首先score是X和W的函数，即：
$$
Score = X.dot(W)
$$
所以，$\mathrm{d}W$必定是由 $\mathrm{d} Score$ 和X计算得出。这里X是(N, 3073)，W是(3073, 10)，所以Score是(N, 10)，而 $\mathrm{d} Score$ 必定与Score的shape相同，所以 $\mathrm{d} Score$ 也是(N, 10)，这样，根据矩阵相乘的维数限制，可以得到
$$
\mathrm{d} W = X.T.dot(\mathrm{d} Score)。
$$
由公式推导可以得到 $\mathrm{d} Score$：
$$
\mathrm{d}s_j = \mathbb{1} \left( s_j - s_{y_i} + \Delta > 0\right)
$$
$$
\mathrm{d}s_{y_i}
 = - \sum_{j \neq y_i}^C  \mathbb{1} \left( s_j - s_{y_i} + \Delta > 0\right)
$$
即对Score的每一列，如果不是true class，且score>0，该位置 $\mathrm{d} Score$ 为1，否则为0；如果是true class，该位置的数值是此列不为0的个数。

#### [Assignment 1: softmax grads的计算](https://github.com/FortiLeiZhang/cs231n/blob/master/code/cs231n/assignment1/softmax.ipynb)

##### 公式推导
还是要stage到score级别，然后再用 $\mathrm{d} W = X.T.dot(\mathrm{d} Score)$来计算 $\mathrm{d} W$，这样可以在推导的时候不用考虑如何计算对两个矩阵相乘。
$$
L_i = - \log \left( \ p_{y_i} \right) = -\log \left(\frac{e^{s_{y_i}}}{\sum_j e^{s_j}} \right )
$$

$L_i$ 对任意 $s_k$ 求导：
$$
\begin{aligned}
\mathrm{d} s_k =& \frac{\partial L_i}{\partial s_k} = - \frac{\partial}{\partial s_k} \left( \log \left(\frac{e^{s_{y_i}}}{\sum_j e^{s_j}} \right ) \right) \newline
=& - \frac{\sum_j e^{s_j}}{e^{s_{y_i}}} \cdot \frac{\left( {e^{s_{y_i}}}\right)^{'} \cdot {\sum_j e^{s_j}} - {e^{s_{y_i}}} \cdot \left( {\sum_j e^{s_j}} \right)^{'}}{\left( {\sum_j e^{s_j}}\right)^2} \newline
=&\frac{\frac{\partial}{\partial s_k}\left( {\sum_j e^{s_j}} \right)}{{\sum_j e^{s_j}}} - \frac{ \frac{\partial }{\partial s_k} \left({e^{s_{y_i}}} \right)}{{e^{s_{y_i}}}} \newline
=&\frac{\frac{\partial}{\partial s_k}\left( e^{s_0} + e^{s_1} + e^{s_{y_i}} + ... \right)}{{\sum_j e^{s_j}}} - \frac{ \frac{\partial }{\partial s_k} \left({e^{s_{y_i}}} \right)}{{e^{s_{y_i}}}}
\end{aligned}
$$
当 $y_i = k$时：
$$
\mathrm{d} s_k = \frac{{e^{s_{y_i}}}}{{\sum_j e^{s_j}}} - 1
$$
当 $y_i \neq k$时：
$$
\mathrm{d} s_k = \frac{{e^{s_k}}}{{\sum_j e^{s_j}}}
$$
综上，
$$
\mathrm{d} s_k = \frac{{e^{s_k}}}{{\sum_j e^{s_j}}} - \mathbb{1} \left( y_i = k \right)
$$

##### [代码实现](https://github.com/FortiLeiZhang/cs231n/blob/master/code/cs231n/assignment1/cs231n/classifiers/softmax.py)

###### softmax_naive
