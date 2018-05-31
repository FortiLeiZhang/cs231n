[Optimization: Stochastic Gradient Descent](http://cs231n.github.io/optimization-1/)

这一节主要讲optimization的相关内容。重点在于各种grads的实现，特别是与矩阵相关的grads的实现，包括公式推导和代码实现。note 3中先给出了svm的grads，以后还会遇到softmax，conv，relu，BN等等各种grads。将会结合作业详细的给出各种grads的公式推导和代码实现。

#### [Assignment 1: SVM grads的计算](https://github.com/FortiLeiZhang/cs231n/blob/master/code/cs231n/assignment1/svm.ipynb)

##### 公式推导

对一个sample $x_i$，svm的loss为：

$$
\begin{aligned}
L_i = & \sum_{j \neq y_i} \max\left( 0, w_j x_i - w_{y_i} x_i + \Delta \right) \newline
= & \max\left( 0, w_0 x_i - w_{y_i} x_i + \Delta \right) + \max\left( 0, w_1 x_i - w_{y_i} x_i + \Delta \right)  + ... + \max\left( 0, w_j x_j - w_{y_i} x_i + \Delta \right) + ...
\end{aligned}
$$

$L_i$ 对 $w_j$ 求导：

$$
\frac{\partial L_i}{\partial w_j} = 0 + 0 + ... +
 \mathbb{1} \left( w_j x_i - w_{y_i} x_i + \Delta > 0\right) \cdot x_i
$$

$L_i$ 对 $w_{y_i}$ 求导：

$$
\begin{aligned}
\frac{\partial L_i}{\partial w_{y_i}} =&
\mathbb{1} \left( w_0 x_i - w_{y_i} x_i + \Delta > 0\right) \cdot (-x_i) +
 \mathbb{1} \left( w_1 x_i - w_{y_i} x_i + \Delta > 0\right) \cdot (-x_i) + ... + \mathbb{1} \left( w_j x_i - w_{y_i} x_i + \Delta > 0\right) \cdot (-x_i) + ... \newline
 =& - \left(  \sum_{j \neq y_i}  \mathbb{1} \left( w_j x_i - w_{y_i} x_i + \Delta > 0\right) \right) \cdot x_i
 \end{aligned}
$$

##### 代码实现
