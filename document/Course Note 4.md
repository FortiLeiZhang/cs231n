[Backpropagation, Intuitions](http://cs231n.github.io/optimization-2/)
=====
Backprop和chain rule，就是用求解微分时的链式法则，将复杂算式的微分计算，一步步分解成小的node，然后用这些基本的node层层叠加，最后得到微分结果。通常做法是先画出computation graph，然后再stage by stage的计算grads，基本的公式是：
>                      down_diff = local_diff * up_diff

其中up_diff是从上一层block传递下来的，local_diff要通过计算得到，并且和输入值有关，两者相乘传递给下一层的block。道理很简单，但是具体代码写起来会遇到各种问题，到时候再见招拆招吧。

### Patterns in backward flow
#### add gate
up_diff不做任何改变均匀的传向两个分支。

#### max gate
up_diff传向输入值大的一个分支，另一个分支为0

#### multiply gate
up_diff与输入值相乘后传向另外一个分支。需要注意的是，这里有一个switch的动作，即一个分支的grad与另外一个分支的输入成正比。所以一个分支的输入如果过大的话，会导致另外一个分支的grad很大，造成梯度爆炸。

具体的，score = wx，x是training data，通常是已知的不变的，所以不会计算对x的grad，只计算对w的grad $\mathrm{d} w$。如果输入数据x很大的话，那么由于w和x要相乘，那么会造成 $\mathrm{d} w$ 很大，这样的后果就是，要么梯度会爆炸，要么要大大降低learning rate，使得学习变慢。所以我们要对原始的输入数据进行预处理，减去均值；同时这也是BN层会加速训练的原因之一。

#### sigmoid gate
$$
\frac{\mathrm{d}}{\mathrm{d} x}  \sigma(x)\ = \left( 1 - \sigma(x) \right) \cdot \sigma(x)
$$

### Gradients for vectorized operations
这又是个头疼的地方，首先要记住的一点是，stage到score，不要妄图直接计算 $\mathrm{dW}$, 先算 $\mathrm{dScore}$，然后通过 **维度分析** 来计算 $\mathrm{dW}$。例如，X是(N, D)，W是(D, C)，那么Score = X.dot(W)是(N, C)。根据 **维度分析**：
$$
\begin{aligned}
\mathrm{dW} &= X.T.\mathrm{dot} (\mathrm{dScore} ) \newline
\mathrm{dX} &= \mathrm{d Score}.\mathrm{dot}(W.T)
\end{aligned}
$$
这样可以省去很多麻烦。

如果实在避免不了计算对vector的grad，那么就要项note里说的，先写出对vector中每一项的grad，然后再去general成vector形式，这里常用到的公式是视频里板书的那个:
$$
\frac{\partial f}{\partial X} = \sum_i \frac{\partial f}{\partial q_i} \cdot \frac{\partial q_i}{\partial X}，
$$
这里要注意的是写代码时np.sum()要对哪个axis进行。

此处有作业
---
#### [Assignment 1: two_layer_net ](https://github.com/FortiLeiZhang/cs231n/blob/master/code/cs231n/assignment1/two_layer_net.ipynb)
难点在于grads的计算，这里详细的把推导过程写写，然后总结出一套简单的算法，以后再用时直接用这套算法就行，省得再去想一遍推导过程。代码在[neural_network](https://github.com/FortiLeiZhang/cs231n/blob/master/code/cs231n/assignment1/cs231n/classifiers/neural_net.py)里。

这个两层的NN是先经过(W1, b1)，然后ReLU一下，在经过(W2, b2)得到score，然后score再经过softmax，得到最后的Loss。它的forward很简单：
```python
layer1_out = X.dot(W1) + b1
relu_out = np.maximum(0, layer1_out)
scores = relu_out.dot(W2) + b2
```
得到score后，再经过softmax得到Loss：
```python
stable_scores = scores - np.max(scores, axis=1, keepdims=True)
correct_score = stable_scores[np.arange(N), y]

loss = -np.sum(np.log(np.exp(correct_score) / np.sum(np.exp(stable_scores), axis=1)))
loss = loss/N + reg * np.sum(W1 * W1) + reg * np.sum(W2 * W2)
```
下一步就是计算对W1, b1, W2, b2的grad了。

首先一定要牢记的是，上来不要妄图直接计算 $\mathrm{d} W$ 或者是 $\mathrm{d} b$，一定要先从XW+b的结果score下手。所以，这里先算Loss对score的grad，在softmax一节已经详细讲过了:
```python
Dscores = np.exp(stable_scores) / np.sum(np.exp(stable_scores), axis=1, keepdims=True)
Dscores[np.arange(N), y] -= 1
Dscores = Dscores / N
```
需要注意的一个细节是最后一步的除以N，在这里除了的话，后面可以不用再除，以防后面计算的时候忘记。然后就要计算 $\mathrm{d} W2$ 和 $\mathrm{d} b2$。这里详细的讲一下Y = XW+b如何算grads。
#### Y = X.dot(W) + b梯度的计算
首先这里的已知量是(X, W, b, Y, dY = $\frac{\partial \mathrm{Loss}}{\partial y}$)，要求出 $d\mathrm{W}$ 和 $\mathrm{d} b$。其中

y (N, C)形如：
$$
\begin{bmatrix}
 y_{11}&   y_{12}&  ... &  y_{1c}\newline
 y_{21}&   y_{22}&  ... &  y_{2c}\newline
      & ...      &  ... & \newline
  y_{n1}&   y_{n2}& ... &  y_{nc}
\end{bmatrix}
$$
X (N, D)形如：
$$
\begin{bmatrix}
 x_{11}&   x_{12}&  ... &  x_{1d}\newline
 x_{21}&   x_{22}&  ... &  x_{2d}\newline
      & ...      &  ... & \newline
  x_{n1}&   x_{n2}& ... &  x_{nd}
\end{bmatrix}
$$
W (D, C)形如
$$
\begin{bmatrix}
w_{11}&   w_{12}&  ... &  w_{1c}\newline
w_{21}&   w_{22}&  ... &  w_{2c}\newline
     & ...      &  ... & \newline
 w_{d1}&   w_{d2}& ... &  w_{dc}
\end{bmatrix}
$$
b (1, C)形如：
$$
[b_1, b_2, ... , b_c]
$$
dY (N, C) 形如：
$$
\begin{bmatrix}
 \mathrm{d} y_{11}&   \mathrm{d} y_{12}&  ... &  \mathrm{d} y_{1c} \newline
 \mathrm{d} y_{21}&   \mathrm{d} y_{22}&  ... &  \mathrm{d} y_{2c}\newline
      & ...      &  ... & \newline
  \mathrm{d} y_{n1}&   \mathrm{d} y_{n2}& ... &  \mathrm{d} y_{nc}
\end{bmatrix}
$$
Y = X.dot(W) + b形如：
$$
\begin{bmatrix}
 y_{11}&   y_{12}&  ... &  y_{1c}\newline
 y_{21}&   y_{22}&  ... &  y_{2c}\newline
      & ...      &  ... & \newline
  y_{n1}&   y_{n2}& ... &  y_{nc}
\end{bmatrix} =
\begin{bmatrix}
 x_{11}&   x_{12}&  ... &  x_{1d}\newline
 x_{21}&   x_{22}&  ... &  x_{2d}\newline
      & ...      &  ... & \newline
  x_{n1}&   x_{n2}& ... &  x_{nd}
\end{bmatrix} *
\begin{bmatrix}
w_{11}&   w_{12}&  ... &  w_{1c}\newline
w_{21}&   w_{22}&  ... &  w_{2c}\newline
     & ...      &  ... & \newline
 w_{d1}&   w_{d2}& ... &  w_{dc}
\end{bmatrix} +
[b_1, b_2, ... , b_c]
$$

##### $\mathrm{d} b$的推导
首先求 $\mathrm{d} b$ 的第一项 $\mathrm{d} b_1$：
$$
\mathrm{d} b_1 = \frac{\partial \mathrm{Loss}}{\partial b_1} = \frac{\partial \mathrm{Loss}}{\partial y} \cdot \frac{\partial y}{\partial b_1} = \sum_i \sum_j \frac{\partial \mathrm{Loss}}{\partial y_{ij}} \cdot \frac{\partial y_{ij}}{\partial b_1}
$$
还记得视频中板书的那个公式么，在这里用到了。
将 $y_{ij}$ 展开：
$$
\begin{aligned}
y_{11} &= x_{11}w_{11} + x_{12}w_{21} + ... + x_{1d}w_{d1} + b_1 \newline
y_{12} &= x_{11}w_{12} + x_{12}w_{22} + ... + x_{1d}w_{d2} + b_2 \newline
&... ... \newline
y_{ij} &= x_{i1}w_{1j} + x_{i2}w_{2j} + ... + x_{id}w_{dj} + b_j
\end{aligned}
$$
由此可以看出，$b_1$仅与 $y_{i1}$ 有关，同样，$b_j$仅与 $y_{ij}$ 有关，并且 $\frac{\partial y_{ij}}{\partial b_j} = 1$ 那么：
$$
\mathrm{d} b_1 = \sum_i \sum_j \frac{\partial \mathrm{Loss}}{\partial y_{ij}} \cdot \frac{\partial y_{ij}}{\partial b_1} = \sum_{i = 1}^{N} \mathrm{d} y_{i1}
$$
就是将 $\mathrm{d} y$ 的第一列所有行相加。同理，$\mathrm{d} b_j$ 就是将 $\mathrm{d} y$ 第j列所有行相加。而 $\mathrm{d} y$ 为(N, C), 由此计算出的 $\mathrm{d} b_j$ 为 (1, C)，正好是b的shape。
```python
grads['b2'] = np.sum(Dscores, axis=0)
```
这里用 **维度分析** 更好解释，正因为 $\mathrm{d} y$ 为(N, C)，而b (1, C)只能与 $\mathrm{d} y$ 有关，所以只能沿着axis=0相加得到。详细的推导摆在这里，以后再遇到按 **维度分析** 的方式直接用就行了。

##### $\mathrm{d} W$的推导

同样以 $\mathrm{d} w_{11}$ 为例
$$
\mathrm{d} w_{11} = \frac{\partial \mathrm{Loss}}{\partial w_{11}} = \frac{\partial \mathrm{Loss}}{\partial y} \cdot \frac{\partial y}{\partial w_{11}} = \sum_i \sum_j \frac{\partial \mathrm{Loss}}{\partial y_{ij}} \cdot \frac{\partial y_{ij}}{\partial w_{11}}
$$
从y的展开式来看，$w_{11}$ 仅与 $y_{i1}$ 有关，而 $\frac{\partial y_{i1}}{\partial w_{11}} = x_{i1}$，所以：
$$
\mathrm{d} w_{11} = \sum_i \sum_j \frac{\partial \mathrm{Loss}}{\partial y_{ij}} \cdot \frac{\partial y_{ij}}{\partial w_{11}} = \sum_{i = 1}^{N} x_{i1} \cdot \mathrm{d} y_{i1}
$$
推广到一般，可得：
$$
\mathrm{d} w  = \begin{bmatrix}
\mathrm{d} w_{11}&   \mathrm{d} w_{12}&  ... &  \mathrm{d} w_{1c} \newline
\mathrm{d} w_{21}&   \mathrm{d} w_{22}&  ... &  \mathrm{d} w_{2c} \newline
     & ...      &  ... & \newline
 \mathrm{d} w_{d1}&   \mathrm{d} w_{d2}& ... &  \mathrm{d} w_{dc}
\end{bmatrix} = \begin{bmatrix}
 x_{11}&   x_{12}&  ... &  x_{1d}\newline
 x_{21}&   x_{22}&  ... &  x_{2d}\newline
      & ...      &  ... & \newline
  x_{n1}&   x_{n2}& ... &  x_{nd}
\end{bmatrix}^{T} * \begin{bmatrix}
 \mathrm{d} y_{11}&   \mathrm{d} y_{12}&  ... &  \mathrm{d} y_{1c} \newline
 \mathrm{d} y_{21}&   \mathrm{d} y_{22}&  ... &  \mathrm{d} y_{2c}\newline
      & ...      &  ... & \newline
  \mathrm{d} y_{n1}&   \mathrm{d} y_{n2}& ... &  \mathrm{d} y_{nc}
\end{bmatrix}
$$
再用 **维度分析** 解释一下，X形如(N, D)，dY形如(N, C)， $\mathrm{d} W$与W相同形如(D, C)，所以 $\mathrm{d} W = X.T.dot(\mathrm{d} Y)$，与公式推导得到的结果一致。
```python
grads['W2'] = relu_out.T.dot(Dscores) + 2 * reg * W2
```

##### $\mathrm{d} X$的推导

既然写了，就把它写全吧。
$$
\mathrm{d} x_{11} = \frac{\partial \mathrm{Loss}}{\partial x_{11}} = \frac{\partial \mathrm{Loss}}{\partial y} \cdot \frac{\partial y}{\partial x_{11}} = \sum_i \sum_j \frac{\partial \mathrm{Loss}}{\partial y_{ij}} \cdot \frac{\partial y_{ij}}{\partial x_{11}}
$$
而 $x_{11}$ 仅与 $y_{1j}$ 有关，且 $\frac{\partial y_{1j}}{\partial x_{11}} = w_{1j}$，所以
$$
\mathrm{d} x_{11} = \sum_i \sum_j \frac{\partial \mathrm{Loss}}{\partial y_{ij}} \cdot \frac{\partial y_{ij}}{\partial x_{11}} = \sum_{j = 1}^{C} w_{1j} \cdot \mathrm{d} y_{1j}
$$
推广到一般：
$$
\mathrm{d} x  = \begin{bmatrix}
\mathrm{d} x_{11}&   \mathrm{d} x_{12}&  ... &  \mathrm{d} x_{1c} \newline
\mathrm{d} x_{21}&   \mathrm{d} x_{22}&  ... &  \mathrm{d} x_{2c} \newline
     & ...      &  ... & \newline
 \mathrm{d} x_{d1}&   \mathrm{d} x_{d2}& ... &  \mathrm{d} x_{dc}
\end{bmatrix} = \begin{bmatrix}
 \mathrm{d} y_{11}&   \mathrm{d} y_{12}&  ... &  \mathrm{d} y_{1c} \newline
 \mathrm{d} y_{21}&   \mathrm{d} y_{22}&  ... &  \mathrm{d} y_{2c}\newline
      & ...      &  ... & \newline
  \mathrm{d} y_{n1}&   \mathrm{d} y_{n2}& ... &  \mathrm{d} y_{nc}
\end{bmatrix} * \begin{bmatrix}
w_{11}&   w_{12}&  ... &  w_{1c}\newline
w_{21}&   w_{22}&  ... &  w_{2c}\newline
     & ...      &  ... & \newline
 w_{d1}&   w_{d2}& ... &  w_{dc}
\end{bmatrix}^{T}
$$
用 **维度分析** 解释一下，W形如(D, C)，X形如(N, D)，dY形如(N, C)， $\mathrm{d} X$与X相同形如(N, D)，所以 $\mathrm{d} X = \mathrm{d} Y.dot(W.T)$，与公式推导得到的结果一致。
```python
Drelu_out = Dscores.dot(W2.T)
```
#### 一些小的细节

##### ReLU梯度的计算
```python
Dlayer1_out = Drelu_out * (layer1_out > 0)
```
##### grad_check函数
```python
for param_name in grads:
    f = lambda W: net.loss(X, y, reg=0.05)[0]
    param_grad_num = eval_numerical_gradient(f, net.params[param_name], verbose=False)
```
这里注意lambda函数最后的那个 **[0]**
```python
    f = lambda W: net.loss(X, y, reg=0.05)[0]
```
因为net.loss函数的返回值是两个：
```python
def loss(self, X, y=None, reg=0.0):
  ...
  return loss, grads
```
这个[0]表示在计算f(x)的时候，只考虑返回值的第一个，即loss。

> Inline Question
>
> Now that you have trained a Neural Network classifier, you may find that your testing accuracy is much lower than the training accuracy. In what ways can we decrease this gap? Select all that apply.
>
>Train on a larger dataset.
>
>Add more hidden units.
>
>Increase the regularization strength.
>
> None of the above.

增加dataset通常来讲可以；增加hidden可能会行，但不一定，因为反而会更加overfit；增大reg strength也可以减小overfit。上述所有措施都是可能，但不能保证一定行。









end
