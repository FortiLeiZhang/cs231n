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
