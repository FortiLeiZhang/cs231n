[Assignment3 | LSTM Captioning](https://github.com/FortiLeiZhang/cs231n/blob/master/code/cs231n/assignment3/LSTM_Captioning.ipynb)
---
上一节介绍了 Vanilla RNN，同 NN 一样，RNN 存在的最大的问题就是梯度消失。这是因为，RNN 中需要学习的参数在一个 time capsule 结束之后才会更新一次，而在同一个 time capsule 中，所有的时刻都会有 loss 产生，因此也都会有对于参数的梯度产生，最后参数更新时的梯度是所有这些梯度之和。梯度的计算式为：
$$
\frac{\partial L}{\partial W} = \sum_{t=1}^{T} \frac{\partial L^{(t)}}{\partial W} = \sum_{t=1}^{T} \sum_{k=0}^{t} \frac{\partial L^{(t)}}{\partial o^{(t)}} \frac{\partial o^{(t)}}{\partial h^{(t)}} \left( \prod_{j=k+1}^{t} \frac{\partial h^{(j)}}{\partial h^{(j-1)}} \right) \frac{\partial h^{(k)}}{\partial W}
$$
通常使用 sigmoid 或者是 tanh 作为激活函数，所以上式中的连乘项可以写成：
$$
\prod_{j=k+1}^{t} \frac{\partial h^{(j)}}{\partial h^{(j-1)}} = \prod_{j=k+1}^{t} \mathrm{tanh}^{'} \cdot W^{t-j}
$$
或者
$$
\prod_{j=k+1}^{t} \frac{\partial h^{(j)}}{\partial h^{(j-1)}} = \prod_{j=k+1}^{t} \mathrm{sigmoid}^{'} \cdot W^{t-j}
$$
因为$\tanh(x) \in (-1, 1)$, 所以 $\tanh'(x) = 1 - \tanh^2(x) \in (0, 1]$;

因为$\sigma(x) \in (0, 1)$, 所以 $\sigma'(x) = \sigma(x) \cdot (1 - \sigma(x)) \in (0, 0.25]$。

所以上述连乘一个小于1的数时，最终会造成梯度的消失。特别是 sigmoid 函数作为激活函数时。

那么，可不可以用 ReLU 函数作激活函数呢？其实是可以的，而且也有很多人在用 ReLU 作 RNN 的激活函数，但是 ReLU 也有自身的缺点。
ReLU 函数的导数要么是 0，要么是 1,。如果是0的话，因为是连乘，所以该时刻以及之前的梯度就一直为 0 了，那么神经元就死掉了。如果 ReLU 的导数在取 1 的范围内的话，连乘中包含一项 W, 在 RNN 的一个 time capsule 中，W 是不变的，只要 W 中有大于 1 的特征值，那么这个连乘就会导致一个数值很大的矩阵；这不像在 CNN 中，随梯度向后传递的 W 是独立的，各个 W 的特征值不会同时大于1或者小于1，所以很大程度上会抵消梯度爆炸的结果。
上述两点实际上都有办法解决。真正选择使用 LSTM 而不用 ReLU 的原因并不是因为梯度消失的问题，而是 LSTM 可以引入长期的记忆，即可以学习长距离的依存关系。

### LSTM
由于梯度消失，Vanilla RNN 仅有短期记忆，而 LSTM 通过 memory cell 控制引入了长期记忆。
![LSTM](https://github.com/FortiLeiZhang/cs231n/raw/master/images/lstm.jpg)

如图所示，与传统的 RNN 相比，LSTM 除了包含原有的 hidden state 以外，还增加了随时间更新的 memory cell。某一时刻的 cell 与 hidden state 有着相同的形状，两者相互依赖于彼此进行状态的更新。具体来看，需要学习的参数 $W_x$ 和 $W_h$ 由 RNN 中的形如 (W, H) 和 (H, H) 变成了 (W, 4H) 和 (H, 4H)，即 (W, f+i+g+o) 和 (H, f+i+g+o)，而 $h(t-1) \cdot W_h + x(t) \cdot W_x$ 的结果也成为形如 (N, f+i+g+o)，其中 f/i/g 用来更新 cell 的状态，得到的新的 cell 状态 C(t) 与 o 一起来更新 h(t)。

##### forget gate
forget gate 是经过一个 sigmoid 函数得到的，所以它的范围在 (0, 1) 之间，并与 C(t-1) 相乘。它的作用是从前一个 cell 中取多少内容放到新的 cell 中去。这里叫 forget gate，我觉着还不如叫 remember gate 合适。

##### gate gate
gate gate 是把新的变换结果经过一个 tanh 函数得到的，它实际上是产生了一组新的记忆。

##### input gate
input gate 是经过一个 sigmoid 函数得到的，所以它的范围在 (0, 1) 之间，并与 gate gate 产生的新的记忆相乘。它的作用是决定新的记忆中有多少内容会被写到新的 memory cell 中去。

新的 memory cell 中的内容由先前cell，forget gate，gate gate 和 input gate 共同决定，即
$$
C(t) = C(t-1) \cdot f + i \cdot g
$$

##### output gate
output gate 经过 sigmoid 函数，与经过 tanh 函数的新的记忆 cell 相乘，得到的记过即为新的 hidden state。即：
$$
h(t) = o \cdot \tanh(C(t))
$$





















end
