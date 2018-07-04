[Assignment3 | RNN Captioning](https://github.com/FortiLeiZhang/cs231n/blob/master/code/cs231n/assignment3/RNN_Captioning.ipynb)
---
这部分实际上做了两件事情，首先建立一个 RNN，然后以此 RNN 为基础，训练一个模型来完成图片 caption 的工作。我感觉作业中的代码先后顺序有些混乱，这里依照自己的理解，把内容重新组织一下。


### Dataset
train 和 val 使用的是 Coco2014，从打印出来的 data 信息来大概浏览一下数据的构成。
```python
train_captions <class 'numpy.ndarray'> (400135, 17) int32
train_image_idxs <class 'numpy.ndarray'> (400135,) int32
train_features <class 'numpy.ndarray'> (82783, 512) float32
idx_to_word <class 'list'> 1004
word_to_idx <class 'dict'> 1004
train_urls <class 'numpy.ndarray'> (82783,) <U63
```
train dataset 中有 82783 张图片，每一张图片对应多个 caption，共有 400135 个caption，每一个 caption 最多包含 17 个整形数字，每一个数字通过 idx_to_word 对应到一个单词。idx_to_word 是一个 list，每一个位置上对应一个单词，其中位置0-3分别是特殊字符 \<NULL>, \<START>，\<END>，\<UNK>。所有的 caption 都是以 \<START> 起，以 \<END> 止，如果不足17个单词，那么在 \<END> 以后补 \<NULL>，不在 idx_to_word 中的记为 \<UNK>。
```python
print(data['train_captions'][1])
print(decode_captions(data['train_captions'][1], data['idx_to_word']))
[  1   4   3 172   6   4  62  10 317   6 114 612   2   0   0   0   0]
<START> a <UNK> view of a kitchen and all of its appliances <END>
```
train feature 是直接取自 VGG16 的第 fc7 层，也就是从 4096 映射到 1000 个 class 的前一层，所以是 4096 维的。这里为了减少计算量，使用 PCA 将维度减小到 512 维。

### Vanilla RNN
![CNN_time_cap](https://github.com/FortiLeiZhang/cs231n/raw/master/images/CNN_time_capsule.jpg)
RNN 一次处理一个长度为 T 的时间序列 x(0), x(t), ... , x(T)，其中，隐状态 h(t) 由该时刻输入 x(t) 和上一时刻隐状态 h(t-1) 共同决定：
$$
h(t) = \mathrm{tg} \left ( W_x \cdot x(t) + W_h \cdot h(t-1) + b_h \right )
$$
输出的 score 由该时刻的隐状态 h(t) 决定：
$$
s(t) = W \cdot h(t) + b
$$
得到的 score 与该时刻的 y(t) 经过 softmax 后得到该时刻的 Loss。
这里需要学习的参数是 $W_x$，$W_h$，$b_h$，$W$ 和 $b$。需要注意的是，在 RNN 处理该长度为 T 的时间序列 x(t) 时，上述参数是 **不变的**。只有处理完该序列后，才会进行参数的更新，并且在进行参数更新时， RNN 的 loss 是在所有时刻 loss 之和，即：
$$
Loss = L(0) + ... + L(t) + ... + L(T)
$$

作业中将 RNN 中的隐状态转移和 loss 计算分成了两部分，首先按时序先将每一时刻的隐状态表示出来，然后在所有隐状态已知的情况下，将各个时刻的 loss 一步求出。这样做在计算 backprop 时要特别注意某一时刻隐状态的梯度是由（此时刻 loss 的梯度 + 下一隐状态的梯度）两部分组成。而所有要更新的参数 $W_x$，$W_h$，$b_h$，$W$ 和 $b$，其所对应的梯度值，要将各个时刻计算得到的值全部叠加起来。

#### RNN: hidden state transition
##### forward
forward 的计算是根据公式：
$$
h(t) = \mathrm{tg} \left ( W_x \cdot x(t) + W_h \cdot h(t-1) + b_h \right )
$$
原作业中用的名称是 next_h，我认为不妥，应该是根据 prev_h 来计算出 current_h。

##### backward
用维度分析法来推导，这里就不赘述了，要注意的是 $W_h$ 是形如 (H, H) 的，不要忘记也要做转置。

#### RNN: affine layer
这一步是将隐状态映射为 score，根据公式：
$$
s(t) = W \cdot h(t) + b
$$
需要注意的是，这里是在已知所有隐状态的前提下，计算在各个时刻的 score，所以，对于某一时刻而言，h(t) 是形如 (N, H)，而一个 time capsule 中所有隐状态 h 是形如(N, T, H)的。这里代码没有什么可说的，将输入 flatten 一下就好。

#### RNN: softmax layer
与先前 softmax 的计算没什么大的差别，要注意的是由于输出的 caption 的长度不相同的，所以标记为 \<NULL> 的单词不计算入总的 loss。因此要引入一个 mask。

#### RNN: single time capsule
##### forward
forward 没什么好说的，在所有时刻一步步调用 rnn_step_forward 即可。

##### backward
backward 要强调的是它的输入参数，原作业中用的是 dh，这里为了更清楚，记为 dLossdh，它是形如 (N, T, H) 的，表示从某一时刻的 loss 传递到该时刻隐状态的梯度值；而此时刻隐状态的梯度值还包括另外一部分，即从后一时刻隐状态传递到该时刻隐状态的梯度值，即：
```python
dcurrent_h = dLossdh[:, i, :] + dprev_h
```
backprop 从最后一个时刻 h(T) 开始，往 h(0) 计算，而最后一个隐状态没有下一时刻，所以 dprev_h 的初始值应该为 0 ：
```python
dprev_h = np.zeros_like(prev_h)
```
另外需要注意的是，在 RNN 的一个 time capsule 中，所有参数都是 **不进行更新的**，而每一个时刻对参数都计算一个梯度，最后要将每一时刻的这些梯度都加起来：
```python
dx[:, i, :], dprev_h, dWx_, dWh_, dbh_ = rnn_step_backward(dcurrent_h, cache)
dWx += dWx_
dWh += dWh_
dbh += dbh_
```

#### RNN: full rnn step
顺带又加了一种实现方式，即在每一时刻，不但计算出下一时刻的隐状态，同时计算出该时刻的 loss。

### Word Embedding
word embedding 的作用实际上是一个空间的映射，它将用 int 型数字编码的单词，映射到一个 D 维度的 float 型数字编码的空间。具体来说，将一个单词用一个整型数字来表示，所有的单词组成了一个词汇表，这个词汇表的词汇量大小是 V，词汇表中每一个单词的编码值应该在 [0, V) 范围内。然后定义一个形如 (V, D) 的映射关系 W。W 中的每一行是一个 D 维的 float 向量，对应一个单词在词汇表中的 offset。这样，就将一个单词由一个整型数字表示映射到一个 D 维的 float 向量。

举个例子，单词 cat 在词汇表中的 offset 为 10，在 W 中，第10行的向量是 $[0.21428571,  0.28571429,  0.35714286]$，那么单词 cat 就映射成了一个三维的向量 $[0.21428571,  0.28571429,  0.35714286]$。

在这里，单词在词汇表中的 offset 是固定的，而映射关系 W 是通过学习而来的参数。

#### forward
```python
N, T = x.shape
V, D = W.shape
out = np.zeros((N, T, D))
for n in range(N):
    for t in range(T):
        out[n, t, :] = W[x[n, t], :]
```
以上是 naive 的实现，如果用 python 的 indexing 的话，一行就够了
```python
out = W[x, :]
```

#### backward
```python
N, T = x.shape
V, D = W.shape
dW = np.zeros_like(W)

for v in range(V):
    for n in range(N):
        for t in range(T):
            if x[n, t] == v:
                dW[x[n, t], :] += dout[n, t, :]
```
同样，先写 naive 形式，用 np.add.at 的话
```python
dW = np.zeros_like(W)
np.add.at(dW, x, dout)
```
np.add.at 在 numpy 的教程里解释的也不多，这里究竟是如何 indexing 的，对照 naive 方法自己体会吧。

### RNN for image captioning
训练一个 RNN 来做 image caption 需要输入图片的 feature 和 caption，并用同样的 caption 做 label。整个过程包含三个学习过程，需要训练三组参数。分别为：1.图片的 feature 向 h(0) 的 projection；2. RNN；3. 单词的 projection。

#### 初始化
##### 图片 feature 向 h(0) 的 projection

输入的图片 feature 是从 VGG 的 FC7 层截取的，原始值是 4096 维的，为了减小计算量，通过 PCA 降维到 512 维，而 RNN 的 h(0) 是 H 维的。从形如 (N, D) 的 feature 映射到 (N, H) 的隐状态，需要一组形如 (D, H) 的参数
```python
self.params['W_feature'] = np.random.randn((input_dim, hidden_dim)) / np.sqrt(input_dim)
self.params['b_feature'] = np.zeros(hidden_dim, )
```

##### RNN
一个 RNN 需要两组参数，一组是隐状态之间的转换参数 $W_x$ 和 $W_h$；另一组是隐状态向 score 的转换参数 $W_s$。各个参数的形状已经在上面的图中标注的很清楚了。
```python
self.params['Wx'] = np.random.randn((wordvec_dim, hidden_dim)) / np.sqrt(wordvec_dim)
self.params['Wh'] = np.random.randn((hidden_dim, hidden_dim)) / np.sqrt(hidden_dim)
self.params['bh'] = np.zeros(hidden_dim, )
self.params['Ws'] = np.random.randn((hidden_dim, vocab_size)) / np.sqrt(hidden_dim)
self.params['bs'] = np.zeros(vocab_size, )
```

##### Word Embedding
进行 word embedding 的参数形如 (V, W)
```python
self.params['W_word'] = np.random.randn((vocab_size, wordvec_dim)) / 100
```
这里要注意的是如何将同一个 caption 拆分成输入的 data 和 label，并且要注意一个 RNN time capsule 中的时序问题。

首先，Coco 的一个 caption 有 17 位，以 \<START> 始，以 \<END> 或者是 \<NULL> 止。这里，作为输入的 caption_in 是取 caption 的前16位 [0, T-1]，所以必定是以 \<START> 开始的，去掉了最后一个单词（无论是 \<END> 还是 \<NULL>）；而作为 label 的 caption_out 取后16位 [1, T]，去掉了开头的 \<START>。所以整个 RNN 所有时刻的输入输出对应关系是
![CNN_in_out_sync](https://github.com/FortiLeiZhang/cs231n/raw/master/images/CNN_in_out_sync.jpg)

需要注意的是，h(0) 作为初始状态，是不参与到 caption 的输出的。




















end
