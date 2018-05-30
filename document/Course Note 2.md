[Linear classification: Support Vector Machine, Softmax](http://cs231n.github.io/linear-classify/)

Linear Classification
---
实现image classification更常用的方法是采用score function + loss function。其中，score function将raw data映射为class score；而loss function衡量所得到的的class score与已知的ground truth差别有多大。所有的NN和CNN问题实际上都是围绕这两个function进行的。

###Score function
最容易想到的就是linear function。假设有张32\*32\*3的图片,属于10个类别其中之一，那么将每一个像素点视为[0, 255]之间的一个数字，得到(32, 32, 3)的一个tensor，然后flatten为32*32*3=3072维的vector X;那么系数矩阵W应为(3072, 10)，X与W矩阵相乘：(1, 3072)\*(3072, 10)=(1, 10)，再加上一个bias项(1, 10)，得到的结果就是X分别属于10个类别中任一个的score。这个想法虽然看上去很简单很原始，但实际上所有基于NN的图像识别最底层的原理都是基于此。这里我有一些改进的想法，放在另外一个project中。

bias trick：将W扩一列，x扩一行恒为1，则可以将bias项统一到W中

数据预处理：所有数据都减去均值。
###Loss function
####Multiclass SVM
如果correct比incorrect的score大出至少一个margin，那么loss为0；否则loss为两者之差加margin，公式如下：
$$L_i = \sum_{j\neq y_i}\max(0, s_j - s_{y_i} + \Delta)，$$
这里 $\Delta$通常设置为1.0，不用纠结，大小没影响。公式很简单，但是视频里问的几个问题比较好，Note里没有，写在这里：
> 1.  What happens to loss if car scores change a bit?

结合上下文，这里想问的是如果correct的score已经足够大到loss已经为0，那么将correct的score再提高一点，对loss有什么影响？显然是没有影响，loss还是0。
> 2. what is the min/max possible loss?

min为0，max为 $+\infty$

> 3. At initialization W is small so all s ≈ 0. What is the loss?

对W进行初始化时，此时的loss应该为C-1，C为目标class的数目。W初始化时，所有系数都是随机的，此时s都近似为0，所以每个class类别的loss都为1，减去正确class的loss，所以为C-1。这个结论很重要，在后面NN和CNN初始化时，我们得到的第一个loss应该为C-1，如果偏差很大的话，说明要么初始化有错，要么network有错。

> 4. What if the sum was over all classes? (including j = y_i)

多加个1

> 5. What if we used mean instead of sum?

没影响

> 6. What if we used
$$L_i = \sum_{j\neq y_i}\max(0, s_j - s_{y_i} + \Delta)^2$$

有影响，平方项对偏差大的项penalty更大。
####Regularization
正则项对系数进行惩罚，使得W的系数尽量小，模型尽量趋向简单，从而避免overfitting问题。除了regularization之外，常用方法还有dropout，BN，stochastic depth，fractional pooling等等。

对于L1/L2正则的选择，L1产生sparse的W，即一项很大，其他尽量为0；L2产生spread的W，即所有项大小都尽量平均。记下这两个例子就行了：L1 [1 0 0 0]；L2 [0.25 0.25 0.25 0.25]。

此处有作业
---
[Assignment 1: SVM](https://github.com/FortiLeiZhang/cs231n/blob/master/code/cs231n/assignment1/svm.ipynb)

作业这里细说说这个svm_loss_vectorized是怎么做的
```python
def svm_loss_vectorized(W, X, y, reg):
    loss, grad = None, None
    N = X.shape[0]
    C = W.shape[1]

    scores = X.dot(W)
    correct_score = scores[np.arange(N), y]

    scores = np.maximum(0, scores - correct_score[:, np.newaxis] + 1.0)
    scores[np.arange(N), y] = 0

    loss = np.sum(scores) / N + reg * np.sum(W * W)

    return loss, grad
```
首先看这一行
```python
correct_score = scores[np.arange(N), y]
```
要从得到的scores(500, 10)中找出正确class的score是多少，这里用到的是numpy里的[Advanced Indexing](https://docs.scipy.org/doc/numpy-1.13.0/reference/arrays.indexing.html)，这一部分极其复杂，到现在每次用到的时候我也是极其头疼。看这个例子，scores用的index，rank 1是np.arange(N)，rank 2是y。这里N=500，所以np.arange(N)和y都是(500, )的vector，y的每一个值在[0, 9]之间。rank 1的500个数和rank 2的500个数两两配对，组成了500个tuple:(1, y1), (2, y2), ..., 其中y1, y2表示y的第一项、第二项的值，所以得到的矩阵是从scores的第一行取y1列的值，第二行取y2列的值，直到第500行。得到的correct_score是(500,)的vector。这里要强调的是rank 1和rank 2 index的维度一定要相等，这样才能两两配对组成一个index tuple。

再来看这一行
```python
scores = np.maximum(0, scores - correct_score[:, np.newaxis] + 1.0)
```
这里用到的是numpy里的[broadcasting](https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)的概念。在学习矩阵论的时候习惯了检查维度匹配，这里突然来个broadcasting给你自动扩展匹配维度，还真的是要适应一阵。具体看这个例子：

scores是(500, 10)，correct_score是(500,)，如果这里你用两者直接相减的话，python会报错提示维度不匹配。这里用np.newaxis给correct_score增加了一个rank，现在correct_score由(500, )变成了(500, 1)，现在两者再相减的话，numpy会自动将correct_score扩展为(500, 10)，多出来的数据是在axis=1上将数据复制10份，正是我们想要的。broadcasting这个概念很重要，在以后的编程中我们会遇到无数次。

SVM的作业先写到这里，还有计算grad和SGD，等视频讲到了再回来补上。
