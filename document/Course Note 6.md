[Neural Networks Part 2: Setting up the Data and the Loss](http://cs231n.github.io/neural-networks-2/)
---
### 数据预处理
常见的数据预处理方法包括：
1. 减均值
2. 除标准差：只有在feature具有相似的标准和度量方式时才有意义
3. PCA：先减均值，再除标准差，然后找所有eigenvalue，最后将原始数据映射到最大N个eigenvalue定义的空间上
4. 白化：将数据映射到eigenvalue定义的空间内，然后再除以eigenvalue

1常用，2视情况而定，3和4不用。

另外需要注意的是，在进行数据预处理的时候，均值、方差仅仅由training数据得到，然后将此均值、方差应用到val/test数据上。如果对所有数据取均值、方差后，再划分train/val/test集，是 **错误的**。

### 权重初始化
1. 不可以全部初始化为0
2. 0均值高斯随机初始化。这里需要注意的是，对于很深的NN，方差会随着layer的深入而减小，直到均值为0，方差也为0，这样造成的问题是backprop回来的grads太小，造成梯度消失。
3. 经过linear function后输出方差的大小正比于输入数据的个数，因此如果输入数据，即mini-batch个数很多的话，输出方差会很大，常用方法是除以 $\sqrt{n}$。这里要注意的是与上一点的区别，这里是为了保证，经过linear function后的输出方差大小近似，而不是正比于输入的数据个数。

目前常用的方法是Kaiming法，将随机方差设为 $\sqrt{2.0/n}$，即：
```python
w = np.random.randn(n) * sqrt(2.0/n)
```
4. 稀疏初始化：仅将很少数目的权重做高斯初始化，其余均设为0
5. bias全部初始化为0
6. batch normalization：作业里会细讲，这里先略过

### 正则化
1. L1/L2 regularization：前面已经提过两次了：L1是sparse，L2是spread
> L2: encouraging the network to use all of its inputs a little rather than some of its inputs a lot.

> L1: leads the weight vectors to become sparse during optimization (i.e. very close to exactly zero).

2. Dropout：作业里细讲
3. 人为的引入stochasticity，例如dropout，data augmentation。

常用的方法是L2 + dropout。

### 代价函数
不同的问题有不同的代价函数，对于classfication而言，常用的代价函数是SVM和softmax。其他问题的代价函数也不是几句话就能总结清楚的，以后用到再说。
