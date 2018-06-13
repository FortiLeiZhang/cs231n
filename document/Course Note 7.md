[Neural Networks Part 3: Learning and Evaluation](http://cs231n.github.io/neural-networks-3/)
---
### 梯度检验
梯度的数值计算方法是：
$$
f^{'} = \frac{\mathrm{d} f(x)}{\mathrm{d} x} = \frac{f(x+h) - f(x-h)}{2h}
$$
梯度检验的方法是：
$$
rel\_err = \frac{|f_a^{'} - f_n^{'}|}{\max \left(|f_a^{'}| + |f_n^{'}|,  \epsilon \right)}
$$
```python
def rel_error(x, y):
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))
```
梯度检验是要注意的点：
1. rel_error应该比1e-7小，如果NN很深的话，因为误差是累积的，所以rel_error可能会大一点。
2. 梯度检验对精度很敏感，要用双精度类型，np.float64；另外，如果梯度数值很小的话，对梯度检验也是不利的，所以尽量使梯度的数值在1.0数量级上。
3. 在奇点处，函数的左右极限是不相等的，梯度检验在奇点处是不准确的。对于NN来说，可能引入奇点的只有relu或者SVM。对于relu或者SVM来说，检验函数是否过奇点很简单：(x+h)和(x-h)符号相反。虽然在NN中函数取值到奇点附近不是一个小概率事件，但是因为梯度检验并不是对所有点都进行，所以不是大问题。
4. 抽样检验。没必要检验所有data和所有参数的grad。仅仅抽样几个sample检验，每个sample检验几个参数就可以了。但是这里要注意的是coverage的问题，因为weights的个数比bias要多很多，所以如果两者合在一起抽样的话，有很大概率bias会检测不到。
5. h太小也会有问题，1e-5即可
6. 不要在训练开始时进行梯度检验，待稳定后再进行。
7. loss和reg分开检验。先令reg=0，检查loss的grad；然后在令loss=0，检查reg项的grad，或者增大reg，那么总的loss和grad都会增大。
8. 避免dropout的随机性，使用相同的random seed来进行dropout。

### Sanity Check
1. softmax最初的loss应该为 $\ln C$，SVM为 $C-1$。如果不是，那很大概率是初始化有问题。
2. reg增大，loss也会增加
3. 在正式训练之前，选取一个小的样本，将reg设为0，然后overfit这个小样本。

### 训练过程中的debug
###### learning rate
![Learning rate](../images/loss_1.jpg)

黄色：loss飞了，lr太大

蓝色：线性下降，lr太小

红色：刚好

绿色：开始下降太陡，然后不再下降，lr太大

###### batch size
![batch size](../images/loss_2.jpg)

噪声太多，说明batch size太小；曲线的趋势说明lr还可以

###### overfitting
train/val的差距太大，表明overfitting，需要加样本，加L2，加dropout等等。

train/val的差距太小，表明模型容量太小，表达能力不足，需要增加层数或者参数。

###### Ratio of weights:updates
updates与weights幅度的比例不要太小，1e-3左右比较合适，即：
$$
\frac{\left \| lr \cdot \mathrm{d} W\right \|^2}{\left \| W\right \|^2} \approx 1e-3
$$

###### 每一层的激活/梯度分布
画出每一层的激活/梯度分布的柱状图，应该符合预期的概率分布。

###### 第一层参数可视化
图像不应该有太多噪声点。

### 超参数的优化
优化超参数是训练当中花费时间最长的一步。这里只是提了几点指导性的方法：
1. 选择一个合适大小的val set，在这个set上做validation，不要做many folds cross-validation。
2. lr和reg的范围一般用指数形式：
```python
learning_rate = 10 ** np.uniform(-6, 1)
```
3. 用random search，不要用grid search
4. 如果最优值落到了边界上，考虑扩大边界
5. 先选择一个大的范围，lr大概在1e-3到1e-5之间，每次test仅做几个epoch，然后逐步缩小范围，精调。

### 参数更新
Course note里写的比较简单，重要的内容都在视频中。
这里罗列了几种SGD参数更新的方法，具体在作业里讲，仅把pytorch函数记录在此。
###### Vanilla SGD
```python
while True:
  dx = computer_gradient(x)
  x += - learning_rate * dx
```
Vanilla SGD实现很简单，但在实际应用中有很多问题：

1. 如果函数在一个维度下降很快(陡)，而在另外一个维度下降很慢(缓)。在陡的维度，函数变化很大，在缓的维度，函数变化很小，所以函数虽然会持续向最小值收敛，但收敛曲线会像在两堵墙之间来回反弹一样zig-zag前进，效率很低。因为要学习的函数通常是几千维的，所以这种情况几乎是肯定发生的。
2. 会卡在局部极小值点或者鞍点，这才是最大的问题。1和2其实是同一个问题，想象一下1的极限情况就是在鞍点处，曲线在一个方向几乎不动，而在其他方向会像打乒乓一样来回反弹。
3. 因为是stochastic的，每次抽样的mini-batch会引入noise，vanilla SGD对这种误差敏感，造成收敛曲线抖动。

###### Momentum SGD
```python
vx = 0
while True:
  dx = computer_gradient(x)
  vx = rho * vx + dx   # running mean of gradients, rho = 0.9 or 0.99
  x += - learning_rate * dx
```
在Vanilla SGD的基础上加入动量(momentum)的概念，这与股票研究中的momentum是一个道理。

1. 由于momentum的存在，在鞍点处，即使某一方向dx等于0，vx仍然不为0，参数依然会更新，收敛曲线有概率冲过鞍点。
2. 同样，由于momentum的存在，在缓的方向收敛速度会增大，减少zig-zag的频率。
3. 会减少noise对收敛的影响。

注意：理论上讲，Vanilla SGD的问题在Momentum SGD中依然会存在，但是因为momentum的引入，会大大缓解。实际上，即使是下面介绍的更先进的方法依然不会完全避免上述问题，但是会使问题出现的概率大大的降低。
###### Nesterov Momentum SGD
```python
old_v = 0
v = 0
while True:
  dx = computer_gradient(x)
  old_v = v
  v = rho * v - learning_rate * dx
  x += - rho * old_v + (1 + rho) * v
```
Momentum SGD的另一个变种。特征在于先对velocity进行更新，然后再做参数更新。

上面三个SGD极其变种仅仅引入了velocity的概念，下面几种参数更新的方法把dx的滑动均值也纳入更新函数中。
###### Adagrad
```python
grad_squared = 0
while True:
  dx = computer_gradient(x)
  grad_squared += dx * dx
  x += - learning_rate * dx / (np.sqrt(grad_squared) + eps)
```
Adagrad的作用是，当dx大时，参数更新时会除以一个较大的数；而dx小时，相应的除以一个较小的数。从而平衡在各个方向上的收敛速度。但是learning rate会随着学习的深入越来越小。

###### RMSprop
```python
grad_squared = 0
while True:
  dx = computer_gradient(x)
  grad_squared = decay_rate * grad_squared + (1 - grad_squared) * dx * dx
  x += - learning_rate * dx / (np.sqrt(grad_squared) + eps)
```
RMSprop引入了一个decay来缓解Adagrad中learning rate随学习深入而减小的问题。

###### Adam
```python
first_moment = 0
second_moment = 0
for t in range(1, num_iterations):
  dx = computer_gradient(x)
  first_moment = beta1 * first_moment + (1 - beta1) * dx
  second_moment = beta2 * second_moment + (1 - beta2) * dx * dx
  first_unbias = first_moment / (1 - beta1 ** t)
  second_unbias = second_moment / (1 - beta2 ** t)
  x += - learning_rate * first_unbias / (np.sqrt(second_unbias) + eps)
```
Adam是将momentum和dx的滑动均值统统考虑进来。注意，这里加入一个unbias项是因为：如果没有unbias项，那么在训练开始的时候，second_moment为0，此时更新x需要除以一个很小的数，导致learning rate会很大。

Best practise：Adam ( beta1 = 0.9，beta2 = 0.999，learning_rate = 1e-3 )

###### Learning rate decay
随着学习的深入，逐步减小learning rate，通常与SGD一起用，Adam事实上本身已经实现了learnig rate decay。实践中，训练开始并不设置learnig rate decay，训练几个epoch后，如果loss不再降低，就要考虑加入learning rate decay。

###### Second-order optimization
使用first-oder和second-oder梯度，但是需要算Hessian矩阵，太复杂很少用。

### Model Ensembles
1. 同一模型，不同初始化参数
2. 最好的几个模型
3. 同一模型在不同时间点上的训练参数
4. 对不同时间点上的模型参数取平均值
