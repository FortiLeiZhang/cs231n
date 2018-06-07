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
![Learning rate](http://cs231n.github.io/assets/nn3/learningrates.jpeg)

黄色：loss飞了，lr太大

蓝色：线性下降，lr太小

红色：刚好

绿色：开始下降太陡，然后不再下降，lr太大

###### batch size
![batch size](http://cs231n.github.io/assets/nn3/loss.jpeg)

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

### 参数更新
这里罗列了几种SGD参数更新的方法，具体在作业里讲，仅把pytorch函数记录在此。
###### SGD
```python
optimizer = torch.optim.SGD(params, lr=0.1, momentum=0, dampening=0, weight_decay=0, nesterov=False)
```
###### Momentum SGD
```python
optimizer = torch.optim.SGD(params, lr=0.1, momentum=0.9, dampening=0, weight_decay=0, nesterov=False)
```
###### Nesterov Momentum SGD
```python
optimizer = torch.optim.SGD(params, lr=0.1, momentum=0.9, dampening=0, weight_decay=0, nesterov=True)
```
以上几种方法的learning rate是不变的，或者是随固定节奏decay的，下面几种方法learning rate随训练的进行而自动进行调整，即learning rate是参数W或者dW的函数：
###### Adagrad
```python
optimizer = torch.optim.Adagrad(params, lr=0.01, lr_decay=0, weight_decay=0, initial_accumulator_value=0)
```
###### RMSprop
```python
optimizer = torch.optim.RMSprop(params, lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
```
###### Adam
```python
optimizer = torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
```

目前常用的是Adam，但是也可以试试SGD+Nesterov。

### 超参数的优化
优化超参数是训练当中花费时间最长的一步。这里只是提了几点指导性的方法：
1. 选择一个合适大小的val set，在这个set上做validation，不要做many folds cross-validation。
2. lr和reg的范围一般用指数形式：
```python
learning_rate = 10 ** np.uniform(-6, 1)
```
3. 用random search，不要用grid search
4. 如果最优值落到了边界上，考虑扩大边界
5. 先选择一个大的范围，每次test仅做几个epoch，然后逐步缩小范围，精调。

### Model Ensembles
1. 同一模型，不同初始化参数
2. 最好的几个模型
3. 同一模型在不同时间点上的训练参数
4. 对不同时间点上的模型参数取平均值

其实这一节看看summary里的点就行了。
