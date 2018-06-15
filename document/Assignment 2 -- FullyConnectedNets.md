[Assignment 2 | FullyConnectedNets](https://github.com/FortiLeiZhang/cs231n/blob/master/code/cs231n/assignment2/FullyConnectedNets.ipynb)
---

### [gradient check函数的不同实现](https://github.com/FortiLeiZhang/cs231n/blob/master/code/cs231n/assignment2/cs231n/gradient_check.py)
到目前为止用到了三个gradient check函数，分别是：grad_check_sparse，eval_numerical_gradient，eval_numerical_gradient_array。

###### eval_numerical_gradient
```python
def eval_numerical_gradient(f, x, h=0.00001)
```
```python
f = lambda W: net.loss(X, y, reg=0.05)[0]
param_grad_num = eval_numerical_gradient(f, net.params[param_name])
```
这里，f函数是net.loss，返回值是一个数字。

###### eval_numerical_gradient_array
```python
def eval_numerical_gradient_array(f, x, df, h=1e-5):
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:

        ix = it.multi_index
        oldval = x[ix]
        x[ix] = oldval + h
        fxph = f(x).copy()
        x[ix] = oldval - h
        fxmh = f(x).copy()
        x[ix] = oldval

        grad[ix] = np.sum((fxph - fxmh) * df)/ (2 * h)
        it.iternext()

    return grad
```
```python
dx_num = eval_numerical_gradient_array(lambda z: affine_forward(x, w, b)[0], x, dout)
```
这里f函数是affine_forward，返回值out是一个矩阵。所以，要对计算出函数值后， 要对其进行深拷贝：
```python
fxph = f(x).copy()
```
并且，每次循环算出的仅仅是对x中的一项进行的微分，其结果也是一个数字，并且
$$
\mathrm{d} x_{ij} = \frac{\partial \mathrm{L}}{\partial x_{ij}} = \sum_p \sum_q \frac{\partial \mathrm{L}}{\partial y_{pq}} \cdot \frac{\partial y_{pq}}{\partial x_{ij}}
$$
所以这里逐项点乘之后要对所有项求和：
```python
grad[ix] = np.sum((fxph - fxmh) * df)/ (2 * h)
```

###### grad_check_sparse
```python
def grad_check_sparse(f, x, analytic_grad, num_checks=10, h=1e-5)
```
这个函数仅仅是用eval_numerical_gradient后，对结果进行抽样检查。注意，这里f函数的返回值要求是一个数字。

> Inline Question 1:
>
>We've only asked you to implement ReLU, but there are a number of different activation functions that one could use in neural networks, each with its pros and cons. In particular, an issue commonly seen with activation functions is getting zero (or close to zero) gradient flow during backpropagation. Which of the following activation functions have this problem? If you consider these functions in the one dimensional case, what types of input would lead to this behaviour?
>1. Sigmoid
>2. ReLU
>3. Leaky ReLU

Sigmoid函数在输入值太大和太小时都会进入饱和区，在此区域grad为0；ReLU函数在输入小于0时，grad为0；Leaky ReLU则不会有grad为0的情况。

### [Solver函数的实现](https://github.com/FortiLeiZhang/cs231n/blob/master/code/cs231n/assignment2/cs231n/solver.py)
第一个要注意的地方是_rest函数为每一个需要更新的param初始化了一个config：
```python
self.optim_configs = {}
for p in self.model.params:
    d = {k: v for k, v in self.optim_config.items()}
    self.optim_configs[p] = d
```
这是因为：如果只采用Vanilla SGD的话，很简单，所有params共享一个learning rate参数。但是，如果采用其他的方法，每一个param在更新时，都有与自己以前状态相关的参数，如momentum，dx的滑动平均等等，所以要为每一个param维护一个属于自己的config。

> Inline Question 2:
>
>Did you notice anything about the comparative difficulty of training the three-layer net vs training the five layer net? In particular, based on your experience, which network seemed more sensitive to the initialization scale? Why do you think that is the case?

五层网络比三层网络难调的多，对weight_scale特别敏感，稍不小心loss就飞了。这是因为网络层数越多，越难以保持数据的方差，BN应该对解决此问题有效果。

> Inline Question 3:
>
> AdaGrad, like Adam, is a per-parameter optimization method that uses the following update rule:
```python
cache += dw**2
w += - learning_rate * dw / (np.sqrt(cache) + eps)
```
>John notices that when he was training a network with AdaGrad that the updates became very small, and that his network was learning slowly. Using your knowledge of the AdaGrad update rule, why do you think the updates would become very small? Would Adam have the same issue?

因为cache是dw平方的叠加，随着学习的深入，这个值会越来越大，而在w更新的过程中需要除以cache，这会导致实际上的learnig rate会越来越小。

### 训练一个好的FC网络
在完成 BN 和 Dropout 后，返回来完成最后一步：用上完成的所有方法来实现一个好的 FC 网络。

首先粗调 learning rate 和 weight scale ，这里先用一个大范围搜索
```python
lr = np.random.uniform(1e-5, 1e-1)
ws = np.random.uniform(1e-5, 1e-1)
```
随机100次，每次跑10个 epoch，最好的结果在
>  Epoch 10 / 10 - best_val_acc: 0.437000
>
>learning_rate is: 1.79e-05, weight_scale is: 7.45e-04

然后将范围缩小
```python
lr = np.random.uniform(1.5e-5, 2.0e-5)
ws = np.random.uniform(7e-4, 8e-4)
```
最好结果为
> Epoch 10 / 10 - best_val_acc: 0.458000
>
> learning_rate is: 1.91e-05, weight_scale is: 7.60e-04

固定 learning_rate = 1.91e-05，weight_scale = 7.60e-04，训练100个 epoch。将 loss_history，train_acc_history，val_acc_history 画出来看看。loss 的噪声太大，说明 batch size 有点小，loss 曲线貌似刚开始趋缓，可以再增加 epoch 看看。train 和 val 的差距开始变大，说明要 overfitting 了，考虑增加 reg。

reg 加到 1.5, epoch 加到 1000，batch size 加到 200，试一下。最后再把hidden dim 增大到100 试试。








end
