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






















end
