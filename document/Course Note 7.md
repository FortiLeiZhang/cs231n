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
#### learning rate
![Learning rate](http://cs231n.github.io/assets/nn3/learningrates.jpeg)

黄色：loss飞了，lr太大

蓝色：线性下降，lr太小

红色：刚好

绿色：开始下降太陡，然后不再下降，lr太大

#### batch size
![batch size](http://cs231n.github.io/assets/nn3/loss.jpeg)

噪声太多，说明batch size太小；曲线的趋势说明lr还可以

#### overfitting
train/val的差距太大，表明overfitting，需要加样本，加L2，加dropout等等。

train/val的差距太小，表明模型容量太小，表达能力不足，需要增加层数或者参数。

#### Ratio of weights:updates
updates与weights幅度的比例不要太小，1e-3左右比较合适，即：
$$
\frac{\left \| lr \cdot \mathrm{d} W\right \|^2}{\left \| W\right \|^2} \approx 1e-3
$$

#### 每一层的激活/梯度分布
画出每一层的激活/梯度分布的柱状图，应该符合预期的概率分布。

#### 第一层参数可视化
图像不应该有太多噪声点。















end
