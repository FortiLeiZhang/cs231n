[Assignment 2 | Dropout](https://github.com/FortiLeiZhang/cs231n/blob/master/code/cs231n/assignment2/Dropout.ipynb)
---
Dropout 就是在 train 的过程中随机的丢掉一部分连接，但是在 test 的时候使用全部的连接。这样做的好处是在 train 的时候引入一些随机性，在 test 的时候相当于用这些随机性进行了 ensemble。起到了 reg 的作用。

>Inline Question 1:
>
>What happens if we do not divide the values being passed through inverse dropout by p in the dropout layer? Why does that happen?

Notes 里讲到了这个问题，如果不在forward的时候 /p , 会导致输出的均值变为输入均值的 p 倍，而dropout只是要保持输入输出均值不变。

> Inline Question 2:
>
>Compare the validation and training accuracies with and without dropout -- what do your results suggest about dropout as a regularizer?

dropout reg 的作用很明显，但是对test 好像帮助不大。

> Inline Question 3:
>
>Suppose we are training a deep fully-connected network for image classification, with dropout after hidden layers (parameterized by keep probability p). How should we modify p, if at all, if we decide to decrease the size of the hidden layers (that is, the number of nodes in each layer)?

这个问题实在是有歧义，是要通过改变 p 来减小size 呢，还是减小 size 以后 p 要如何调整。

实际上这里的 p 是 keep 的概率，不是 drop 的概率，如果要通过 p 来减小 size，那么就要减小 p，要注意的是即使 p 减小，也不会对 test 时的size 产生影响。








end
