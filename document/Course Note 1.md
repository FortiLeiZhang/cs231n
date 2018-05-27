[Image Classification: Data-driven Approach, k-Nearest Neighbor, train/val/test splits](http://cs231n.github.io/classification/)
=====
关于KNN
-----
首先讲Nearest Neighbor，然后扩展到KNN。就是将所有的training data映射到R维空间中，然后定义某种距离度量计算方法，比如L1、L2等等。然后将要predict的data依同样方法映射到该R维空间中，找出与该点距离最近的点，NN的话就找最近的一个点，KNN的话就由K个点投票决定。
KNN的优点在于简单易实现，并且train的时间复杂度是O(1)，因为train的时候不需要任何计算，只是单纯的记住所有training data。但是，它的缺点也很突出，那就是它在predict的时候时间复杂度为O(N)，因为需要计算predict点与所有train data的距离。这与我们期望的恰恰相反，因为我们可以接受长时间的训练过程，但是预测过程希望越快越好。KNN的另外一个缺点是，如果空间维度很高，即R很大的时候，如果要“填满”该R维空间，我们需要随R指数级增长的训练数据点。

L1/L2的选择
----
在notes里简单的提了一句：

> In particular, the L2 distance is much more unforgiving than the L1 distance when it comes to differences between two vectors. That is, the L2 distance prefers many medium disagreements to one big one.

这里的意思是L2通常选择多个平均的、中等大小的变量，而L1倾向于选择一个大的变量，而将其他变量趋于0。这个点在后面的regulation中还会细讲，这里记住这个例子就可以了。
> L2: [0.25, 0.25, 0.25, 0.25]
>
> L1: [1, 0, 0, 0]

上述两个list之和均为1，但是L2倾向于把weight平均分配给4个参数，而L1倾向把权重都给1个参数，而另其他参数为0。

另外在视频中还提了一句，L1定义的距离与坐标系(coordinate frame)的选择有关，如果将坐标系旋转的话，距离是会变的；而L2距离与坐标系选择无关。如果一个vector有它特定的物理上的意义的话，可以选择L1，而L2更普适一些。

Validation
---
将所有数据分为train/val/test三组，使用train训练，用val调整超参数，在最后的最后，才可以使用test，并且test只允许使用这一次，并将这一次的结果作为最终结果上报。否则得到的classifier会overfitting，或者结果不准确，有cheat的嫌疑。
>Evaluate on the test set only a single time, at the very end.

所谓5-fold cross validation就是将所有的train data均匀分成5份，每次取4份做train，另外一份做val，重复五次，将五次结果平均。这样做的话每个数据都做了四次train，一次val。这样做的缺点是太expensive，NN中通常不用。注意，在这个过程中，test是不参与其中的。一定先将test set拿出来放到一边，不到最后交结果的时候不要碰它。
