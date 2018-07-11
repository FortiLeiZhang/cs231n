[Assignment3 | Network Visualization](https://github.com/FortiLeiZhang/cs231n/blob/master/code/cs231n/assignment3/NetworkVisualization-PyTorch.ipynb)
---

在这里只解决与 NN 有关的问题，与 PyTorch 有关的问题，集中到另外的一个 project 专门学习。

### Saliency Maps

Saliency map 表示的是图片中的每一个像素对最终分类 score 的影响有多大。计算 saliency map 时，首先使用一个训练好的 model，计算出输入图片的对正确 class 的 score 或者是最终的 loss，然后 backprop 出 score/loss 相对于输入像素点的梯度，也就是 dx，然后取 dx 所有 channel 中绝对值最大的作为该点的 saliency value。所以，对于 (3, H, W) 的图片，saliency map 是 (H, W) 的。

这里用的是正确 class 的 score 来进行 backprop 的，所反映出的是图片中的每一个像素对作出正确分类的影响有多大。同样，也可以用任意的 class 的 score 来进行 backprop，如果这么做的话，反映的是如果该图片被错误的分类成了某个 class，那么图片中的像素点对错误分类的影响有多大。

总之，saliency map 反映的是图片中的每一个像素对分类的影响有多大。
> Inline Question
>
> A friend of yours suggests that in order to find an image that maximizes the correct score, we can perform gradient ascent on the input image, but instead of the gradient we can actually use the saliency map in each step to update the image. Is this assertion true? Why or why not?

这是不对的，因为 saliency map 在计算时取了三个 channel 的绝对值中的最大值，是形如 (H, W) 的，如果要作梯度的来 update image 的话，应该用形如 (3, H, W) 的。

### Fooling Images
生产 fooling image 的方法和更新学习参数是一样的，即对目标类别的 score 作 backprop 得到梯度 dx，根据梯度对图片作 gradient ascent，不断重复此过程，直到 model 被欺骗产生我们想要得到的分类。

可以看到，产生的 fooling image 与原始的 image 人的肉眼根本无法区别，两者的区别用图片表示出来类似白噪声。

### Class Visualization
方法同上，但是这次是从一个白噪声 image 开始，通过 gradient ascent 不断向想要得到的分类类别逼近。只不过这里加了一个 L2 reg 项。
