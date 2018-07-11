Lecture 12 | Visualizing and Understanding
---
首先，这一节所要讲述的问题，在视频的[50:26]明确的提到：
> This whole field of trying to Visualizing intermediates is kind of in response to a common criticism of deep learning. So a common criticism of deep learning is like： you've got this big black box network, you trained it on gradient descent, you got a good number, and that's great. But we don't trust the network because we don't understand as people why it's making the decisions that it is making. So a lot of these type of visualization techniques were developed to try and address that and try to understand as people why the network are making their various classification decisions a bit more.

也就是说，这一节中所讲到的方法，主要是用来给人解释，NN 为什么可以很好的处理图像识别，分类等问题。到目前为止，我们所做的工作就是把大量图片喂给一个 NN，然后设计 loss 函数，计算梯度，根据梯度下降来更新 NN 的参数使得 loss 减小，直到得到一个满意的模型。但是，整个过程就像一个黑盒子，我们只能看到输入进黑盒子的数据，和黑盒子产生的输出结果，但是黑盒子里面发生了什么，我们一无所知。这一节给出了一些方法来帮助我们探索这个黑盒子内部发生了什么。

作者给出了很多方法，最后的 deep dream 和 style transfer 也很 cool，但是我不认为这一节是重点。CNN 里面究竟是什么样子，或许以后我们会遇到这个问题，但现在还是先让科学家去思考这个问题吧。
