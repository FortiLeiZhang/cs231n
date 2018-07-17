Lecture 15 | Efficient Methods and Hardware for Deep Learning
---
Stanford 之所以是 Stanford，就在于他能请到的这些 guest speaker。这一节讲如何让 DNN 算的更快，内存用的更少，能耗更低。其中介绍了算法上的优化和硬件上的优化，算法上的优化又可以进一步的用在硬件优化上。对硬件不熟悉，但是算法上的优化很吸引人，特别是第一部分 Algorithms for Efficient Inference 中的几点：Pruning, Weight Sharing, Quantization, Low Rank Approximation, Binary/Ternary Net, Winograd Transformation。我对上述方法的初步理解就是，DNN 中的参数不需要太精确，有时候用32位或者64位的精度反而适得其反。甚至在 Pruning 中，把比较小的参数直接令其为0，这样既减少了参数，也减少了计算量，效果还不差。
