# 初衷

* （表明该整理的目的，为后续工作展开做一些指引）
*  本文不完全包括了一些NN Fixed-Point Training的可能实现方向，为后续实验指路
*  本文的最终目的是面向FPGA等嵌入式设备的NN 部署/训练，的一个软件仿真工具。为了Scalability,会尝试实现一些较为General的定点训练技巧
   *  故可能**不会包含**BinaryNetwork等较为*激进/专门化(需要额外硬件计算结构设计)*的实现方式
   *  以及诸如MobileNet, ShuffleNet 等轻量化网络设计
*  Quantization的两种主要方式
   *  基于CodeBook的(Deep Compression) ：实际参数还是高精度的，无法利用定点计算进行加速，仅能减少存储
   *  基于定点数(Fixed Point表示)，(IBM的FP8也可以归入此类) ： 可利用定点计算加速，故本文主要采取该方式
*  目前预计的几种场景
   *  Post-Training Quantization : 在完成FP训练之后的压缩，产生定点的W/A进行部署
      *  Example：Distillation，EntropyConstraintQ， IncrementalQ
   *  Quantize-Aware Training ： 在训练过程中考虑定点的影响，在训练中采取Fixed,产生定点的W/A进行部署
      *  Example： StraightThroughActivation的方法（训练时用Fixed Inference，但是梯度是对应的全精度副本做）
   *  Fixed-Point Training: 训练过程中进行**纯定点(W/G/A)**，模拟在纯定点设备上进行训练
      *  Example：WAGE
      *  ~~有点激进，不知道是否能实现~~

# Methods

> 从自己的出发点对看到的一些定点相关工作的方法与思想的纯主观归纳（可能存在偏差甚至错误）

> 该种划分方式没有什么道理，只是我强行区分的罢了）

## A. Post-Training Quantization

> 该类方法最大的特点就是利用已经训练好的模型进行压缩，与Quantized-Aware Training相对比有几分 2-Stage的意味，优势在于可以利用已经基本确定的参数分布去分析，采取量化策略,对大模型比较有效，但是小模型会崩

* Deep Compression
  * 利用了参数分布的知识，采用K-Means
* [Fixed Point Quantization of Deep Convolutional Network](https://arxiv.org/abs/1511.06393)
  * 高通 基于SQNR,前Deep Compression的上古时期，没有什么大的亮点
* [Entropy Constraint Scalar Quantization](https://www.mdpi.com/1099-4300/18/12/449)
  * 对每一个参数的Gradient做泰勒展开并且舍弃高阶项，化简得到哪些参数对最终Loss重要，以此作为剪枝或者量化的依据(选取聚类中心)
  * ~~也可以作为剪枝的依据~~
  * 和剪枝的这篇，有一定相关性[Designing Energy-Efficient Convolutional Neural Networks using Energy-Aware Pruning](https://arxiv.org/abs/1611.05128)
    * 贪心的剪去对最后Loss影响不大的 (也就是TCP(Transfer Channel Prunning)中的剪枝方式)
    * [NISP: Pruning Networks using Neuron Importance Score Propagation](https://arxiv.org/abs/1711.05908)也会涉及
* Incremental Quantization
  * 分组-量化-Finetune的流程，每次retrain只有当前组被量化，迭代直到所有参数都被量化（介于A与B之间）
* [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)
  * 利用一个很大的Tecaher来引导低比特网络
* Other Methods    (~~More Of a Digest Not Important~~)
  * [TWN](https://arxiv.org/abs/1605.04711)
    * TWN - 最小化全精度weight与Ternary Weight之间的L2 Norm
  * [Retraining-Based Iterative Weight Quantization for Deep Neural Networks](https://arxiv.org/abs/1805.11233)
  * ⭐[Post training 4-bit quantization of convolutional networks for rapid-deployment(NIPS 2019)](https://arxiv.org/abs/1810.05723)
    * Intel, No Need To Finetune On Full Dataset
  * [And the Bit Goes Down: Revisiting the Quantization of Neural Networks](https://arxiv.org/abs/1907.05686)


## B. Quantize-Aware-Training

> 相比于第一类，该类方法的主要优势在于1-Stage，简化了训练过程

* 早期的一些Binary/XNORNet均属于此类，大部分基于StraightThroughActivation的思想，即认为定点过程的导数为1
  * XNORNet 对WA二值化，加上一个L1Norm的mean作为每层的ScalingFactor（其改进DoReFa加上了G的）
  * [WRPN-Intel-ICLR2018](https://openreview.net/pdf?id=B1ZvaaeAZ)
    * 低比特WA(全精度G)，但是让网络更wide(增多了FeatureMap数量)
* [Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference](https://arxiv.org/abs/1712.05877)
  * 算是Quantize—Aware Training的开山了，包含了一个浮点的Scale Factor
  * 非对称量化(有一个零点，以及一个浮点的Scale - 逐层)
  * Merge Conv-BN
* ⭐[PACT](https://arxiv.org/abs/1805.06085)
  * 训练中Quantize Activation，训练一个activation clipping parameter(修改Relu的clip范围s)(也就是在训练中找FixScale)
* [Mixed Precision Training Of ConvNets Using Integer Operations-ICLR2918](https://arxiv.org/pdf/1802.00930.pdf)
  * Intel
  * 16bit Training
* Other Works
  * [Accurate & Efficient 2-bit QNN](https://www.semanticscholar.org/paper/ACCURATE-AND-EFFICIENT-2-BIT-QUANTIZED-NEURAL-Choi-Venkataramani/c3cb27f9ef7176658f37b607e75cc2c37f5e0ea8)
    * Quantize with Shortcut / with PACT
  * [Training Quantized Network with Auxiliary Gradient Module](https://arxiv.org/abs/1903.11236)
    * 额外的fullPrecision梯度模块(解决residue的skip connection不好定的问题，目的前向完全fix point)，有几分用一个FP去杠杆起低比特网络的意味
  * [Mixed Precision Training With 8-bit Floating Point](https://arxiv.org/abs/1905.12334)
    * WAGE All FP8
    * 对比了RNE(Round2NearestEven)&sStochastic Rounding 
  * [Learning to Quantize Deep Networks by Optimizing Quantization Intervals with Task Loss](http://openaccess.thecvf.com/content_CVPR_2019/papers/Jung_Learning_to_Quantize_Deep_Networks_by_Optimizing_Quantization_Intervals_With_CVPR_2019_paper.pdf)
    * 学了Quantize Interval
  * [Accumulation bit-width Scaling](https://arxiv.org/abs/1901.06588)
    * IBM ICLR 2019, 找Accumulator可以压到多少⭐



## C. (Full) Fixed-Point Training

> 纯定点的训练,大部分都是对一些经典的部分做一些简化。以及对梯度量化也会归入此类(目的是训练的加速，而不仅仅是为了部署) 

* [DoReFa](https://arxiv.org/abs/1606.06160)
  * 算是最先提出低比特训练
  * 每个Layer有一个ScalingFactor
* [WAGE - Training & Inference with Integers in DNN](https://arxiv.org/abs/1802.04680)
* ⭐ [Scalable Methods for 8-bit Training of Neural Networks](https://arxiv.org/abs/1805.11046)
  * Intel(AIPG) NIPS2018 (WAG8)(RangeBN)
* [Training Deep Neural Networks with 8-bit Floating Point Numbers](https://papers.nips.cc/paper/7994-training-deep-neural-networks-with-8-bit-floating-point-numbers.pdf))
  * FP8
* Other Methods
  * [Per-Tensor-Quantization of BackProp](https://arxiv.org/abs/1812.11732)
    * ICLR2019, Precision Assignment (好多数学假设分析),给出了一种确定每层位宽的方法
  * [Hybrid 8-bit Training](https://papers.nips.cc/paper/8736-hybrid-8-bit-floating-point-hfp8-training-and-inference-for-deep-neural-networks)
    * FP8的后续，对不同的组件提出不同的exponential bit与mantissa bit的划分方式

# Ideas

* 把WA压缩到能放到片上能够显著提升硬件设计性能(显而易见)
  * 这可能也是BNN系列的比较繁荣的原因
* Huge Batch Size可以帮助Binary的训练(原理上对低比特同理?)
* Rounding Methods - Neareset/Stochastic/Biased Rounding
* Fine-Grained能显著提高低比特Inference
  * 这也是目前Fixed-Point Training的一定能work的办法之一
* 从模型压缩角度看，硬件部署训练的几大难点(和不太elegant的地方)
  * 依赖大batch对存储要求高
  * 随机rounding，硬件难实现



### Stochastic Roudning Related

* Stochastic Rounding对保证Convergence重要
  * 但是WRPN作者文中表示自己利用了Full-Precision Grad所以不关键
  * Scalable Methods for 8-bit Training of Neural Networks也提到了 
  * Mixed Precision Training With 8-bit Floating Point文中对RNE和Stochastic Rounding做了一个对比
    * 比如imagenet Res18，问题不大，但是Res50会明显Overfiting
    * 作者归因为Noisy Gradient错误指引了模型的前进方向，weight被错误更新了
    * 同时发现L2范数会激烈增大，导致了Gradient变得更加Noisy，从而Vicious Circle
    * 结论是**RNE**的方法对Gradient的Quantization Noise不是很有效
    * 认为stochastic rounding参考了被丢掉bit的信息，更稳定

### BN Related

* WAGE首先提出BN是训练瓶颈
* [L1BN (Linear BN) ](https://arxiv.org/pdf/1802.09769.pdf)
    * 将BN所有的操作都转化为线性
* [RangeBN](https://arxiv.org/pdf/1802.09769.pdf)
    * 将BN的Var转化为一个|Max-Min|*(1/sqrt(2ln(n)))
* 假设BN的running mean/var已经稳定
    * 然后把他当作常数来计算


# Genre

## Binary及其延申(极低比特)

> 从一开始的BNN延申开来的一系列work，基本都是利用了二值之后乘法变bitwise，对硬件部署采用非传统运算单元。

* [BNN](https://arxiv.org/abs/1602.02830)
* [BinaryConnect](https://arxiv.org/abs/1511.00363)
* [TernaryNet(TWN)](https://arxiv.org/abs/1605.04711)
* [XNorNet](https://arxiv.org/pdf/1603.05279.pdf)
* [ABCNet](https://arxiv.org/abs/1711.11294)
* [WRPN-Intel-ICLR2018](https://openreview.net/pdf?id=B1ZvaaeAZ)
* [DoReFaNet](https://arxiv.org/pdf/1606.06160.pdf)
* [TTQ(Trained Ternary Quantization)](https://arxiv.org/pdf/1612.01064.pdf)
* [Simultaneously Optimizing Weight and Quantizer of Ternary Neural Network using Truncated Gaussian Approximation](https://arxiv.org/abs/1810.01018)
* [Training Competitive Binary Neural Networks from Scratch](https://arxiv.org/abs/1812.01965)
* ~~最后这篇文章放在这里只是为了告诉大家这个领域到了2019年还在蓬勃发展~~

## 量化方法（低比特）

> 使用相对较为传统的比特数(4,6,8)，在具体量化方式，以及训练方式入手

* [Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference](https://arxiv.org/abs/1712.05877)
* [PACT](https://arxiv.org/abs/1805.06085)
* [Incremental Quantization](https://ieeexplore.ieee.org/document/4476718/)
* [Model compression via distillation and quantization](https://arxiv.org/abs/1802.05668)
* [Training Deep Neural Networks with 8-bit Floating Point Numbers](https://papers.nips.cc/paper/7994-training-deep-neural-networks-with-8-bit-floating-point-numbers.pdf)

## 理论分析
* [Training Quantized Nets: A Deeper Understanding](https://arxiv.org/abs/1706.02379)
* [Towards The Limits Of Network Quantization](https://openreview.net/pdf?id=rJ8uNptgl)
* [Accumulation bit-width Scaling](https://arxiv.org/abs/1901.06588)
* [Per-Tensor-Quantization of BackProp](https://arxiv.org/abs/1812.11732)
* [Understanding Straight-Through Estimator in Training Activation Quantized Neural Nets](https://arxiv.org/abs/1903.05662)
* [An Empirical study of Binary Neural Networks' Optimisation](https://openreview.net/pdf?id=rJfUCoR5KX)
* [Scalable Methods for 8-bit Training of Neural Networks | Part 5](https://arxiv.org/abs/1805.11046)


## ~~奇技淫巧~~

* [ernGrad: Ternary Gradients to Reduce Communication in Distributed Deep Learning](https://papers.nips.cc/paper/6749-terngrad-ternary-gradients-to-reduce-communication-in-distributed-deep-learning)
* [Deep Learning with Low Precision by Half-wave Gaussian Quantization](https://arxiv.org/abs/1702.00953)
* [Learning low-precision neural networks without Straight-Through Estimator (STE)](https://arxiv.org/abs/1903.01061)
* [SWALP: Stochastic Weight Averaging in Low-Precision Training](https://arxiv.org/abs/1904.11943)
* [Analysis Of Quantized MOdels-ICLR2019](https://openreview.net/forum?id=ryM_IoAqYX)
* [Training Quantized Network with Auxiliary Gradient Module](https://arxiv.org/abs/1903.11236)
* [Learning to Quantize Deep Networks by Optimizing Quantization Intervals with Task Loss](https://arxiv.org/abs/1808.05779)
* [And the Bit Goes Down: Revisiting the Quantization of Neural Networks](https://arxiv.org/abs/1907.05686)

# Docs

> 看一下大公司主流的压缩工具都提供了什么功能

## Tensorflow Lite

> tf.contrib.quantize & [Tensorflow Lite](https://www.tensorflow.org/lite/guide)

* 提供了一个[Post-Training-Quantize的工具](https://www.tensorflow.org/lite/performance/post_training_quantization)
  * 看上去是很直接的Quantize没有用到什么技巧，标准数据格式之间的转变(float16/int8)  **没有自定义的数据格式**
  * 文档中直接写到 ```If you want Higher Performance, Use Quantize-aware Training```
  * 没有Finetune的过程？
* [Quantize-Aware Training](https://github.com/tensorflow/tensorflow/tree/r1.14/tensorflow/contrib/quantize)
  * 只支持部分网络结构(合理)以卷积为主，还有一些RNN
  * 这里默认Fold了Conv和BN(我们所说的MergeConvBNs)
  * 有一个TOCO(Tf Lite Optimizing Converter)工具可以直接将训练好的FrozenGraph转化为真正的定点模型

## PyTorch

> [Quantization Tool](https://pytorch.org/docs/stable/quantization.html?highlight=quantize)

* 支持PerTensor和PerChannel的量化，采用带zeropoint的rounding
* Quantize Aware Training at ```torch.nn.qat torch.nn.intrinsic.qat```
* 提供了很多Observer
* 支持Symmetrical和Asymmetrical的量化





---

# Groups (Low-Bit Training)

* Intel (AIPG)
* KAIST (Korea)
* IBM
* Kaust (迪拜的一个学校...)

# TODO

* 实现对BN的魔改
  * 首先研究它的定点Beahaviour
    * ~~还需要再过一过文献中，统计以下目前对BN的认识(除了不能定点之外，大家都是怎么处理的)~~
* 实现Stochastic rounding
* 实现PACT
* 对目前的几种Range方法实现分组
* 对WA做Clamp实现，作为超参数加入
* (?)  能否从TernaryNet这一系列中提取出一些可以参考的优化点（比如训练出一个Range）

# References

* [Awesome-Model-Compression](https://github.com/csyhhu/Awesome-Deep-Neural-Network-Compression/blob/master/Paper/PaperByConference.md)
* [Blog N0.0](https://blueardour.github.io/2019/04/29/model-compression-summary.html)
* [TensorBoard Lite Doc](https://www.tensorflow.org/lite/performance/post_training_quantization)
* [Distiller Doc](https://nervanasystems.github.io/distiller/quantization.html)

