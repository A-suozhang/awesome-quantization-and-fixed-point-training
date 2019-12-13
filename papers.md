* [Training Quantized Network with Auxiliary Gradient Module]()
    * 额外的fullPrecision梯度模块(解决residue的skip connection不好定的问题，目的前向完全fix point)，有几分用一个FP去杠杆起低比特网络的意味
    * ![](https://github.com/A-suozhang/MyPicBed/raw/master/img/20191210113140.png)
      * 这里的Adapter是一个1x1Conv
      * FH共享卷积层参数，但是有独立的BN，在训练过程中Jointly Optimize
      * 用H这个网络跳过Shortcut，让梯度回流
    * 好像是一个很冗长的方法...


* [Scalable Methods for 8-bit Training of Neural Networks]()
  * Intel (AIPG)
  * 涉及到了几篇其他的工作
    * [Mixed Precision Training of Convnet](https://arxiv.org/pdf/1802.00930.pdf)
      * 16bit训练，没有精度损失
      * 用了DFP (Dynamic Fixed Point)
    * [L1-Norm Batch Normalization for Efficient Training of Deep Neural Networks Shuang](https://arxiv.org/pdf/1802.09769.pdf)
  * RangeBN
    * ![](https://github.com/A-suozhang/MyPicBed/raw/master/img/20191210171314.png)
    * (~~所以我们之前爆炸很正常~~，但是前向定BN不定grad为啥没事呢...)
    * 核心思想是用输入的Max-Min来代替方差，再加上一个ScaleAdjustTerm（固定值1/sqrt(2*ln(n))）
      * 证明了在Gaussian情况下
        * ![](https://github.com/A-suozhang/MyPicBed/raw/master/img/20191210172516.png)
    * Backward只有y的导数是低比特的，而W的是全精度的
      * （作者argue说只有y的计算是Sequential的，所以另外一个部分组件的计算不要求很快，所以可以全精度...）
  * Part5 Theoretical Analysis
    * ![](https://github.com/A-suozhang/MyPicBed/raw/master/img/20191210184548.png)
    * ⭐对层的敏感度分析有参考价值
  * GEMMLOWP
    * ![](https://github.com/A-suozhang/MyPicBed/raw/master/img/20191210164913.png)
    * ![](https://github.com/A-suozhang/MyPicBed/raw/master/img/20191210165035.png)
      * 按照chunk确定，来规避大的Dynamic Range
    * ![](https://github.com/A-suozhang/MyPicBed/raw/master/img/20191210165123.png)

* [Mixed Precision Training of CNN using Interger]
  * DFP(Dynamic Fixed Point)
  * ![](https://github.com/A-suozhang/MyPicBed/raw/master/img/20191210194007.png)



---

* [TTQ]()
  * Han
  * DoReFa(Layer-Wise Scaling Factor:L1Norm)
  * TWN(Ternary Weight Net)(View As Optmize Problem Of Minimizing L2 Norm Between)
  * ![](https://github.com/A-suozhang/MyPicBed/raw/master/img/20191210130401.png)
    * t是一个超参数，对所有层保持一致
  * TWN的步骤融入了Quantize-Aware Training
  * Scaling Factor
    * DoReFa直接取L1 Norm的mean
    * TWN对fp32的Weight，最小化L2范数（Xnor也是）
    * TTQ这里是训练出来的（所以并不是来自整个参数的分布，而是独立的参数）
  * ![](https://github.com/A-suozhang/MyPicBed/raw/master/img/20191210132214.png)

---


* 早期的一些二值化网络的延申
  * XnorNet文章是同时提了BinaryWeightedNetwork和XNORNet
    * 向量乘变为二值向量做bitcount
  * 后面是TernaryNet（TWN）
    * 二值变三值
  * ABCNet (Accurate Binary Conv)
    * 用多个Binary的LinearCombinatoin来代表全精度weight
  * DoReFa
    * 也可以加速反向梯度
    * 和xnor的区别是，xnor的取saclefactor是逐channel的而它是逐layer的
      * ~~Dorefa作者说xnor的方式做不了backprop加速，但是xnor作者在原文中说他也binarize了梯度~~
  * TTQ （我全都训）

---

