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

* [Learning to Quantize Deep Networks by Optimizing Quantization Intervals with Task Loss](http://openaccess.thecvf.com/content_CVPR_2019/papers/Jung_Learning_to_Quantize_Deep_Networks_by_Optimizing_Quantization_Intervals_With_CVPR_2019_paper.pdf)
	* Train a Quantize Interval, Orune & Clipping Together(这tm不就是clip吗，clip到0就叫prune吗，学到了)
	* ![](https://github.com/A-suozhang/MyPicBed/raw/master/img/20191213191652.png)
	* 将一个Quantizer分为一个Transformer(将元素映射到[-1,1]或者是[0,1])和一个Discretezer
		* ![](https://github.com/A-suozhang/MyPicBed/raw/master/img/20191213192019.png)
		* 我理解是把QuantizeStep也就是qD作为一个参数(parameterize it)？
		* ![](https://github.com/A-suozhang/MyPicBed/raw/master/img/20191213192826.png)
		* 这个演示只演示了正半轴，负半轴对称
		* 这个cx，dx也要去学习?（因为作者文中并没有提到这两个怎么取得）
		* 作者强调了把这个step也参数化，然后直接从最后的Los来学习这些参数，而不是通过接近全精度副本的L2范数（这个思想和And The Bit Goes Down类似） 
		* ![](https://github.com/A-suozhang/MyPicBed/raw/master/img/20191213192259.png)
			* 虽然这个变换血复杂，但是作者argue说只是为了inference，前向时候用的是固定过的参数，所以无所谓
			* 这个更新用到也是STE，但是我理解都这么非线性了，STE还能work吗？
		* Activation的Quantizer与其不同，那个次方的参数gamma是1了
			* ![](https://github.com/A-suozhang/MyPicBed/raw/master/img/20191213192512.png)


---

* [Mixed Precision Training of CNN using Interger]
  * DFP(Dynamic Fixed Point)
  * ![](https://github.com/A-suozhang/MyPicBed/raw/master/img/20191210194007.png)

---

* [Post Training 4 Bit Quantization of Conv Network For Rapid Deployment](https://arxiv.org/pdf/1810.05723.pdf)
  * Intel (AIPG)
  * No Need For Finetune / Whole Dataset
    * Privacy & Off-The-Shelf (Avoid Retraining) - Could Achieve 8 bit
  * 3 Methods
    * 1. Analutical Clipping For Integer Quantization
      * Analytical threshold for cliping Value
      * 假设QuantizationNoise是一个关于高斯或者[拉普拉斯分布](https://zh.wikipedia.org/wiki/%E6%8B%89%E6%99%AE%E6%8B%89%E6%96%AF%E5%88%86%E5%B8%83)的函数
      * 对传统的，在min/max之间均匀量化的情况，round到middle point
        * ![](https://github.com/A-suozhang/MyPicBed/raw/master/img/20191213143844.png)
      * 则MSE是
        * ![](https://github.com/A-suozhang/MyPicBed/raw/master/img/20191213143814.png)
      * Quantization Noise
        * ![](https://github.com/A-suozhang/MyPicBed/raw/master/img/20191213144138.png)
      * Clipping Noise
        * ![](https://github.com/A-suozhang/MyPicBed/raw/master/img/20191213144206.png)
      * Optimal Clipping 
        * ![](https://github.com/A-suozhang/MyPicBed/raw/master/img/20191213144243.png)
        * **We Could Just Use This**
    * 2. Per Channel Bit Allocation
      * Overall MSE min
      * Regular Per Channel Quantization,每一层有一个自己的Scale和offset，本文*不同的Channel选用了不同的bitwidth*
      * 只要保证最后平均每层还是4bit（有点流氓哈）
        * ![](https://github.com/A-suozhang/MyPicBed/raw/master/img/20191213144616.png)
      * 转化为一个优化问题，我总共有B个bit可以分配给N个channe，我希望最后的MSE最小
      * 拉格朗日乘子法
        * ![](https://github.com/A-suozhang/MyPicBed/raw/master/img/20191213144836.png)
        * 计算出最终的最佳分配，给每层的Bi
          * ![](https://github.com/A-suozhang/MyPicBed/raw/master/img/20191213144918.png)
      * *Assume That: Optimial Quantization Step Size is (Range)^(2/3)*
    * 3. After Quantization Will be a bias in mean/var
      * ![](https://github.com/A-suozhang/MyPicBed/raw/master/img/20191213145006.png)
  * Related Works
    * ACIQ是这篇文章的方法，更早 (Propose Activation Clip post training)
      * 更早有人用KL散度去找Clipping Value（需要用全精度模型去训练，花时间）
      * 本质都是去Handle Statistical Outlier
        * 有人分解Channel 
        * ![](https://github.com/A-suozhang/MyPicBed/raw/master/img/20191213143501.png)
          * 细节没清楚，放这边看之后有咩有时间看

* [Accurate & Efficient 2 bit QNN](cn.bing.com/?toHttps=1&redig=7C8B48CACF7748ACB6C926F9E0DBECE4)
  * PACT + SAWB
    * SAWB(Statistical Aware Weight Bining)-目的是为了有效的利用分布的统计信息(其实也就是一二阶统计量)
    * 优化的目标还是最小化量化误差(新weight和原weight的L2范数)
    * 说之前取Scale用所有参数的mean的方式只有在分布Normal的时候才成立
    *  ![](https://github.com/A-suozhang/MyPicBed/raw/master/img/20191213150637.png)
    * 参数C1，C2是线性拟合出来的，根据bitwidth
      * 选取了几种比较常见的分布(Gauss,Uniform,Laplace,Logistic,Triangle,von Mises)
      * ![](https://github.com/A-suozhang/MyPicBed/raw/master/img/20191213150944.png)
      * (我理解这个好像只是在证明一阶和二阶统计量就足够拟合了)图里面的点是每个分布的最佳Scale
        * 上面的实验是对应一种量化间隔，作者后面的实验又说明对于多个量化间隔同理
        * 所以最大的贡献其实是**对于实际的分布，用实际分布的统计量去找这个最佳的Scaling**
  * 文中分析了PACT和Relu一样有表达能力
  * 还给了一个SystemDeisgn的Insght

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

* [TernGrad](https://papers.nips.cc/paper/6749-terngrad-ternary-gradients-to-reduce-communication-in-distributed-deep-learning.pdf)
  * 数学证明了收敛(假设是梯度有界)
  * 是From Scratch的
  * 做了Layer-Wise的Ternary/以及Gradient Clipping
  * ![](https://github.com/A-suozhang/MyPicBed/raw/master/img/20191213172324.png) 
    * 其中bt是一个Random Binary Vector
    * Stochastic Rounding
    * ![](https://github.com/A-suozhang/MyPicBed/raw/master/img/20191213172521.png)
  * Scale Sharing
    * ![](https://github.com/A-suozhang/MyPicBed/raw/master/img/20191213172635.png)
    * ![](https://github.com/A-suozhang/MyPicBed/raw/master/img/20191213172728.png)
  * ![](https://github.com/A-suozhang/MyPicBed/raw/master/img/20191213172838.png)
  * 他能work我们不能work？（前向是全精度的？）
 

---


* 早期的一些二值化网络的延申
  * XnorNet文章是同时提了BinaryWeightedNetwork和XNORNet
    * 向量乘变为二值向量做bitcount
    * 反向传播的时候也可以把梯度编程Ternary，但是需要取Max作为ScalingFactor而不是最大值
    * ![](https://github.com/A-suozhang/MyPicBed/raw/master/img/20191213153040.png)
      * 很神奇 
      * [Source Code](https://github.com/allenai/XNOR-Net/blob/master/models/alexnetxnor.lua) 
      * 别人的一个复现![](https://github.com/A-suozhang/MyPicBed/raw/master/img/20191213154204.png)
      * 参考了[这个issue](https://github.com/allenai/XNOR-Net/issues/4)是参与前向运算的
  * **注意只有XNORNet是input和weight全部都是二值，所以可以用Bitcount的运算！而后续的TWN和TTQ都没有提这个事情，而他们是只对Weight做了三值！**
  * 后面是TernaryNet（TWN）
    * 二值变三值
    * ![](https://github.com/A-suozhang/MyPicBed/raw/master/img/20191213163328.png)
    * ![](https://github.com/A-suozhang/MyPicBed/raw/master/img/20191213163424.png)
    * ![](https://github.com/A-suozhang/MyPicBed/raw/master/img/20191213163501.png)
  * ABCNet (Accurate Binary Conv)
    * 用多个Binary的LinearCombinatoin来代表全精度weight
  * DoReFa
    * 也可以加速反向梯度
    * 和xnor的区别是，xnor的取saclefactor是逐channel的而它是逐layer的
      * ~~Dorefa作者说xnor的方式做不了backprop加速，但是xnor作者在原文中说他也binarize了梯度~~
  * TTQ （我全都训）
    * ![](https://github.com/A-suozhang/MyPicBed/raw/master/img/20191213162015.png)
    * Backprop
      * ![](https://github.com/A-suozhang/MyPicBed/raw/master/img/20191213162414.png)
      * ![](https://github.com/A-suozhang/MyPicBed/raw/master/img/20191213164311.png)
      * ![](https://github.com/A-suozhang/MyPicBed/raw/master/img/20191213162446.png)
      * ![](https://github.com/A-suozhang/MyPicBed/raw/master/img/20191213165007.png)
      * ![](https://github.com/A-suozhang/MyPicBed/raw/master/img/20191213162700.png)

---

### [AMC-AutoML For Model Compression and Accleration on Mobile Devices]
* ECCV 2018
* Replace Handcrafted heuristics & rule-based Policy via AutoML
* Also Perform Better

---

* ![](https://github.com/A-suozhang/MyPicBed/raw/master/img/20191214170306.png)
	* Process Pretrained Model Layer-By-Layer
	* RL Agent receive Embedding Of an Layer,outtput prune ratio at
	* 强调了网络layers并不是independent的
	* Rule-Based couldn't Transfer(无法应对新的架构)
		* Deeper Net, More Search Space
	* 对于每一个Layer，做一个Encoding，之后，喂给RL Agent，其输出一个PruneRatio，剪完之后再给到下一层
* 两种方式
	* Resource-Constraint: 限制资源Flops，只去尝试采样出来的那些模块
		* action space(prune ratio) 被限制为了只有低于resource的才会被采样 
	* Acc-Constraint : 
* 在包括Classification和Detcetion的多个地方做了实验
* **本质上是用一种非常黑盒的方式直接从一个pretrain模型中学习出一个Agent来给出每一层的PruneRatio**
  * 对于Resource-Constraint的场景，采用了直接将搜索空间加一个限制的方式来做
  * 这种方式理论上很难训练出来，但是其实验结果证明了可以搜出一个架构（但是需要搜多久呢？）这个Agent是针对某一个网络的（等会它的weight需要确定吗？）
  * *我们是否可以说明对AutoPrunning这个问题其实没有那么复杂，我们可以用这个近似线性的模型直接拟合出以一个能够学下去的模型*  
    * 这样的一个问题就是不能保证泛化能力（也就是我们的拟合系数不能泛化过去）这个是否可以通过实验证明，其实这个问题没有这么复杂？
  * 它利用RL Agent对每一层的敏感度进行了一个黑盒的建模
    * 但是我们的所谓Fit的模块就是显式的去找这个敏感度
    * *等会我们说的那个假设是各层之间需要时独立的，AMC的背景里提了每一层可能时不独立，它的层间建模方式有一些诡异*
      * ![](https://github.com/A-suozhang/MyPicBed/raw/master/img/20191214181936.png)
      * ![](https://github.com/A-suozhang/MyPicBed/raw/master/img/20191214182333.png)
      * ![](https://github.com/A-suozhang/MyPicBed/raw/master/img/20191214182533.png)
        * 好像就只是很敷衍的表示这个层间信息我们融入进来了
* 我觉得Floor这个点非常有意义，甚至可以单独领出来思考，因为一些黑盒子，是很难显式的建模出一些很明显的先验的，比如说16~20   
  * 而Amc的Agent输入Embedding中是包含了Flops信息的，会有一些非常explicit的rule，**黑盒子建模得不偿失**
* Fine-Grained Action Space,如果是discrete action的话，很难explore出来，同时忽略了order信息，所以用DDPG，来在连续域做拟合
* DDPG Agent接收到的是一个StreamOfLayer，整个网络过完了之后收到一个reward(使用的Acc是finetune之前的)
    * 限定action space，直接将Reward设置为-Error
    * ![](https://github.com/A-suozhang/MyPicBed/raw/master/img/20191214191907.png)
      * 这样暴力的方法对实际硬件设计，Flops和PruneRatio严格正相关，但是硬件其实不一定，这是一个很明确的事情，可以直接加上（？）
* ![](https://github.com/A-suozhang/MyPicBed/raw/master/img/20191214193113.png)
  * 做了这样的一个假设，我感觉还是挺暴力的，回头应该看一下这个结论是哪里来的
* 设置的时候并没有把BN加入到Conv层中
* ![](https://github.com/A-suozhang/MyPicBed/raw/master/img/20191214195522.png)
  * 感觉也没有差太多啊，因为大家都要Finetune的嘛
  
---

* 有一个小问题，就是我们的敏感度分析，的确没有办法建模出多层之间的影响，这个要么要说明可以看作独立，要么要加层间
  * 现在就是对每一层单独考虑，但是我觉得我们如果步长比较小的话问题不大
* 我感觉需要问一下涛哥关于DDPG的setback以及它这种直接砍搜索空间的方法合不合理
* AMC的建模是对网络结构的建模，如果weight发生改变是不是这个就不是很成立了？需要重新再训练一个agent
  * agent是不是weight-invariant的
* [AutoPruner: An End-to-End Trainable Filter Pruning Method forEfficient Deep Model Inference](https://arxiv.org/pdf/1805.08941.pdf)
  * 在Finetune的过程当中完成Prun
* [ECC: Platform-Independent Energy-Constrained Deep Neural Network Compression via a Bilinear Regression Mode](https://arxiv.org/pdf/1812.01803.pdf)
  * 有一些类似，不过他们Regressiond的是Energy，而且最后实验是在TX上做的
  * 是整个模型压缩，而不是layer-by-layer
  * relatedwork列举了几个工作比较有价值
  * ![](https://github.com/A-suozhang/MyPicBed/raw/master/img/20191214233728.png)
* [Automated Pruning for Deep Neural Network Compression](https://ieeexplore.ieee.org/abstract/document/8546129)
  * 可导的AutoPrune
* [AutoCompress: An Automatic DNN Structured Pruning Framework for Ultra-High Compression Rates](https://www.researchgate.net/profile/Zhiyuan_Xu9/publication/334316382_AutoSlim_An_Automatic_DNN_Structured_Pruning_Framework_for_Ultra-High_Compression_Rates/links/5ddf9aab4585159aa44f1634/AutoSlim-An-Automatic-DNN-Structured-Pruning-Framework-for-Ultra-High-Compression-Rates.pdf)
  * 提出了一个超级牛逼的Automated Framework **AutoCompress**
  * 用了Advanced Pruning方法**ADMM**

---

* [Layer Compensated Prunning For Resource Constraint CNN](https://arxiv.org/pdf/1810.00518.pdf)
  * 之前大多都是确定每层需要剪多少，然后在层内排序；这里看成一个Global Sorting的问题
  * 用了Meta Learning，找到了最好的solution？
* [Efficient Neural Network Compression](http://openaccess.thecvf.com/content_CVPR_2019/papers/Kim_Efficient_Neural_Network_Compression_CVPR_2019_paper.pdf)
  * 快速的找SVD分解对应的那个Rank
  * ~~把他挂在这里只是因为这个题目实在有点霸气...~~
  * [AutoRank: Automated Rank Selection for Effective Neural Network Customization](https://mlforsystems.org/assets/papers/isca2019/MLforSystems2019_Mohammad_Samragh.pdf)
* [Leveraging Filter Correlation For Deep Model Compression](https://arxiv.org/abs/1811.10559)
  * 可以认为他们认为Filter是怎么相关的来支持我们的独立性假设
* [SNN Compression](https://arxiv.org/abs/1911.00822)
* [PruneTrain: fast neural network training by dynamic sparse model reconfiguration](https://dl.acm.org/citation.cfm?id=3356156)
  * Lasso的那篇文章
* [EPNAS: Efficient Progressive Neural Architecture Search](https://arxiv.org/abs/1907.04648)
* [Mapping Neural Networks to FPGA-Based IoT Devices for Ultra-Low Latency Processing](https://www.mdpi.com/1424-8220/19/13/2981)
  * 不是9102年还发这种文章
* [SQuantizer: Simultaneous Learning for Both Sparse and Low-precision Neural Networks](https://arxiv.org/abs/1812.08301)
  * 尝试一步到位，但是好像没有引用
* [Band-limited Training and Inference for Convolutional Neural Networks](http://proceedings.mlr.press/v97/dziedzic19a.html)
* [PocketFlow: An Automated Framework for Compressing and Accelerating Deep Neural Networks](https://openreview.net/forum?id=H1fWoYhdim)
  * 也是automation，但是主要贡献点和我们不是特别一样（Tecent NIPS2018）
* [Cascaded Projection: End-To-End Network Compression and Acceleration](http://openaccess.thecvf.com/content_CVPR_2019/papers/Minnehan_Cascaded_Projection_End-To-End_Network_Compression_and_Acceleration_CVPR_2019_paper.pdf)
  * CVPR2019 Low-Ranlk Projection
* [Low Precision Constant Parameter CNN on FPGA](https://arxiv.org/pdf/1901.04969.pdf)
  * 纯FPGA的文章，摘要很短，很好奇
* [Reconstruction Error Aware Pruning for Accelerating Neural Networks](https://link.springer.com/chapter/10.1007/978-3-030-33720-9_5)
  * 这个所谓的reconstruct error是啥，是不是可以融入到我们的敏感度分析之中？ 
* [ReForm: Static and Dynamic Resource-Aware DNN Reconfiguration Framework for Mobile Device](https://dl.acm.org/citation.cfm?id=3324696)
  * DAC19
* [Network Pruning via Transformable Architecture Search](https://arxiv.org/abs/1905.09717)
* [AutoML for Architecting Efficient and Specialized Neural Networks](https://ieeexplore.ieee.org/abstract/document/8897011)
  * 有点牛逼，说有autoprunning和automixedprecision
* [BANANAS: Bayesian Optimization with Neural Architectures for Neural Architecture Search](https://arxiv.org/abs/1910.11858)
* [Accelerate CNN via Recursive Bayesian Pruning](http://openaccess.thecvf.com/content_ICCV_2019/html/Zhou_Accelerate_CNN_via_Recursive_Bayesian_Pruning_ICCV_2019_paper.html)
* [A Deep Learning-based Radar and Camera Sensor Fusion Architecture for Object Detection](https://ieeexplore.ieee.org/abstract/document/8916629/)
* [Neural Network Pruning with Residual-Connections and Limited-Data](https://arxiv.org/abs/1911.08114)
* [AutoQB: AutoML for Network Quantization and Binarization on Mobile Devices](https://arxiv.org/abs/1902.05690)
* [PruneTrain: Gradual Structured Pruning from Scratch for Faster Neural Network Training](https://arxiv.org/abs/1901.09290)