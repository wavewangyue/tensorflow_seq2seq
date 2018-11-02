# Tensorflow中的Seq2Seq全家桶

查看完整文档：[https://zhuanlan.zhihu.com/p/47929039](https://zhuanlan.zhihu.com/p/47929039)

### 引言

听说以后公司那边用 Tensorflow，最近就转回 Tensorflow学习一下，发现很久以前 Tensorflow 把 seq2seq 的接口又重新升级了一下，也加了一些功能，变成了一个物美价廉的全家桶（tf.contrib.seq2seq）。所以来感受一下，顺便做个记录

除了最基本的 Seq2Seq 模型搭建之外，主要是对全家桶接口里的 Teacher Forcing，Attention，Beam Search，Sequence Loss 这样一些比较实用的配件（其实也不算配件，已经是现在 seq2seq 模型的基本要求了）做了一下研究，顺手实践了一下

另外，又在不使用 Tensorflow 提供的的 Seq2Seq 接口的情况下用手实现了一下这些功能，体会一下区别

### 源代码

一共4个代码文件：

* model_seq2seq_contrib.py：用全家桶实现的 seq2seq 模型
* model_seq2seq.py：不用全家桶手写的 seq2seq 模型
* train_seq2seq.py：模型训练代码
* infer_seq2seq.py：模型测试代码

### 还是先上结论

tensorflow 所提供的这个 seq2seq 全家桶功能还是很强大，很多比如 Beam Search 这些实现起来需要弯弯绕绕写一大段，很麻烦的事情，直接调个接口，一句话就能用，省时省力，很nice

优点就是封装的很猛，简单看一眼文档，没有教程也能拿过来用。缺点就是封装的太猛了，太傻瓜式了，特别是像 Attention 这类比较重要的东西，一封起来就看不到数据具体是怎么流动的，会让用户失去很多对模型的理解力，可控性也减少了很多，比如我现在还没发现怎么输出 attention score（。。[尴尬捂脸]，如果有知道的请教我一下，感激不尽）

有得必有失，想要简便快捷拿过来就用使用，不想花时间去学习原理再去一行行码字，就要失去一些对模型的控制力和理解，正常。总的来说这个全家桶还是很好用，很强大，给了不熟练 Tensorflow 或不熟悉 seq2seq 的玩家一个 3 分钟上手 30 分钟上天的机会。但是使用的同时最好了解一下原理，毕竟如果真的把深度学习变成了简单的调包游戏，那这游戏以后很难上分啊

上一句话写给能看到的人，也写给我自己

![](https://pic1.zhimg.com/80/v2-162d4ff280e1261544de57920eeab6e0_hd.jpg)
