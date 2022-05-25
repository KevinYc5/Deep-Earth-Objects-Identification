# Deep Earth Objects Identification

## 基于CNN的断层识别

断层识别是通过将地震振幅体数据视为二维图像并考虑二维体积中相邻地震剖面之间的空间特性，以二维空间体积中的各个位置为中心取二维邻域子块作为图像输入，将断层识别的问题转化为图像分类问题。卷积神经网络（Convolutional Neural Networks, CNN）是一类包含卷积计算且具有深度结构的前馈神经网络，是深度学习（deep learning）的代表算法之一。卷积神经网络具有表征学习能力，能够按其阶层结构对输入信息进行平移不变分类，因此也被称为平移不变人工神经网络。卷积神经网络在计算机视觉领域应用较广，可用于解决图像分类问题。

### 效果

![](D:/资料/课程/第二学期/基于人工智能的深地目标搜索/Deep-Earth-Objects-Identification/pic/demo_earth1.png)



## 基于Unet的盐丘识别

 盐丘是由于盐岩和石膏向上流动并挤入围岩，使上覆岩层发生拱曲隆起而形成的一种构造，它是一种具有重要意义的底辟构造。盐丘识别即是通过某种方法从复杂的地震数据中圈定出盐丘的位置，识别出盐丘的形状。 可将盐丘识别视为目标识别分割问题。U-Net 基于 FCN（Fully Convolutional Networks）进行了改进并使用了数据增强（ data augmentation）可以训练一些相对较小的数据样本。

### 效果

![](D:\资料\课程\第二学期\基于人工智能的深地目标搜索\Deep-Earth-Objects-Identification\pic\demo_earth2.png)

### 代码框架

| 文件名                  | 功能                                                         |
| ----------------------- | ------------------------------------------------------------ |
| util.py                 | getData函数输入切片数据section以及patch中心点所在的位置，以及size大小得到一个以pos为中心的patch（边界的像素点采用填充0），getPatch函数返回一个包含所有patch的numpy数组 |
| preprocess.py           | raintestSplit函数和trainvalidationSplit函数将数据划分为训练数据，验证数据和测试数据，processTraindata函数将训练数据和验证数据转化为符合网络输入要求的数据集重组为断层标签数据（Importfile函数用于导入mat格式的数据，get_balanced函数用于平衡样本中的数据类别） |
| train_deepearthfault.py | 定义神经网络相关的参数 ，画出训练过程中的训练损失变化， 画出训练过程中的验证集指标变化 |
| inference_salthills.py  | 用于定义U-net网络模型结构。结构由三部分组成：下采样，池化实现；上采样，反卷积实现和最后层的softmax输出概率图 |
| process_img.py          | 文件用于二值化结果图，此代码单独运行观察最后二值化的结果，再调整阈值大小以得到一个较好的结果，不需要重新跑模型 |
| train_salthills.py      | 用于训练模型。得到一个生成器，以batch = 2的速率生成增强后的数据，利用ModelCheckpoint函数保存模型，fit_generator函数用于模型调参优化，最后绘制acc-loss曲线 |

