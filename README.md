# ResNet_keras_custom
Building ResNet by keras after reading ResNet paper.

### ResNet论文学习与keras实现
***
#### 简述
深度网络大多是整合低、中、高层特征，依靠端到端分类器的多层结构。

作者经过研究发现，在随着网络结构逐渐变深的情况下，会出现一种**退化 (Degradation)** 现象。表现为，随着网络深度的增加，准确率会逐渐上升并达到饱和，随后会急剧的下降。这种退化现象并不是由过拟合引起的。

作者提出一种残差块结构.
这种结构方法表示出一个特性：**深层网络模型不会比它对应的浅层结构产生更高的误差。**其中，恒等映射x称为**shortcut**。
残差结构跟 **“highway networks”** 有着相似的结构。
* highway有gating functions，触发效果的gate是数据独立的且有着自身的参数，并且当gate关闭的时候，highway结构就没有残差的功能了。
* 本文提出的残差结构是不需要 **训练参数（parameter-free）** 的，并且这种恒等映射永远不会关闭，也就是一直会存在残差的功能。

#### 残差学习
##### 残差
假设：多层的模型输出可以近似复杂函数，那么也能近似残差表示函数，如$H(x)-x$（假设输出与映射同纬度）。

于是作者提出看法：与其堆叠层数去近似原来的输出H(x)，不如去近似残差输出函数$F(x)=H(x)-x$。也就是网络的输出变为两部分的叠加：F(x)+x。
* 当shortcut表现为恒等映射的时候，那么可以得到上述结构的结果：深层模型得到的训练误差不会比对应的浅层误差更高。并且当恒等映射（浅层输出）是最优的时候，我们只需要将网络的输出F(x)置0即可。

虽然实际情况中，恒等映射x不一定是最优的，但这种结构更容易去调整输出。

假设期望的最优输出更接近于恒等映射x而不是0映射，那么测量输出的扰动会更加容易。
如假设在W1参数情况下，H(x)=1.1, x=1, F(x)=0.1；当参数改变，在W2参数情况下，H(x)=1.2, x=1, F(x)=0.2，那么计算变化率：
$$\sigma_1=(1.2-1.1)/1.1=0.09$$
$$\sigma_2=(0.2-0.1)/0.1=1$$
可以看出，残差结构下的变化更容易测量。

##### Identity Mapping by Shortcuts
考虑两种情况：
* 输入输出同维度：
则residual block的输出表示为：$y=F(x, {W_i})+x$ 。所以残差输出为 $F=W_1\sigma (W_2x)$，$\sigma$是ReLU，所以输出y=F+x。
* 输入输出不同维度：
则residual block的输出表示为：$y=F(x, {W_i}) + W_sx$。即需要对恒等映射x进行一个维度的变换。

#### 网络结构
* Plain network以VGG为基础，均使用3x3的卷积，存在两个原则：
1. 对于输出相同的层，卷积核的个数应该相同；
2. 对于输出不同的层，当size减半的时候，卷积核的个数应该扩大为2倍以保留相同的时间复杂度。
* 结构中所有的下采样均使用stride=2的卷积来进行操作。
* 最后是全局的average pooling和全连接。
* 注意，结构中的虚线shortcut表示维度尺寸发生变化。

##### 残差网络
当维度发生变化的时候，有两种操作选择：
1. shortcut仍然保持恒等映射，增加的维度用0来填充。
2. shortcut进行1x1卷积操作来增加维度，步长strides选择为2。

#### 初始化参数
* **数据增强**，进行随机裁剪、水平翻转、减去均值等操作。
* 采用**batch normalization**，顺序为conv-bn-relu。
* SGD，batch=256。
* learning rate选择0.1，当错误率不变的时候乘以因数0.1。
* l2正则化，weight_decay=0.0001。
* momentum=0.9。

### keras实现
ResNet:
[https://github.com/seiei17/ResNet_keras_custom](https://github.com/seiei17/ResNet_keras_custom)

Se ResNet:
[https://github.com/seiei17/SENet_keras_custom](https://github.com/seiei17/SENet_keras_custom)
