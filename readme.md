## 神经网络篇
> 模型的训练：导入数据 -> 数据进行归一化处理 -> 特征和标签 -> 划分训练集和验证集 -> 搭建神经网络 -> 进行训练并导出模型 -> 对训练集和验证集评价指标进行图形化
>
> 模型的验证： 导入数据 -> 数据进行归一化处理 -> 特征和标签 -> 划分训练集和验证集 -> 导入模型 -> 进行训练 -> 格式化预测数据 -> （回归任务：反归一化） -> 计算评价指标rmse和mape
>

#### 激活函数
1. Sigmoid 函数

![image](https://cdn.nlark.com/yuque/__latex/1622c082073681d20e5e72c4904914a7.svg)

适用于简单分类任务，缺点：反向传播训练有梯度消失的问题、非0对称

![](https://cdn.nlark.com/yuque/0/2025/png/29704292/1750658328633-006a02e6-f0ca-40f1-97e2-143ab5b3357f.png)

![](C:\Users\xhyu10\AppData\Roaming\Typora\typora-user-images\image-20250618100620592.png)

2. Tanh函数

![image](https://cdn.nlark.com/yuque/__latex/9218a1f9f63350bd92bdf0d6184d7b59.svg)

收敛快，解决了非0对称，缺点：反向传播梯度消失

![](https://cdn.nlark.com/yuque/0/2025/png/29704292/1750658361592-b6046a38-70c8-4074-b9fe-96b6e6c8322d.png)

![](C:\Users\xhyu10\AppData\Roaming\Typora\typora-user-images\image-20250618100914573.png)

3. ReLU函数

![image](https://cdn.nlark.com/yuque/__latex/1ae0cde4ed28679751d232b3b5c5d3ad.svg)

解决了梯度消失问题，计算简单，缺点：训练可能出现神经元死亡

![](https://cdn.nlark.com/yuque/0/2025/png/29704292/1750658384442-5d5aa18c-993e-4be6-887e-af0765616802.png)

![](C:\Users\xhyu10\AppData\Roaming\Typora\typora-user-images\image-20250618101141043.png)

4. Leaky ReLU函数

![image](https://cdn.nlark.com/yuque/__latex/3e8bf45a2fa87da2605a7b024f496db1.svg)

![](C:\Users\xhyu10\AppData\Roaming\Typora\typora-user-images\image-20250618101336163.png)

解决了ReLU的神经元死亡问题

![](https://cdn.nlark.com/yuque/0/2025/png/29704292/1750658417524-ddf6b049-4266-4849-b590-986aa1ac2f91.png)

5. SoftMax函数

![image](https://cdn.nlark.com/yuque/__latex/83a7ec1eb28cfc5a124007e5c8691ca7.svg)

![](C:\Users\xhyu10\AppData\Roaming\Typora\typora-user-images\image-20250618101604996.png)![](https://cdn.nlark.com/yuque/0/2025/png/29704292/1750658446070-071fe85b-9680-463d-b73d-8c60134d832e.png)

#### 损失函数
1. 回归任务

![image](https://cdn.nlark.com/yuque/__latex/87dafc19aad2f8df3e3866053432b554.svg)

2. 分类任务

![image](https://cdn.nlark.com/yuque/__latex/2b42ce1dae4b0500aacdf34cb4af312e.svg)

#### 数据类型
深度学习常用的数据类型有**DataFrame(Pandas)**、 **Array(Numpy)**、**tensor**、**list**、**map**

```python
import pandas as pd
import numpy as np

# DataFrame(Pandas) iloc切片
df = pd.read_csv('fileName')
a = np.array([1, 2, 3])

# DataFrame -> Array
data = np.array(data)

# Array -> torch
data = torch.from_numpy(data)
data = torch.tensor(data)

# list
image_paths = ['data/001.jpg', 'data/002.jpg']  

# map
model_config = {
    'hidden_dim': 512,
    'num_layers': 6,
    'dropout': 0.1
}
```



### 1.全连接网络（FCNN）
 每个神经元都与前一层和后一层的所有神经元相连接，形成一个密集的连接结构。

![](C:\Users\xhyu10\AppData\Roaming\Typora\typora-user-images\image-20250618095709297.png)![](https://cdn.nlark.com/yuque/0/2025/png/29704292/1750658465700-46fdfbfa-823c-407f-8f53-f135536d689c.png)

分类任务：loss + accuracy 回归任务： loss

### 2.卷积神经网络（CNN）![](C:\Users\xhyu10\AppData\Roaming\Typora\typora-user-images\image-20250618105712212.png)![](https://cdn.nlark.com/yuque/0/2025/png/29704292/1750658497005-f80677c3-afe4-4fb6-a1e6-4c677bce354e.png)
+ 卷积层：用来提取图像的底层特征
+ 池化层：防止过拟合，将数据维度减小
+ 全连接层：汇总卷积层和池化层得到的图像的底层特征和信息

**特征图大小**

![image](https://cdn.nlark.com/yuque/__latex/f775642bf66fb6b7bd45d6c52b41f0e4.svg)

![image](https://cdn.nlark.com/yuque/__latex/53d8e14d7bcf488ed75f920779ec29be.svg)

`H`/`W` 是输入特征图的高 / 宽；`P` 是填充（Padding）；`FH`/`FW` 是卷积核的高 / 宽；`S` 是步长

**1.卷积运算**

**单通道**

![](C:\Users\xhyu10\AppData\Roaming\Typora\typora-user-images\image-20250618110018982.png)![](https://cdn.nlark.com/yuque/0/2025/png/29704292/1750658526992-f53fe3fe-d9f7-4332-ad18-4acf3942490c.png)

**多通道（通达数是由卷积核的数量决定的）**

![](C:\Users\xhyu10\AppData\Roaming\Typora\typora-user-images\image-20250618105937768.png)![](https://cdn.nlark.com/yuque/0/2025/png/29704292/1750658569685-e8151938-995c-4ead-8523-a8753cdf6724.png)

**2.池化运算**

1. 最大池化运算

![](C:\Users\xhyu10\AppData\Roaming\Typora\typora-user-images\image-20250618111737197.png)![](https://cdn.nlark.com/yuque/0/2025/png/29704292/1750658621381-e82e187c-eee0-4ed7-9f9c-930d061eae93.png)

2. 平均池化运算

![](C:\Users\xhyu10\AppData\Roaming\Typora\typora-user-images\image-20250618111810820.png)![](https://cdn.nlark.com/yuque/0/2025/png/29704292/1750658632876-4adfcf5d-e994-465d-b3a8-ea97d00cbd7e.png)

### 3.循环神经网络（RNN）
与传统的前馈神经网络不同，RNN在处理每个输入时都会保留一个隐藏状态，该隐藏状态会被传递到下一个时间步，以便模型能够**记忆**之前的信息。**权重共享**

![](C:\Users\xhyu10\AppData\Roaming\Typora\typora-user-images\image-20250618135436957.png)![](https://cdn.nlark.com/yuque/0/2025/png/29704292/1750658650927-f0eabf80-bb87-423a-b2db-710f7ba16eba.png)![](https://cdn.nlark.com/yuque/0/2025/png/29704292/1750658666426-053acf67-c69b-426d-8365-6375bc130a14.png)

![](C:\Users\xhyu10\AppData\Roaming\Typora\typora-user-images\image-20250618135532849.png)

![image](https://cdn.nlark.com/yuque/__latex/b814443584d61218309a86db43fb0ef1.svg)

![image](https://cdn.nlark.com/yuque/__latex/a29784b6f024161585e3a186dd9d51f1.svg)

**基于时间反向传播（BPTT，Backpropagation Through Time ）** 

![image](https://cdn.nlark.com/yuque/__latex/d3cc48bc7b6cc55289f072f949c3f065.svg)

![image](https://cdn.nlark.com/yuque/__latex/268d87fe63cb5e26b066a6b637925cc6.svg)

![image](https://cdn.nlark.com/yuque/__latex/2ccc2d009b36f045ecbcf4b408ea342f.svg)

![image](https://cdn.nlark.com/yuque/__latex/dd71f6708dd8af9acac2db631fb915c2.svg)

**RNN梯度消失和梯度爆炸情况**

### 4.长短时记忆网络（LSTM）
传统的 RNN 在处理长序列数据时面临着严重的**梯度消失**问题，这使得网络难以学习到长距离的依赖关系。LSTM 作为一种特殊的 RNN 架构应运而生，有效地解决了这一难题。

![](C:\Users\xhyu10\AppData\Roaming\Typora\typora-user-images\image-20250618141311607.png)![](https://cdn.nlark.com/yuque/0/2025/png/29704292/1750658707407-5ac9be8d-6b8b-4af7-a130-5955d930be3a.png)

#### 遗忘门
![](C:\Users\xhyu10\AppData\Roaming\Typora\typora-user-images\image-20250618141540975.png)![](https://cdn.nlark.com/yuque/0/2025/png/29704292/1750658720050-43055781-7dee-4be9-be5c-df454ce5454f.png)

#### 输入门
![](C:\Users\xhyu10\AppData\Roaming\Typora\typora-user-images\image-20250618141818765.png)![](https://cdn.nlark.com/yuque/0/2025/png/29704292/1750658731544-d4344aa8-e8c3-469b-ad58-3afe7a184c90.png)

#### 细胞状态进行更新：根据遗忘门Ct-1和输入门(当前时刻)的结果，对细胞状态Ct进行更新
![](C:\Users\xhyu10\AppData\Roaming\Typora\typora-user-images\image-20250618142631825.png)![](https://cdn.nlark.com/yuque/0/2025/png/29704292/1750658747549-0fa84ab8-4731-4578-9b6e-3170731c76c4.png)

#### 输出门
![](C:\Users\xhyu10\AppData\Roaming\Typora\typora-user-images\image-20250618142725318.png)![](https://cdn.nlark.com/yuque/0/2025/png/29704292/1750658763303-1a62c67c-c978-419c-bd51-216e3be862c6.png)

### 5.LeNet-5 & AlexNet网络
#### 1）LeNet-5网络结构图
![](https://cdn.nlark.com/yuque/0/2025/png/29704292/1750661659005-4210e25a-8089-4ff1-8055-da56a4e210cd.png)

5层神经网络：2层卷积层+3层全连接层，使用的激活函数的sigmoid

![](https://cdn.nlark.com/yuque/0/2025/png/29704292/1750661842114-7b935a26-55d7-4657-82e1-c9ef88ca6120.png)

#### 2） AlexNet网络结构
![](https://cdn.nlark.com/yuque/0/2025/png/29704292/1750662067408-818d5362-f64d-406c-a8e9-f764d73a328d.png)

![](https://cdn.nlark.com/yuque/0/2025/png/29704292/1750661998869-58a1ba90-b68b-4d3a-9871-0852dc88ceaa.png)

8层神经网络：5层卷积层+3层全连接层，使用的激活函数的ReLu激活函数

![](https://cdn.nlark.com/yuque/0/2025/png/29704292/1750662242852-21ef50e7-b7d1-4383-906b-33bbab9e7873.png)

##### Dropout操作
<font style="color:rgb(77, 77, 77);">先随机选择其中的一些</font>**<font style="color:rgb(77, 77, 77);">神经元</font>**<font style="color:rgb(77, 77, 77);">并将其临时丢弃，然后再进行本次的训练和优化。在下一次迭代中，继续随机隐藏一些神经元，直至训练结束。</font>**<font style="color:rgb(77, 77, 77);">防止过拟合</font>**<font style="color:rgb(77, 77, 77);">（全连接层数据参数较多）。</font>

![](https://cdn.nlark.com/yuque/0/2025/png/29704292/1750662826927-4bda45f3-cdfc-474b-9f3b-f83cd798efcc.png)

扩增数据集：水平翻转、随机裁切、主成分分析PCA

##### LRN正则化
对局部的值进行归一化操作，使其较大的值变得更大，增加局部的对比度。

![image](https://cdn.nlark.com/yuque/__latex/117d9d1c9f31fa8fb311dcdb930da59f.svg)

形象可以理解为

![image](https://cdn.nlark.com/yuque/__latex/3db3935136d00561f24dcaabea99896b.svg)

### 6.VGG-16网络
<font style="color:rgb(51, 51, 51);">VGGNet可以看成是加深版的AlexNet，把网络分成了</font>**<font style="color:rgb(51, 51, 51);">5段</font>**<font style="color:rgb(51, 51, 51);">，每段都把多个尺寸为3×3的卷积核串联在一起，每段卷积接一个尺寸2×2的最大池化层，最后面接3个全连接层和一个softmax层，所有隐层的激活单元都采用ReLU函数。</font>

<font style="color:rgb(51, 51, 51);">VGG-16：13层卷积层+3层全连接成</font>

![](https://cdn.nlark.com/yuque/0/2025/png/29704292/1750927402915-f7a87484-db90-4d71-9993-3b9e7dfd25cb.png)

### 7.<font style="color:rgb(79, 79, 79);">GoogLeNet网络</font>
GoogLeNet 的创新点：

+ 引入了 Inception 结构（融合不同尺度的特征信息）
+ 使用1x1的卷积核进行降维以及映射处理 
+ 添加两个辅助分类器帮助训练
+ 丢弃全连接层，使用平均池化层（大大减少模型参数，除去两个辅助分类器，网络大小只有vgg的1/20）

#### Inception结构
![](https://cdn.nlark.com/yuque/0/2025/png/29704292/1751010222394-7def1d50-649e-46b3-acb6-a9b9d35bc482.png)

**1x1的卷积核：不改变长和宽，只通过卷积核的数量来改变通道的数量。**

**卷积核通道数的降维和升维，减少网络参数**

#### 全局平均池化GAP
![](https://cdn.nlark.com/yuque/0/2025/png/29704292/1751011695117-ec858b43-7993-4700-b011-5e857021fcac.png)

全局平均池化是对每个特征图的所有元素进行平均操作，将每个特征图缩减为一个标量值。其主要作用包括：

1. **减少维度**：将特征图从HxWxC维度缩减为1x1xC，减少计算量和参数数量。
2. **防止过拟合**：通过减少全连接层的使用，降低过拟合风险。
3. **简化模型**：GAP层可以直接连接到全连接层或分类器，简化网络结构。

#### 网络结构
![](https://cdn.nlark.com/yuque/0/2025/png/29704292/1751011987737-d91761e4-a657-4ef3-86b8-e7e75d91cfe0.png)

#### <font style="color:rgba(0, 0, 0, 0.85);">GoogLeNet两个额外的小型分类器</font>
反向传播中<font style="color:rgba(0, 0, 0, 0.85);">解决深度神经网络训练中的</font>**<font style="color:rgb(0, 0, 0) !important;">梯度消失</font>**<font style="color:rgba(0, 0, 0, 0.85);">和</font>**<font style="color:rgb(0, 0, 0) !important;">特征表示退化</font>**<font style="color:rgba(0, 0, 0, 0.85);">问题，同时起到</font>**<font style="color:rgb(0, 0, 0) !important;">正则化</font>**<font style="color:rgba(0, 0, 0, 0.85);">的效果。</font>

![](https://cdn.nlark.com/yuque/0/2025/png/29704292/1751267378964-a5001419-84a8-4eae-9fe1-aa40066d7d15.png)

![](https://cdn.nlark.com/yuque/0/2025/png/29704292/1751267461593-7d525a73-268e-462f-a8b2-2af750e0c9bf.png)

### 8.ResNet网络
#### <font style="color:rgb(77, 77, 77);">1）网络结构</font>
<font style="color:rgb(77, 77, 77);">在ResNet网络中有如下几个亮点：</font>

+ <font style="color:rgba(0, 0, 0, 0.75);">提出</font><font style="color:rgb(78, 161, 219);">residual</font><font style="color:rgba(0, 0, 0, 0.75);">结构（残差结构），并搭建超深的网络结构(突破1000层)</font>
+ <font style="color:rgba(0, 0, 0, 0.75);">使用批量规范化</font><font style="color:rgb(78, 161, 219);">Batch Normalization</font><font style="color:rgba(0, 0, 0, 0.75);">加速训练(丢弃dropout)</font>

![](https://cdn.nlark.com/yuque/0/2025/png/29704292/1751352022522-7eac0991-916b-4675-94aa-3604356edffa.png)

#### <font style="color:rgb(77, 77, 77);">2）两种残差结构</font>
<font style="color:rgb(77, 77, 77);">左边的残差结构是针对层数较少网络，例如ResNet18层和ResNet34层网络。右边是针对网络层数较多的网络，例如ResNet101，ResNet152等。</font>

![](https://cdn.nlark.com/yuque/0/2025/png/29704292/1751351954383-1a6a186b-4a95-4c56-9782-b864176ab1f4.png)

**<font style="color:rgb(77, 77, 77);">Inception</font>**<font style="color:rgb(77, 77, 77);">中加法是通道数相加，但是</font>**<font style="color:rgba(0, 0, 0, 0.75);">residual</font>**<font style="color:rgba(0, 0, 0, 0.75);">中加法是不变，输入=输出，输入和输出的形状都相同，</font><font style="color:rgb(77, 77, 77);">左边左图</font>**<font style="color:rgba(0, 0, 0, 0.75);">WHC都相同</font>**<font style="color:rgba(0, 0, 0, 0.75);">，右图输入和输出WH相同，</font>**<font style="color:rgba(0, 0, 0, 0.75);">通道数不同（1*1卷积改变通道数）。</font>**

![](https://cdn.nlark.com/yuque/0/2025/png/29704292/1751352159429-e739c6e9-2e30-421f-8340-dd7d57c471b5.png)

#### 3）Batch Normalization批量规范化层
![](https://cdn.nlark.com/yuque/0/2025/png/29704292/1751353476653-c31b8d63-5078-41d7-9066-59a868f85156.png)



