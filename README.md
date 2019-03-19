Tensorflow implementations with python
=

# 一、从神经细胞到神经网络

## 1.1生物神经网络

人工神经网络ANN的设计实际上是从生物体的神经网络结构获得的灵感。生物神经网络一般是指生物的大脑神经元，细胞，触电等组成的网络，
用于产生生物的意识，帮助生物进行思考和行动。

神经细胞构是构成神经系统的基本单元，简称为神经元。神经元主要由三部分构成：①细胞体；②轴突；③树突。如下图所示

![神经元结构](https://github.com/Joker2770/My_TensorFlow_Lab/blob/master/Resource/nerve_cell.jpg)

## 1.2单层感知器
([_01_PerceptronLearnRule.py](https://github.com/Joker2770/My_TensorFlow_Lab/blob/master/src/_01_PerceptronLearnRule.py) 
& [_02_PerceptronLearnRule_matrix.py](https://github.com/Joker2770/My_TensorFlow_Lab/blob/master/src/_02_PerceptronLearnRule_matrix.py) 
& [_03_SinglePerceptron.py](https://github.com/Joker2770/My_TensorFlow_Lab/blob/master/src/_03_SinglePerceptron.py))

### 1.2.1单层感知器介绍

受到生物神经网络的启发，计算机学家Frank Rosenblatt在20世纪60年代提出了一种模拟生物神经网络的的人工神经网络结构，
称为感知器(Perceptron)。单层感知器结构与神经元结构一致。

![单层感知器结构](https://github.com/Joker2770/My_TensorFlow_Lab/blob/master/Resource/单层感知器表示.png)

图中x1，x2，x3为输入信号，类似于生物神经网络中的树突。
w1，w2，w3分别为x1，x2，x3的权值，它可以调节输入信号的值的大小，让输入信号变大(w＞0)，不变(w=0)或者减小(w＜0)。
可以理解为生物神经网络中的输入阻碍，树突的信号不一定是100%传递到细胞核的，可能某些树突的信号有80%传递到细胞核，
某些树突的信号有50%传递到细胞核，某些树突的信号被完全阻碍。
公式表示细胞的输入信号在细胞核的位置进行汇总，然后再加上该细胞本身自带的信号b，b一般称为偏置值(Bias)。
f(x)称为激活函数，可以理解为信号在轴突上进行的变化。在单层感知器中使用的激活函数是sign(x)激活函数。该函数的特点是当x＞0时，
输出值为1；当x＝0时，输出值为0,；当x＜0时，输出值为-1。sign(x)函数图像如图

![sign函数图像](https://github.com/Joker2770/My_TensorFlow_Lab/blob/master/Resource/sign.jpg)

y就是，为单层感知器的输出结果。

### 1.2.2单层感知器计算举例

假如有一个单层感知器有3个输入x1,x2,x3，同时已知b=-0.6，w1=w2=w3=0.5，那么根据单层感知器的计算公式我们就可以得到如图计算结果。

![单层感知器计算结果](https://github.com/Joker2770/My_TensorFlow_Lab/blob/master/Resource/单层感知器计算结果.png)

![单层感知器计算过程](https://github.com/Joker2770/My_TensorFlow_Lab/blob/master/Resource/单层感知器计算过程.png)

### 1.2.3单层感知器的另一种表达形式

单层感知器的另一种表达形式如图

![单层感知器另一种表示](https://github.com/Joker2770/My_TensorFlow_Lab/blob/master/Resource/单层感知器另一种表示.jpg)

其实这种表达形式跟上面的单层感知器是一样的。只不过是把偏置值b变成了输入w0×x0，其中x0=1。所以w0×x0实际上就是w0，
把公式展开得到：w1×x1+w2×x2+w3×x3+w0。所以这两个单层感知器的表达不一样，但是计算结果是一样的。此种表达形式更加简洁，更适合使用矩阵来进行运算。

### 1.2.4单层感知器的学习规则 

在1.2.3中我们已知单层感知器表达式可以写成：

![单层感知器表达式](https://github.com/Joker2770/My_TensorFlow_Lab/blob/master/Resource/单层感知器表达式.jpg)

(1.1)中y表示感知器的输出
f是sign激活函数
n是输入信号的个数
i=0,1,2...

![第i个权值的变化](https://github.com/Joker2770/My_TensorFlow_Lab/blob/master/Resource/第i个权值的变化.jpg)

(1.2)中 ***ΔWi*** 表示第i个权值的变化；
***η*** 表示学习率(Learning rate)，用来调节权值变化的大小；
t是正确的标签(target)。

因为单层感知器的激活函数为sign函数，所以t和y的取值都为±1

t=y时， ***ΔWi*** 为0；t=1，y=-1时， ***ΔWi*** 为2；t=-1，y=1时， ***ΔWi*** 为-2。
由(1.2)可以推出：

![权值变化推导](https://github.com/Joker2770/My_TensorFlow_Lab/blob/master/Resource/权值变化推导.jpg)

权值的调整公式为：

![权值的调整公式](https://github.com/Joker2770/My_TensorFlow_Lab/blob/master/Resource/权值的调整公式.jpg)

### 1.2.5单层感知器的学习规则计算举例

假设有一个单层感知器有三个输入x0=1，x1=0，x2=-1，权值w0=-5，w1=0，w2=0，学习率=1，正确的标签t=1。(注意在这个例子中偏置值b用w0×x0来表示，x0的值固定为1)

Step1：我们首先计算感知器的输出：

![计算感知器输出](https://github.com/Joker2770/My_TensorFlow_Lab/blob/master/Resource/计算感知器输出.jpg)

由于y=-1与正确的标签t=1不相同，所以需要对感知器中的权值进行调节。

![对感知器中的权值进行调节](https://github.com/Joker2770/My_TensorFlow_Lab/blob/master/Resource/对感知器中的权值进行调节.jpg)

Step2：重新计算感知器的输出：

![重新计算感知器输出](https://github.com/Joker2770/My_TensorFlow_Lab/blob/master/Resource/重新计算感知器的输出.jpg)

由于y=-1与正确的标签t=1不相同，所以需要对感知器中的权值进行调节。

![对感知器中的权值进行调节1](https://github.com/Joker2770/My_TensorFlow_Lab/blob/master/Resource/对感知器中的权值进行调节1.jpg)

Step3：重新计算感知器的输出：

![重新计算感知器输出1](https://github.com/Joker2770/My_TensorFlow_Lab/blob/master/Resource/重新计算感知器的输出1.jpg)

由于y=1与正确的标签t=1相同，说明感知器经过训练后得到了我们想要的结果，我们就可以结束训练了。

单层感知器学习规则计算，python实现代码为[_01_PerceptronLearnRule.py](https://github.com/Joker2770/My_TensorFlow_Lab/blob/master/src/_01_PerceptronLearnRule.py)，结果如下：

~~~
-5 0 0
-3 0 -2
-1 0 -4
done
~~~

单层感知器学习规则计算举例(矩阵计算)，python实现代码为[_02_PerceptronLearnRule_matrix.py](https://github.com/Joker2770/My_TensorFlow_Lab/blob/master/src/_02_PerceptronLearnRule_matrix.py)，结果如下

~~~
[[-5]
[ 0]
[ 0]]
[[-3]
[ 0]
[-2]]
[[-1]
[ 0]
[-4]]
done
~~~

### 1.2.6学习率

学习率是人为设定的一个值，主要是在训练阶段用来控制模型参数调整的快慢。关于学习率主要有3个要点需要注意：

_1.η取值一般取0-1之间；_ <br>
_2.太大容易造成权值调整不稳定；_  <br>
_3.学习率太小，模型参数调整太慢，迭代次数太多。_  <br>

你可以想象一下在洗热水澡的时候：如果每次调节的幅度很大，那水温要不就是太热，要不就是太冷，很难得到一个合适的水温；
如果一开始的时候水很冷，每次调节的幅度都非常小，那么需要调节很多次，花很长时间才能得到一个合适的水温。学习率的调整也是这样一个道理。

图1.2.6表示不同大小的学习率对模型训练的影响：

![不同大小的学习率对模型训练的影响](https://github.com/Joker2770/My_TensorFlow_Lab/blob/master/Resource/不同大小的学习率对模型训练的影响.jpg)

**图1.2.6 不同大小的学习率对模型训练的影响**

图中的纵坐标loss代表代价函数，在后面的章节中有更详细的介绍，这里我们可以把它近似理解为模型的预测值与真实值之间的误差，
我们训练模型的主要目的就是为了降低loss值，减少模型预测值与真实值之间的误差。横坐标Epoch代表模型的迭代周期，
把所有训练数据都训练一遍可以称为迭代了一个周期。

从图中我们可以看到，如果使用非常大的学习率来训练模型，loss会一直处于一个比较大的位置，模型不能收敛，这肯定不是我们想要的结果。
如果使用比较大的学习率来训练模型，loss会下降很快，但是最后loss最终不能得到比较比较小的值，所以结果也不理想。
如果使用比较小的学习率来训练模型，模型收敛的速度会很慢，需要等待很长时间模型才能收敛。最理想的结果是使用合适的学习率来训练模型，
使用合适的学习率，模型的loss值下降得比较快，并且最后的loss也能够下降到一个比较小的位置，结果最理想。

看到这里大家可能会有一个疑问，学习率的值到底取多少比较合适？这个问题其实是没有明确答案的，需要根据建模的经验以及测试才能找到合适的学习率。
不过学习率的选择也有一些小的trick可以使用，比如说最开始我们设置一个学习率为0.01，经过测试我们发现学习率太小了需要调大一点，那么我们可以改成0.03。
如果0.03还需要调大，我们可以调到0.1。同理，如果0.01太大了，需要调小，那么我们可以调到0.003。如果0.003还需要调小，我们可以调到0.001。
所以常用的学习率可以选择：
1，0.3，0.1，0.03，0.01，0.003，0.001，0.0003，0.0001 ...

当然这也不是绝对的，其他的学习率的取值你也可以去尝试。

### 1.2.7模型的收敛条件

通常模型的收敛条件可以有以下3个：

_1. loss小于某个预先设定的较小的值；_ <br>
_2. 两次迭代之间权值的变化已经很小了；_ <br>
_3. 设定最大迭代次数，当迭代超过最大次数就停止。_ <br>

第一种很容易理解，模型的训练目的就是为了减少loss值，那么我们可以设定一个比较小的数值，每一次训练的时候我们都同时计算一下loss值的大小，
当loss值小于某个预先设定的阈值，就可以认为模型收敛了。那么就可以结束训练。

第二种的意思是，每一次训练我们可以记录模型权值的变化，如果我们发现两次迭代之间模型的权值变化已经很小，那么说明模型已经几乎不需要做权值地调整了，
那么就可以认为模型收敛，可以结束训练。

第三种是用得最多的方式。我们可以预先设定一个比较大的模型迭代周期，比如迭代100次，或者10000次，或者1000000次等（需要根据实际情况来选择）。
模型完成规定次数的训练之后我们就可以认为模型训练完毕。

### 1.2.8超参数(hyperparameters)和参数(parameters)的区别

超参数是机器学习或者深度学习中经常用到的一个概念，我们可以认为是根据经验来人为设置的一些模型的参数。比如说前面提到的学习率，学习率需要根据经验来人为设置。
比如模型的迭代次数，也是需要在模型训练之前预先进行人为设置。

而前面提到的权值和偏置值则是参数，一般我们会给权值和偏置值进行随机初始化赋值。模型在训练过程中根据训练数量会不断调节这些参数，进行自动学习。

### 1.2.9单层感知器分类案例

题目：假设我们有4个数据2维的数据，数据的特征分别是(3,3),(4,3),(1,1),(2,1)。(3,3),(4,3)这两个数据的标签为1，(1,1),(2,1)这两个数据的标签为-1。构建神经网络来进行分类。

思路：我们要分类的数据是2维数据，所以只需要2个输入节点，我们可以把神经元的偏置值也设置成一个输入节点，使用1.2.3中的方式。这样我们需要3个输入节点。

输入数据有4个(1,3,3),(1,4,3),(1,1,1),(1,2,1)，每个数据开头的1表示偏置值b。
数据对应的标签为(1,1,-1,-1)
初始化权值w1,w2,w3取0到1的随机数
学习率lr(learning rate)设置为0.1
激活函数为sign函数
我们可以构建一个单层感知器如图1.2.9.1所示：

![单层感知器](https://github.com/Joker2770/My_TensorFlow_Lab/blob/master/Resource/单层感知器.jpg)

**图1.2.9.1 单层感知器**

单层感知器案例[_03_SinglePerceptron.py](https://github.com/Joker2770/My_TensorFlow_Lab/blob/master/src/_03_SinglePerceptron.py)，输出结果为:

~~~
epoch: 1
weights: [[-0.06390143]
 [ 0.58589308]
 [ 0.25649672]]
epoch: 2
weights: [[-0.16390143]
 [ 0.43589308]
 [ 0.15649672]]
epoch: 3
weights: [[-0.26390143]
 [ 0.28589308]
 [ 0.05649672]]
epoch: 4
weights: [[-0.36390143]
 [ 0.13589308]
 [-0.04350328]]
epoch: 5
weights: [[-0.31390143]
 [ 0.28589308]
 [ 0.10649672]]
epoch: 6
weights: [[-0.41390143]
 [ 0.13589308]
 [ 0.00649672]]
Finished

~~~

![单层感知器案例输出](https://github.com/Joker2770/My_TensorFlow_Lab/blob/master/Resource/单层感知器案例输出.png)

因为权值的初始化使用的是随机的初始化方式，所以每一次训练的周期以及画出来的图可能都是不一样的。这里我们可以看到单层感知器的一个问题，
虽然单层感知器可以顺利地完成分类任务，但是使用单层感知器来做分类的时候，最后得到的分类边界距离蓝色的边界点比较近，而距离黄色的边界点比较远，
并不是一个特别理想的分类效果。

图1.2.9.2中的分类效果应该才是比较理想的分类效果，分界线距离黄色和蓝色两个类别边界点的距离是差不多的。

![单层感知器比较理想的分类边界](https://github.com/Joker2770/My_TensorFlow_Lab/blob/master/Resource/单层感知器比较理想的分类边界.jpg)

**图1.2.9.2 单层感知器比较理想的分类边界**

## 1.3线性神经网络

### 1.3.1线性神经网络介绍

线性神经网络跟单层感知器非常类似，只是把单层感知器的sign激活函数改成了purelin函数: <br>
 ***y = x             (1.5)*** <br>
purelin函数也称为线性函数，函数图像为图3.1：

![线性函数](https://github.com/Joker2770/My_TensorFlow_Lab/blob/master/Resource/线性函数.jpg)

_图3.1 线性函数_ <br>

### 1.3.2线性神经网络分类案例

参考1.2.9中的案例，我们这次使用线性神经网络来完成相同的任务。线性神经网络的程序跟单层感知器的程序非常相似，大家可以思考一下需要修改哪些地方。
代码[_04_LinearNN.py](https://github.com/Joker2770/My_TensorFlow_Lab/blob/master/src/_04_LinearNN.py)：线性神经网络案例

程序的输出结果为：

![线性神经网络案例结果](https://github.com/Joker2770/My_TensorFlow_Lab/blob/master/Resource/线性神经网络案例结果.jpg)

线性神经网络的程序有两处是对单层感知器程序进行了修改。
第一处是在train()函数中，将Y = np.sign(np.dot(X,W))改成了Y = np.dot(X,W)。因为线性神经网络的激活函数是y=x，所以这里就不需要np.sign()了。
第二处是在for i in range(100)中，把原来的：
~~~
# 训练100次
for i in range(100):
    # 更新一次权值
    train()
    # 打印当前训练次数
    print ('epoch:',i + 1)
    # 打印当前权值
    print ('weights:',W)
    # 计算当前输出
	Y = np.sign(np.dot(X,W))
    # .all()表示Y中的所有值跟T中所有值都对应相等，结果才为真
    if (Y == T).all():
        print ('Finished')
        # 跳出循环
        break
~~~
改成了：
~~~
# 训练100次
for i in range(100):
    # 更新一次权值
    train()
~~~
在单层感知器中，当y等于t时，Δw=0就会为0，模型训练就结束了，所以可以提前跳出循环。单层感知器使用的模型收敛条件是两次迭代模型的权值已经不再发生变化，则可以认为模型收敛。
而在线性神经网络中，y会一直逼近t的值，不过一般不会得到等于t的值，所以可以对模型不断进行优化。线性神经网络使用的模型收敛条件是设置一个最大迭代次数
，当训练了一定次数后就可以认为模型收敛了。
对比单层感知器和线性神经网络所得到的结果，我们可以看得出线性神经网络所得到的结果会比单层感知器得到的结果更理想。但是线性神经网络也还不够优秀
，当使用它处理非线性问题的时候，它就不能很好完成工作了。

### 1.3.3线性神经网络处理异或问题

首先我们先来回顾一下异或运算。 <br> 
_0与0异或等于0_ <br> 
_0与1异或等于1_ <br> 
_1与0异或等于1 1与1异或等于0_ <br>

代码[_05_LinearXOR.py](https://github.com/Joker2770/My_TensorFlow_Lab/blob/master/src/_05_LinearXOR.py)：线性神经网络-异或问题

程序的输出结果为：

![线性神经网络_异或问题](https://github.com/Joker2770/My_TensorFlow_Lab/blob/master/Resource/线性神经网络_异或问题.jpg)

从结果我们能够看出用一条直线并不能把异或问题中的两个类别给划分开来，因为这是一个非线性的问题，可以使用非线性的方式来进行求解。
其中一种方式是我们可以给神经网络加入非线性的输入。代码[_05_LinearXOR.py](https://github.com/Joker2770/My_TensorFlow_Lab/blob/master/src/_05_LinearXOR.py)
中的输入信号只有3个信号x0,x1,x2，我们可以利用这3个信号得到带有非线性特征的输入：
x0,x1,x2,x1×x1,x1×x2,x2×x2，其中x1×x1,x1×x2,x2×x2为非线性特征。神经网络结构图如图3.2所示：

![神经网络结构](https://github.com/Joker2770/My_TensorFlow_Lab/blob/master/Resource/神经网络结构.jpg)

_图3.2 引入非线性输入的线性神经网络_ <br>

代码[_06_nLinearXOR.py](https://github.com/Joker2770/My_TensorFlow_Lab/blob/master/src/_06_nLinearXOR.py)：线性神经网络引入非线性特征解决异或问题

程序的输出结果为：
~~~
[[-0.98650596] [0.990989 ] [0.990989 ][-0.99302749]]
~~~

![线性神经网络引入非线性特征解决异或问题结果](https://github.com/Joker2770/My_TensorFlow_Lab/blob/master/Resource/线性神经网络引入非线性特征解决异或问题结果.jpg)

从输出的预测值我们可以看出，预测值与真实标签的数值是非常接近的，几乎相等，说明预测值很符合我们想要的结果。而从输出图片中也能观察到两条曲线的内部是负样本所属的类别，
两条曲线的外部是正样本所属的类别。这两条曲线很好地把两个类别区分开了。
线性神经网络可以通过引入非线性的输入特征来解决非线性问题，但这并不是一种非常好的解决方案。
下一章节我们将介绍一种新的神经网络，BP(Back Propagation)神经网络，通过学习BP神经网络我们可以获得更好的解决问题的思路。

## 2.1BP神经网络介绍及发展背景

BP(back propagation)神经网络是1986年由Rumelhart和McClelland为首的科学家提出的概念，他们在《Parallel Distributed Processing》一书中对BP神经网络进行了详细的分析。
BP神经网络是一种按照误差逆向传播算法训练的多层前馈神经网络，它是20世纪末期神经网络算法的核心，也是如今深度学习算法的基础。
感知器对人工神经网络的发展发挥了极大的作用，但是它的结构只有输入层和输出层，不能解决非线性问题的求解。Minsky和Papert在颇具影响力的《Perceptron》一书中指出，
简单的感知器只能求解线性问题，能够求解非线性问题的网络应该具有隐藏层，但是对隐藏层神经元的学习规则还没有合理的理论依据。从前面介绍的感知器学习规则来看，
其权值的调整取决于期望输出与实际输出之差：

![BP网络权值调整公式](https://github.com/Joker2770/My_TensorFlow_Lab/blob/master/Resource/BP网络权值调整公式.jpg)

但是对于各个隐藏层的节点来说，不存在已知的期望输出，因而该学习规则不能用于隐藏层的权值调整。
BP算法的基本思想是，学习过程由信号的正向传播和误差的反向传播两个过程组成。
正向传播时，把样本的特征从输入层进行输入，信号经过各个隐藏层逐层处理后，最后从输出层传出。对于网络的实际输出与期望输出之间的误差，把误差信号从最后一层逐层反传，
从而获得各个层的误差信号，再根据误差信号来修正各个层神经元的权值。
这种信号正向传播与误差反向传播，然后各个层调整权值的过程是周而复始地进行的。权值不断调整的过程，也就是网络学习训练的过程。
进行此过程直到网络输出误差减小到预先设置的阈值以下，或者是训练了一定的次数之后为止。

## 2.2代价函数

代价函数也称为损失函数，英文称为loss function或cost function，有些地方我们会看到使用loss表示代价函数的值，有些地方我们会看到用cost表示代价函数的值。
为了统一规范，本书中我们统一使用代价函数这个名字，英文使用loss。
代价函数并没有准确的定义，一般我们可以理解为是一个人为定义的函数，我们可以利用这个函数来优化模型的参数。
最简单常见的一个代价函数是均方差(mean-square error, MSE)代价函数：

![MSE代价函数](https://github.com/Joker2770/My_TensorFlow_Lab/blob/master/Resource/MSE代价函数.jpg)

矩阵可以用大写字母来表示，这里的T表示真实标签，Y表示网络输出。

T-Y可以到每个训练样本与真实标签的误差。误差的值有正有负，我们可以求平方，把所有的误差值都变成正的，然后除以2N。这里2没有特别的含义，
主要是我们对均方差代价函数求导的时候，公式中的2次方的2可以跟分母中的2约掉，使得公式推导看起来更加整齐简洁。
N表示训练样本的个数(注意这里的N是一个大于0的整数，不是矩阵)，除以N表示求每个样本误差平均的平均值。

公式可以用矩阵形式来表达，也可以拆分为用∑来累加各个训练样本的真实标签与网络输出的误差的平方。

## 2.3梯度下降法

### 2.3.1 梯度下降法(Gradient Descent)介绍
在求解机器学习算法的模型参数时，梯度下降法是最常用的方法之一。在讲解梯度下降法之前我们先来了解一下导数（derivative）、偏导数（partial derivative）、方向导数（directional derivative）和梯度(gradient)的概念。

导数 —— 导数的概念就如图2.1所示：

![导数](https://github.com/Joker2770/My_TensorFlow_Lab/blob/master/Resource/导数.jpg)

_图2.3.1 导数_ <br>

导数的定义如下：

![导数的定义](https://github.com/Joker2770/My_TensorFlow_Lab/blob/master/Resource/导数的定义.jpg)

总的来说，导数反映的是函数 y = f(x) 在x轴上某一点处沿x轴正方向的变化率/变化趋势。也就是在x轴上的某一点初，如果f '(x)>0，说明f(x)的函数值在x点沿x轴正方向是趋向于增加的；
如果f '(x)<0，说明f(x)的函数值在x点沿x轴正方向是趋向于减小的。

偏导数 —— 偏导数的定义如下：

![偏导数的定义](https://github.com/Joker2770/My_TensorFlow_Lab/blob/master/Resource/偏导数的定义.jpg)

可以看到，导数与偏导数本质是一致的，都是当自变量的变化量趋近于0时，函数值的变化量与自变量变化量比值的极限。直观地说，
偏导数也就是函数在某一点上沿坐标轴正方向的的变化率。

区别在于： 　
导数，指的是一元函数中，函数y = f(x)在某一点处沿x轴正方向的变化率； 　
偏导数，指的是多元函数中，函数

![多元函数](https://github.com/Joker2770/My_TensorFlow_Lab/blob/master/Resource/多元函数.jpg)

在某一点处沿某一坐标轴

![某一坐标轴](https://github.com/Joker2770/My_TensorFlow_Lab/blob/master/Resource/某一坐标轴.jpg)

正方向的变化率。

方向导数 —— 方向导数的定义如下：

![方向导数的定义](https://github.com/Joker2770/My_TensorFlow_Lab/blob/master/Resource/方向导数的定义.jpg)

其中

![某个方向](https://github.com/Joker2770/My_TensorFlow_Lab/blob/master/Resource/某个方向.jpg)

表示某个方向

在前面导数和偏导数的定义中，均是沿坐标轴正方向讨论函数的变化率。那么当我们讨论函数沿任意方向的变化率时，也就引出了方向导数的定义，即：某一点在某一趋近方向上的导数值。

通俗的解释是： 　
我们不仅要知道函数在坐标轴正方向上的变化率（即偏导数），而且还要设法求得函数在其他特定方向上的变化率。而方向导数就是函数在其他特定方向上的变化率。

梯度 —— 梯度的定义如下：

![梯度的定义](https://github.com/Joker2770/My_TensorFlow_Lab/blob/master/Resource/梯度的定义.jpg)

对于

![多元函数1](https://github.com/Joker2770/My_TensorFlow_Lab/blob/master/Resource/多元函数1.jpg)

上的某一点来说存在很多个方向导数，梯度的方向是函数在某一点增长最快的方向，梯度的模则是该点上方向导数的最大值，梯度的模等于：

![梯度的模](https://github.com/Joker2770/My_TensorFlow_Lab/blob/master/Resource/梯度的模.jpg)

这里注意三点：
梯度是一个向量，即有方向有大小
梯度的方向是最大方向导数的方向
梯度的值是最大方向导数的值

梯度下降法—— 既然在变量空间的某一点处，函数沿梯度方向具有最大的变化率，那么在优化代价函数的时候，就可以沿着负梯度方向去减小代价函数的值。计算过程可以描述如下：

![梯度下降法](https://github.com/Joker2770/My_TensorFlow_Lab/blob/master/Resource/梯度下降法.jpg)

Repeat表示不断重复

![梯度下降参数调整](https://github.com/Joker2770/My_TensorFlow_Lab/blob/master/Resource/梯度下降参数调整.jpg)

表示参数调整 <br>
***η*** 表示学习率。

### 2.3.2 梯度下降法(Gradient Descent)二维例子

2.2中我们已经知道了代价函数的定义，代价函数的值越小，说明模型的预测值越接近真实标签的值。代价函数中的预测值y是跟神经网络中的参数w和b相关的。
我们可以先考虑一个简单的情况，假如神经网络只有一个参数w，参数w与代价函数loss的关系如图2.2所示：

![参数w与代价函数loss的关系图](https://github.com/Joker2770/My_TensorFlow_Lab/blob/master/Resource/参数w与代价函数loss的关系图.jpg)

_图2.1 参数w与代价函数loss的关系图_ <br>

假设w的初始值是-3，我们需要使用梯度下降法来不断优化w的取值，使得loss值不断减少，首先我们应该先计算w=-3时的梯度，如图2.3所示：

![w为-3时的导数](https://github.com/Joker2770/My_TensorFlow_Lab/blob/master/Resource/w为-3时的导数.jpg)

_图2.3 w为-3时的导数_ <br>

从图2.3中我们可以看出，当w为-3时，w所处位置的梯度应该是一个负数，梯度下降法在优化代价函数的时候，
是沿着负梯度方向去减小代价函数的值，所以负梯度是一个正数，w的值应该变大。根据梯度下降法的优化公式：

![梯度下降法的优化公式](https://github.com/Joker2770/My_TensorFlow_Lab/blob/master/Resource/梯度下降法的优化公式.jpg)

学习率η一般是一个大于0的数，∂f/∂w为负数，我们可以判断出w的值会变大。变大的数值跟学习率大小η有关，也跟函数f在w处的梯度大小有关。

假设w变大移动到了w=2的位置，我们需要再次计算w=2时的梯度，如图2.4所示： 

![w为2时的导数](https://github.com/Joker2770/My_TensorFlow_Lab/blob/master/Resource/w为2时的导数.jpg)

_图2.4  w为2时的导数_ <br>

从图2.4中我们可以看出，当w为2时，w所处位置的梯度应该是一个正数，梯度下降法在优化代价函数的时候，是沿着负梯度方向去减小代价函数的值，
所以负梯度是一个负数，w的值应该变小。根据梯度下降法的优化公式2.7。

学习率η一般是一个大于0的数，∂f/∂w为正数，我们可以判断出w的值会变小。变小的数值跟学习率大小η有关，也跟函数f在w处的梯度大小有关。

我们可以发现不管w处于那一个位置，当w向着负梯度的方向进行移动时，实际上就是向着可以使loss值减小的方向进行移动。这就有点类似一个小球在山坡上面，
它总是往坡底的方向进行移动，只不过它每一次是移动一步，这个步子的大小会受到学习率和所处位置梯度的大小所影响。

### 2.3.3 梯度下降法(Gradient Descent)三维例子

我们可以再考虑一个稍微复杂一点点的情况，假如神经网络有两个参数w1和w2，参数w1和w2与代价函数loss的关系如图2.5所示： 

![w1和w2与loss的关系图](https://github.com/Joker2770/My_TensorFlow_Lab/blob/master/Resource/w1和w2与loss的关系图.jpg)

_图2.5 w1和w2与loss的关系图_ <br>

我们在图中随机选取两个w1和w2的初始值p1和p2，然后从p1,p2这两个初始位置开始使用梯度下降法优化网络参数，得到如图2.6中的结果：

![从p1p2初始点开始优化网络](https://github.com/Joker2770/My_TensorFlow_Lab/blob/master/Resource/从p1p2初始点开始优化网络.jpg)

_图2.6 从p1,p2初始点开始优化网络_ <br>

图2.6中可以看到网络参数的优化过程其实就是p1,p2两个“小球“从初始点开始，每次移动一步，不断向坡地进行移动。在这个过程中整个网络的loss值是在不断变小的。

同时我们还可以观察到一个现象，p1“小球“最后走到了图中的全局最小值(global minimum)，而p2“小球”最后走到的位置是一个局部极小值(local minimum)。
说明我们在使用梯度下降法的时候不同的初始值的选取可能会影响到最后的结果，有些时候我们可以得到loss的全局最小值，或者称为全局最优解。
而有些时候我们得到的结果可能是loss的局部极小值，或者称为局部最优解。这算是梯度下降法存在的一个问题。

## 2.4 常用激活函数讲解

神经网络的激活函数其实有很多种，在前面的章节中我们介绍过两种激活函数，sign函数和purelin函数。sign函数也称为符号函数，因为sign(x)中x＞0，
函数结果为1；sign(x)中x＜0，函数结果为-1。purelin函数也称为线性函数，表达式为y=x。这两种激活函数在处理复杂非线性问题的时候都不能得到很好的结果，
线性函数的分类边界也是线性的，所以不能区别非线性的复杂边界，比如一条直线不能区分异或问题的两个类别。下面我们介绍几个在BP神经网络中常用的非线性激活函数，
sigmoid函数，tanh函数，softsign函数和ReLU函数，使用这些非线性激活函数可以帮助我们解决复杂的非线性问题。

### 2.4.1 sigmoid函数

sigmoid函数 —— sigmoid函数也称为逻辑函数(logical function)，函数的公式为：

![sigmoid函数公式](https://github.com/Joker2770/My_TensorFlow_Lab/blob/master/Resource/sigmoid函数公式.jpg)

函数图像如图2.7所示：

![sigmoid函数图像](https://github.com/Joker2770/My_TensorFlow_Lab/blob/master/Resource/sigmoid函数图像.png)

_图2.7 sigmoid函数图像_ <br>

图中我们可以看出函数的取值范围是0-1之间，当x趋向于-∞的时候函数值趋向于0；当x趋向于+∞的时候函数值趋向于1。

### 2.4.2 tanh函数

tanh函数 —— tanh函数也称为双曲正切函数，函数的公式为： 

![双曲正切函数公式](https://github.com/Joker2770/My_TensorFlow_Lab/blob/master/Resource/双曲正切函数公式.jpg)

函数图像如图2.8所示： 

![tanh函数图像](https://github.com/Joker2770/My_TensorFlow_Lab/blob/master/Resource/tanh函数图像.png)

_图2.8 tanh函数图像_ <br>

图中我们可以看出函数的取值范围是-1-1之间，当x趋向于-∞的时候函数值趋向于-1；当x趋向于+∞的时候函数值趋向于1。

### 2.4.3 softsign函数

softsign函数 —— softsign函数的公式为：

![softsign函数的公式](https://github.com/Joker2770/My_TensorFlow_Lab/blob/master/Resource/softsign函数的公式.jpg)

函数图像如图2.9所示：

![softsign函数图像](https://github.com/Joker2770/My_TensorFlow_Lab/blob/master/Resource/softsign函数图像.jpg)

_图2.9 softsign函数图像_ <br>

图中我们可以看出函数的取值范围是-1-1之间，当x趋向于-∞的时候函数值趋向于-1；当x趋向于+∞的时候函数值趋向于1。

我们可以通过图2.10对比一下这三种函数的区别：

![三种激活函数的区别](https://github.com/Joker2770/My_TensorFlow_Lab/blob/master/Resource/三种激活函数的区别.jpg)

_图2.10 三种激活函数的区别_ <br>

它们这三个激活函数都是S形函数，形状相似，只不过sigmoid函数取值范围是0-1之间，tanh函数和softsign函数取值范围是-1-1之间。
我们还可以观察到softsign函数相对于tanh函数而言过渡更加平滑，在x等于0附近函数的数值改变更缓慢。

### 2.4.4ReLU函数

ReLU函数—— ReLU(The Rectified Linear Unit)函数的公式为 

![ReLU函数公式](https://github.com/Joker2770/My_TensorFlow_Lab/blob/master/Resource/ReLU函数公式.jpg)

函数图像如图2.11所示： 

![ReLU函数图像](https://github.com/Joker2770/My_TensorFlow_Lab/blob/master/Resource/ReLU函数图像.jpg)

_图2.11 ReLU函数图像_ <br>

当x小于0时，y等于0。当x大于0时，y等于x。ReLU的中文名称是校正线性单元，虽然在x小于0时函数是线性的，x大于0时函数也是线性的，
但是组合起来之后，函数就具有了非线性的特征。这种非线性的特征是怎么体现的呢，我们可以观察一下下面的一系列图片，首先看到图2.12： 

![使用tanh作为激活函数的分类边界](https://github.com/Joker2770/My_TensorFlow_Lab/blob/master/Resource/使用tanh作为激活函数的分类边界.jpg)

_图2.12 使用tanh作为激活函数的分类边界_ <br>

图2.12使用的是tanh作为激活函数训练出来的分类模型，其实使用sigmoid或者softsign函数也可以得到类似结果。我使用了带有4个隐藏层的神经网训练了出了这个模型，
图中有两个类别的数据，并且我们可以观察到一个类似椭圆形的分类边界把两个类别给区分开了。我们再观察图2.13：

![使用ReLu作为激活函数的分类边界](https://github.com/Joker2770/My_TensorFlow_Lab/blob/master/Resource/使用ReLu作为激活函数的分类边界.jpg)

_图2.13 使用ReLu作为激活函数的分类边界_ <br>

我使用带有4个隐藏层的神经网络训练出了这个模型。我们发现使用ReLU激活函数得到的分类边界跟使用tanh激活函数得到分类边界是差不多的，并不能看出ReLU函数的特点。
同样的一个学习任务和数据，我改变了神经网络的层数，只使用2个隐藏层，依然使用ReLU激活函数得到了图2.14的结果：

![使用ReLU作为激活函数的分类边界1](https://github.com/Joker2770/My_TensorFlow_Lab/blob/master/Resource/使用ReLU作为激活函数的分类边界1.jpg)

_图2.14 使用ReLU作为激活函数的分类边界

我们观察图2.14可以得到一些结论：

* 我们可以发现ReLU激活函数所描绘出来的边界其实是一条一条的直线构成的，不存在曲线。图2.13中的边界看起来像一个椭圆，实际上它也有有一段一段很小的直线构成的。

* 神经网络的层数会影响模型的拟合效果，层数越多，模型就可以拟合出更复杂的分类边界。

模型的拟合效果其实还跟其他一些因素相关，比如说每一层隐藏层的神经元越多，那么模型的拟合能力也就越强。模型训练的周期越多，模型的拟合能力就越强。
关于模型拟合强弱的问题，再后面的章节中我们还会进一步讨论。

另外我们再来看一下ReLU应用于回归预测时的特点，我看一下图2.15和图2.16：

![使用tanh激活函数训练的回归模型](https://github.com/Joker2770/My_TensorFlow_Lab/blob/master/Resource/使用tanh激活函数训练的回归模型.jpg)

_图2.15 使用tanh激活函数训练的回归模型_ <br>

![使用ReLU激活函数训练的回归模型](https://github.com/Joker2770/My_TensorFlow_Lab/blob/master/Resource/使用ReLU激活函数训练的回归模型.jpg)

_图2.16 使用ReLU激活函数训练的回归模型_ <br>

我们发现了跟分类中类似的情况，tanh激活函数得到的回归线是一条曲线，而ReLU激活函数得到的是由一段一段直线构成的回归线。

## 2.5 BP网络模型和公式推导

### 2.5.1BP网络模型

假设我们有一个2层(统计神经网络层数的时候一般输入层忽略不计)的神经网络如图2.17所示：

![BP神经网络](https://github.com/Joker2770/My_TensorFlow_Lab/blob/master/Resource/BP神经网络.jpg)

_图2.17 BP神经网络_ <br>

该网络的输入向量为X = （x1，x2，...，xi，...，xn），图中x0 = 1表示输入层偏置值。隐藏层输出向量为Y1 = (y11，y12，...，y1j，...，y1m)，图中y10 = 1表示隐藏层偏置值。
输出层输出向量为Y2 = (y21，y22，...，y2k，...，y2l)。期望输出T = （t1，t2，...，tk，...，tl）。输入层到隐藏层之间的权值用矩阵W1表示，W1ij表示W1矩阵中第i行第j列的权值。
隐藏层到输出层之间的权值用矩阵W2表示，W2jk表示W2矩阵中第j行第k列的权值。另外我们定义net1为隐藏层中权值乘以输入层信号的总和，
net1j表示隐藏层中第j个神经元得到的输入信号总和。net2为输出层中权值乘以隐藏层信号的总和，net2k表示输出层中第k个神经元得到的输入信号总和。

对于隐藏层有： 

![BP隐藏层](https://github.com/Joker2770/My_TensorFlow_Lab/blob/master/Resource/BP隐藏层.jpg)

对于输出层有：

![BP输出层](https://github.com/Joker2770/My_TensorFlow_Lab/blob/master/Resource/BP输出层.jpg)

公式2.13和2.15中的激活函数假设我们都使用sigmoid函数，sigmoid函数的公式在上文中的公式2.8。sigmoid函数具有连续、可导的特点，它的导数为：

![sigmoid函数导数](https://github.com/Joker2770/My_TensorFlow_Lab/blob/master/Resource/sigmoid函数导数.jpg)

### 2.5.2BP算法推导

根据上文中提到的代价函数，当网络输出与期望输出不同时，会存在输出误差E，为了简单我们只计算一个样本的均方差公式，如果是计算多个样本可以求所有样本代价函数的平均值。
一个样本的均方差公式定义如下： 

![一个样本的均方差公式](https://github.com/Joker2770/My_TensorFlow_Lab/blob/master/Resource/一个样本的均方差公式.jpg)

将以上误差定义式展开至隐藏层： 

![误差定义式展开至隐藏层](https://github.com/Joker2770/My_TensorFlow_Lab/blob/master/Resource/误差定义式展开至隐藏层.jpg)

再进一步展开至输入层： 

![进一步展开至输入层](https://github.com/Joker2770/My_TensorFlow_Lab/blob/master/Resource/进一步展开至输入层.jpg)

从公式2.18和2.19中可以看出，网络的误差E是跟神经网络各层权值W1ij和W2jk相关的，因此调整各层的权值，就可以改变误差E的值。我们的目标就是要得到比较小的误差值，
所以我们可以采用梯度下降法来最小化误差E的值。根据梯度下降法，我们可以得到： 

![采用梯度下降法来最小化误差E的值](https://github.com/Joker2770/My_TensorFlow_Lab/blob/master/Resource/采用梯度下降法来最小化误差E的值.jpg)

在下面的推导过程中均默认对于隐藏层有i =0，1，2，...，n；j = 1，2，...，m；对于输出层有：j =0，1，2，...，m；k = 1，2，...，l。
根据微积分的链式法则可以得到，对于隐藏层有： 

![对于隐藏层微积分的链式法则](https://github.com/Joker2770/My_TensorFlow_Lab/blob/master/Resource/对于隐藏层微积分的链式法则.jpg)

根据微积分的链式法则可以得到，对于输出层有： 

![对于输出层微积分的链式法则](https://github.com/Joker2770/My_TensorFlow_Lab/blob/master/Resource/对于输出层微积分的链式法则.jpg)

我们可以定义一个误差信号，命名为δ(delta)，令： 

![误差信号δ(delta)](https://github.com/Joker2770/My_TensorFlow_Lab/blob/master/Resource/误差信号δ(delta).jpg)

综合公式2.12,2.22,2.24，可以得到输入层到隐藏层的权值调整公式为：

![输入层到隐藏层的权值调整公式](https://github.com/Joker2770/My_TensorFlow_Lab/blob/master/Resource/输入层到隐藏层的权值调整公式.jpg)

综合公式2.14,2.23,2.25，可以得到隐藏层到输出层的权值调整公式为： 

![隐藏层到输出层的权值调整公式](https://github.com/Joker2770/My_TensorFlow_Lab/blob/master/Resource/隐藏层到输出层的权值调整公式.jpg)

可以看出在公式2.26和2.27中，只要求出δ1j和δ2k的值，就可以计算出Δw1ij和Δw2jk的值了。

对于隐藏层，δ1j可以展开为： 

![对于隐藏层δ1j展开](https://github.com/Joker2770/My_TensorFlow_Lab/blob/master/Resource/对于隐藏层δ1j展开.jpg)

对于输出层，δ2k可以展开为：

![对于输出层δ2k展开](https://github.com/Joker2770/My_TensorFlow_Lab/blob/master/Resource/对于输出层δ2k展开.jpg)

在公式2.28和2.29中，求网络误差对各层输出的偏导，对于输出层：

![网络误差对输出层输出的偏导](https://github.com/Joker2770/My_TensorFlow_Lab/blob/master/Resource/网络误差对输出层输出的偏导.jpg)

对于隐藏层：

![网络误差对隐藏层输出的偏导](https://github.com/Joker2770/My_TensorFlow_Lab/blob/master/Resource/网络误差对隐藏层输出的偏导.jpg)

将公式2.30带入公式2.29，再根据sigmoid函数的求导公式2.16，可以得到： 

![根据sigmoid函数的求导公式得到](https://github.com/Joker2770/My_TensorFlow_Lab/blob/master/Resource/根据sigmoid函数的求导公式得到.jpg)

将公式2.32带入2.27中，得到隐藏层到输出层权值调整： 

![隐藏层到输出层权值调整](https://github.com/Joker2770/My_TensorFlow_Lab/blob/master/Resource/隐藏层到输出层权值调整.jpg)

将公式2.33带入2.26中，得到输入层到隐藏层权值调整：

![输入层到隐藏层权值调整](https://github.com/Joker2770/My_TensorFlow_Lab/blob/master/Resource/输入层到隐藏层权值调整.jpg)

对于一个多层的神经网络，假设一共有h个隐藏层，按顺序将各隐藏层节点数分别记为：m1，m2，...，mh，输入神经元个数为n，输出神经元个数为l；
各隐藏层输出分别记为：Y1，Y2，...，Yh，输入层的输入记为：X，输出层的输出记为：Yh+1；各层权值矩阵分别记为：W1，W2，...，Wh+1，
W1表示输入层到一个隐藏层的权值矩阵，Wh+1表示最后一个隐藏层到输出层的权值矩阵；各层学习信号分别记为：δ1，δ2，...，δh+1，δh+1表示输出层计算出的学习信号；
则各层权值调整计算公式为：

对于输出层：

![输出层权值调整计算公式](https://github.com/Joker2770/My_TensorFlow_Lab/blob/master/Resource/输出层权值调整计算公式.jpg)

对于第h隐藏层：

![第h隐藏层权值调整计算公式](https://github.com/Joker2770/My_TensorFlow_Lab/blob/master/Resource/第h隐藏层权值调整计算公式.jpg)

按照以上规律逐层类推，则第一个隐藏层的权值调整公式为：

![第一个隐藏层的权值调整公式](https://github.com/Joker2770/My_TensorFlow_Lab/blob/master/Resource/第一个隐藏层的权值调整公式.jpg)

### 2.5.3BP算法推导的补充说明

我们已经从头到尾详细推导了一遍BP算法的整个流程，在这一小节中对BP算法再做两点补充说明：

#### 1.网络的偏置值。

在上文中我们的推导过程一直是使用权值w来进行计算的，如果我们把偏置值独立出来，那么偏置值的参数应该怎么调整呢？

我们可以看到公式2.26以及2.27，在公式2.26中，把i的取值设置为0，并且我们知道x0 = 1，所以我们可以得到： 

![239](https://github.com/Joker2770/My_TensorFlow_Lab/blob/master/Resource/239.jpg)

在公式2.26中，把j的取值设置为0，并且我们知道y0 = 1，所以我们可以得到：

![240](https://github.com/Joker2770/My_TensorFlow_Lab/blob/master/Resource/240.jpg)

如果是把偏置值单独拿出来计算的话就是公式2.39和2.40的表达式。

#### 2.用矩阵形式来表达BP学习算法。

下面我们直接给出BP学习算法矩阵表达形式的结果，具体推导过程跟上文中的推导过程类似，不过会涉及到矩阵求导的相关知识，大家有兴趣的话可以自己推导一下。
如果是把BP学习算法写成矩阵的形式来表达，假设一共有h个隐藏层。输入数据的矩阵为X，X中的每一行表示一个数据，列表示数据的特征。比如我们一次性输入3个数据，
每个数据有4个特征，那么X就是一个3行4列的矩阵。

各隐藏层输出分别记为：Y1，Y2，...，Yh，输出层的输出记为：Yh+1。Y中的每一个行表示一个数据的标签。比如我们有3个数据，每个数据有1个标签，那么Y就是一个3行1列的矩阵。

各层权值矩阵分别记为：W1，W2，...，Wh+1，W1表示输入层到一个隐藏层的权值矩阵，Wh+1表示最后一个隐藏层到输出层的权值矩阵。权值矩阵的行等于前一层的神经元个数，
权值矩阵的列对应于后一层的神经元个数。比如在输入层和第一个隐藏层之间的权值矩阵是W1，输入层有3个神经元，第一个隐藏层有10个神经元，那么W1就是一个3行10列的矩阵。

各层学习信号分别记为：δ1，δ2，...，δh+1，δh+1表示输出层计算出的学习信号。

对于输出层的学习信号δh+1： 

![对于输出层的学习信号](https://github.com/Joker2770/My_TensorFlow_Lab/blob/master/Resource/对于输出层的学习信号.jpg)

公式2.41中的“ 。”符号是element-wise multiplication，意思是矩阵中的元素对应相乘。例如下面的例子： 

![矩阵中的元素对应相乘](https://github.com/Joker2770/My_TensorFlow_Lab/blob/master/Resource/矩阵中的元素对应相乘.jpg)

对于第h隐藏层的学习信号δh： 

![对于第h隐藏层的学习信号δh](https://github.com/Joker2770/My_TensorFlow_Lab/blob/master/Resource/对于第h隐藏层的学习信号δh.jpg)

对于第1隐藏层的学习信号δ1：

![对于第1隐藏层的学习信号δ1](https://github.com/Joker2770/My_TensorFlow_Lab/blob/master/Resource/对于第1隐藏层的学习信号δ1.jpg)

对于输出层的权值矩阵Wh+1：

![对于输出层的权值矩阵](https://github.com/Joker2770/My_TensorFlow_Lab/blob/master/Resource/对于输出层的权值矩阵.jpg)

对于第h隐藏层权值矩阵Wh： 

![对于第h隐藏层权值矩阵Wh](https://github.com/Joker2770/My_TensorFlow_Lab/blob/master/Resource/对于第h隐藏层权值矩阵Wh.jpg)

对于第1隐藏层权值矩阵W1：

![对于输出层的权值矩阵W1](https://github.com/Joker2770/My_TensorFlow_Lab/blob/master/Resource/对于输出层的权值矩阵W1.jpg)

## 2.6 梯度消失与梯度爆炸

### 2.6.1 梯度消失

根据上文BP算法中的推导，我们从公式2.44,2.45,2.46中可以知道，权值的调整ΔW是跟学习信号δ相关的。同时我们从2.41,2.42,2.43中可以知道在学习信号δ 表达式中存在f ' (x)。也就是说激活函数的导数会影响学习信号δ的值，而学习信号δ的值会影响权值调整ΔW的值。那么激活函数的值越大，ΔW的值就越大；激活函数的值越小，ΔW的值也就越小。

假设激活函数为sigmoid函数，前文中我们已经知道了sigmoid函数的表达式为： 

![sigmoid函数公式](https://github.com/Joker2770/My_TensorFlow_Lab/blob/master/Resource/sigmoid函数公式.jpg)

，sigmoid函数的导数为：f ' (x) = f(x)[1-f(x)]，我们可以画出sigmoid函数的导数图像为图2.18： 

![sigmoid函数的导数图像](https://github.com/Joker2770/My_TensorFlow_Lab/blob/master/Resource/sigmoid函数的导数图像.jpg)

_图2.18 sigmoid函数导数_ <br>

这里我们发现当x=0时，sigmoid函数导数可以取得最大值0.25。x取值较大或较小时，sigmoid函数的导数很快就趋向于0。不管怎么样，sigmoid函数的导数都是一个小于1的数，
学习信号δ乘以一个小于1的数，那么δ就会减小。学习信号从输出层一层一层向前反向传播的时候，每传播一层学习信号就会变小一点，经过多层传播后，学习信号就会接近于0，
从而使得权值ΔW调整接近于0。ΔW接近于0那就意味着该层的参数不会发生改变，不能进行优化。参数不能优化，那整个网络就不能再进行学习了。
学习信号随着网络传播逐渐减小的问题也被称为梯度消失(vanishing gradient)的问题。

我们再考虑一下tanh函数的导数，tanh函数的表达式为：

![双曲正切函数公式](https://github.com/Joker2770/My_TensorFlow_Lab/blob/master/Resource/双曲正切函数公式.jpg)

tanh函数的导数为： f ' (x) = 1- (f(x))2，tanh函数的导数如图2.19：

![tanh函数导数图像](https://github.com/Joker2770/My_TensorFlow_Lab/blob/master/Resource/tanh函数导数图像.jpg) 

_图2.19 tanh函数导数_ <br>

tanh函数导数图像看起来比sigmoid函数要好一些，x=0时，tanh函数导数可以取得最大值1。x取值较大或较小时，tanh函数的导数很快就趋向于0。不管怎么样，
tanh函数导数的取值总是小于1的，所以tanh作为激活函数也会存在梯度消失的问题。

对于softsign函数，softsign函数的表达式为：

![softsign函数的公式](https://github.com/Joker2770/My_TensorFlow_Lab/blob/master/Resource/softsign函数的公式.jpg)

softsign函数的导数为：

![softsign函数的导数](https://github.com/Joker2770/My_TensorFlow_Lab/blob/master/Resource/softsign函数的导数.jpg)

softsign函数的导数如图2.20： 

![softsign函数导数图像](https://github.com/Joker2770/My_TensorFlow_Lab/blob/master/Resource/softsign函数导数图像.jpg)

_图2.20 softsign函数导数_ <br>

softsign函数x=0时，softsign函数导数可以取得最大值1。x取值较大或较小时，softsign函数的导数很快就趋向于0。不管怎么样，softsign函数导数的取值总是小于1的，
所以softsign作为激活函数也会存在梯度消失的问题。

### 2.6.2 梯度爆炸

当我们使用sigmoid,tanh和softsign作为激活函数时，它们的导数取值范围都是小于等于1的，所以会产生梯度消失的问题。那么我们可能会想到，
如果使用导数大于1的函数作为激活函数，情况会如何？

如果学习信号δ乘以一个大于1的数，那么δ就会变大。学习信号从输出层一层一层向前反向传播的时候，每传播一层学习信号就会变大一点，经过多层传播后，
学习信号就会接近于无穷大，从而使得权值ΔW调整接近于无穷大。ΔW接近于无穷大那就意味着该层的参数，处于一种极不稳定的状态，那么网络就不能正常工作了。
学习信号随着网络传播逐渐增大的问题也被称为梯度爆炸(exploding gradient)的问题。

### 2.6.3 使用ReLU函数解决梯度消失和梯度爆炸的问题

我们知道ReLU的表达式为：f(x) = max(0,x)。当x小于0时，f(x)的取值为0；当x大于0时，f(x)的取值等于x。ReLU函数的导数如图2.21： 

![ReLU函数的导数图像](https://github.com/Joker2770/My_TensorFlow_Lab/blob/master/Resource/ReLU函数的导数图像.jpg)

_图2.21 ReLU函数导数_ <br>

前面我们讨论了当激活函数的导数小于1时，网络会产生梯度消失，激活函数的导数大于1时，网络会产生梯度爆炸。那么当我们使用ReLU作为激活函数的时候，x小于0时，
ReLU的导数为0；x大于0时，ReLU的导数为1。导数为1是一个很好的特性，不会使得学习信号越来越小，也不会让学习信号越来越大，可以让学习信号比较稳定地从后向前传播。
解决了梯度消失和梯度爆炸的问题，同时计算方便，可以加速网络的训练。

ReLU也存在缺点，由于x小于0的部分f(x)的取值恒定为0，会导致一些神经元无法激活。

不过总的来说，ReLU还是优点大于缺点的，所以在一些比较深层的神经网络中，通常都会使用ReLU作为神经网络的激活函数。

除此之外可能我们还会想到线性激活函数y = x 的导数也是等于1，也可以克服梯度消失和梯度爆炸的问题。不过之前我们也多次提到线性激活函数不会能描绘非线性的分类边界，
所以无法处理非线性问题。因此线性函数依旧不算是一个好的激活函数。 

![loss图像](https://github.com/Joker2770/My_TensorFlow_Lab/blob/master/Resource/loss图像.jpg)

![accuracy图像](https://github.com/Joker2770/My_TensorFlow_Lab/blob/master/Resource/accuracy图像.jpg)

## 2.7 使用BP神经网络解决异或问题

代码[_07_BPXOR.py](https://github.com/Joker2770/My_TensorFlow_Lab/blob/master/src/_07_BPXOR.py)：BP神经网络解决异或问题

程序的输出结果为：
~~~
epochs 0 loss 0.21914938776385912
epochs 5000 loss 0.11848943046512897
epochs 10000 loss 0.07584832296969335
epochs 15000 loss 0.02081529755970813
epochs 20000 loss 0.006791104652688612
epochs 25000 loss 0.00347537604347784
epochs 30000 loss 0.002214137161812754
epochs 35000 loss 0.0015846969612142681
epochs 40000 loss 0.0012168649734302002
epochs 45000 loss 0.000979035374693999
epochs 50000 loss 0.0008141367915345103
epochs 55000 loss 0.0006938273165481596
epochs 60000 loss 0.0006025823118787628
epochs 65000 loss 0.0005312435415994395
epochs 70000 loss 0.00047408582121212625
epochs 75000 loss 0.0004273593960833713
epochs 80000 loss 0.00038851258530058115
epochs 85000 loss 0.00035575261105233317
epochs 90000 loss 0.0003277853028390947
epochs 95000 loss 0.0003036544434920989
epochs 100000 loss 0.00028263911503724915
~~~

![BPXOR](https://github.com/Joker2770/My_TensorFlow_Lab/blob/master/Resource/BPXOR.png)

预测结果：
~~~
output
[[0.02169402]
 [0.97973978]
 [0.97647214]
 [0.02405968]]
0
1
1
0
~~~


# 二、一元二次方程([_08_OnePowerDistance.py](https://github.com/Joker2770/My_TensorFlow_Lab/blob/master/src/_07_OnePowerDistance.py))

# 三、mnist数据集([mnist](https://github.com/Joker2770/My_TensorFlow_Lab/blob/master/src/mnist))

# 四、fashion_mnist数据集([fashion_mnist](https://github.com/Joker2770/My_TensorFlow_Lab/blob/master/src/fashion_mnist))

