Tensorflow implementations with python
=

# 1、从神经细胞到神经网络

现在人们热议的神经网络算法结构实质是一种仿生技术，灵感来源于神经细胞。神经细胞构是构成神经系统的基本单元，简称为神经元。神经元主要由三部分构成：①细胞体；②轴突；③树突。如下图所示

![神经元结构](https://github.com/Joker2770/My_TensorFlow_Lab/blob/master/Resource/nerve_cell.jpg)

## 1.1单层感知器

### 1.1.1单层感知器介绍

受到生物神经网络的启发，计算机学家Frank Rosenblatt在20世纪60年代提出了一种模拟生物神经网络的的人工神经网络结构，称为感知器(Perceptron)。单层感知器结构与神经元结构一致。

![单层感知器结构](https://github.com/Joker2770/My_TensorFlow_Lab/blob/master/Resource/单层感知器表示.png)

图中x1，x2，x3为输入信号，类似于生物神经网络中的树突。
w1，w2，w3分别为x1，x2，x3的权值，它可以调节输入信号的值的大小，让输入信号变大(w＞0)，不变(w=0)或者减小(w＜0)。可以理解为生物神经网络中的输入阻碍，树突的信号不一定是100%传递到细胞核的，可能某些树突的信号有80%传递到细胞核，某些树突的信号有50%传递到细胞核，某些树突的信号被完全阻碍。
公式表示细胞的输入信号在细胞核的位置进行汇总，然后再加上该细胞本身自带的信号b，b一般称为偏置值(Bias)。
f(x)称为激活函数，可以理解为信号在轴突上进行的变化。在单层感知器中使用的激活函数是sign(x)激活函数。该函数的特点是当x＞0时，输出值为1；当x＝0时，输出值为0,；当x＜0时，输出值为-1。sign(x)函数图像如图

![sign函数图像](https://github.com/Joker2770/My_TensorFlow_Lab/blob/master/Resource/sign.jpg)

y就是，为单层感知器的输出结果。

### 1.1.2单层感知器计算举例

假如有一个单层感知器有3个输入x1,x2,x3，同时已知b=-0.6，w1=w2=w3=0.5，那么根据单层感知器的计算公式我们就可以得到如图计算结果。

![单层感知器计算结果](https://github.com/Joker2770/My_TensorFlow_Lab/blob/master/Resource/单层感知器计算结果.png)

![单层感知器计算过程](https://github.com/Joker2770/My_TensorFlow_Lab/blob/master/Resource/单层感知器计算过程.png)

### 1.1.3单层感知器的另一种表达形式

单层感知器的另一种表达形式如图

![单层感知器另一种表示](https://github.com/Joker2770/My_TensorFlow_Lab/blob/master/Resource/单层感知器另一种表示.jpg)

其实这种表达形式跟上面的单层感知器是一样的。只不过是把偏置值b变成了输入w0×x0，其中x0=1。所以w0×x0实际上就是w0，把公式展开得到：w1×x1+w2×x2+w3×x3+w0。所以这两个单层感知器的表达不一样，但是计算结果是一样的。此种表达形式更加简洁，更适合使用矩阵来进行运算。

### 1.1.4单层感知器的学习规则 

在1.1.3中我们已知单层感知器表达式可以写成：



# 2、一元二次方程(OnePowerDistance.py)

