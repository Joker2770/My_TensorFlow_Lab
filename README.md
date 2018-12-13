Tensorflow implementations with python
=

# 1、从神经细胞到神经网络(PerceptronLearnRule.py & PerceptronLearnRule_matrix.py)

## 1.1生物神经网络

人工神经网络ANN的设计实际上是从生物体的神经网络结构获得的灵感。生物神经网络一般是指生物的大脑神经元，细胞，触电等组成的网络，用于产生生物的意识，帮助生物进行思考和行动。

神经细胞构是构成神经系统的基本单元，简称为神经元。神经元主要由三部分构成：①细胞体；②轴突；③树突。如下图所示

![神经元结构](https://github.com/Joker2770/My_TensorFlow_Lab/blob/master/Resource/nerve_cell.jpg)

## 1.2单层感知器

### 1.2.1单层感知器介绍

受到生物神经网络的启发，计算机学家Frank Rosenblatt在20世纪60年代提出了一种模拟生物神经网络的的人工神经网络结构，称为感知器(Perceptron)。单层感知器结构与神经元结构一致。

![单层感知器结构](https://github.com/Joker2770/My_TensorFlow_Lab/blob/master/Resource/单层感知器表示.png)

图中x1，x2，x3为输入信号，类似于生物神经网络中的树突。
w1，w2，w3分别为x1，x2，x3的权值，它可以调节输入信号的值的大小，让输入信号变大(w＞0)，不变(w=0)或者减小(w＜0)。可以理解为生物神经网络中的输入阻碍，树突的信号不一定是100%传递到细胞核的，可能某些树突的信号有80%传递到细胞核，某些树突的信号有50%传递到细胞核，某些树突的信号被完全阻碍。
公式表示细胞的输入信号在细胞核的位置进行汇总，然后再加上该细胞本身自带的信号b，b一般称为偏置值(Bias)。
f(x)称为激活函数，可以理解为信号在轴突上进行的变化。在单层感知器中使用的激活函数是sign(x)激活函数。该函数的特点是当x＞0时，输出值为1；当x＝0时，输出值为0,；当x＜0时，输出值为-1。sign(x)函数图像如图

![sign函数图像](https://github.com/Joker2770/My_TensorFlow_Lab/blob/master/Resource/sign.jpg)

y就是，为单层感知器的输出结果。

### 1.2.2单层感知器计算举例

假如有一个单层感知器有3个输入x1,x2,x3，同时已知b=-0.6，w1=w2=w3=0.5，那么根据单层感知器的计算公式我们就可以得到如图计算结果。

![单层感知器计算结果](https://github.com/Joker2770/My_TensorFlow_Lab/blob/master/Resource/单层感知器计算结果.png)

![单层感知器计算过程](https://github.com/Joker2770/My_TensorFlow_Lab/blob/master/Resource/单层感知器计算过程.png)

### 1.2.3单层感知器的另一种表达形式

单层感知器的另一种表达形式如图

![单层感知器另一种表示](https://github.com/Joker2770/My_TensorFlow_Lab/blob/master/Resource/单层感知器另一种表示.jpg)

其实这种表达形式跟上面的单层感知器是一样的。只不过是把偏置值b变成了输入w0×x0，其中x0=1。所以w0×x0实际上就是w0，把公式展开得到：w1×x1+w2×x2+w3×x3+w0。所以这两个单层感知器的表达不一样，但是计算结果是一样的。此种表达形式更加简洁，更适合使用矩阵来进行运算。

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

单层感知器学习规则计算，python实现代码为PerceptronLearnRule.py，结果如下：

~~~
-5 0 0
-3 0 -2
-1 0 -4
done
~~~

单层感知器学习规则计算举例(矩阵计算)，python实现代码为PerceptronLearnRule_matrix.py，结果如下

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

		**1.η取值一般取0-1之间；**  
		**2.太大容易造成权值调整不稳定；**  
		**3.学习率太小，模型参数调整太慢，迭代次数太多。**  

你可以想象一下在洗热水澡的时候：如果每次调节的幅度很大，那水温要不就是太热，要不就是太冷，很难得到一个合适的水温；
如果一开始的时候水很冷，每次调节的幅度都非常小，那么需要调节很多次，花很长时间才能得到一个合适的水温。学习率的调整也是这样一个道理。

# 2、一元二次方程(OnePowerDistance.py)

