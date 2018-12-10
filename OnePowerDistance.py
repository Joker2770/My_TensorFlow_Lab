#-*-coding:utf-8-*-
#date:2018/12/10

#在这个例子中，我们构造一 个满足一元二次函数y = ax 2 +b 的原始数据，然后构建一个最简单的神经网络，
#仅包含一个输 入层、一个隐藏层和一个输出层。通过TensorFlow将隐藏层和输出层的weights 和biases 的值 学习出来，
#看看随着训练次数的增加，损失值是不是不断在减小。


#首先来生成输入数据。我们假设最后要学习的方程为y = x 2 − 0.5，我们来构造满足这个方 程的一堆x 和y ，同时加入一些不满足方程的噪声点。

import tensorflow as tf 
import numpy as np # 构造满足一元二次方程的函数 
# 为了使点更密一些，我们构建了300个点，分布在-1到1 区间，直接采用np生成等差数列的方法，
#并将结果为300个点的一维数组，转换为300×1的二维数组 
x_data = np.linspace(-1,1,300)[:, np.newaxis] 
noise = np.random.normal(0, 0.05, x_data.shape) # 加入一些噪声点，使它与x_data的维度一致，并且 拟合为均值为0、方差为0.05的正态分布 
y_data = np.square(x_data) - 0.5 + noise # y = x^2 – 0.5 + 噪声

#接下来定义x 和y 的占位符来作为将要输入神经网络的变量：
xs = tf.placeholder(tf.float32, [None, 1]) 
ys = tf.placeholder(tf.float32, [None, 1])

#这里我们需要构建一个隐藏层和一个输出层。作为神经网络中的层，输入参数应该有4个 变量：输入数据、输入数据的维度、输出数据的维度和激活函数。
#每一层经过向量化（y = weights ×x + biases ）的处理，并且经过激活函数的非线性化处理后，最终得到输出数据。
#下面来定义隐藏层和输出层，示例代码如下：
def add_layer(inputs, in_size, out_size, activation_function=None):  
	# 构建权重：in_size×out_size大小的矩阵  
	weights = tf.Variable(tf.random_normal([in_size, out_size]))   
	# 构建偏置：1×out_size的矩阵  
	biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)   
	# 矩阵相乘  
	Wx_plus_b = tf.matmul(inputs, weights) + biases   
	if activation_function is None:    
		outputs = Wx_plus_b  
	else:  
		outputs = activation_function(Wx_plus_b) 
	return outputs # 得到输出数据 
# 构建隐藏层，假设隐藏层有10个神经元 
h1 = add_layer(xs, 1, 20, activation_function=tf.nn.relu) 
# 构建输出层，假设输出层和输入层一样，有1个神经元 
prediction = add_layer(h1, 20, 1, activation_function=None)

#接下来需要构建损失函数：计算输出层的预测值和真实值间的误差，对二者差的平方求和 再取平均，得到损失函数。运用梯度下降法，以0.1的效率最小化损失：
# 计算预测值和真实值间的误差 
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1])) 
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

#我们让TensorFlow训练1000次，每50次输出训练的损失值：

init = tf.global_variables_initializer() # 初始化所有变量 
sess = tf.Session() 
sess.run(init) 
for i in range(1000): # 训练1000次  
	sess.run(train_step, feed_dict={xs: x_data, ys: y_data})  
	if i % 50 == 0: # 每50次打印出一次损失值
		print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))


#以上就是最简单的利用TensorFlow的神经网络训练一个模型的过程，目标就是要训练出 权重值来使模型拟合y = x 2 − 0.5的系数1和−0.5，
#通过损失值越来越小，可以看出训练参数越 来越逼近目标结果。按照标准的步骤，接下来应该评估模型，
#就是把学习出来的系数weights、 biase进行前向传播后和真值y = x 2 − 0.5的结果系数进行比较，根据相近程度计算准确率。这里 省略了评估过程。
