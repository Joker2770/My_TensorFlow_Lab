#-*-coding:utf-8-*-
#date:2018/12/23

'''
MIT License

Copyright (c) 2018 Joker2770

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

import numpy  as  np
import  matplotlib.pyplot  as  plt
# 定义输入，我们习惯上用一行代表一个数据
X = np.array([[1,3,3],[1,4,3],[1,1,1],[1,2,1]])
# 定义标签，我们习惯上用一行表示一个数据的标签
T =np.array([[1],[1],[-1],[-1]])
# 权值初始化，3行1列
# np.random.random可以生成0-1的随机数
W = np.random.random([3,1])
# 学习率设置
lr = 0.1
# 神经网络输出
Y = 0
# 更新一次权值
def  train():
	# 使用全局变量X,Y,W,lr
	global X,Y,W,lr
	# 同时计算4个数据的预测值
	# Y的形状为(4,1)-4行1列
	Y = np.dot(X,W)
	# T - Y得到4个的标签值与预测值的误差E。形状为(4,1)
	E = T - Y
	# X.T表示X的装置矩阵，形状为(3,4)
	# 我们一共有4个数据，每个数据3个值。定义第i个数据的第j个特征值为xij
	# 如第1个数据，第2个值为x12
	# X.T.dot(T - Y)为一个3行1列的数据：
	# 第1行等于：x00×e0+x10×e1+x20×e2+x30×e3，它会调整第1个神经元对应的权值
	# 第2行等于：x01×e0+x11×e1+x21×e2+x31×e3，它会调整第2个神经元对应的权值
	# 第3行等于：x02×e0+x12×e1+x22×e2+x32×e3，它会影调整3个神经元对应的权值
	# X.shape表示X的形状X.shape[0]得到X的行数，表示有多少个数据
	# X.shape[1]得到列数，表示每个数据有多少个特征值。
	delta_W =lr*(X.T.dot(E))/X.shape[0]
	W =W + delta_W
	print(W)
	print("\n")
# 训练100次
for i in range(100):
	# 更新一次权值
	train()
#————————以下为画图部分————————#
# 正样本的xy坐标
x1 = [3,4]
y1 = [3,3]
# 负样本的xy坐标
x2 = [1,2]
y2 = [1,1]
# 计算分界线的斜率以及截距
# 因为正负样本的分界是0，所以分界线的表达式可以写成：
# w0 × x0 + w1 × x1 + w2 × x2 = 0
# 其中x0为1，我们可以把x1，x2分别看成是平面坐标系中的x和y
# 可以得到：w0 + w1×x + w2 × y = 0
# 从而推导出：y = -w0/w2 - w1×x/w2，因此可以得到
k = - W[1] / W[2]
d = - W[0] / W[2]
# 设定两个点
xdata = (0,5)
# 通过两个点来确定一条直线，用红色的线来画出分界线
plt.plot(xdata, (xdata*k+d), 'r')
# 用蓝色的点画出正样本
plt.scatter(x1,y1,c='b')
# 用黄色的点来画出负样本
plt.scatter(x2,y2,c='y')
# 显示图案
plt.show()