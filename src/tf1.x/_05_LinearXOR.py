#-*-coding:utf-8-*-
#date:2018/12/24

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
# 输入数据
# 4个数据分别对应0与0异或，0与1异或，1与0异或，1与1异或
X = np.array([[1,0,0],[1,0,1],[1,1,0],  [1,1,1]])
# 标签，分别对应4种异或情况的结果
# 注意这里我们使用-1作为负标签
T = np.array([[-1],[1],[1],[-1]])
# 权值初始化，3行1列
# np.random.random可以生成0-1的随机数
W = np.random.random([3,1])
# 学习率设置
lr = 0.1
# 神经网络输出
Y = 0
# 更新一次权值
def train():
	# 使用全局变量X,Y,W,lr
	global X,Y,W,lr
	# 计算网络预测值
	Y = np.dot(X,W)
	# 计算权值的改变
	delta_W =lr*(X.T.dot(T - Y)) / X.shape[0]
	# 更新权值
	W =W + delta_W
	print(W)
	print("\n")
# 训练100次
for i in range(100):
	#更新一次权值
	train()
#————————以下为画图部分————————#
# 正样本
x1 = [0,1]
y1 = [1,0]
# 负样本
x2 =[0,1]
y2 = [0,1]
# 计算分界线的斜率以及截距
k = - W[1] / W[2]
d = - W[0] / W[2]
# 设定两个点
xdata = (-2,3)
# 通过两个点来确定一条直线，用红色的线来画出分界线
plt.plot(xdata,xdata*k + d,'r')
# 用蓝色的点画出正样本
plt.scatter(x1,y1,c='b')
# 用黄色的点来画出负样本
plt.scatter(x2,y2,c='y')
# 显示图案
plt.show()