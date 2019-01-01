#-*-coding:utf-8-*-
#date:2018/12/11

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

# 导入numpy 科学计算包
import numpy as np

# 定义输入
x0 = 1
x1 = 0
x2 = -1
# 定义权值
w0 = -5
w1 = 0
w2 = 0
# 定义正确的标签
t = 1
# 定义学习率lr(learning rate)
lr = 1
# 定义偏置值
b = 0

# 循环一个比较大的次数，比如100
for i in range(100):
	# 打印权值
	print(w0,w1,w2)
	# 计算感知器的输出
	y = np.sign(w0 * x0 + w1 * x1 + w2*x2)
	# 如果感知器输出不等于正确的标签
	if(y != t):
		# 更新权值
		w0 = w0 + lr * (t-y) * x0
		w1 = w1 + lr * (t-y) * x1
		w2 = w2 + lr * (t-y) * x2
	# 如果感知器输出等于正确的标签
	else:
		# 训练结束
		print('done')
		# 退出循环
		break