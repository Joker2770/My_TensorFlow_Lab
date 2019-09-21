#-*-coding:utf-8-*-
#date:2019/01/01

#@title MIT License
#
# Copyright (c) 2017 François Chollet
# Copyright (c) 2018 Joker2770
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import os
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
#输出当前使用TensorFlow版本号
print(tf.__version__)

#导入和加载数据
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

#每张图像都映射到一个标签。由于数据集中不包含类别名称，因此将它们存储在此处
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
				'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
				
#训练集中有 60000 张图像，每张图像都表示为 28x28 像素
#train_images.shape
#(60000, 28, 28)
#测试集中有 10000 张图像。同样，每张图像都表示为 28x28 像素
#test_images.shape
#(10000, 28, 28)
#对数据进行预处理
#检查训练集中的第一张图像，就会发现像素值介于 0 到 255 之间
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

#我们将这些值缩小到 0 到 1 之间，然后将其馈送到神经网络模型。
#为此，将图像组件的数据类型从整数转换为浮点数，然后除以 255
train_images = train_images / 255.0
test_images = test_images / 255.0

#显示训练集中的前 25 张图像，并在每张图像下显示类别名称。验证确保数据格式正确无误
plt.figure(figsize=(10,10))
for i in range(25):
	plt.subplot(5,5,i+1)
	plt.xticks([])
	plt.yticks([])
	plt.grid(False)
	plt.imshow(train_images[i], cmap=plt.cm.binary)
	plt.xlabel(class_names[train_labels[i]])
plt.show()

def create_model():
	#第一层 tf.keras.layers.Flatten 将图像格式从二维数组（28x28 像素）转换成一维数组（28 * 28 = 784 像素）。
	#可以将该层视为图像中像素未堆叠的行，并排列这些行。该层没有要学习的参数；它只改动数据的格式。
	#在扁平化像素之后，该网络包含两个 tf.keras.layers.Dense 层的序列。这些层是密集连接或全连接神经层。
	#第一个 Dense 层具有 128 个节点（或神经元）。第二个（也是最后一个）层是具有 10 个节点的 softmax 层，
	#该层会返回一个具有 10 个概率得分的数组，这些得分的总和为 1。
	#每个节点包含一个得分，表示当前图像属于 10 个类别中某一个的概率。
	model = keras.Sequential([
		keras.layers.Flatten(input_shape=(28, 28)),
		keras.layers.Dense(128, activation=tf.nn.relu),
		keras.layers.Dense(10, activation=tf.nn.softmax)])
	#损失函数 - 衡量模型在训练期间的准确率。我们希望尽可能缩小该函数，以“引导”模型朝着正确的方向优化。
	#优化器 - 根据模型看到的数据及其损失函数更新模型的方式。
	#指标 - 用于监控训练和测试步骤。以下示例使用准确率，即图像被正确分类的比例。
	model.compile(optimizer=tf.train.AdamOptimizer(),
				loss='sparse_categorical_crossentropy',
				metrics=['accuracy'])
	return model

# include the epoch in the file name. (uses `str.format`)
checkpoint_path = "training_2/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
latest = tf.train.latest_checkpoint(checkpoint_dir)
print("Latest model:",latest)

# Create checkpoint callback
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
													save_weights_only=True,
													# Save weights, every 5-epochs.
													period=5,
													verbose=1)

# Create a basic model instance
model = create_model()
if latest != None:
	model.load_weights(latest)
	print("Load latest model: ", latest)
model.summary()
loss, acc = model.evaluate(test_images, test_labels)
print("Untrained model, accuracy: {:5.2f}%".format(100*acc))
print("Training...")

#使模型与训练数据“拟合”			  
model.fit(train_images,
			train_labels,
			epochs=10,
			callbacks = [cp_callback],
			validation_data = (test_images, test_labels))
			
#评估准确率
#比较一下模型在测试数据集上的表现			
model = create_model()
if latest != None:
	model.load_weights(latest)
else:
	model.load_weights("training_2/cp-0005.ckpt")
test_loss, test_acc = model.evaluate(test_images, test_labels)
print("Test loss: ", test_loss)
#模型在测试数据集上的准确率略低于在训练数据集上的准确率。训练准确率和测试准确率之间的这种差异表示出现过拟合。
print('Test accuracy:', test_acc)