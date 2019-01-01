#-*-coding:utf-8-*-
#date:2018/12/30

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

import os
import tensorflow as tf
import matplotlib.pyplot as plt
# use tensorflow 1.9+
print("tensorflow version: %s"%tf.__version__)

#导入和加载数据
mnist = tf.keras.datasets.mnist		
(train_images, train_labels),(test_images, test_labels) = mnist.load_data()

#每张图像都映射到一个标签，画图时用
class_names = ['0', '1', '2', '3', '4',
				'5', '6', '7', '8', '9']

#预处理数据
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

#必须先对数据进行预处理，然后再训练网络。如果您检查训练集中的第一张图像，就会发现像素值介于 0 到 255 之间
#我们将这些值缩小到 0 到 1 之间，然后将其馈送到神经网络模型。为此，将图像组件的数据类型从整数转换为浮点数，然后除以 255
train_images, test_images = train_images / 255.0, test_images / 255.0

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

# Returns a short sequential model
def create_model():
	'''网络中的第一层 tf.keras.layers.Flatten 将图像格式从二维数组（28x28 像素）转换成一维数组（28 * 28 = 784 像素）。
	可以将该层视为图像中像素未堆叠的行，并排列这些行。该层没有要学习的参数；它只改动数据的格式。
	在扁平化像素之后，该网络包含两个 tf.keras.layers.Dense 层的序列。这些层是密集连接或全连接神经层。
	第一个 Dense 层具有 512 个节点（或神经元）。第二个（也是最后一个）层是具有 10 个节点的 softmax 层，
	该层会返回一个具有 10 个概率得分的数组，这些得分的总和为 1。
	每个节点包含一个得分，表示当前图像属于 10 个类别中某一个的概率。'''
	model = tf.keras.models.Sequential([
										tf.keras.layers.Flatten(input_shape=(28, 28)),
										tf.keras.layers.Dense(512, activation=tf.nn.relu),
										tf.keras.layers.Dropout(0.2),
										tf.keras.layers.Dense(10, activation=tf.nn.softmax)])
	'''损失函数 - 衡量模型在训练期间的准确率。我们希望尽可能缩小该函数，以“引导”模型朝着正确的方向优化。
	优化器 - 根据模型看到的数据及其损失函数更新模型的方式。
	指标 - 用于监控训练和测试步骤。以下示例使用准确率，即图像被正确分类的比例。'''
	model.compile(optimizer=tf.keras.optimizers.Adam(),
					loss=tf.keras.losses.sparse_categorical_crossentropy,
					metrics=['accuracy'])
								
	return model

checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create checkpoint callback
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

# Create a basic model instance
model = create_model()
model.summary()
loss, acc = model.evaluate(test_images, test_labels)
print("Untrained model, accuracy: {:5.2f}%".format(100*acc))

#将训练数据馈送到模型中，在本示例中为 train_images 和 train_labels 数组。
#模型学习将图像与标签相关联。
#我们要求模型对测试集进行预测，在本示例中为 test_images 数组。我们会验证预测结果是否与 test_labels 数组中的标签一致。
#调用 model.fit 方法，使模型与训练数据“拟合”
model.fit(train_images, train_labels,
			epochs = 10,
			validation_data = (test_images, test_labels),
			callbacks = [cp_callback])  # pass callback to training
			
#加载权重，比较一下模型在测试数据集上的表现
model = create_model()
model.load_weights(checkpoint_path)
model.summary()
loss,acc = model.evaluate(test_images, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))

