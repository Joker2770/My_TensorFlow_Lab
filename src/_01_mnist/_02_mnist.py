#-*-coding:utf-8-*-
#date:2019/01/01

'''
MIT License

Copyright (c) 2019 Joker2770

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

#构建神经网络需要先配置模型的层，然后再编译模型
def create_model():
	#设置层
	# Returns a short sequential model
	model = tf.keras.models.Sequential([
										tf.keras.layers.Flatten(input_shape=(28, 28)),
										tf.keras.layers.Dense(512, activation=tf.nn.relu, input_shape=(784,)),
										tf.keras.layers.Dropout(0.2),
										tf.keras.layers.Dense(10, activation=tf.nn.softmax)])
	#编译模型
	model.compile(optimizer=tf.keras.optimizers.Adam(),
					loss=tf.keras.losses.sparse_categorical_crossentropy,
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
model.fit(train_images, train_labels,
			epochs = 10,
			callbacks = [cp_callback],
			validation_data = (test_images, test_labels))  # pass callback to training
			
model = create_model()
if latest != None:
	model.load_weights(latest)
else:
	model.load_weights("training_2/cp-0005.ckpt")
model.summary()
loss,acc = model.evaluate(test_images, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))

