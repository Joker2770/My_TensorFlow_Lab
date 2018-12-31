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
print("tensorflow version: %s"%tf.__version__)
# use tensorflow 1.9+
mnist = tf.keras.datasets.mnist		

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Returns a short sequential model
def create_model():
	model = tf.keras.models.Sequential([
										tf.keras.layers.Flatten(),
										tf.keras.layers.Dense(512, activation=tf.nn.relu),
										tf.keras.layers.Dropout(0.2),
										tf.keras.layers.Dense(10, activation=tf.nn.softmax)])
	model.compile(optimizer='adam',
					loss='sparse_categorical_crossentropy',
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
model.fit(x_train, y_train,  epochs = 10,
          validation_data = (x_test, y_test),
          callbacks = [cp_callback])  # pass callback to training

