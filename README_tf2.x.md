# 一、自动求导机制
在机器学习中，我们经常需要计算函数的导数。TensorFlow 提供了强大的 自动求导机制 来计算导数。以下代码展示了如何使用tf.GradientTape()计算函数  在  时的导数：
import tensorflow as tf

~~~
x = tf.Variable(initial_value=3.)
with tf.GradientTape() as tape:     # 在 tf.GradientTape() 的上下文内，所有计算步骤都会被记录以用于求导
    y = tf.square(x)
y_grad = tape.gradient(y, x)        # 计算y关于x的导数
print([y, y_grad])
~~~

输出:

~~~
1[array([9.], dtype=float32), array([6.], dtype=float32)]
~~~

这里x是一个初始化为 3 的 变量 （Variable），使用tf.Variable()声明。与普通张量一样，变量同样具有形状、类型和值三种属性。使用变量需要有一个初始化过程，可以通过在tf.Variable()中指定initial_value参数来指定初始值。这里将变量x初始化为3.。变量与普通张量的一个重要区别是其默认能够被 TensorFlow 的自动求导机制所求导，因此往往被用于定义机器学习模型的参数。

tf.GradientTape()是一个自动求导的记录器，在其中的变量和计算步骤都会被自动记录。在上面的示例中，变量x和计算步骤y = tf.square(x)被自动记录，因此可以通过y_grad = tape.gradient(y, x)求张量y对变量x的导数。