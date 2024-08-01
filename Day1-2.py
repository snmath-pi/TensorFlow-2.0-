import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
# physical_devices = tf.config.list_physical_devices('PU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)
#Initialization

x = tf.constant(3, shape=(1, 1),
                dtype=tf.float32)
x=tf.constant([[1, 2, 3],
               [4, 5, 6]])
x = tf.ones((3, 3))
x=tf.zeros((2, 3))
x=tf.eye(3) # I (eye) for identity
        #matrix
x=tf.random.normal((3,3),mean=0,
                   stddev=1)
x=tf.random.uniform((1,3),

                    minval=0,
                    maxval=1)
x = tf.range(start=1, limit=10,
             delta=2)

x = tf.cast(x, dtype=tf.float64)
#Mathematical

x = tf.constant([1, 2, 3])
y = tf.constant([9, 8, 7])

z = tf.add(x, y)
# print(z)
# z = x + y

z = tf.subtract(x, y)
# print(z)
#z = x - y

z = tf.divide(x, y)
# print(z)
z = tf.multiply(x, y)

# print(tf.tensordot(x, y, axes=1))

z = tf.reduce_sum(x*y, axis=0)

# print(z)
z = x ** 5
# print(z)
# z = x @ y
# print(z)



#INdexing
x = tf.constant([1, 2, 3])
print(x[1:2])
#Reshaping


x = tf.range(9)
print(x)
x = tf.reshape(x, (3, 3))
print(x)

x = tf.transpose(x, perm=[1, 0])
print(x)
