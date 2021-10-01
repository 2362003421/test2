import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

print(tf.test.is_gpu_available())
# Tensor -> ndarray
# t = tf.constant([3, 1, 2, 4], tf.float32, name='t')
# session = tf.compat.v1.Session()
# array = t.eval(session=session)
# print(array)

# ndarray -> Tensor
# array = np.array([1, 2, 3], np.float32)
# t = tf.convert_to_tensor(array, tf.float32, name='t')
# print(t)

# 张量的尺寸
# t = tf.constant(
#     [
#         [1, 2, 3],
#         [4, 5, 6]
#     ], tf.float32, name='t')
# s = t.get_shape()
# print(s)
# print(type(s))
# print(s[0])

# 读取图片
# image = tf.read_file("tianyi.jpg", 'r')
# image_tensor = tf.image.decode_jpeg(image)
# shape = tf.shape(image_tensor)
# session = tf.Session()
# print(session.run(image_tensor))
# print(session.run(shape))

# 类型转换
# t = tf.constant(
#     [
#         [0, 2, 0]
#         ,
#         [0, 0, 1]
#     ]
#     , tf.float32)
# session = tf.Session()
# r = tf.cast(t, tf.bool)
# print(session.run(r))

# 取值
# t1 = tf.constant([1, 2, 3, 4, 5], tf.float32)
# t = tf.slice(t1, [2], [3])
# session = tf.Session()
# print(session.run(t))

# 转置
# x = tf.constant(
#     [
#         [1, 2, 3],
#         [4, 5, 6]
#     ], tf.float32)
# r = tf.transpose(x, perm=[1, 0])
# session = tf.Session()
# print(session.run(r))

# 变形
# x = tf.constant(
#     [
#         [
#             [1, 2], [4, 5], [6, 7]
#         ],
#         [
#             [8, 9], [10, 11], [12, 13]
#         ]
#     ], tf.float32)
# t = tf.reshape(x, [4, 1, -1])
# session = tf.Session()
# print(session.run(t))

# 创建 Variable 对象
# v = tf.Variable(tf.constant([2, 3], tf.float32))
# 创建会话
# session = tf.Session()
# "Variable 对象初始化
# session.run(tf.global_variables_initializer())
# 打印
# print(session.run(v))
# session.run(v.assign_add([10, 20]))
# v.assign_add([10, 20])
# print(session.run(v))

print(tf.__version__)
