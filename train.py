import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
from tensorlayer.prepro import *
from tensorlayer.cost import *
import numpy as np
import scipy
from scipy.io import loadmat
import time
import os
import re
import nltk
from utils import *
from model import *
import pickle
import argparse
import model

# 用pickle.load导入数据，怎么dump的就怎么load
with open("_vocab.pickle", 'rb') as f:
    vocab = pickle.load(f)
with open("_image_train.pickle", 'rb') as f:
    images_train_256, images_train = pickle.load(f)
with open("_image_test.pickle", 'rb') as f:
    images_test_256, images_test = pickle.load(f)
with open("_n.pickle", 'rb') as f:
    n_captions_train, n_captions_test, n_captions_per_image, n_images_train, n_images_test = pickle.load(f)
with open("_caption.pickle", 'rb') as f:
    captions_ids_train, captions_ids_test = pickle.load(f)

# images_train_256 = np.array(images_train_256)
# images_test_256 = np.array(images_test_256)
images_train = np.array(images_train)
images_test = np.array(images_test)

print('导入完成')

ni = int(np.ceil(np.sqrt(batch_size)))
tl.files.exists_or_mkdir("samples/step1_gan-cls")
tl.files.exists_or_mkdir("samples/step_pretrain_encoder")
tl.files.exists_or_mkdir("checkpoint")
save_dir = "checkpoint"

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default="train", help='train, train_encoder, translation')
args = parser.parse_args()

t_real_image = tf.placeholder('float32', [batch_size, image_size, image_size, 3], name='real_image')
t_wrong_image = tf.placeholder('float32', [batch_size, image_size, image_size, 3], name='wrong_image')
t_real_caption = tf.placeholder(dtype=tf.int64, shape=[batch_size, None], name='real_caption_input')
t_wrong_caption = tf.placeholder(dtype=tf.int64, shape=[batch_size, None], name='wrong_caption_input')
t_z = tf.placeholder(tf.float32, [batch_size, z_dim], name='z_noise')

# 卷积层
net_cnn = cnn_encoder(t_real_image, is_train=True, reuse=False)
# output 返回的是一个矩阵
x = net_cnn.outputs
# dynamic_rnn
v = rnn_embed(t_real_caption, is_train=True, reuse=False).outputs
x_w = cnn_encoder(t_wrong_image, is_train=True, reuse=True).outputs
v_w = rnn_embed(t_wrong_caption, is_train=True, reuse=True).outputs

alpha = 0.2  # margin alpha
# 计算损失
rnn_loss = tf.reduce_mean(tf.maximum(0., alpha - cosine_similarity(x, v) + cosine_similarity(x, v_w))) + tf.reduce_mean(
    tf.maximum(0., alpha - cosine_similarity(x, v) + cosine_similarity(x_w, v)))
generator_txt2img = model.generator_txt2img_resnet
discriminator_txt2img = model.discriminator_txt2img_resnet

# 创建一个DynamicRnn
net_rnn = rnn_embed(t_real_caption, is_train=False, reuse=True)

net_fake_image, _ = generator_txt2img(t_z, net_rnn.outputs, is_train=True, reuse=False,
                                      batch_size=batch_size)
# NOISE ON RNN
# net_fake_image = net_fake_image + tf.random_normal(shape=net_fake_image.get_shape(),
#                                                    dtype=tf.float32,
#                                                    mean=0, stddev=0.02)
net_d, disc_fake_image_logits = discriminator_txt2img(net_fake_image.outputs, net_rnn.outputs, is_train=True,
                                                      reuse=False)
_, disc_real_image_logits = discriminator_txt2img(t_real_image, net_rnn.outputs, is_train=True, reuse=True)
_, disc_mismatch_logits = discriminator_txt2img(
    # t_wrong_image,
    t_real_image,
    # net_rnn.outputs,
    rnn_embed(t_wrong_caption, is_train=False, reuse=True).outputs,
    is_train=True, reuse=True)

# testing inference for txt2img
net_g, _ = generator_txt2img(t_z,
                             rnn_embed(t_real_caption, is_train=False, reuse=True).outputs,
                             is_train=False, reuse=True, batch_size=batch_size)

d_loss1 = tl.cost.sigmoid_cross_entropy(disc_real_image_logits, tf.ones_like(disc_real_image_logits), name='d1')
d_loss2 = tl.cost.sigmoid_cross_entropy(disc_mismatch_logits, tf.zeros_like(disc_mismatch_logits), name='d2')
d_loss3 = tl.cost.sigmoid_cross_entropy(disc_fake_image_logits, tf.zeros_like(disc_fake_image_logits), name='d3')
d_loss = d_loss1 + (d_loss2 + d_loss3) * 0.5
# d_loss = tf.log(d_loss1) + (tf.log(1 - d_loss2) + tf.log(1 - d_loss3)) * 0.5
g_loss = (tl.cost.sigmoid_cross_entropy(disc_fake_image_logits, tf.ones_like(disc_fake_image_logits), name='g'))

lr = 0.0002
lr_decay = 0.0002
# lr_decay = 0.00005
decay_every = 50
beta1 = 0.5

cnn_vars = tl.layers.get_variables_with_name('cnn', True, True)
rnn_vars = tl.layers.get_variables_with_name('rnn', True, True)
d_vars = tl.layers.get_variables_with_name('discriminator', True, True)
g_vars = tl.layers.get_variables_with_name('generator', True, True)

with tf.variable_scope('learning_rate'):
    lr_v = tf.Variable(lr_decay, trainable=False)
    lr_g = tf.Variable(lr, trainable=False)

d_optim = tf.train.RMSPropOptimizer(lr_v).minimize(d_loss, var_list=d_vars)
g_optim = tf.train.RMSPropOptimizer(lr_g).minimize(g_loss, var_list=g_vars)
grads, _ = tf.clip_by_global_norm(tf.gradients(rnn_loss, rnn_vars + cnn_vars), 10)
optimizer = tf.train.RMSPropOptimizer(lr_v)
rnn_optim = optimizer.apply_gradients(zip(grads, rnn_vars + cnn_vars))

sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
tl.layers.initialize_global_variables(sess)

# load the latest checkpoints
net_rnn_name = os.path.join(save_dir, 'net_rnn.npz')
net_cnn_name = os.path.join(save_dir, 'net_cnn.npz')
net_g_name = os.path.join(save_dir, 'net_g.npz')
net_d_name = os.path.join(save_dir, 'net_d.npz')

load_and_assign_npz(sess=sess, name=net_rnn_name, model=net_rnn)
load_and_assign_npz(sess=sess, name=net_cnn_name, model=net_cnn)
load_and_assign_npz(sess=sess, name=net_g_name, model=net_g)
load_and_assign_npz(sess=sess, name=net_d_name, model=net_d)

sample_size = batch_size
sample_seed = np.random.normal(loc=0.0, scale=1.0, size=(sample_size, z_dim)).astype(np.float32)
# sample_seed = np.random.uniform(low=-1, high=1, size=(sample_size, z_dim)).astype(np.float32)]
n = int(sample_size / ni)
sample_sentence = ["the flower shown has yellow anther red pistil and bright red petals."] * n + \
                  ["this flower has petals that are yellow, white and purple and has dark lines"] * n + \
                  ["the petals on this flower are white with a yellow center"] * n + \
                  ["this flower has a lot of small round pink petals."] * n + \
                  ["this flower is orange in color, and has petals that are ruffled and rounded."] * n + \
                  ["the flower has yellow petals and the center of it is brown."] * n + \
                  ["this flower has petals that are blue and white."] * n + \
                  [
                      "these white flowers have petals that start off white in color and end in a yellow towards the tips."] * n

# sample_sentence = captions_ids_test[0:sample_size]
for i, sentence in enumerate(sample_sentence):
    sentence = preprocess_caption(sentence)
    sample_sentence[i] = [vocab.word_to_id(word) for word in nltk.tokenize.word_tokenize(sentence)] + [
        vocab.end_id]  # add END_ID
    # sample_sentence[i] = [vocab.word_to_id(word) for word in sentence]
    # print(sample_sentence[i])
sample_sentence = tl.prepro.pad_sequences(sample_sentence, padding='post')
n_epoch = 600
print_freq = 1
n_batch_epoch = int(n_images_train / batch_size)
# exit()
for epoch in range(0, n_epoch + 1):
    start_time = time.time()

    if epoch % 50 == 0:
        lr_decay = lr_decay * 0.75
        sess.run(tf.assign(lr_v, lr_decay))

    if epoch % 100 == 0:
        lr = lr * 0.8
        sess.run(tf.assign(lr_g, lr))

    for step in range(n_batch_epoch):
        step_time = time.time()
        # 每一轮抽取batch_size个训练数据
        # idexs = [i for i in range(batch_size * step * 10 + epoch % 10, batch_size * (1 + step) * 10 + epoch % 10, 10)]
        idexs = get_random_int(min=0, max=n_captions_train - 1, number=batch_size)
        b_real_caption = captions_ids_train[idexs]
        b_real_caption = tl.prepro.pad_sequences(b_real_caption, padding='post')
        # get real image
        b_real_images = images_train[
            np.floor(np.asarray(idexs).astype('float') / n_captions_per_image).astype('int')]
        # save_images(b_real_images, [ni, ni], 'samples/step1_gan-cls/train_00.png')
        # get wrong caption
        idexs1 = get_random_int(min=0, max=n_captions_train - 1, number=batch_size)

        b_wrong_caption = captions_ids_train[idexs1]
        b_wrong_caption = tl.prepro.pad_sequences(b_wrong_caption, padding='post')
        # get wrong image
        idexs2 = get_random_int(min=0, max=n_images_train - 1, number=batch_size)
        b_wrong_images = images_train[idexs2]
        # get noise
        b_z = np.random.normal(loc=0.0, scale=1.0, size=(sample_size, z_dim)).astype(np.float32)
        # b_z = np.random.uniform(low=-1, high=1, size=[batch_size, z_dim]).astype(np.float32)

        b_real_images = threading_data(b_real_images, prepro_img,
                                       mode='train')  # [0, 255] --> [-1, 1] + augmentation
        b_wrong_images = threading_data(b_wrong_images, prepro_img, mode='train')
        # updates text-to-image mapping
        # print(idexs)
        # print([
        #     np.floor(np.asarray(idexs).astype('float') / n_captions_per_image).astype('int')])
        errRNN, _ = sess.run([rnn_loss, rnn_optim], feed_dict={
            t_real_image: b_real_images,
            t_wrong_image: b_wrong_images,
            t_real_caption: b_real_caption,
            t_wrong_caption: b_wrong_caption,
            t_z: b_z})

        # updates D
        if step % 5 != 4:
            errD, _ = sess.run([d_loss, d_optim], feed_dict={
                t_real_image: b_real_images,
                t_wrong_image: b_wrong_images,
                t_wrong_caption: b_wrong_caption,
                t_real_caption: b_real_caption,
                t_z: b_z})
        # updates G
        errG, _ = sess.run([g_loss, g_optim], feed_dict={
            t_real_image: b_real_images,
            t_wrong_image: b_wrong_images,
            t_wrong_caption: b_wrong_caption,
            t_real_caption: b_real_caption,
            t_z: b_z})

        print("Epoch: [%2d/%2d] [%4d/%4d] time: %4.4fs, d_loss: %.4f, g_loss: %.4f, rnn_loss: %.4f" \
              % (epoch, n_epoch, step, n_batch_epoch, time.time() - step_time, errD, errG, errRNN))

    if (epoch + 1) % print_freq == 0:
        print(" ** Epoch %d took %fs" % (epoch, time.time() - start_time))
        img_gen, rnn_out = sess.run([net_g.outputs, net_rnn.outputs], feed_dict={
            t_real_caption: sample_sentence,
            t_z: sample_seed})

        # img_gen = threading_data(img_gen, prepro_img, mode='rescale')
        save_images(img_gen, [ni, ni], 'samples/step1_gan-cls/train_{:03d}.png'.format(epoch))

    # save model
    if (epoch != 0) and (epoch % 10) == 0:
        tl.files.save_npz(net_cnn.all_params, name=net_cnn_name, sess=sess)
        tl.files.save_npz(net_rnn.all_params, name=net_rnn_name, sess=sess)
        tl.files.save_npz(net_g.all_params, name=net_g_name, sess=sess)
        tl.files.save_npz(net_d.all_params, name=net_d_name, sess=sess)
        print("[*] Save checkpoints SUCCESS!")

    if (epoch != 0) and (epoch % 100) == 0:
        tl.files.save_npz(net_cnn.all_params, name=net_cnn_name + str(epoch), sess=sess)
        tl.files.save_npz(net_rnn.all_params, name=net_rnn_name + str(epoch), sess=sess)
        tl.files.save_npz(net_g.all_params, name=net_g_name + str(epoch), sess=sess)
        tl.files.save_npz(net_d.all_params, name=net_d_name + str(epoch), sess=sess)
