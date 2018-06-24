"""
Code for training
Inspired by
https://github.com/machrisaa/tensorflow-vgg/issues/11#issuecomment-262748616
https://github.com/machrisaa/tensorflow-vgg/issues/11#issuecomment-262793514
"""
import numpy as np
import tensorflow as tf
from Vgg.vgg19_trainable import Vgg19Resizable
from Vgg.utils import load_image, gen_batch, get_label

WEIGHTS_FILE = './weights/vgg19.npy'
TRAIN_FILE = 'pcb_900/train.lst'
N_BATCH = 64
N_CLASS = 2
L_RATE = 1e-4
MAX_ITER = 10000

# Prepare training data
train_list = [item.strip() for item in open(TRAIN_FILE, encoding='utf-8').readlines()]
gen = gen_batch(train_list, N_BATCH)

# Define default graph
graph = tf.Graph()
with graph.as_default():
    images = tf.placeholder(tf.float32, [N_BATCH, 224, 224, 3])
    labels = tf.placeholder(tf.float32, [N_BATCH, N_CLASS])
    train_mode = tf.placeholder(tf.bool)

    vgg = Vgg19Resizable(WEIGHTS_FILE)
    vgg.build(images, train_mode)

    cost = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=vgg.prob)
    train = tf.train.AdamOptimizer(L_RATE).minimize(cost)
    correct_prediction = tf.equal(tf.argmax(vgg.prob, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Train loop
with tf.Session(graph=graph) as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(MAX_ITER):
        cur_imgs = next(gen)
        batch_img_raw = list(map(load_image, cur_imgs))
        batch_img_arr = np.array(batch_img_raw).reshape((N_BATCH, 224, 224, 3))
        batch_lab_raw = list(map(get_label, cur_imgs))
        batch_lab_arr = np.array(batch_lab_raw)

        feed_dict = {
            images: batch_img_arr,
            labels: batch_lab_arr,
            train_mode: True
        }

        cost_val = cost.eval(feed_dict=feed_dict, session=sess)
        print('cross entropy: ', cost_val)
        sess.run(train, feed_dict=feed_dict)
        if i % 10 == 0:
            with open('./cost.txt', 'a') as f:
                f.write(str(cost_val) + '\n')
