"""
Code for training
Inspired by
https://github.com/machrisaa/tensorflow-vgg/issues/11#issuecomment-262748616
https://github.com/machrisaa/tensorflow-vgg/issues/11#issuecomment-262793514
"""
from os.path import join as join_path
from os.path import isdir
from os import makedirs
import numpy as np
import tensorflow as tf
from Vgg.vgg19_trainable import Vgg19Resizable
from Vgg.utils import load_image, gen_batch, get_label

TASK = 'conf_1'
N_BATCH = 64
N_CLASS = 2
L_RATE = 1e-4
MAX_ITER = 10000

PARENT = './pcb_900'
WEIGHTS_FILE = './weights/vgg19.npy'
TRAIN_FILE = join_path(PARENT, 'train.lst')
SAVED_WEIGHTS = './pcb_900/{}/weights'.format(TASK)
SAVED_SUMMARY = './pcb_900/{}/summary'.format(TASK)

if __name__ == '__main__':

    if not isdir(SAVED_SUMMARY):
        makedirs(SAVED_SUMMARY)

    if not isdir(SAVED_WEIGHTS):
        makedirs(SAVED_WEIGHTS)

    # Prepare training data
    train_list = [item.strip() for item in open(TRAIN_FILE, encoding='utf-8').readlines()]
    gen = gen_batch(train_list, N_BATCH)

    # Define default graph
    graph = tf.Graph()
    with graph.as_default():
        images = tf.placeholder(tf.float32, [N_BATCH, 224, 224, 3])
        labels = tf.placeholder(tf.float32, [N_BATCH, N_CLASS])
        train_mode = tf.placeholder(tf.bool)

        # initialize network
        vgg = Vgg19Resizable(WEIGHTS_FILE)
        vgg.build(images, train_mode)

        with tf.name_scope('cost'):
            with tf.name_scope('cost_fn'):
                cost = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=vgg.prob)
            with tf.name_scope('cost'):
                cost_avg = tf.reduce_mean(tf.cast(cost, tf.float32))
        tf.summary.scalar('cost_avg', cost_avg)

        with tf.name_scope('train'):
            train_step = tf.train.AdamOptimizer(L_RATE).minimize(cost)

        with tf.name_scope('accuracy'):
            with tf.name_scope('correct_prediction'):
                correct_prediction = tf.equal(tf.argmax(vgg.prob, 1), tf.argmax(labels, 1))
            with tf.name_scope('accuracy'):
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

    # define default session
    with tf.Session(graph=graph) as sess:
        sess.run(tf.global_variables_initializer())

        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(join_path(SAVED_SUMMARY, 'train'), graph=sess.graph)
        test_writer = tf.summary.FileWriter(join_path(SAVED_SUMMARY, 'test'), graph=sess.graph)

        # Train loop
        for i in range(MAX_ITER):
            # get train batches
            cur_batch = next(gen)
            batch_img_raw = list(map(load_image, cur_batch))
            batch_img_arr = np.array(batch_img_raw).reshape((N_BATCH, 224, 224, 3))
            batch_lab_raw = list(map(get_label, cur_batch))
            batch_lab_arr = np.array(batch_lab_raw)

            feed_dict = {
                images: batch_img_arr,
                labels: batch_lab_arr,
                train_mode: True
            }

            if i % 10 == 0:
                # Record summaries and test-set accuracy
                accu_val, summary = sess.run([accuracy, merged], feed_dict=feed_dict)
                test_writer.add_summary(summary, i)
                print('Step: {} | Accuracy: {:0.5f}'.format(i, accu_val))
            else:
                # Record train set summaries, and train
                cost_val, summary, _ = sess.run([cost_avg, merged, train_step], feed_dict=feed_dict)
                train_writer.add_summary(summary, i)
                print('Step: {} | Loss: {:0.5} | L_rate: {}'.format(i, cost_val, L_RATE))

            # save weights
            if i % 500 == 0:
                vgg.save_npy(sess, join_path(SAVED_WEIGHTS, 'step_{}'.format(i)))
