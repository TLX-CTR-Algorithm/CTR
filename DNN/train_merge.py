
import tensorflow as tf
from DNN import model2
from DNN import model
import logging
import utils
import pandas as pd
import os
import numpy as np
import re
import math
from DNN import flags

slim = tf.contrib.slim
FLAGS, unparsed = flags.parse_args()

if not os.path.exists(FLAGS.dnn_log_dir):
    os.mkdir(FLAGS.dnn_log_dir)

#设置日志打印格式
logger = logging.getLogger()
if FLAGS.debug_level  == 'DEBUG':
    logger.setLevel(logging.DEBUG)
elif FLAGS.debug_level  == 'INFO':
    logger.setLevel(logging.INFO)
sh = logging.StreamHandler()
sh.setFormatter(logging.Formatter('%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s'))
fl = logging.FileHandler(FLAGS.dnn_log_path)
fl.setFormatter(logging.Formatter('%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s'))
logger.addHandler(sh)
logger.addHandler(fl)

#临时配置
#BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = 'D:\\Documents\\GitCTR\\CTR'
labels_path = os.path.join(BASE_DIR + "/output/ffm_data/train_label.txt")
v_labels_path = os.path.join(BASE_DIR + "/output/ffm_data/vaild_label.txt")
ffm_train_path = os.path.join(BASE_DIR + "/output/ffm_data/train_pred.txt")
ffm_valid_path = os.path.join(BASE_DIR + "/output/ffm_data/vaild_pred.txt")
ffm_test_path = os.path.join(BASE_DIR + "/output/ffm_data/train_pred.txt")
train_path = os.path.join(BASE_DIR + "/output/dnn_data/train.txt")
valid_path = os.path.join(BASE_DIR + "/output/dnn_data/valid.txt")
test_path = os.path.join(BASE_DIR + "/output/dnn_data/train.txt")
learning_rate = 0.02
logfrequency = 10000
batch_size = 10000

def gendata(path):
    data = np.loadtxt(path, delimiter=',')
    return data[:,np.newaxis]


def train_model(batch_size=batch_size):
    #训练数据
    #dnn_inputs = gendata(train_path)
    ffm_inputs = gendata(ffm_train_path)
    labels = gendata(labels_path)
    #inputs = np.concatenate([dnn_inputs,ffm_inputs],axis=1)
    inputs = ffm_inputs

    #校验数据
    #v_dnn_inputs = gendata(valid_path)
    v_ffm_inputs = gendata(ffm_valid_path)
    valid_labels = gendata(v_labels_path)
    #valid_inputs = np.concatenate([v_dnn_inputs,v_ffm_inputs],axis=1)
    valid_inputs = v_ffm_inputs

    #构建单层网络
    tensor_inputs = tf.placeholder(tf.float32, [None,1], name='tensor_inputs')
    #tensor_inputs = tf.placeholder(tf.float32, [None, 2], name='tensor_inputs')
    tensor_labels = tf.placeholder(tf.float32, [None,1], name='labels')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    global_step = tf.Variable(0, trainable=False, name='global_step', dtype=tf.int64)

    prediction = slim.fully_connected(tensor_inputs, 1, activation_fn=tf.nn.sigmoid, scope='prediction')

    log_loss = tf.losses.log_loss(labels=tensor_labels, predictions=prediction)
    loss = tf.reduce_mean(log_loss)


    #构建优化器
    optimizer = tf.train.FtrlOptimizer(learning_rate)
    gradients = optimizer.compute_gradients(loss)
    train_step = optimizer.apply_gradients(gradients, global_step=global_step)

    logging.debug('debug:{} '.format(inputs.shape))
    logging.debug('debug:{} '.format(labels.shape))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        logging.debug('Initialized')

        logging.debug('debug:{} '.format(inputs.shape))
        logging.debug('debug:{} '.format(labels.shape))
        #使用train数据集进行训练
        feed_dict = {tensor_inputs: inputs, tensor_labels: labels}
        gstep, _, train_prediction, train_loss = sess.run([global_step, train_step, prediction, loss], feed_dict)

        #if gstep % logfrequency == 0:
            # 每间隔指定的频率打印日志并存储checkpoint文件
        logging.debug('train: step [{0}] loss [{1}] '.format(gstep, train_loss ))

        #使用valid数据集进行验证
        feed_dict = {tensor_inputs: valid_inputs, tensor_labels: valid_labels}
        gstep, _, valid_prediction, valid_loss = sess.run([global_step, train_step, prediction, loss], feed_dict)
        logging.debug('train: step [{0}] loss [{1}] '.format(gstep, valid_loss))

if __name__ == '__main__':
    train_model()

















