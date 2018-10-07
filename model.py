# 基于densenet网络结构，实现训练网络结构

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

slim = tf.contrib.slim


# 实现模型的构建以及损失函数交叉熵的计算
class Model():
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    # 正态分布生成器,输入参数为:标准差standard deviation
    def trunc_normal(stddev):
        return tf.truncated_normal_initializer(stddev=stddev)

    # 实现了H_l()Composite function,此处暂时按照以下四个步骤构建，视需要和作用进行调整
    def bn_act_conv_drp(current, num_outputs, scope='block'):
        current = slim.batch_norm(current, scope=scope + '_bn')
        current = tf.nn.relu(current)
        current = slim.fully_connected(current, num_outputs, scope=scope + '_fully')
        current = slim.dropout(current, scope=scope + '_dropout')
        return current

    def block(net, layers, growth, scope='block'):
        for idx in range(layers):
            bottleneck = bn_act_conv_drp(net, 4 * growth,
                                         scope=scope + '_botfully' + str(idx))
            tmp = bn_act_conv_drp(bottleneck, growth,
                                  scope=scope + '_tmpfullly' + str(idx))
            net = tf.concat(axis=1, values=[net, tmp])
        return net

    # 对于由全连接神经元构建的类densenet网络结构，由于denseblock会并联前几层的输出，所以会导致特征维数下降缓慢，所以需要transiton层实现特征降维
    def transition(current, num_outputs, scope='trans'):
        current = slim.batch_norm(current, scope=scope + '_bn')
        current = slim.fully_connected(current, num_outputs, scope=scope + '_tsfully')
        return current

    def densenet(self,inputs, final_endpoint='Predictions', num_classes=2, is_training=False,
                 dropout_keep_prob=0.8,
                 scope='densenet'):
        '''
        :param inputs: 待训练样本数据：batch_size * 特征向量维数
        :param final_endpoint: 输出结果索引
        :param num_classes: 最终分类数，由于点击率预测是一个0-1问题，所以这里默认为2
        :param is_training:
        :param dropout_keep_prob: 训练过程中进行dropout操作时保留神经元的比率
        :param scope:densenet
        :return:
        '''
        growth = 128
        outunits = 2048
        outrate = 0.5

        end_points = {}

        with tf.variable_scope(scope, 'DenseNet', [inputs, num_classes]):
            with slim.arg_scope(self.bn_drp_scope(is_training=is_training,
                                             keep_prob=dropout_keep_prob)) as ssc:

                #
                # batchsize * 2048
                end_point = 'fully_0a_1024'
                outunits = outunits * outrate
                net = slim.fully_connected(inputs, outunits, scope=end_point)
                end_points[end_point] = net
                if end_point == final_endpoint: return net, end_points

                # batchsize * 1024
                # denseblock1 layers=6
                end_point = 'denseblock1'
                layers1 = 6
                net = self.block(net, layers1, growth, scope=end_point)
                end_points[end_point] = net
                if end_point == final_endpoint: return net, end_points
                # batchsize * ( 1024 + 6k ) (batchsize * 1792 )
                # transition layer1
                end_point = 'translayer1'
                outunits = outunits * outrate
                net = self.transition(net, outunits, scope=end_point)
                end_points[end_point] = net
                if end_point == final_endpoint: return net, end_points
                # batchsize * 896

                # denseblock2 layers=12
                end_point = 'denseblock2'
                # 考虑到整体特征维度一直在降低，所以增长率也适当降低
                layers2 = 6
                growth = 0.5 * growth
                net = self.block(net, layers2, growth, scope=end_point)
                end_points[end_point] = net
                if end_point == final_endpoint: return net, end_points
                # batchsize * ( 896 + 6 * 0.5 * k ) (batchsize * 1280 )
                # transition layer2
                end_point = 'translayer2'
                outunits = outunits * outrate
                net = self.transition(net, outunits, scope=end_point)
                end_points[end_point] = net
                if end_point == final_endpoint: return net, end_points
                # batchsize * 640

                # denseblock3 layers=6
                end_point = 'denseblock3'
                layers3 = 6
                growth = 0.5 * growth
                net = self.block(net, layers3, growth, scope=end_point)
                end_points[end_point] = net
                if end_point == final_endpoint: return net, end_points
                # batchsize * ( 640 + 6 * 0.5 * 0.5 * k ) (batchsize * 832 )
                # transition layer3
                end_point = 'translayer3'
                outunits = outunits * outrate
                net = self.transition(net, outunits, scope=end_point)
                end_points[end_point] = net
                if end_point == final_endpoint: return net, end_points
                # batchsize * 416

                # Classification layer
                # 分类器进行两次降维最终达到分类目的
                end_point = 'classifer_pool'
                net = slim.fully_connected(net, 104, scope=scope + '_globalclass1')
                logits = slim.fully_connected(net, num_classes, scope=scope + '_globalclass2')
                end_points[end_point] = logits
                if end_point == final_endpoint: return net, end_points
                end_point = 'Predictions'
                end_points[end_point] = slim.softmax(logits, scope=end_point)

        return logits, end_points

    def bn_drp_scope(is_training=True, keep_prob=0.8):
        keep_prob = keep_prob if is_training else 1
        with slim.arg_scope(
                [slim.batch_norm],
                scale=True, is_training=is_training, updates_collections=None):
            with slim.arg_scope(
                    [slim.dropout],
                    is_training=is_training, keep_prob=keep_prob) as bsc:
                return bsc

    def build(self):
        self.default_inputs_size = 2048
        self.inputs = tf.placeholder(tf.int32, name='inputs')
        self.label = tf.placeholder(tf.int32, name='label')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        with tf.variable_scope('builddensenet'):
            logits, end_points = self.densenet(inputs= self.inputs)

        #计算交叉熵损失函数
        with tf.variable_scope():
            labels = self.label
            loss = slim.losses.softmax_cross_entropy(logits, labels, weights=1.0)
            self.loss = tf.reduce_mean(loss)

        with tf.variable_scope():
            self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)













