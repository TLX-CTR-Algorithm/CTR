# 基于densenet网络结构，实现训练网络结构

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

slim = tf.contrib.slim


# 实现模型的构建以及损失函数交叉熵的计算
class Model():
    def __init__(self, learning_rate, oridata_dim, embed_max, embed_dim=128, decay_steps=5000, decay_rate=0.96):
        self.learning_rate = learning_rate
        self.layers = [6, 6, 6]
        self.oridata_dim = oridata_dim
        self.embed_dim = embed_dim
        self.embed_max = embed_max
        self.growth = 128
        self.outunits = 2048
        self.outrate = 0.5
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate

    def embeding(self, embed_inputs):
        with tf.name_scope("embedding"):
            embed_inputs = tf.cast(embed_inputs, tf.int32)
            oridata_dim = self.oridata_dim
            embed_matrix = tf.Variable(tf.random_uniform([self.embed_max, self.embed_dim], -1, 1), name="embed_matrix")
            embed_layer = tf.nn.embedding_lookup(embed_matrix, embed_inputs, name="embed_layer")
            embed_layer = tf.reshape(embed_layer, [-1, self.embed_dim * oridata_dim])
        return embed_layer

    # 正态分布生成器,输入参数为:标准差standard deviation
    def trunc_normal(self, stddev):
        return tf.truncated_normal_initializer(stddev=stddev)

    # 实现了H_l()Composite function,此处暂时按照以下四个步骤构建，视需要和作用进行调整
    def bn_act_conv_drp(self, current, num_outputs, scope='block'):
        current = slim.batch_norm(current, scope=scope + '_bn')
        current = tf.nn.relu(current)
        current = slim.fully_connected(current, num_outputs, scope=scope + '_fully')
        current = slim.dropout(current, scope=scope + '_dropout')
        return current

    def block(self, net, layers, growth, scope='block'):
        growth=int(growth)
        for idx in range(layers):
            bottleneck = self.bn_act_conv_drp(net, 4 * growth,
                                         scope=scope + '_botfully' + str(idx))
            tmp = self.bn_act_conv_drp(bottleneck, growth,
                                  scope=scope + '_tmpfullly' + str(idx))
            net = tf.concat(axis=1, values=[net, tmp])
        return net

    # 对于由全连接神经元构建的类densenet网络结构，由于denseblock会并联前几层的输出，所以会导致特征维数下降缓慢，所以需要transiton层实现特征降维
    def transition(self, current, num_outputs, scope='trans'):
        current = slim.batch_norm(current, scope=scope + '_bn')
        current = slim.fully_connected(current, num_outputs, scope=scope + '_tsfully')
        return current

    def densenet(self, continous_inputs, categorial_inputs, final_endpoint='Predictions', num_classes=1, is_training=False, dropout_keep_prob=0.8, scope='densenet'):
        '''
        :param inputs: 待训练样本数据：batch_size * 特征向量维数
        :param final_endpoint: 输出结果索引
        :param num_classes: 最终分类数，由于点击率预测是一个0-1问题，所以这里默认为2
        :param is_training:
        :param dropout_keep_prob: 训练过程中进行dropout操作时保留神经元的比率
        :param scope:densenet
        :return:
        '''

        end_points = {}

        with tf.variable_scope(scope, 'DenseNet', [continous_inputs, categorial_inputs, num_classes]):
            with slim.arg_scope(self.bn_drp_scope(is_training=is_training, keep_prob=dropout_keep_prob)) as ssc:

                # 进行embeding
                end_point = 'embeding'
                branch_embed = self.embeding(categorial_inputs)
                end_points[end_point] = branch_embed

                # 连续型特征和非连续特征需要拆分处理
                # 对类别型特征进行embeding处理，连续型特征直接和embeding后的数据连接后输入网络中
                end_point = 'continous_net'
                branch_continous_2 = slim.fully_connected(continous_inputs, 40, scope=end_point)
                end_point = 'concat_con_cat'
                net = tf.concat([branch_continous_2, branch_embed], 1, name=end_point)

                #reshape
                net = slim.fully_connected(net, self.default_inputs_size, scope=end_point)

                # batchsize * 2048
                end_point = 'fully_0a_1024'
                outunits = int(self.outunits * self.outrate)
                net = slim.fully_connected(net, outunits, scope=end_point)
                end_points[end_point] = net
                if end_point == final_endpoint: return net, end_points

                # batchsize * 1024
                # denseblock1 layers=6
                end_point = 'denseblock1'
                layers1 = self.layers[0]
                net = self.block(net, layers1, self.growth, scope=end_point)
                end_points[end_point] = net
                if end_point == final_endpoint: return net, end_points
                # batchsize * ( 1024 + 6k ) (batchsize * 1792 )
                # transition layer1
                end_point = 'translayer1'
                outunits = int(outunits * self.outrate)
                net = self.transition(net, outunits, scope=end_point)
                end_points[end_point] = net
                if end_point == final_endpoint: return net, end_points
                # batchsize * 896

                # denseblock2 layers=12
                end_point = 'denseblock2'
                # 考虑到整体特征维度一直在降低，所以增长率也适当降低
                layers2 = self.layers[1]
                growth = 0.5 * self.growth
                net = self.block(net, layers2, growth, scope=end_point)
                end_points[end_point] = net
                if end_point == final_endpoint: return net, end_points
                # batchsize * ( 896 + 6 * 0.5 * k ) (batchsize * 1280 )
                # transition layer2
                end_point = 'translayer2'
                outunits = int(outunits * self.outrate)
                net = self.transition(net, outunits, scope=end_point)
                end_points[end_point] = net
                if end_point == final_endpoint: return net, end_points
                # batchsize * 640

                # denseblock3 layers=6
                end_point = 'denseblock3'
                layers3 = self.layers[2]
                growth = 0.5 * growth
                net = self.block(net, layers3, growth, scope=end_point)
                end_points[end_point] = net
                if end_point == final_endpoint: return net, end_points
                # batchsize * ( 640 + 6 * 0.5 * 0.5 * k ) (batchsize * 832 )
                # transition layer3
                end_point = 'translayer3'
                outunits = int(outunits * self.outrate)
                net = self.transition(net, outunits, scope=end_point)
                end_points[end_point] = net
                if end_point == final_endpoint: return net, end_points
                # batchsize * 416

                # Classification layer
                # 分类器进行两次降维最终达到分类目的
                end_point = 'classifer'
                net = slim.fully_connected(net, 104, scope=scope + '_globalclass1')
                logits = slim.fully_connected(net, num_classes, activation_fn=None, scope=scope + '_globalclass2')
                # logits = slim.fully_connected(net, num_classes, scope=scope + '_globalclass2')
                end_points[end_point] = logits
                if end_point == final_endpoint: return net, end_points
                end_point = 'prediction'
                #end_points[end_point] = slim.softmax(logits, scope=end_point)
                end_points[end_point] = tf.nn.sigmoid(logits, name=end_point)

        return logits, end_points

    def bn_drp_scope(self, is_training=True, keep_prob=0.8):
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
        self.continous_inputs = tf.placeholder(tf.float32, [None,4], name='continous_inputs')
        self.categorial_inputs = tf.placeholder(tf.float32, name='categorial_inputs')
        self.label = tf.placeholder(tf.float32, name='label')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.global_step = tf.Variable(0, trainable=False, name='self.global_step', dtype=tf.int64)

        with tf.variable_scope('builddensenet'):
            self.logits, self.end_points = self.densenet(self.continous_inputs, self.categorial_inputs, dropout_keep_prob=self.keep_prob)
            self.log_loss = tf.losses.log_loss(labels=self.label, predictions=self.end_points['prediction'])
            #self.loss = self.log_loss
            self.loss = tf.reduce_mean(self.log_loss)

        step_learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps, self.decay_rate, staircase=True, name='step_learning_rate')
        optimizer = tf.train.FtrlOptimizer(step_learning_rate)
        self.gradients = optimizer.compute_gradients(self.loss)
        self.train_step = optimizer.apply_gradients(self.gradients, global_step=self.global_step)

        with tf.name_scope("score"):
            correct_prediction = tf.equal(tf.to_float(self.end_points['prediction'] > 0.5), self.label)
            self.accuracy = tf.reduce_mean(tf.to_float(correct_prediction), name="accuracy")
        self.auc = tf.metrics.auc(labels=self.label, predictions=self.end_points['prediction'])

        #summaries
        grad_summaries = []
        for g, v in self.gradients:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name.replace(':', '_')), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name.replace(':', '_')),
                                                     tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        loss_summary = tf.summary.scalar("loss", self.loss)
        acc_summary = tf.summary.scalar("accuracy", self.accuracy)

        self.train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
