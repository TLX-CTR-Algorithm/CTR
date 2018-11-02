import tensorflow as tf
from DNN import flags

slim = tf.contrib.slim
FLAGS, unparsed = flags.parse_args()

class Model():
    def __init__(self, learning_rate, oridata_dim, embed_max, embed_dim=128, decay_steps=5000, decay_rate=0.96):
        self.learning_rate = learning_rate
        self.outunits = [1024, 128, 8]
        self.oridata_dim = oridata_dim
        self.embed_dim = int(embed_dim)
        self.embed_max = int(embed_max)
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate

    def embeding(self,embed_inputs):
        with tf.name_scope("embedding"):
            embed_inputs = tf.cast(embed_inputs,tf.int32)
            oridata_dim = self.oridata_dim
            embed_matrix = tf.Variable(tf.random_uniform([self.embed_max, self.embed_dim], -1, 1), name="embed_matrix")
            embed_layer = tf.nn.embedding_lookup(embed_matrix, embed_inputs, name="embed_layer")
            embed_layer = tf.reshape(embed_layer, [-1, self.embed_dim * oridata_dim])
        return embed_layer

    def construction(self,continous_inputs, categorial_inputs, ffm_logits, keep_prob):
        end_points = {}

        #进行embeding
        end_point = 'embeding'
        #embed_inputs = inputs[:,config.encod_cat_index_begin:config.encod_cat_index_end]
        branch_embed = self.embeding(categorial_inputs)
        end_points[end_point] = branch_embed

        # 连续型特征和非连续特征需要拆分处理
        # 对类别型特征进行embeding处理，连续型特征直接和embeding后的数据连接后输入网络中
        end_point = 'continous_net'
        #branch_continous = inputs[:,0:config.encod_cat_index_begin]
        #branch_continous = inputs[:, 0:4]
        branch_continous_2 = slim.fully_connected(continous_inputs,40, activation_fn=None, scope=end_point)
        #branch_continous_2 = slim.fully_connected(continous_inputs, 40, weights_initializer=tf.initializers.random_uniform(),scope=end_point)
        end_point = 'concat_con_cat'
        net = tf.concat([branch_continous_2,branch_embed],1,name=end_point)
        net = slim.batch_norm(net, scope=end_point + '_bn2')

        end_point = 'fully_con1'
        #net = slim.fully_connected(net, self.outunits[0], activation_fn=None, scope=end_point)
        net = slim.fully_connected(net, self.outunits[0], scope=end_point)
        end_points[end_point] = net

        end_point = 'fully_con2'
        #net = slim.fully_connected(net, self.outunits[1], activation_fn=None, scope=end_point)
        net = slim.fully_connected(net, self.outunits[1], scope=end_point)
        end_points[end_point] = net

        # dropout层
        end_point = 'dropout'
        net = slim.dropout(net, keep_prob=keep_prob, scope=end_point)
        end_points[end_point] = net

        end_point = 'logits'
        #logits = slim.fully_connected(net, self.outunits[2], activation_fn=None, scope=end_point)
        logits = slim.fully_connected(net, self.outunits[2], scope=end_point)
        end_points[end_point] = logits

        end_point = 'merge'
        net = tf.concat([logits, ffm_logits], 1, name=end_point)
        logits = slim.fully_connected(net, self.outunits[2], scope=end_point)
        end_points[end_point] = logits

        #模型融合的时候只需要使用logits进行融合即可，这里求prediction主要是为了便于单独训练用，从而对模型性能进行验证
        end_point = 'prediction'
        prediction = slim.fully_connected(net, 1, activation_fn=tf.nn.sigmoid, scope=end_point)
        end_points[end_point] = prediction

        #with open('./log/graph.log', 'w')as graph:
        #    graph.write(str(tf.get_default_graph().as_graph_def()))

        return logits, end_points

    def build(self):
        self.continous_inputs = tf.placeholder(tf.float32, [None,FLAGS.encod_cat_index_begin], name='continous_inputs')
        #self.categorial_inputs = tf.placeholder(tf.float32, [None,26], name='categorial_inputs')
        self.categorial_inputs = tf.placeholder(tf.float32, name='categorial_inputs')
        self.ffm_logits = tf.placeholder(tf.float32, [None,1], name='ffm_logits')
        self.label = tf.placeholder(tf.float32, name='label')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.global_step = tf.Variable(0, trainable=False, name='self.global_step', dtype=tf.int64)

        with tf.variable_scope('fullyconnect_3'):
            self.logits, self.end_points = self.construction(self.continous_inputs, self.categorial_inputs, self.ffm_logits, self.keep_prob)
            self.log_loss = tf.losses.log_loss(labels=self.label,predictions=self.end_points['prediction'])
            self.loss = tf.reduce_mean(self.log_loss)

        step_learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps, self.decay_rate, staircase=True, name='step_learning_rate')
        #self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss,global_step=self.global_step)
        #self.optimizer = train_op.apply_gradients()
        optimizer = tf.train.FtrlOptimizer(step_learning_rate)
        #结果证明使用adam优化器，在这个问题上效果比Ftrl优化器差很多
        #optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.gradients = optimizer.compute_gradients(self.loss)
        self.train_step = optimizer.apply_gradients(self.gradients, global_step=self.global_step)

        #Accuracy
        with tf.name_scope("score"):
            correct_prediction = tf.equal(tf.to_float(self.end_points['prediction'] > 0.5), self.label)
            self.accuracy = tf.reduce_mean(tf.to_float(correct_prediction), name="accuracy")
        self.auc = tf.metrics.auc(self.label, self.end_points['prediction'])

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
        #self.train_summary_op = tf.summary.merge([loss_summary, grad_summaries_merged])
