import tensorflow as tf
import config

slim = tf.contrib.slim

class Model():
    def __init__(self, learning_rate,oridata_dim):
        self.learning_rate = learning_rate
        self.outunits = [1024, 128, 8]
        self.dim_embedding = 5
        self.dim_embedding=5
        self.oridata_dim = oridata_dim

    def embeding(self,embed_inputs, oridata_dim=23):
        with tf.name_scope("embedding"):
            #oridata_dim = embed_inputs.shape[1]
            #oridata_dim = config.oridata_dim
            oridata_dim = self.oridata_dim
            embed_matrix = tf.Variable(tf.random_uniform([config.embed_max, config.embed_dim], -1, 1), name="embed_matrix")
            embed_layer = tf.nn.embedding_lookup(embed_matrix, embed_inputs, name="embed_layer")
            embed_layer = tf.reshape(embed_layer, [-1, config.embed_dim * oridata_dim])
        return embed_layer

    def construction(self, inputs,keep_prob):
        end_points = {}

        #进行embeding
        end_point = 'embeding'
        net = self.embeding(inputs)
        end_points[end_point] = net

        end_point = 'fully_con1'
        net = slim.fully_connected(net, self.outunits[0], activation_fn=None, scope=end_point)
        end_points[end_point] = net

        end_point = 'fully_con2'
        net = slim.fully_connected(net, self.outunits[1], activation_fn=None, scope=end_point)
        end_points[end_point] = net

        # dropout层
        end_point = 'dropout'
        net = slim.dropout(net, keep_prob=keep_prob, scope=end_point)
        end_points[end_point] = net

        end_point = 'logits'
        logits = slim.fully_connected(net, self.outunits[2], activation_fn=None, scope=end_point)
        end_points[end_point] = logits

        #模型融合的时候只需要使用logits进行融合即可，这里求prediction主要是为了便于单独训练用，从而对模型性能进行验证
        end_point = 'prediction'
        prediction = slim.fully_connected(net, 2, activation_fn=tf.nn.sigmoid, scope=end_point)
        end_points[end_point] = prediction

        return logits, end_points

    def build(self):
        self.inputs = tf.placeholder(tf.int32, name='inputs')
        self.label = tf.placeholder(tf.int32, name='inputs')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.global_step = tf.Variable(0, trainable=False, name='self.global_step', dtype=tf.int64)

        with tf.variable_scope('fullyconnect_3'):
            self.logits, self.end_points = self.construction(self.inputs, self.keep_prob)
            self.loss = tf.losses.log_loss(labels=self.label,predictions=self.end_points['prediction'])

        self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss,global_step=self.global_step)
        #self.optimizer = train_op.apply_gradients()
