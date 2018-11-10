import tensorflow as tf
from DNN import model2
from DNN import model
import logging
import utils
import pandas as pd
import os
import math
import numpy as np
import re
from DNN import flags

slim = tf.contrib.slim
FLAGS, unparsed = flags.parse_args()

def get_tensors(loaded_graph):
    continous_inputs = loaded_graph.get_tensor_by_name("continous_inputs:0")
    categorial_inputs = loaded_graph.get_tensor_by_name("categorial_inputs:0")
    label = loaded_graph.get_tensor_by_name("label:0")
    keep_prob = loaded_graph.get_tensor_by_name("keep_prob:0")
    prediction = loaded_graph.get_tensor_by_name("fullyconnect_3/prediction/Sigmoid:0")
    ffm_logist = loaded_graph.get_tensor_by_name("ffm_logits:0")

    return continous_inputs, categorial_inputs, label, keep_prob,prediction

def test_model(batch_size=FLAGS.batch_size):
    loaded_graph = tf.Graph()
    with tf.Session(graph=loaded_graph) as sess:
        loader = tf.train.import_meta_graph(FLAGS.model_ouput_dir + 'model.ckpt-6670.meta')
        loader.restore(sess, FLAGS.model_ouput_dir + 'model.ckpt-6670')
        model_continous_inputs, model_categorial_inputs, model_label, model_keep_prob,model_prediction, model_ffm_logist = get_tensors(loaded_graph)

        inputs = utils.gendata(flag='test', train_path=FLAGS.encod_train_path, vaild_path=FLAGS.encod_vaild_path, test_path=FLAGS.encod_test_path)
        ffm = utils.genffm(flag='test',train_path=FLAGS.ffm_train_path,vaild_path=FLAGS.ffm_valid_path,test_path=FLAGS.ffm_test_path)
        batches = utils.genbatch(inputs, batch_size=FLAGS.batch_size)
        batches2 = utils.genbatch(ffm, batch_size=FLAGS.batch_size)


        pre_file = open('./log/test_prediction.txt',"wb")
        for step in range(math.ceil(len(inputs)/batch_size)):
            batch_inputs = next(batches)
            batch_ffm = next(batches2)
            continous_inputs = batch_inputs[:, 0:FLAGS.encod_cat_index_begin]
            categorial_inputs = batch_inputs[:, FLAGS.encod_cat_index_begin:FLAGS.encod_cat_index_end]
            feed_dict = {model_categorial_inputs: categorial_inputs, model_continous_inputs: continous_inputs,model_keep_prob: FLAGS.keep_prob, model_ffm_logist:batch_ffm}
            prediction = sess.run( model_prediction, feed_dict=feed_dict)
            np.savetxt(pre_file, prediction)
        pre_file.close()

if __name__ == '__main__':
    test_model()