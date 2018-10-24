import tensorflow as tf
from DNN import model2
from DNN import model
import logging
import utils
import pandas as pd
import os
import numpy as np
import re
from DNN import flags

slim = tf.contrib.slim
FLAGS, unparsed = flags.parse_args()

if not os.path.exists(FLAGS.dnn_log_dir):
    os.mkdir(FLAGS.dnn_log_dir)
#设置日志打印格式
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
sh = logging.StreamHandler()
sh.setFormatter(logging.Formatter('%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s'))
fl = logging.FileHandler(FLAGS.dnn_log_path)
fl.setFormatter(logging.Formatter('%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s'))
logger.addHandler(sh)
logger.addHandler(fl)
#logging.basicFLAGS(level=logging.DEBUG,datefmt='%a, %d %b %Y %h:%M:%S',)

#模型训练函数
def train_model(batch_size=FLAGS.batch_size):
    #获取训练数据
    inputs, lables = utils.gendata(flag='train',train_path=FLAGS.encod_train_path,vaild_path=FLAGS.encod_vaild_path,test_path=FLAGS.encod_test_path)
    categorial_data = inputs[:,FLAGS.encod_cat_index_begin:FLAGS.encod_cat_index_end]
    logging.debug('oridata_dim:{}'.format(categorial_data.shape[1]))

    try:
        dictsizes = pd.read_csv(FLAGS.dictsizefile)
        dictsize_list = np.array(dictsizes)
        embed_max = sum(dictsize_list[:,1])
    except:
        embed_max = np.max(categorial_data[-1])
    logging.debug('embed_max:{}'.format(embed_max))

    #获取校验数据
    valid_inputs,valid_labels = utils.gendata(flag='valid',train_path=FLAGS.encod_train_path,vaild_path=FLAGS.encod_vaild_path,test_path=FLAGS.encod_test_path)

    #构建网络模型
    dnnmodel = model.Model(learning_rate=FLAGS.learning_rate, oridata_dim=categorial_data.shape[1], embed_max=embed_max, embed_dim=FLAGS.embed_dim )
    dnnmodel.build()

    #如果没有checkpoint文件则需要对所有变量进行初始化
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        logging.debug('Initialized')

        #校验加载ckp文件
        try:
            saver = tf.train.Saver(max_to_keep=5)
            checkpoint_path = tf.train.latest_checkpoint(FLAGS.model_ouput_dir)
            saver.restore(sess, checkpoint_path)
            logging.debug('restore from [{0}]'.format(checkpoint_path))
        except Exception:
            logging.debug('nocheck point found...')

        train_summary_dir = os.path.join(FLAGS.summary_dir, "train")
        try:
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)
        except:
            os.mkdir(train_summary_dir)
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        #会按照配置内容来进行最大step之内的训练,到达Max_step自动停止训练
        global_step = 0
        epoch=0
        while 1 == 1:
            # 使用训练数据进行模型训练
            batches = utils.genbatch(inputs, lables, batch_size=FLAGS.batch_size)
            #logging.info('epoch:{}'.format(epoch))
            for step in range(len(inputs) // batch_size):
                batch_inputs,batch_lables = next(batches)
                continous_inputs = batch_inputs[:, 0:FLAGS.encod_cat_index_begin]
                categorial_inputs = batch_inputs[:,FLAGS.encod_cat_index_begin:FLAGS.encod_cat_index_end]
                feed_dict = { dnnmodel.categorial_inputs:categorial_inputs, dnnmodel.continous_inputs:continous_inputs, dnnmodel.label:batch_lables, dnnmodel.keep_prob:FLAGS.keep_prob }
                #with tf.Session() as sess:
                global_step, _, logits, loss, accuracy, summaries = sess.run([dnnmodel.global_step, dnnmodel.train_step, dnnmodel.logits, dnnmodel.loss, dnnmodel.accuracy, dnnmodel.train_summary_op], feed_dict=feed_dict)
                train_summary_writer.add_summary(summaries, step)
                if global_step % FLAGS.logfrequency == 0:
                    #每间隔指定的频率打印日志并存储checkpoint文件
                    logging.info('train: step [{0}] loss [{1}] accuracy [{2}]'.format(global_step, loss, accuracy))
                    try:
                        saver.save(sess, os.path.join(FLAGS.model_ouput_dir, "model.ckpt"), global_step=global_step)
                    except:
                        os.mkdir(FLAGS.model_ouput_dir)
                        saver.save(sess, os.path.join(FLAGS.model_ouput_dir, "model.ckpt"), global_step=global_step)
                if global_step >= FLAGS.Max_step:
                    break

            logging.info('----------------------valid-----------------------')
            #使用验证数据，验证模型性能
            valid_batches = utils.genbatch(valid_inputs, valid_labels, batch_size=FLAGS.batch_size)
            for step in range(len(valid_inputs) // batch_size):
                batch_valid_inputs,batch_valid_lables = next(valid_batches)
                valid_continous_inputs = batch_valid_inputs[:, 0:FLAGS.encod_cat_index_begin]
                valid_categorial_inputs = batch_valid_inputs[:,FLAGS.encod_cat_index_begin:FLAGS.encod_cat_index_end]
                feed_dict = { dnnmodel.categorial_inputs:valid_categorial_inputs, dnnmodel.continous_inputs:valid_continous_inputs, dnnmodel.label:batch_valid_lables, dnnmodel.keep_prob:FLAGS.keep_prob }
                #with tf.Session() as sess:
                valid_global_step, logits, loss, accuracy = sess.run([dnnmodel.global_step, dnnmodel.logits, dnnmodel.loss, dnnmodel.accuracy], feed_dict=feed_dict)
                #if valid_global_step % FLAGS.logfrequency == 0:
                if step % FLAGS.logfrequency == 0:
                    #每间隔指定的频率打印日志并存储checkpoint文件
                    logging.info('valid: step [{0}] loss [{1}] accuracy [{2}]'.format(global_step, loss, accuracy))
                    #saver.save(sess, os.path.join(FLAGS.model_ouput_dir, "model.ckpt"), global_step=global_step)

            if global_step >= FLAGS.Max_step:
                break
            logging.info('next epoch')
            epoch = epoch + 1

if __name__ == '__main__':
    #train部分
    #参数
    for i in dir(FLAGS):
        if re.match(r"_.*",i):
            pass
        else:
            logging.info('{}:{}'.format(i,getattr(FLAGS,i)))

    train_model()


