import tensorflow as tf
#from model2 import Model
import model2
import config
import logging
import sample
import pandas as pd
import os
import numpy as np

slim = tf.contrib.slim

#设置日志打印格式
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %h:%M:%S',
                    )

#需要考虑添加summary部分代码

#模型训练函数
def train_model(batch_size=config.batch_size):
    #获取训练数据
    inputs, lables = sample.gendata(flag='train')
    categorial_data = inputs[:,config.encod_cat_index_begin:config.encod_cat_index_end]
    logging.debug('oridata_dim:{}'.format(categorial_data.shape[1]))
    dictsizes = pd.read_csv(config.dictsizefile)
    dictsize_list = np.array(dictsizes)
    embed_max = sum(dictsize_list[:,1])
    logging.debug('embed_max:{}'.format(embed_max))

    #获取校验数据
    valid_inputs,valid_labels = sample.gendata(flag='valid')

    #构建网络模型
    dnnmodel = model2.Model(learning_rate=config.learning_rate, oridata_dim=categorial_data.shape[1], embed_max=embed_max )
    dnnmodel.build()

    #如果没有checkpoint文件则需要对所有变量进行初始化
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        logging.debug('Initialized')

        #校验加载ckp文件
        try:
            saver = tf.train.Saver(max_to_keep=5)
            checkpoint_path = tf.train.latest_checkpoint(config.model_ouput_dir)
            saver.restore(sess, checkpoint_path)
            logging.debug('restore from [{0}]'.format(checkpoint_path))
        except Exception:
            logging.debug('nocheck point found...')

        train_summary_dir = os.path.join(config.summary_dir, "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        #会按照配置内容来进行最大step之内的训练,到达Max_step自动停止训练
        global_step = 0
        epoch=0
        while 1 == 1:
            # 使用训练数据进行模型训练
            batches = sample.genbatch(inputs, lables, batch_size=config.batch_size)
            #logging.info('epoch:{}'.format(epoch))
            for step in range(len(inputs) // batch_size):
                batch_inputs,batch_lables = next(batches)
                continous_inputs = batch_inputs[:, 0:config.encod_cat_index_begin]
                categorial_inputs = batch_inputs[:,config.encod_cat_index_begin:config.encod_cat_index_end]
                feed_dict = { dnnmodel.categorial_inputs:categorial_inputs, dnnmodel.continous_inputs:continous_inputs, dnnmodel.label:batch_lables, dnnmodel.keep_prob:config.keep_prob }
                #with tf.Session() as sess:
                global_step, _, logits, loss, accuracy, summaries = sess.run([dnnmodel.global_step, dnnmodel.train_step, dnnmodel.logits, dnnmodel.loss, dnnmodel.accuracy, dnnmodel.train_summary_op], feed_dict=feed_dict)
                train_summary_writer.add_summary(summaries, step)
                if global_step % config.logfrequency == 0:
                    #每间隔指定的频率打印日志并存储checkpoint文件
                    logging.info('train: step [{0}] loss [{1}] accuracy [{2}]'.format(global_step, loss, accuracy))
                    saver.save(sess, os.path.join(config.model_ouput_dir, "model.ckpt"), global_step=global_step)
                if global_step >= config.Max_step:
                    break

            logging.info('----------------------valid-----------------------')
            #使用验证数据，验证模型性能
            valid_batches = sample.genbatch(valid_inputs, valid_labels, batch_size=config.batch_size)
            for step in range(len(valid_inputs) // batch_size):
                batch_valid_inputs,batch_valid_lables = next(valid_batches)
                valid_continous_inputs = batch_valid_inputs[:, 0:config.encod_cat_index_begin]
                valid_categorial_inputs = batch_valid_inputs[:,config.encod_cat_index_begin:config.encod_cat_index_end]
                feed_dict = { dnnmodel.categorial_inputs:valid_categorial_inputs, dnnmodel.continous_inputs:valid_continous_inputs, dnnmodel.label:batch_valid_lables, dnnmodel.keep_prob:config.keep_prob }
                #with tf.Session() as sess:
                valid_global_step, logits, loss, accuracy = sess.run([dnnmodel.global_step, dnnmodel.logits, dnnmodel.loss, dnnmodel.accuracy], feed_dict=feed_dict)
                #if valid_global_step % config.logfrequency == 0:
                if step % config.logfrequency == 0:
                    #每间隔指定的频率打印日志并存储checkpoint文件
                    logging.info('valid: step [{0}] loss [{1}] accuracy [{2}]'.format(global_step, loss, accuracy))
                    #saver.save(sess, os.path.join(config.model_ouput_dir, "model.ckpt"), global_step=global_step)

            if global_step >= config.Max_step:
                break
            logging.info('next epoch')
            epoch = epoch + 1

if __name__ == '__main__':
    #train部分
    train_model()







