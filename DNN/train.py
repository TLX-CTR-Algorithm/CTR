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
#logging.basicFLAGS(level=logging.DEBUG,datefmt='%a, %d %b %Y %h:%M:%S',)

#模型训练函数
def train_model(batch_size=FLAGS.batch_size):
    #获取训练数据
    inputs, lables = utils.gendata(flag='train',train_path=FLAGS.encod_train_path,vaild_path=FLAGS.encod_vaild_path,test_path=FLAGS.encod_test_path)
    categorial_data = inputs[:,FLAGS.encod_cat_index_begin:FLAGS.encod_cat_index_end]
    logging.debug('oridata_dim:{}'.format(categorial_data.shape[1]))
    count_data = lables.shape[0]
    logging.debug('count_data:{}'.format(count_data))

    try:
        dictsizes = pd.read_csv(FLAGS.dictsizefile)
        dictsize_list = np.array(dictsizes)
        embed_max = sum(dictsize_list[:,1])
    except:
        embed_max = int(np.max(categorial_data) + 1)
    logging.debug('embed_max:{}'.format(embed_max))

    #获取校验数据
    valid_inputs,valid_labels = utils.gendata(flag='valid',train_path=FLAGS.encod_train_path,vaild_path=FLAGS.encod_vaild_path,test_path=FLAGS.encod_test_path)

    #构建网络模型
    if FLAGS.model_flag == 'model':
        dnnmodel = model.Model(learning_rate=FLAGS.learning_rate, oridata_dim=categorial_data.shape[1], embed_max=embed_max, embed_dim=FLAGS.embed_dim, decay_steps=FLAGS.decay_steps, decay_rate=FLAGS.decay_rate )
    elif FLAGS.model_flag == 'model2':
        dnnmodel = model2.Model(learning_rate=FLAGS.learning_rate, oridata_dim=categorial_data.shape[1], embed_max=embed_max, embed_dim=FLAGS.embed_dim, decay_steps=FLAGS.decay_steps, decay_rate=FLAGS.decay_rate)
    dnnmodel.build()

    #如果没有checkpoint文件则需要对所有变量进行初始化
    #with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
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
        epoch = 1
        while 1 == 1:
            # 使用训练数据进行模型训练
            batches = utils.genbatch(inputs, lables, batch_size=FLAGS.batch_size)
            train_loss_list=[]
            train_auc_list = []
            train_accuracy_list = []
            for step in range(len(inputs) // batch_size):
                batch_inputs,batch_lables = next(batches)
                continous_inputs = batch_inputs[:, 0:FLAGS.encod_cat_index_begin]
                categorial_inputs = batch_inputs[:,FLAGS.encod_cat_index_begin:FLAGS.encod_cat_index_end]
                feed_dict = { dnnmodel.categorial_inputs:categorial_inputs, dnnmodel.continous_inputs:continous_inputs, dnnmodel.label:batch_lables, dnnmodel.keep_prob:FLAGS.keep_prob }
                #with tf.Session() as sess:
                global_step, _, logits, loss, accuracy, summaries, auc, end_points, labels = sess.run([dnnmodel.global_step, dnnmodel.train_step, dnnmodel.logits, dnnmodel.loss, dnnmodel.accuracy, dnnmodel.train_summary_op, dnnmodel.auc, dnnmodel.end_points, dnnmodel.label], feed_dict=feed_dict)
                train_summary_writer.add_summary(summaries, step)
                train_loss_list.append(loss)
                train_auc_list.append(auc[0])
                train_accuracy_list.append(accuracy)
                #np.savetxt('./log/tlogits.log', end_points['logits'])
                #np.savetxt('./log/tpre.log', end_points['prediction'])
                #np.savetxt('./log/tlabels.log', labels)
                if global_step % FLAGS.logfrequency == 0:
                    #每间隔指定的频率打印日志并存储checkpoint文件
                    logging.debug('train: step [{0}] loss [{1}] auc [{2}] accuracy [{3}]'.format(global_step, loss, auc, accuracy))
                    try:
                        saver.save(sess, os.path.join(FLAGS.model_ouput_dir, "model.ckpt"), global_step=global_step)
                    except:
                        os.mkdir(FLAGS.model_ouput_dir)
                        saver.save(sess, os.path.join(FLAGS.model_ouput_dir, "model.ckpt"), global_step=global_step)
                #if global_step >= FLAGS.Max_step or global_step > epoch * batch_size:
                if global_step >= FLAGS.Max_step:
                    break
            train_loss = np.mean(train_loss_list)
            train_auc = np.mean(train_auc_list, 0)
            train_accuracy = np.mean(train_accuracy_list)

            logging.debug('----------------------valid-----------------------')
            #使用验证数据，验证模型性能
            if FLAGS.valid_switch == 0:
                valid_continous_inputs = valid_inputs[:, 0:FLAGS.encod_cat_index_begin]
                valid_categorial_inputs = valid_inputs[:, FLAGS.encod_cat_index_begin:FLAGS.encod_cat_index_end]
                feed_dict = {dnnmodel.categorial_inputs: valid_categorial_inputs,
                             dnnmodel.continous_inputs: valid_continous_inputs, dnnmodel.label: valid_labels,
                             dnnmodel.keep_prob: FLAGS.keep_prob}
                valid_global_step, logits, loss, accuracy, auc, end_points, labels = sess.run([dnnmodel.global_step, dnnmodel.logits, dnnmodel.loss, dnnmodel.accuracy, dnnmodel.auc, dnnmodel.end_points, dnnmodel.label], feed_dict=feed_dict)
                logging.debug(
                    'valid: step [{0}] loss [{1}] auc [{2}] accuracy [{3}]'.format(global_step, loss, auc,
                                                                                   accuracy))
            else:
                valid_batches = utils.genbatch(valid_inputs, valid_labels, batch_size=FLAGS.batch_size)
                loss_list = []
                auc_list = []
                accuracy_list = []
                for step in range(len(valid_inputs) // batch_size):
                    batch_valid_inputs,batch_valid_lables = next(valid_batches)
                    valid_continous_inputs = batch_valid_inputs[:, 0:FLAGS.encod_cat_index_begin]
                    valid_categorial_inputs = batch_valid_inputs[:,FLAGS.encod_cat_index_begin:FLAGS.encod_cat_index_end]
                    feed_dict = { dnnmodel.categorial_inputs:valid_categorial_inputs, dnnmodel.continous_inputs:valid_continous_inputs, dnnmodel.label:batch_valid_lables, dnnmodel.keep_prob:FLAGS.keep_prob }
                    valid_global_step, logits, loss, accuracy, auc, end_points, labels = sess.run([dnnmodel.global_step, dnnmodel.logits, dnnmodel.loss, dnnmodel.accuracy, dnnmodel.auc, dnnmodel.end_points, dnnmodel.label], feed_dict=feed_dict)

                    loss_list.append(loss)
                    auc_list.append(auc[0])
                    accuracy_list.append(accuracy)
                    #np.savetxt('./log/logits.log', end_points['logits'])
                    #np.savetxt('./log/pre.log', end_points['prediction'])
                    #np.savetxt('./log/labels.log', labels)
                    #if step % FLAGS.logfrequency == 0:
                        #每间隔指定的频率打印日志并存储checkpoint文件
                     #   logging.info('valid: step [{0}] loss [{1}] auc [{2}] accuracy [{3}]'.format(global_step, loss, auc, accuracy))
                valid_loss = np.mean(loss_list)
                valid_auc = np.mean(auc_list,0)
                valid_accuracy = np.mean(accuracy_list)
                logging.debug( 'valid: step [{0}] loss [{1}] auc [{2}] accuracy [{3}]'.format(global_step, valid_loss, valid_auc, valid_accuracy))

            #epoch = (global_step * batch_size) // count_data
            epoch = math.ceil((global_step * batch_size) / count_data)
            logging.debug('has completed epoch:{}'.format(epoch))

            logging.info('epoch [{0}] train_loss [{1}] valid_loss [{2}] train_auc [{3}] valid_auc [{4}] train_accuracy [{5}] valid_accuracy [{6}]'.format(
                epoch, train_loss, valid_loss, train_auc, valid_auc, train_accuracy, valid_accuracy
            ))
            if epoch >= FLAGS.Max_epoch or global_step >= FLAGS.Max_step:
                break

if __name__ == '__main__':
    #train部分
    #参数
    for i in dir(FLAGS):
        if re.match(r"_.*",i):
            pass
        else:
            logging.info('{}:{}'.format(i,getattr(FLAGS,i)))

    train_model()


