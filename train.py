import tensorflow as tf
#from model2 import Model
import model2
import config
import logging
import sample
import os

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
    inputs, lables = sample.gendata(is_training=True)

    logging.debug('oridata_dim:'.format(inputs.shape[1]))
    #构建网络模型
    dnnmodel = model2.Model(learning_rate=config.learning_rate, oridata_dim=inputs.shape[1])
    dnnmodel.build()

    #如果没有checkpoint文件则需要对所有变量进行初始化
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        logging.debug('Initialized')

        #校验加载ckp文件
        try:
            saver = tf.train.Saver(max_to_keep=5)
            checkpoint_path = tf.train.latest_checkpoint(config.output_dir)
            saver.restore(sess, checkpoint_path)
            logging.debug('restore from [{0}]'.format(checkpoint_path))
        except Exception:
            logging.debug('nocheck point found...')

        #会按照配置内容来进行最大step之内的训练,到达Max_step自动停止训练
        global_step = 0
        while 1 == 1:
            batches = sample.genbatch( inputs, lables, batch_size=config.batch_size)
            #进行1个epoch的训练
            #for step in range(len(inputs)/batch_size):
            for step in range(len(inputs) // batch_size):
                batch_inputs,batch_lables = next(batches)
                feed_dict = { dnnmodel.inputs:batch_inputs, dnnmodel.label:batch_lables, dnnmodel.keep_prob:config.keep_prob }
                #with tf.Session() as sess:
                global_step, _, logits, loss = sess.run([dnnmodel.global_step, dnnmodel.train_step, dnnmodel.logits, dnnmodel.loss], feed_dict=feed_dict)
                if global_step % config.logfrequency == 0:
                    #每间隔指定的频率打印日志并存储checkpoint文件
                    logging.info('step [{0}] loss [{1}]'.format(global_step, loss))
                    saver.save(sess, os.path.join(config.output_dir, "model.ckpt"), global_step=global_step)
                if global_step >= config.Max_step:
                    break
            if global_step >= config.Max_step:
                break

if __name__ == '__main__':
    #train部分
    train_model()







