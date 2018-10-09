
import tensorflow as tf
import model
import config
import pandas as pd

slim = tf.contrib.slim

#需要添加校验checkpoint文件的代码
#需要考虑添加summary部分代码

def gendata( is_training=True ):
    #with open(config.train_path,encoding='utf-8') as trainf:
    #此处代码需要根据输入数据形态进行改进
    if is_training:
        with pd.read_csv(config.train_path,encoding='utf-8') as trainf:
            train_x = trainf[:-2]
            train_y = trainf[-1]
            return train_x,train_y
    else:
        with pd.read_csv(config.test_path,encoding='utf-8') as testf:
            test_x = testf[:-1]
            return test_x

if __name__ == '__main__':
    #train部分
    inputs,lables = gendata()
    with tf.Seesion() as initsess:
        initsess.run(tf.global_variables_initializer())
    with tf.Session() as sess:
        feed_dict={ model.inputs:inputs, model.label:lables, model.keep_prob:0.8 }
        sess.run(model.train_step,feed_dict=feed_dict)
