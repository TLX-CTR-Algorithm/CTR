import config
import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import sys

#日志配置
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %h:%M:%S',
                    )

def analyze_detail(data):
    categorical_features = data.select_dtypes(include=["int64"]).columns
    for col in categorical_features:
        logging.debug('\n{0}属性的不同取值和出现的次数: \n{1}'.format(col,data[col].value_counts()))

def analyze( **kwargs ):

    datadict={}
    datadict.update(kwargs)

    for key in datadict:
        data=datadict[key]
        if data.__class__.__name__ == 'DataFrame':
            # pandas显示设置
            pd.set_option('display.width', 300)
            pd.set_option('display.max_colwidth', 100)
            pd.set_option('display.max_columns', 30)

            # logging.debug层级日志会输出需要分析的数据的各个特性,可以注释掉不需要的属性
            logging.debug('\n{0}.shape: \n{1}'.format(key,data.shape))
            #logging.debug('\n{0}.value: \n{1}'.format(key, data))
            logging.debug('\n{0}.head: \n{1}'.format(key,data.head(5)))

        if data.__class__.__name__ == 'ndarray':
            #np.set_printoptions(threshold=1000000)
            logging.debug('\n{0}.shape: \n{1}'.format(key, data.shape))
            logging.debug('\n{0}.value: \n{1}'.format(key, data[0:2]))


def standard(data):
    encoder = StandardScaler()
    #encoder.fit(np.unique(data.values))
    #analyze(uniquevale=np.unique(data.values))
    #encoderdata = data.apply(encoder.transform)
    encoderdata = encoder.fit_transform(data)

    return encoderdata

def splitfealabdata(data,flag='train'):
    feature_data = data[:,0:-1]
    if flag == 'train'or flag == 'valid':
        label_data = data[:,-1]
        return feature_data, label_data
    elif flag == 'test':
        return feature_data
    else:
        logging.error('arguments of function splitdata must be train,test or valid')
        sys.exit()

def splitdata(data,flag='train'):
    continous_data = data[:, 0:config.encod_cat_index_begin]
    categorial_data = data[:,config.encod_cat_index_begin:config.encod_cat_index_end]
    if flag == 'train'or flag == 'valid':
        label_data = data[:,-1]
        return continous_data,categorial_data, label_data
    elif flag == 'test':
        return continous_data,categorial_data
    else:
        logging.error('arguments of function splitdata must be train,test or valid')
        sys.exit()

def genbatch(feature_data,label_data=None,batch_size=200):
    for start in range(0, len(feature_data), batch_size):
        end = min(start + batch_size, len(feature_data))
        if label_data is None:
            yield feature_data[start:end]
        else:
            yield feature_data[start:end], label_data[start:end]

#获取数据,视需求可调用其中的数据分析功能
def gendata(flag='train'):

    if flag == 'train':
        train_data = np.loadtxt(config.encod_train_path,delimiter=',')
        # 数据拆分
        train_continous_data, train_categorial_data, train_data_label = splitdata(train_data)
        train_continous_standard_data = standard(train_continous_data)
        train_feature_data = np.concatenate([train_continous_standard_data,train_categorial_data],axis=1)
        return train_feature_data, train_data_label
    elif flag == 'valid':
        valid_data = np.loadtxt(config.encod_vaild_path,delimiter=',')
        valid_continous_data, valid_categorial_data, valid_data_label = splitdata(valid_data, flag='valid')
        valid_continous_standard_data = standard(valid_continous_data)
        valid_feature_data = np.concatenate([valid_continous_standard_data, valid_categorial_data], axis=1)
        return valid_feature_data, valid_data_label
    elif flag == 'test':
        test_data = np.loadtxt(config.encod_test_path,delimiter=',')
        test_continous_data, test_categorial_data = splitdata(test_data, flag='valid')
        test_continous_standard_data = standard(test_continous_data)
        test_feature_data = np.concatenate([test_continous_standard_data, test_categorial_data], axis=1)
        return test_feature_data
    else:
        logging.error('arguments of function gendata must be train,test or valid')
        sys.exit()

if __name__ == '__main__':
    train_data, train_lable = gendata()
    analyze(train_data=train_data)





