import config
import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

#日志配置
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %h:%M:%S',
                    )

def analyze_detail(data):
    #categorical_features = data.select_dtypes(include=["object"]).columns
    categorical_features = data.select_dtypes(include=["int64"]).columns
    for col in categorical_features:
        logging.debug('\n{0}属性的不同取值和出现的次数: \n{1}'.format(col,data[col].value_counts()))

def analyze( **kwargs ):

    datadict={}
    datadict.update(kwargs)
    #logging.debug('params:{}'.format(params))

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

def splitdata(data,is_train=True):
    #num_data = data[config.keyofnum]
    #object_data = data[config.keyofobject]
    feature_data = data.drop(config.keyoflable,axis=1).astype(str)
    if is_train:
        label_data = data[config.keyoflable]
        #return num_data, object_data, label_data
        return feature_data, label_data
    else:
        #return num_data, object_data
        return feature_data

def objnumeric(data):
    encoder = LabelEncoder()
    encoder.fit(np.unique(data.values))
    analyze(uniquevale=np.unique(data.values))
    encoderdata = data.apply(encoder.transform)

    return encoderdata

def genbatch(feature_data,label_data=None,batch_size=200):
    for start in range(0, len(feature_data), batch_size):
        end = min(start + batch_size, len(feature_data))
        if label_data is None:
            yield feature_data[start:end]
        else:
            yield feature_data[start:end], label_data[start:end]

#获取数据,视需求可调用其中的数据分析功能
def gendata(is_training=True):

    if is_training:
        train_data = pd.read_csv(config.train_path)
        # 数据拆分
        #train_num_data, train_obj_data, train_data_label = splitdata(train_data)
        train_feature_data, train_data_label = splitdata(train_data)

        analyze(train_feature_data=train_feature_data)
        # 数字化编码
        #train_encodata = objnumeric(train_obj_data)
        # analyze(train_encodata=train_encodata)
        #train_data = np.concatenate((np.array(train_num_data, dtype=np.int64), train_encodata), axis=1)
        train_data = objnumeric(train_feature_data)
        train_lable = np.array(train_data_label, dtype=np.int64)
        # analyze(train_data=train_data)
        return train_data, train_lable
    else:
        test_data = pd.read_csv(config.test_path)
        # 数据拆分
        test_num_data, test_obj_data = splitdata(test_data,is_train=False)
        # 数字化编码
        test_encodata  = objnumeric(test_obj_data)
        #analyze(train_num_data=train_num_data)
        test_data  = np.concatenate((np.array(test_num_data, dtype=np.int64), test_encodata), axis=1)
        #analyze(test_data=test_data)
        return test_data

if __name__ == '__main__':
    gendata()






