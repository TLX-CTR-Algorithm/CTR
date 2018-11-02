import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
import numpy as np
import pickle
import collections
import datetime
import os
import sys
import random
import hashlib
import csv
import logging
from DNN import flags
FLAGS, unparsed = flags.parse_args()


def en_dummy(data, cate_vn_list):
    # one_hot 编码
    for feature in cate_vn_list:
        le = LabelEncoder()
        data[feature] = le.fit_transform(data[feature])
        max_ = data[feature].max()
        data[feature] = (data[feature] - max_) * (-1)

    data = data.loc[:, cate_vn_list]
    en_d = preprocessing.OneHotEncoder()
    en_d.fit(data)
    data = en_d.transform(data).toarray()
    result = pd.DataFrame(data)
    return result

def one_hot_representation(sample, fields_dict, array_length):
    """
    One hot presentation for every sample data
    :param fields_dict: fields value to array index
    :param sample: sample data, type of pd.series
    :param array_length: length of one-hot representation
    :return: one-hot representation, type of np.array
    """
    array = np.zeros([array_length])
    for field in fields_dict:
        # get index of array
        if field == 'hour':
            field_value = int(str(sample[field])[-2:])
        else:
            field_value = sample[field]
        ind = fields_dict[field][field_value]
        array[ind] = 1
    return array



NR_BINS = 1000000


def hashstr(input):
    '''
    对特征hash
    '''
    return str(int(hashlib.md5(input.encode('utf8')).hexdigest(), 16) % (NR_BINS - 1) + 1)


def gen_folder(dnsp_data_dir, fe_gen_dir, model_data_dir):
    '''
    构建文件夹
    '''
    if not os.path.exists(dnsp_data_dir):
        os.makedirs(dnsp_data_dir)
    if not os.path.exists(fe_gen_dir):
        os.makedirs(fe_gen_dir)
    if not os.path.exists(model_data_dir):
        os.makedirs(model_data_dir)


def def_user(row):
    '''
    定位用户，作为用户id
    '''
    if row['device_id'] == 'a99f214a':
        user = 'ip-' + row['device_ip'] + '-' + row['device_model']
    else:
        user = 'id-' + row['device_id']

    return user


def gen_userid(data):

    user = np.where(data['device_id'] == 'a99f214a', 'ip-' + data['device_ip']
                    + '-' + data['device_model'], 'id-' + data['device_id'])
    return user


def is_app(site_id):
    '''
    判断用户访问方式，是否为app
    '''
    return True if site_id == '85f751fd' else False



def to_weekday(date):
    '''
    判断日期为星期几
    '''
    week = datetime.datetime.strptime(
        str(date // 100), '%y%m%d').strftime('%a')
    return week


def down_sampling(tr_path, label, outpath):
    '''
    数据下采样
    '''
    tr_data = pd.read_csv(os.path.join(tr_path, 'train.csv'))
    temp_0 = tr_data[label] == 0
    data_0 = tr_data[temp_0]
    temp_1 = tr_data[label] == 1
    data_1 = tr_data[temp_1]
    sampler = np.random.permutation(data_0.shape[0])[:data_1.shape[0]]
    data_0_ed = data_0.iloc[sampler, :]
    data_downsampled = pd.concat([data_1, data_0_ed], ignore_index=True)
    del tr_data
    del data_0
    del data_1
    del data_0_ed
    data_downsampled.to_csv(os.path.join(
        outpath, 'dnsp_train.csv'), index=None)
    del data_downsampled


def scan(path):
    '''
    统计设备id，设备ip，用户，用户-时间的频数
    '''
    id_cnt = collections.defaultdict(int)
    ip_cnt = collections.defaultdict(int)
    user_cnt = collections.defaultdict(int)
    user_hour_cnt = collections.defaultdict(int)
    for i, row in enumerate(csv.DictReader(open(path)), start=1):
        # print(row)
        user = def_user(row)

        id_cnt[row['device_id']] += 1  # 统计device_id各特征值计数,反映该设备浏览广告数目
        ip_cnt[row['device_ip']] += 1  # 统计device_ip各特征值计数，反映该ip浏览广告数目
        user_cnt[user] += 1  # 用户计数，各浏览者浏览广告数目，反映具体人广告推送的情况
        user_hour_cnt[user + '-' + row['hour']] += 1  # 组合具体人与时间，反映具体人的活动时间分布
    return id_cnt, ip_cnt, user_cnt, user_hour_cnt


# 数据标准化
def standard(data):
    encoder = StandardScaler()
    encoderdata = encoder.fit_transform(data)
    return encoderdata

# 数据拆分，默认数据最后一列数据为label
def splitfealabdata(data,flag='train'):
    feature_data = data[:,0:-1]
    if flag == 'train'or flag == 'valid':
        label_data = data[:,-1]
        return feature_data, label_data
    elif flag == 'test':
        return feature_data
    else:
        logging.error('arguments of function splitfealabdata must be train,test or valid')
        sys.exit()

# 连续型数据和类别型数据拆分，默认数据最后一列数据为label
def splitdata(data,flag='train',index_begin=4,index_end=30):
    encod_cat_index_begin = index_begin
    encod_cat_index_end = index_end
    continous_data = data[:, 0:encod_cat_index_begin]
    categorial_data = data[:,encod_cat_index_begin:encod_cat_index_end]
    if flag == 'train'or flag == 'valid':
        label_data = data[:,-1]
        return continous_data,categorial_data, label_data
    elif flag == 'test':
        return continous_data,categorial_data
    else:
        logging.error('arguments of function splitdata must be train,test or valid')
        sys.exit()

# 根据batchsize获取数据
def genbatch(feature_data,label_data=None,batch_size=200):
    for start in range(0, len(feature_data), batch_size):
        end = min(start + batch_size, len(feature_data))
        if label_data is None:
            yield feature_data[start:end]
        else:
            yield feature_data[start:end], label_data[start:end]

# 获取特征工程处理后的数据
def gendata(flag='train',train_path='output/model_data/train.txt',vaild_path='output/model_data/valid.txt',test_path='output/model_data/test.txt'):
    encod_train_path = train_path
    encod_vaild_path = vaild_path
    encod_test_path = test_path

    if flag == 'train':
        train_data = np.loadtxt(encod_train_path,delimiter=',')
        # 数据拆分
        train_continous_data, train_categorial_data, train_data_label = splitdata(train_data,index_begin=FLAGS.encod_cat_index_begin,index_end=FLAGS.encod_cat_index_end)
        train_continous_standard_data = standard(train_continous_data)
        train_feature_data = np.concatenate([train_continous_standard_data,train_categorial_data],axis=1)
        return train_feature_data, train_data_label
    elif flag == 'valid':
        valid_data = np.loadtxt(encod_vaild_path,delimiter=',')
        valid_continous_data, valid_categorial_data, valid_data_label = splitdata(valid_data, flag='valid',index_begin=FLAGS.encod_cat_index_begin,index_end=FLAGS.encod_cat_index_end)
        valid_continous_standard_data = standard(valid_continous_data)
        valid_feature_data = np.concatenate([valid_continous_standard_data, valid_categorial_data], axis=1)
        return valid_feature_data, valid_data_label
    elif flag == 'test':
        test_data = np.loadtxt(encod_test_path,delimiter=',')
        test_continous_data, test_categorial_data = splitdata(test_data, flag='test',index_begin=FLAGS.encod_cat_index_begin,index_end=FLAGS.encod_cat_index_end)
        test_continous_standard_data = standard(test_continous_data)
        test_feature_data = np.concatenate([test_continous_standard_data, test_categorial_data], axis=1)
        return test_feature_data
    elif flag == 'ffm':
        filename_queue = tf.train.string_input_producer([encod_ffm_path])
        reader = tf.FixedLengthRecordReader()
        ffm_data = reader.read(filename_queue)
        return ffm_data
    else:
        logging.error('arguments of function gendata must be train,test or valid')
        sys.exit()

# 获取FFM模型的数据
def genffm(flag='train',train_path='output/model_data/train_ffm.txt.bin',vaild_path='output/model_data/valid_ffm.txt.bin',test_path='output/model_data/test_ffm.txt.bin'):
    if flag == 'train':
        ffm_path = train_path
    elif flag == 'valid':
        ffm_path = vaild_path
    elif flag == 'test':
        ffm_path = test_path
    else:
        logging.error('arguments of function gendata must be train,test or valid')
        sys.exit()
    #filename_queue = tf.train.string_input_producer([ffm_path])
    #reader = tf.FixedLengthRecordReader(record_bytes=1)
    #ffm_data = reader.read(filename_queue)
    ffm_data = np.fromfile(ffm_path, dtype=np.int32)
    return ffm_data[:,np.newaxis]

