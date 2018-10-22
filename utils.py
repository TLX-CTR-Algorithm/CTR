import pandas as pd
from sklearn.preprocessing import LabelEncoder
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

continous_features = range(1, 5)
categorial_features = range(5, 31)
continous_clip = [21356, 9789, 557, 51]

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


def create_feature(data_path, datadir, filename=None):
    '''
    构建新特征，生成数据集
    data_path:采样后数据位置
    datadir:生成数据存储位置
    '''
    if filename == 'dnsp_train.csv':
        data_df = pd.read_csv(data_path + filename)
    else:
        filename = 'test.csv'
        data_df = pd.read_csv(data_path + filename)
    print("finished loading raw data, ", filename, data_df.shape)

    data_df['user'] = gen_userid(data=data_df)
    device_id_cnt, device_ip_cnt, user_cnt, user_hour_cnt = scan(
        data_path + filename)

    data_df['user_count'] = data_df['user'].map(user_cnt)
    data_df['user_hour'] = data_df['user'] + '-' + data_df['hour'].map(str)
    data_df['smooth_user_hour_count'] = data_df['user_hour'].map(user_hour_cnt)
    data_df['device_id_count'] = data_df['device_id'].map(device_id_cnt)
    data_df['device_ip_count'] = data_df['device_ip'].map(device_ip_cnt)
    data_df['day'] = np.round(data_df.hour %
                              10000 / 100).astype('int')  # 生成日期，某天
    data_df['hour_n'] = np.round(data_df.hour % 100)  # 抽取单位时
    data_df['weekday'] = list(map(to_weekday, data_df.hour))[0]
    data_df['app_or_web'] = 0  # 生成app，web识别特征
    data_df.loc[data_df.app_id.values == 'ecad2386', 'app_or_web'] = 1
    data_df['C15_C16'] = np.add(data_df.C15.map(
        str), data_df.C16.map(str))  # 组合图形尺寸
    data_df['pub_id'] = np.where(data_df['site_id'].map(is_app),
                                 data_df['app_id'], data_df['site_id'])
    data_df['pub_domain'] = np.where(data_df['site_id'].map(is_app), data_df['app_domain'],
                                     data_df['site_domain'])
    data_df['pub_category'] = np.where(data_df['site_id'].map(is_app), data_df['app_category'],
                                       data_df['site_category'])

    print("finished creating feature, ", filename, data_df.shape)

    vn_list = ['click', 'device_id_count', 'device_ip_count', 'user_count',
               'smooth_user_hour_count', 'C1', 'C14', 'C17', 'C18', 'C19',
               'C21', 'app_category', 'app_domain', 'app_id', 'banner_pos',
               'device_conn_type', 'device_id', 'device_ip', 'device_model',
               'device_type', 'site_category', 'site_domain', 'site_id', 'day',
               'hour_n', 'weekday', 'app_or_web', 'C15_C16', 'pub_id',
               'pub_domain', 'pub_category']

    if filename == 'dnsp_train.csv':
        data_df = data_df.loc[:, vn_list]
        data_df.to_csv(datadir + 'fe_add_train_data.csv', index=None, header=0)
        print('fe_add_train_data.csv is in ' + datadir)
    else:
        vn_list.remove('click')
        data_df = data_df.loc[:, vn_list]
        data_df.to_csv(datadir + 'fe_add_test_data.csv', index=None, header=0)
        print('fe_add_test_data.csv is in ' + datadir)


# 生成连续型特征
class ContinuousFeatureGenerator:
    """
    Normalize the integer features to [0, 1] by min-max normalization
    """

    def __init__(self, num_feature):  # 初始化，与传入特征长度一致的列表，用来存放最大最小值
        self.num_feature = num_feature
        self.min = [sys.maxsize] * num_feature
        self.max = [-sys.maxsize] * num_feature

    def build(self, datafile, continous_features):
        with open(datafile, 'r') as f:
            for line in f:
                features = line.rstrip('\n').split(',')  # 从文本中抽取数据，并根据分隔符分开
                for i in range(0, self.num_feature):  # 估计是除去首列id列，循环处理前面13个连续特征
                    val = features[continous_features[i]]  # 该行i对应的特征的值
                    if val != '':
                        val = int(val)  # 向下取整
                        # 若大于分割点，则以分割点为值，也就是去除异常值，超过95%为异常点
                        if val > continous_clip[i]:
                            val = continous_clip[i]
                        # 与该位置的值比较，取较小值，找出该位置的最小值
                        self.min[i] = min(self.min[i], val)
                        # 与该位置的值比较，取较大值，找出该位置的最大值
                        self.max[i] = max(self.max[i], val)

    def gen(self, idx, val):  # val为值，idx是连续型变量的位置，从0开始
        if val == '':
            return 0.0
        val = float(val)
        # 对应位置做最大最小化
        return (val - self.min[idx]) / (self.max[idx] - self.min[idx])


# 分类型变量生成
# cutoff 超参数，利用cutoff对特征的特征值计数进行处理，低于该值都归为一类
class CategoryDictGenerator:
    """
    Generate dictionary for each of the categorical features
    """

    def __init__(self, num_feature):
        self.dicts = []
        self.num_feature = num_feature  # 类别型特征的数目
        for i in range(0, num_feature):  # 列表中元素依次为对应位置的特征及计数的字典
            self.dicts.append(collections.defaultdict(int))
# 获取分类型特征的符合cutoff条件的特征值的名称，且是降序的

    def build(self, datafile, categorial_features, cutoff=0):  # categorial_features序列，cutoff切割点,传入
        with open(datafile, 'r') as f:
            for line in f:
                features = line.rstrip('\n').split(',')
                for i in range(0, self.num_feature):
                    if features[categorial_features[i]] != '':  # 对应位置的特征的值
                        # 该值对应字典的值加1，计数，字典里统计各种特征值的计数
                        self.dicts[i][features[categorial_features[i]]] += 1
        for i in range(0, self.num_feature):
            self.dicts[i] = filter(lambda x: x[1] >= cutoff,
                                   self.dicts[i].items())  # 根据分割点保留符合条件的特征值对，将新的字典放回

            self.dicts[i] = sorted(
                self.dicts[i], key=lambda x: (-x[1], x[0]))  # 字典按值降序排列，放回列表
            # 将特征值与对应计数分开成两个，对应的元组，前面是特征值，后面是计数
            vocabs, _ = list(zip(*self.dicts[i]))
            # 新字典，键是特征值，值是序号？放回列表
            self.dicts[i] = dict(zip(vocabs, range(1, len(vocabs) + 1)))
            self.dicts[i]['<unk>'] = 0  # 对应位置的特征的字典中添加特殊值的键，值为0

    def gen(self, idx, key):  # idx为特征对应的位置，key为特征值
        if key not in self.dicts[idx]:  # 获取序号？
            res = self.dicts[idx]['<unk>']
        else:
            res = self.dicts[idx][key]
        return res

    def dicts_sizes(self):  # 返回每个特征的符合条件的特征值的个数，返回为列表
        return list(map(len, self.dicts))


def preprocess(datadir, outdir, continous_features, categorial_features):  # 预处理
    """
    分割数据集，用于ffm,GBDT,fcn
    """

    dists = ContinuousFeatureGenerator(len(continous_features))
    dists.build(os.path.join(datadir, 'fe_add_train_data.csv'),
                continous_features)  # 得到了最大最小值

    dicts = CategoryDictGenerator(len(categorial_features))
    dicts.build(
        os.path.join(datadir, 'fe_add_train_data.csv'), categorial_features, cutoff=25)  # 200 50

    dict_sizes = dicts.dicts_sizes()  # 每个特征符合条件的特征值个数列表，长度等于分类型特征个数
    categorial_feature_offset = [0]
    for i in range(1, len(categorial_features)):  # 1，分类型特征个数,1-26,25个  ？
        offset = categorial_feature_offset[i - 1] + dict_sizes[i - 1]
        categorial_feature_offset.append(offset)

    random.seed(0)

    # 分割数据集
    train_ffm = open(os.path.join(outdir, 'train_ffm.txt'), 'w')
    valid_ffm = open(os.path.join(outdir, 'valid_ffm.txt'), 'w')

    train_lgb = open(os.path.join(outdir, 'train_lgb.txt'), 'w')
    valid_lgb = open(os.path.join(outdir, 'valid_lgb.txt'), 'w')

    with open(os.path.join(outdir, 'train.txt'), 'w') as out_train:
        with open(os.path.join(outdir, 'valid.txt'), 'w') as out_valid:
            with open(os.path.join(datadir, 'fe_add_train_data.csv'), 'r') as f:
                for line in f:
                    features = line.rstrip('\n').split(',')
                    continous_feats = []  # 连续型，GBDT
                    continous_vals = []  # 连续型
                    for i in range(0, len(continous_features)):  # 连续型两个列表，一个ffm，一个神经网络

                        # 实际位置，该行实际值，最大最小化
                        val = dists.gen(i, features[continous_features[i]])
                        continous_vals.append(
                            "{0:.6f}".format(val).rstrip('0').rstrip('.'))  # 取六位小数
                        continous_feats.append(
                            "{0:.6f}".format(val).rstrip('0').rstrip('.'))  # ('{0}'.format(val))

                    categorial_vals = []
                    categorial_lgb_vals = []
                    for i in range(0, len(categorial_features)):  # 分类型两个列表，一个GBDT，一个神经网络
                        # ？序号 + 该特征的特征值个数，累积的
                        val = dicts.gen(
                            i, features[categorial_features[i]]) + categorial_feature_offset[i]
                        categorial_vals.append(str(val))
                        # 感觉想label in code ?
                        val_lgb = dicts.gen(
                            i, features[categorial_features[i]])
                        categorial_lgb_vals.append(str(val_lgb))

                    continous_vals = ','.join(continous_vals)
                    categorial_vals = ','.join(categorial_vals)
                    label = features[0]  # 首列不是id，而是标签
                    if random.randint(0, 9999) % 10 != 0:  # 九成训练
                        out_train.write(','.join(
                            [continous_vals, categorial_vals, label]) + '\n')
                        train_ffm.write('\t'.join(label) + '\t')
                        train_ffm.write('\t'.join(
                            ['{}:{}:{}'.format(ii, ii, val) for ii, val in enumerate(continous_vals.split(','))]) + '\t')
                        train_ffm.write('\t'.join(
                            ['{}:{}:1'.format(ii + 4, str(np.int32(val) + 4)) for ii, val in enumerate(categorial_vals.split(','))]) + '\n')

                        train_lgb.write('\t'.join(label) + '\t')
                        train_lgb.write('\t'.join(continous_feats) + '\t')
                        train_lgb.write('\t'.join(categorial_lgb_vals) + '\n')

                    else:  # 一成校验
                        out_valid.write(','.join(
                            [continous_vals, categorial_vals, label]) + '\n')
                        valid_ffm.write('\t'.join(label) + '\t')
                        valid_ffm.write('\t'.join(
                            ['{}:{}:{}'.format(ii, ii, val) for ii, val in enumerate(continous_vals.split(','))]) + '\t')
                        valid_ffm.write('\t'.join(
                            ['{}:{}:1'.format(ii + 4, str(np.int32(val) + 4)) for ii, val in enumerate(categorial_vals.split(','))]) + '\n')

                        valid_lgb.write('\t'.join(label) + '\t')
                        valid_lgb.write('\t'.join(continous_feats) + '\t')
                        valid_lgb.write('\t'.join(categorial_lgb_vals) + '\n')
    train_ffm.close()
    print('finished train_ffm')
    valid_ffm.close()
    print('finished valid_ffm')

    train_lgb.close()
    print('finished train_lgb')
    valid_lgb.close()
    print('finished valid_lgb')

    test_ffm = open(os.path.join(outdir, 'test_ffm.txt'), 'w')  # 读取
    test_lgb = open(os.path.join(outdir, 'test_lgb.txt'), 'w')

    with open(os.path.join(outdir, 'test.txt'), 'w') as out:  # 同样编码的新预测集
        with open(os.path.join(datadir, 'fe_add_test_data.csv'), 'r') as f:  # ？真预测集
            for line in f:
                features = line.rstrip('\n').split(',')

                continous_feats = []
                continous_vals = []
                for i in range(0, len(continous_features)):
                    val = dists.gen(i, features[continous_features[i] - 1])
                    continous_vals.append(
                        "{0:.6f}".format(val).rstrip('0').rstrip('.'))
                    continous_feats.append(
                        "{0:.6f}".format(val).rstrip('0').rstrip('.'))  # ('{0}'.format(val))

                categorial_vals = []
                categorial_lgb_vals = []
                for i in range(0, len(categorial_features)):
                    val = dicts.gen(i,
                                    features[categorial_features[i] -
                                             1]) + categorial_feature_offset[i]
                    categorial_vals.append(str(val))

                    val_lgb = dicts.gen(
                        i, features[categorial_features[i] - 1])
                    categorial_lgb_vals.append(str(val_lgb))

                continous_vals = ','.join(continous_vals)
                categorial_vals = ','.join(categorial_vals)

                out.write(','.join([continous_vals, categorial_vals]) + '\n')

                test_ffm.write('\t'.join(['{}:{}:{}'.format(
                    ii, ii, val) for ii, val in enumerate(continous_vals.split(','))]) + '\t')
                test_ffm.write('\t'.join(
                    ['{}:{}:1'.format(ii + 4, str(np.int32(val) + 4)) for ii, val in enumerate(categorial_vals.split(','))]) + '\n')

                test_lgb.write('\t'.join(continous_feats) + '\t')
                test_lgb.write('\t'.join(categorial_lgb_vals) + '\n')

    test_ffm.close()
    print('finished test_ffm')
    test_lgb.close()
    print('finished test_lgb')

