import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
import numpy as np
import pickle
import collections
import datetime
import config
import os
import random



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

#判断日期为星期几	
def to_weekday(date):
    week = datetime.datetime.strptime(str(date // 100),'%y%m%d').strftime('%a')
    return week

#数据下采样
def down_sampling(tr_data,label):
    temp_0=tr_data[label]== 0
    data_0=tr_data[temp_0] 
    temp_1=tr_data[label]==1
    data_1=tr_data[temp_1] 
    data_0_ed=data_0[0:len(data_1)]
    data_downsampled=pd.concat([data_1,data_0_ed])
    return data_downsampled

#对训练集，预测集新建特征
def create_feature(train_path,test_path ,datadir,dn_sample = True):
    
    if dn_sample:
        train_df = pd.read_csv(train_path + 'train.csv')
        train_df = down_sampling(tr_data = train_df,label = 'click') #数据下采样

        test_df = pd.read_csv(test_path + 'test.csv')
    else:
        nrow = 4000000
        train_df = pd.read_csv(train_path + 'train.csv',nrows = nrow)
        test_df = pd.read_csv(test_path + 'test.csv',nrows = nrow)


    test_df['click'] = 0 

    train_data = pd.concat([train_df, test_df],ignore_index=True)

    print("finished loading raw data, ", train_data.shape)
    train_data['day']=np.round(train_data.hour % 10000 / 100).astype('int') #生成日期，某天
    train_data['hour_n'] = np.round(train_data.hour % 100) #抽取单位时
    train_data['weekday'] = list(map(to_weekday,train_data.hour))[0]
    train_data['app_or_web'] = 0 #生成app，web识别特征
    train_data.ix[train_data.app_id.values=='ecad2386', 'app_or_web'] = 1
    train_data['C15_C16'] = np.add(train_data.C15.map(str),train_data.C16.map(str)) #组合图形尺寸
    print("finished creating feature, ", train_data.shape)

    vn_list = ['click','C1', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21',
            'app_category', 'app_domain', 'app_id', 'banner_pos', 'device_conn_type',
            'device_id','device_ip', 'device_model','device_type', 'site_category', 
            'site_domain', 'site_id', 'day','hour_n', 'weekday', 'app_or_web', 'C15_C16']

    train_data2 = train_data.loc[:len(train_df),vn_list]
    test_data2 = train_data.loc[len(train_df):,vn_list].drop('click',axis = 1)

    train_data2.to_csv(datadir + 'fe_add_train_data.csv',index = None,header = 0)
    test_data2.to_csv(datadir + 'fe_add_test_data.csv',index = None,header = 0)
    print("finished exporting train_data/test_data after increasing features")



# 分类型变量生成 
#cutoff 超参数，利用cutoff对特征的特征值计数进行处理，低于该值都归为一类
class CategoryDictGenerator:
    """
    Generate dictionary for each of the categorical features
    """

    def __init__(self, num_feature):
        self.dicts = []
        self.num_feature = num_feature #类别型特征的数目
        for i in range(0, num_feature):#列表中元素依次为对应位置的特征及计数的字典
            self.dicts.append(collections.defaultdict(int))
#获取分类型特征的符合cutoff条件的特征值的名称，且是降序的
    def build(self, datafile, categorial_features, cutoff=0):#categorial_features序列，cutoff切割点,传入
        with open(datafile, 'r') as f:
            for line in f:
                features = line.rstrip('\n').split(',')
                for i in range(0, self.num_feature):
                    if features[categorial_features[i]] != '':#对应位置的特征的值
                        self.dicts[i][features[categorial_features[i]]] += 1 #该值对应字典的值加1，计数，字典里统计各种特征值的计数
        for i in range(0, self.num_feature):
            self.dicts[i] = filter(lambda x: x[1] >= cutoff,
                                   self.dicts[i].items())# 根据分割点保留符合条件的特征值对，将新的字典放回

            self.dicts[i] = sorted(self.dicts[i], key=lambda x: (-x[1], x[0]))#字典按值降序排列，放回列表
            vocabs, _ = list(zip(*self.dicts[i])) #将特征值与对应计数分开成两个，对应的元组，前面是特征值，后面是计数
            self.dicts[i] = dict(zip(vocabs, range(1, len(vocabs) + 1))) #新字典，键是特征值，值是序号？放回列表
            self.dicts[i]['<unk>'] = 0 #对应位置的特征的字典中添加特殊值的键，值为0

    def gen(self, idx, key):#idx为特征对应的位置，key为特征值
        if key not in self.dicts[idx]:#获取序号？
            res = self.dicts[idx]['<unk>']
        else:
            res = self.dicts[idx][key]
        return res

    def dicts_sizes(self):#返回每个特征的符合条件的特征值的个数，返回为列表
        return list(map(len, self.dicts))
		

#对分类型特征编码，同时分割处训练集，校验集，预测集
#预测集中首列为click,全0
#
def preprocess(datadir, outdir,categorial_features):#预处理
    dicts = CategoryDictGenerator(len(categorial_features))
    dicts.build(os.path.join(datadir, 'fe_add_train_data.csv'), categorial_features, cutoff=20)#200 50
    
    dict_sizes = dicts.dicts_sizes()#每个特征符合条件的特征值个数列表，长度等于分类型特征个数
    categorial_feature_offset = [0]
    
    for i in range(1, len(categorial_features)):#1，分类型特征个数,1-26,25个  ？ 
        offset = categorial_feature_offset[i - 1] + dict_sizes[i - 1]
        categorial_feature_offset.append(offset)

    random.seed(0)
    
    train_ffm = open(os.path.join(outdir, 'train_ffm.txt'), 'w')
    valid_ffm = open(os.path.join(outdir, 'valid_ffm.txt'), 'w')

    train_lgb = open(os.path.join(outdir, 'train_lgb.txt'), 'w')
    valid_lgb = open(os.path.join(outdir, 'valid_lgb.txt'), 'w')
    
    with open(os.path.join(outdir, 'train.txt'), 'w') as out_train:
        with open(os.path.join(outdir, 'valid.txt'), 'w') as out_valid:
            with open(os.path.join(datadir, 'fe_add_train_data.csv'), 'r') as f: #新增变量后训练数据读取
                for line in f:
                    features = line.rstrip('\n').split(',')
                
                    categorial_vals = []
                    categorial_lgb_vals = []
                    for i in range(0, len(categorial_features)):#分类型两个列表，一个GBDT，一个神经网络
                        val = dicts.gen(i, features[categorial_features[i]]) + categorial_feature_offset[i] #？序号 + 该特征的特征值个数，累积的
                        categorial_vals.append(str(val))
                        val_lgb = dicts.gen(i, features[categorial_features[i]]) # 感觉想label in code ?
                        categorial_lgb_vals.append(str(val_lgb))
                    
                    categorial_vals = ','.join(categorial_vals)
                    label = features[0] #首列不是id，而是标签
                
                    if random.randint(0, 9999) % 10 != 0: #九成训练
                        out_train.write(','.join([categorial_vals, label]) + '\n')
                        train_ffm.write('\t'.join(label) + '\t')
                        train_ffm.write('\t'.join(['{}:{}:1'.format(ii, str(np.int32(val))) for ii, val in enumerate(categorial_vals.split(','))]) + '\n')

                        train_lgb.write('\t'.join(label) + '\t')
                        train_lgb.write('\t'.join(categorial_lgb_vals) + '\n')
                    
                    else:
                        out_valid.write(','.join([categorial_vals, label]) + '\n')
                        valid_ffm.write('\t'.join(label) + '\t')
                        valid_ffm.write('\t'.join(['{}:{}:1'.format(ii , str(np.int32(val))) for ii, val in enumerate(categorial_vals.split(','))]) + '\n')

                        valid_lgb.write('\t'.join(label) + '\t')
                        valid_lgb.write('\t'.join(categorial_lgb_vals) + '\n')
                    
    print("finished train")
    train_ffm.close()
    print("finished train_ffm")
    valid_ffm.close()
    print("finished valid_ffm")

    train_lgb.close()
    print("finished train_lgb")
    valid_lgb.close()
    print("finished valid_lgb")
    
    test_ffm = open(os.path.join(outdir, 'test_ffm.txt'), 'w') #读取
    test_lgb = open(os.path.join(outdir, 'test_lgb.txt'), 'w')

    with open(os.path.join(outdir, 'test.txt'), 'w') as out: #同样编码的新预测集
        with open(os.path.join(datadir, 'fe_add_test_data.csv'), 'r') as f:#新增变量后预测数据读取
            for line in f:
                features = line.rstrip('\n').split(',')

                categorial_vals = []
                categorial_lgb_vals = []
                for i in range(0, len(categorial_features)):
                    val = dicts.gen(i,
                                    features[categorial_features[i] -
                                             1]) + categorial_feature_offset[i]
                    categorial_vals.append(str(val))

                    val_lgb = dicts.gen(i, features[categorial_features[i] - 1])
                    categorial_lgb_vals.append(str(val_lgb))

                categorial_vals = ','.join(categorial_vals)

                out.write(','.join([categorial_vals]) + '\n')

                test_ffm.write('\t'.join(
                    ['{}:{}:1'.format(ii, str(np.int32(val))) for ii, val in enumerate(categorial_vals.split(','))]) + '\n')

                test_lgb.write('\t'.join(categorial_lgb_vals) + '\n')

    print("finished test")
    test_ffm.close()
    print("finished test_ffm")
    test_lgb.close()
    print("finished test_lgb")
	
