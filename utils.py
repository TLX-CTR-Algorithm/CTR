import pandas as pd
import numpy as np
import collections
import datetime
import os
import sys
import random
import csv
import config
import logging

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

def gen_continous_clip(tr_path,filename):
    tr_data = pd.read_csv(os.path.join(tr_path, filename))
    continous_clip = []
    for i in range(1,7):
        perc = np.percentile(tr_data.iloc[:,i],95)
        continous_clip.append(perc)
    del tr_data
    return continous_clip
  
  
def down_sampling(tr_path, label, outpath):
    '''
    数据下采样
    '''
    tr_data = pd.read_csv(os.path.join(tr_path, 'train.csv'))
    print('train data is loaded,down_sampling start')
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
    sampler = np.random.permutation(len(data_downsampled))
    data_downsampled=data_downsampled.take(sampler)
    data_downsampled.to_csv(os.path.join(
        outpath, 'train.csv'), index=None)
    print('train data is loaded,down_sampling end')
    del data_downsampled

    
def up_sampling(tr_path, label, outpath):
    '''
    数据上采样,只取最后最近三天数据
    '''
    tr_data = pd.read_csv(os.path.join(tr_path, 'train.csv'))
    print('train data is loaded,up_sampling start')
    tr_data = tr_data.loc[tr_data['hour'] >= 14102800, :]
    temp_0 = tr_data[label] == 0
    data_0 = tr_data[temp_0]
    temp_1 = tr_data[label] == 1
    data_1 = tr_data[temp_1]
    del tr_data
    sampler = np.random.randint(data_1.shape[0], size=data_0.shape[0])
    data_1_ed = data_1.iloc[sampler, :]
    data_upsampled = pd.concat([data_1_ed, data_0], ignore_index=True)
    del data_0
    del data_1
    del data_1_ed
    data_upsampled.to_csv(os.path.join(
        outpath, 'train.csv'), index=None)
    print('train data is loaded,up_sampling end')
    del data_upsampled

    
def scan(path, is_trian):
    '''
    统计设备id，设备ip，用户，用户-时间的频数，各设备id的点击率，各设备ip的点击率
    '''
    id_cnt = collections.defaultdict(int)
    id_cnt_1 = collections.defaultdict(int)
    ip_cnt = collections.defaultdict(int)
    ip_cnt_1 = collections.defaultdict(int)
    user_cnt = collections.defaultdict(int)
    user_hour_cnt = collections.defaultdict(int)
    file = open(path)
    for i, row in enumerate(csv.DictReader(file), start=1):
        # print(row)
        user = def_user(row)

        id_cnt[row['device_id']] += 1  # 统计device_id各特征值计数,反映该设备浏览广告数目
        ip_cnt[row['device_ip']] += 1  # 统计device_ip各特征值计数，反映该ip浏览广告数目
        if is_trian:
                id_cnt_1[row['device_id']] += int(row['click'])
                ip_cnt_1[row['device_ip']] += int(row['click'])

        user_cnt[user] += 1  # 用户计数，各浏览者浏览广告数目，反映具体人广告推送的情况
        user_hour_cnt[user + '-' + row['hour']] += 1  # 组合具体人与时间，反映具体人的活动时间分布
    file.close()
    return  id_cnt, ip_cnt, user_cnt, user_hour_cnt,id_cnt_1, ip_cnt_1
