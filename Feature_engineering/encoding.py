import pandas as pd
import numpy as np
import scipy as sc
import sys
import utils
import config
import xgboost as xgb

train_data0 = pd.read_csv(config.train_path)
test_data = pd.read_csv(config.test_path)

train_data0['tr_or_te'] = 'tr'
test_data['tr_or_te'] = 'te'

test_data['click'] = 0 
train_data = pd.concat([train_data0, test_data])
click = train_data['click'].values
tr_or_te = train_data['tr_or_te'].values

print("finished loading raw data, ", train_data.shape)

print("to add some basic features ...")
train_data['day']=np.round(train_data.hour % 10000 / 100) 
train_data['hour1'] = np.round(train_data.hour % 100)
train_data['day_hour'] = (train_data.day.values - 21) * 24 + train_data.hour1.values
train_data['day_hour_prev'] = train_data['day_hour'] - 1
train_data['day_hour_next'] = train_data['day_hour'] + 1
train_data['app_or_web'] = 0
train_data.ix[train_data.app_id.values=='ecad2386', 'app_or_web'] = 1
train_data['C15_C16'] = np.add(train_data.C15.values,train_data.C16.values)

train_data_cy = train_data.copy()

print("to encode categorical features using mean responses from earlier days -- univariate")

cate_vn_list = [ 'hour', 'C1', 'banner_pos', 'site_id', 'site_domain',
       'site_category', 'app_id', 'app_domain', 'app_category', 'device_id',
       'device_ip', 'device_model', 'device_type', 'device_conn_type', 'C14',
       'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21','day','hour1','day_hour',
        'day_hour_prev','day_hour_next','app_or_web','C15_C16']

#哑编码数据
data_dum = utils.en_dummy(data=train_data_cy[cate_vn_list], cate_vn_list = cate_vn_list)
data_dum['click'] = click
data_dum['tr_or_te'] = tr_or_te

t0 = data_dum.loc[data_dum['tr_or_te'] == 'tr',].drop('tr_or_te',axis = 1)
h0 = data_dum.loc[data_dum['tr_or_te'] == 'te',].drop('tr_or_te',axis = 1)