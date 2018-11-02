import numpy as np
import pandas as pd
import config
import pickle
import logging
from collections import Counter

"""
site_id, site_domain, app_id, app_domain, device_model,
device_ip, device_id fields,C14,C17,C19,C21, one-hot 
计算频次
其他的的fields 用one-hot-encoding
"""

# one-hot encoding directly
click = set()
hour = set()
C1 = set()
banner_pos = set()
site_category = set()
app_category = set()
device_type = set()
device_conn_type = set()
C15 = set()
C16 = set()
C18 = set()
C20 = set()

hour = set(range(24))

# one-encoding by frequency bucket
C14 = []
C17 = []
C19 = []
C21 = []
site_id = []
site_domain = []
app_id = []
app_domain = []
device_model = []
device_ip = []
device_id = []



train = pd.read_csv(config.train_path,chunksize=10000)
test = pd.read_csv(config.test_path, chunksize=10000)

for data in train:
    # get fields
    click_v = set(data['click'].values)
    click = click | click_v

    C1_v = set(data['C1'].values)
    C1 = C1 | C1_v

    C15_v = set(data['C15'].values)
    print(C15_v)
    C15 = C15 | C15_v
    print(C15)

    C16_v = set(data['C16'].values)
    C16 = C16 | C16_v

    C18_v = set(data['C18'].values)
    C18 = C18 | C18_v

    C20_v = set(data['C20'].values)
    C20 = C20 | C20_v

    banner_pos_v = set(data['banner_pos'].values)
    banner_pos = banner_pos | banner_pos_v

    site_category_v = set(data['site_category'].values)
    site_category = site_category | site_category_v

    app_category_v = set(data['app_category'].values)
    app_category = app_category | app_category_v

    device_type_v = set(data['device_type'].values)
    device_type = device_type | device_type_v

    device_conn_type_v = set(data['device_conn_type'].values)
    device_conn_type = device_conn_type | device_conn_type_v



# save dictionaries
with open(config.data_path+'click.pkl', 'wb') as f:
    pickle.dump(click, f)

with open(config.data_path+'hour.pkl', 'wb') as f:
    pickle.dump(hour, f)

with open(config.data_path+'C1.pkl', 'wb') as f:
    pickle.dump(C1, f)

with open(config.data_path+'C15.pkl', 'wb') as f:
    pickle.dump(C15, f)

with open(config.data_path+'C16.pkl', 'wb') as f:
    pickle.dump(C16, f)

with open(config.data_path+'C18.pkl', 'wb') as f:
    pickle.dump(C18, f)

with open(config.data_path+'C20.pkl', 'wb') as f:
    pickle.dump(C20, f)

with open(config.data_path+'banner_pos.pkl', 'wb') as f:
    pickle.dump(banner_pos, f)

with open(config.data_path+'site_category.pkl','wb') as f:
    pickle.dump(site_category, f)

with open(config.data_path+'app_category.pkl','wb') as f:
    pickle.dump(app_category, f)

with open(config.data_path+'device_type.pkl','wb') as f:
    pickle.dump(device_type, f)

with open(config.data_path+'device_conn_type.pkl','wb') as f:
    pickle.dump(device_conn_type, f)

# other feature
C14 = dict()
C17 = dict()
C19 = dict()
C21 = dict()
site_id = dict()
site_domain = dict()
app_id = dict()
app_domain = dict()
device_model = dict()
device_id = dict()
device_ip = dict()

count = 0
for data in train:
    C14_list = data['C14'].values
    for k, v in Counter(C14_list).items():
        if k in C14.keys():
            C14[k] += v
        else:
            C14[k] = v

    C17_list = data['C17'].values
    for k, v in Counter(C17_list).items():
        if k in C17.keys():
            C17[k] += v
        else:
            C17[k] = v

    C19_list = data['C19'].values
    for k, v in Counter(C19_list).items():
        if k in C19.keys():
            C19[k] += v
        else:
            C19[k] = v

    C21_list = data['C21'].values
    for k,v in Counter(C21_list).items():
        if k in C21.keys():
            C21[k] += v
        else:
            C21[k] = v

    site_id_list = data['site_id'].values
    for k,v in Counter(site_id_list).items():
        if k in site_id.keys():
            site_id[k] += v
        else:
            site_id[k] = v

    site_domain_list = data['site_domain'].values
    for k,v in Counter(site_domain_list).items():
        if k in site_domain.keys():
            site_domain[k] += v
        else:
            site_domain[k] = v

    app_id_list = data['app_id'].values
    for k,v in Counter(app_id_list).items():
        if k in app_id.keys():
            app_id[k] += v
        else:
            app_id[k] = v

    app_domain_list = data['app_domain'].values
    for k,v in Counter(app_domain_list).items():
        if k in app_domain.keys():
            app_domain[k] += v
        else:
            app_domain[k] = v

    device_model_list = data['device_model'].values
    for k,v in Counter(device_model_list).items():
        if k in device_model.keys():
            device_model[k] += v
        else:
            device_model[k] = v

    device_id_list = data['device_id'].values
    for k,v in Counter(device_id_list).items():
        if k in device_id.keys():
            device_id[k] += v
        else:
            device_id[k] = v

    device_ip_list = data['device_ip'].values
    for k,v in Counter(device_ip_list).items():
        if k in device_ip.keys():
            device_ip[k] += v
        else:
            device_ip[k] = v

    count += 1
    if count % 100 == 0:
        print('{} has finished'.format(count))


for data in test:
    C14_list = data['C14'].values
    for k,v in Counter(C14_list).items():
        if k in C14.keys():
            C14[k] += v
        else:
            C14[k] = v

    C17_list = data['C17'].values
    for k,v in Counter(C17_list).items():
        if k in C17.keys():
            C17[k] += v
        else:
            C17[k] = v

    C19_list = data['C19'].values
    for k,v in Counter(C19_list).items():
        if k in C19.keys():
            C19[k] += v
        else:
            C19[k] = v

    C21_list = data['C21'].values
    for k,v in Counter(C21_list).items():
        if k in C21.keys():
            C21[k] += v
        else:
            C21[k] = v

    site_id_list = data['site_id'].values
    for k,v in Counter(site_id_list).items():
        if k in site_id.keys():
            site_id[k] += v
        else:
            site_id[k] = v

    site_domain_list = data['site_domain'].values
    for k,v in Counter(site_domain_list).items():
        if k in site_domain.keys():
            site_domain[k] += v
        else:
            site_domain[k] = v

    app_id_list = data['app_id'].values
    for k,v in Counter(app_id_list).items():
        if k in app_id.keys():
            app_id[k] += v
        else:
            app_id[k] = v

    app_domain_list = data['app_domain'].values
    for k,v in Counter(app_domain_list).items():
        if k in app_domain.keys():
            app_domain[k] += v
        else:
            app_domain[k] = v

    device_model_list = data['device_model'].values
    for k,v in Counter(device_model_list).items():
        if k in device_model.keys():
            device_model[k] += v
        else:
            device_model[k] = v

    device_id_list = data['device_id'].values
    for k,v in Counter(device_id_list).items():
        if k in device_id.keys():
            device_id[k] += v
        else:
            device_id[k] = v

    device_ip_list = data['device_ip'].values
    for k,v in Counter(device_ip_list).items():
        if k in device_ip.keys():
            device_ip[k] += v
        else:
            device_ip[k] = v


# save dictionaries
with open(config.data_path+'C14.pkl','wb') as f:
    pickle.dump(C14,f)

with open(config.data_path+'C17.pkl','wb') as f:
    pickle.dump(C17,f)

with open(config.data_path+'C19.pkl','wb') as f:
    pickle.dump(C19,f)

with open(config.data_path+'C21.pkl','wb') as f:
    pickle.dump(C21,f)

with open(config.data_path+'site_id.pkl','wb') as f:
    pickle.dump(site_id,f)

with open(config.data_path+'site_domain.pkl','wb') as f:
    pickle.dump(site_domain,f)

with open(config.data_path+'app_id.pkl','wb') as f:
    pickle.dump(app_id,f)

with open(config.data_path+'app_domain.pkl','wb') as f:
    pickle.dump(app_domain,f)

with open(config.data_path+'device_model.pkl','wb') as f:
    pickle.dump(device_model,f)

with open(config.data_path+'device_id.pkl','wb') as f:
    pickle.dump(device_id,f)

with open(config.data_path+'device_ip.pkl','wb') as f:
    pickle.dump(device_ip,f)
