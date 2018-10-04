import pandas as pd
import numpy as np
import scipy as sc
import sys


raw_data_path = '/data/barnett007/ctr-data/'

def en_dummy(data,cate_vn_list):
    for feature in cate_vn_list:
        le = LabelEncoder()
        data[feature] = le.fit_transform(data[feature])
        max_ = data[feature].max()
        data[feature] = (data[feature] - max_) * (-1)

    data = data.loc[:,cate_vn_list]
    en_d = preprocessing.OneHotEncoder()
    en_d.fit(data)
    data = en_d.transform(data).toarray()
    result = pd.DataFrame(data)
    return result