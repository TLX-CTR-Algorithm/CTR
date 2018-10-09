import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
import numpy as np
import pickle


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




