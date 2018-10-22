import pandas as pd
import numpy as np
import scipy as sc
import utils
import datetime
import collections
import os
import random
import config_path

train_path = config_path.train_path
test_path = config_path.test_path

dnsp_data_dir = config_path.dnsp_data_dir
fe_gen_dir = config_path.fe_gen_dir
model_data_dir = config_path.model_data_dir

utils.gen_folder(dnsp_data_dir, fe_gen_dir, model_data_dir)

continous_features = range(1, 5)
categorial_features = range(5, 31)
continous_clip = [1740, 6, 2, 2]  # 待调整

utils.down_sampling(tr_path=train_path, label='click', outpath=dnsp_data_dir)
utils.create_feature(data_path=dnsp_data_dir, datadir=fe_gen_dir, filename='dnsp_train.csv')
utils.create_feature(data_path=test_path, datadir=fe_gen_dir, filename='test.csv')

utils.preprocess(datadir=fe_gen_dir, outdir=model_data_dir,continous_features=continous_features,categorial_features=categorial_features)
