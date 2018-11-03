import pandas as pd
import numpy as np
import scipy as sc
import utils
import datetime
import collections
import os
import random
import config_path
import preprocess

train_path = config_path.train_path
test_path = config_path.test_path

upsp_data_dir = config_path.upsp_data_dir
fe_gen_dir = config_path.fe_gen_dir
model_data_dir = config_path.model_data_dir

utils.gen_folder(dnsp_data_dir, fe_gen_dir, model_data_dir)

continous_features = range(1, 7)
categorial_features = range(7, 31)


utils.up_sampling(tr_path=train_path, label='click', outpath=upsp_data_dir)

preprocess.create_feature(data_path=upsp_data_dir, datadir=fe_gen_dir, filename='train.csv',is_train = True)
preprocess.create_feature(data_path=test_path, datadir=fe_gen_dir, filename='test.csv', is_train = False)

continous_clip = utils.gen_continous_clip(tr_path = fe_gen_dir,filename = 'fe_add_train_data.csv')

preprocess.preprocess(datadir=fe_gen_dir, outdir=model_data_dir,continous_features=continous_features,categorial_features=categorial_features,continous_clip =continous_clip)


