
import pandas as pd
import numpy as np
import scipy as sc
import utils
import datetime
import collections
import os
import random
import config

train_path = config.train_data_dir + '/'
test_path = config.test_data_dir + '/'

data_dir = config.data_dir + '/'
outdir = config.output_dir

trainfilename = config.trainfilename
testfilename = config.testfilename

continous_features = config.continous_features
categorial_features = config.categorial_features
continous_clip = config.continous_clip

utils.create_feature(data_path=train_path, datadir=data_dir,
                     filename=trainfilename)
utils.create_feature(data_path=train_path, datadir=data_dir,
                     filename=testfilename)

dict_sizes=utils.preprocess(datadir=data_dir, outdir=outdir,
                 continous_features=continous_features,
                 categorial_features=categorial_features)

print (dict_sizes)
pddict_sizes = pd.DataFrame(dict_sizes)
pddict_sizes.to_csv(outdir + '/dictsize.csv',encoding = 'gbk')