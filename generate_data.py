
import pandas as pd
import numpy as np
import scipy as sc
import utils
import datetime
import collections
import os
import random

train_path = './data/'
test_path = './data/'

data_dir = './data/'
outdir = './output'

continous_features = range(1, 5)
categorial_features = range(5, 31)
continous_clip = [1740, 6, 2, 2] #待调整

utils.create_feature(data_path=train_path, datadir=data_dir,
                     filename='train.csv')
utils.create_feature(data_path=train_path, datadir=data_dir,
                     filename='test.csv')

utils.preprocess(datadir=data_dir, outdir=outdir,
                 continous_features=continous_features,
                 categorial_features=categorial_features)
