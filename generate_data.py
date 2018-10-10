
import pandas as pd
import numpy as np
import scipy as sc
import utils
import datetime
import collections
import os
import random

train_path = '/data/barnett007/ctr-data/'
test_path = '/data/barnett007/ctr-data/'

data_dir = '/data/barnett007/ctr-data/'
outdir = '/output'

categorial_features = range(1, 27)

utils.create_feature(train_path,test_path,datadir = data_dir ,nrow = 20000)
utils.preprocess(data_dir,outdir = outdir,categorial_features=categorial_features)

