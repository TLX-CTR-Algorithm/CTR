import pandas as pd
import numpy as np
import scipy as sc
import sys
import utils
import xgboost as xgb
import config

	
#reading training,test
train_data0 = pd.read_csv(utils.raw_data_path + "train.csv")
test_data = pd.read_csv(utils.raw_data_path + "test.csv")
data_dir = utils.raw_data_path
outdir = config.outdir


categorial_features = range(1,27) #26维分类型变量

#对训练集，预测集混合，挑选并新建特征，完成后分离保存
utils.create_feature(train_path,test_path,datadir = data_dir ,nrow = nrow)

#对数据编码，生成用于神经网络，ffm，lightgbm的训练集，校验集，预测集
#预测集中带有label，需要后续处理
#data_dir 新增特征后训练集，预测集所在位置
utils.preprocess(data_dir,outdir = outdir,categorial_features=categorial_features)