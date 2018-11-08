import os
import time

'''
模型的基本配置 
BASE_DIR 本地路径
train_path 训练路径
test_path 测试路径

'''

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
train_path = os.path.join(BASE_DIR, "avazu_CTR/train.csv")
test_path = os.path.join(BASE_DIR, "avazu_CTR/test.csv")
data_path = os.path.join(BASE_DIR, "avazu_CTR/sets/")
field2count = os.path.join(BASE_DIR, "avazu_CTR/field2count/")


print(test_path)

# 深度网络相关配置
debug_level='INFO'
# 路径和文件配置
encod_train_path = os.path.join(BASE_DIR, "output/model_data/train.txt")
encod_vaild_path = os.path.join(BASE_DIR, "output/model_data/valid.txt")
encod_test_path = os.path.join(BASE_DIR, "output/model_data/test.txt")
ffm_train_path = os.path.join(BASE_DIR, "output/model_data/train_pred.txt")
ffm_valid_path = os.path.join(BASE_DIR, "output/model_data/vaild_pred.txt")
ffm_test_path = os.path.join(BASE_DIR, "output/model_data/test_pred.txt")
dictsizefile = os.path.join(BASE_DIR, "output/model_data/dictsize.csv")
model_ouput_dir = os.path.join(BASE_DIR, "DNN/model_output/")
summary_dir = os.path.join(BASE_DIR, "DNN/summary/")
dnn_log_file = 'train_' + time.strftime('%Y%m%d', time.localtime(time.time())) + '.log'
dnn_log_dir = os.path.join(BASE_DIR, "DNN/log/")
dnn_log_path = os.path.join(dnn_log_dir, dnn_log_file)
encod_cat_index_begin = 6
encod_cat_index_end = 30
valid_switch = 1
model_flag = 'model2'
# 训练参数
batch_size = 100
keep_prob = 0.8
logfrequency = 10
Max_step = 2000000000
Max_epoch = 50
embed_dim = 128
learning_rate = 0.01
decay_rate = 0.96
decay_steps = 5000
oridata_dim = 23
