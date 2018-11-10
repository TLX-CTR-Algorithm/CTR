import os

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
# 路径和文件配置
encod_train_path = os.path.join(BASE_DIR, "output/train.txt")
encod_vaild_path = os.path.join(BASE_DIR, "output/valid.txt")
encod_test_path = os.path.join(BASE_DIR, "output/test.txt")
dictsizefile = os.path.join(BASE_DIR, "output/dictsize.csv")
model_ouput_dir = os.path.join(BASE_DIR, "model_output/")
summary_dir = os.path.join(BASE_DIR, "summary/")
encod_cat_index_begin = 4
encod_cat_index_end = 30
# 训练参数
batch_size = 1000
keep_prob = 0.8
logfrequency = 10
Max_step = 20000000
embed_dim = 128
learning_rate = 0.01
oridata_dim = 23
