
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn import preprocessing
import sample
import config

ids = tf.Variable([[10,20,30],
                   [40,50,60],
                   [70,80,90]])

params = tf.get_variable(
                    'embedding', [50, 10])

#pddata = pd.DataFrame(
#    {
#        'a': [10,20,30],
#        'b': [40,50,60],
#        'c': [70,80,90]
#    }
#)

pddata = pd.DataFrame(
    {
        'a': ['小','朱','朱'],
        'b': ['sha','mao','ssh'],
        'c': ['70','80','90'],
        'd': ['A','B','C']
    }
)

tsdata = tf.Variable(
[           [10,20,30],
            [40,50,60],
            [70,80,90]
]
)

tsfloatdata = tf.Variable(
[           [1.56,2.01,3.50],
            [4.4,5.5,6.6],
            [70,80,90]
]
)


'''
tsdata = tf.constant(
[
            [10,20,30],
            [40,50,60],
            [70,80,90]
]
)
'''
#tsdata = tf.constant(10)

#sample.analyze(pddata)
#print (pddata.head())
#print (pddata.values)
#print (np.unique(pddata.values))

#print('-----------------------')
#print (pddata.values + )
#pddata2=pddata.apply(lambda x.values: str(x.values) +  )
#print (pddata2)

'''
le = preprocessing.LabelEncoder()
le.fit(np.unique(pddata.values))
output = pddata.apply(le.transform)
#print (output)


#ndarray test
arr = np.array(pddata)
tensorarr = tf.Variable(arr)
#print (arr)
#print (len(arr))

#embed_matrix = tf.Variable(tf.random_uniform([config.embed_max, config.embed_dim], -1, 1), name="embed_matrix")
#print (embed_matrix.shape)
#print (embed_matrix.eval)

#print (arr)
#print (arr[1])
#print (arr[:,0:-1])

dictsizes = pd.read_csv(config.dictsizefile)
print (dictsizes.__class__.__name__)

dictsize_np = np.array(dictsizes)
print (dictsize_np.__class__.__name__)

dictsize_list = list(dictsize_np[:,1])
print (dictsize_list.__class__.__name__)

print (dictsize_list)
embed_max = sum( dictsize_list )
#embed_max = dictsize_list.sum()
print (embed_max)
print (sum(dictsize_np[:,1]))

#testfile=np.loadtxt('./output/test.txt')


inputs = tf.placeholder(tf.float32, name='inputs')
print (config.encod_cat_index_begin)
branch_continous = inputs[:,0:config.encod_cat_index_begin]
print ('input.shape:{}'.format(branch_continous.shape))
'''


tsdata2 = tf.cast(tsdata,dtype=tf.float32)
tsfloatdata2 = tf.cast(tsfloatdata,dtype=tf.int32)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    #print (tensorarr[0].eval())
    print (sess.run(tsdata))
    print(tsdata.eval())
    print (tsfloatdata.eval())
    print (sess.run(tsfloatdata2))
    #print (tsdata[:,0:2].eval())
    #print(tsdata[0,1,2].eval())






