
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn import preprocessing
import sample

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

#sample.analyze(pddata)
#print (pddata.head())
#print (pddata.values)
#print (np.unique(pddata.values))

#print('-----------------------')
#print (pddata.values + )
#pddata2=pddata.apply(lambda x.values: str(x.values) +  )
#print (pddata2)

le = preprocessing.LabelEncoder()
le.fit(np.unique(pddata.values))
output = pddata.apply(le.transform)
#print (output)


#ndarray test
arr = np.array(pddata)
print (arr)
print (len(arr))


