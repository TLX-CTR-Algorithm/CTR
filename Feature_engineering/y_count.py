import csv, sys, collections
from config import *

'''
查看正负样本的量
'''

counts = collections.defaultdict(lambda: [0, 0, 0])
# counts 默认为[0,0,0]

# feature_name = csv.reader()
for i, row in enumerate(csv.DictReader(open(train_path))):
    label = row['click']
    # print(label)
    if str(label) == '0':
        # 负样本
        counts['Pos'][0] += 1
    else:
        # 正样本
        counts['Neg'][1] += 1
    counts['count'][2] += 1
    if i % 1000000 == 0:
        sys.stderr.write('{}m\n'.format(int(i/1000000)))

print('Field,Value,Neg,Pos,Total,Ratio')
for key, (neg, pos, total) in sorted(counts.items(), key=lambda x: x[1][2]):
    if total < 10:
        continue
    ratio = round(float(pos)/total, 5)
    print(key+','+str(neg)+','+str(pos)+','+str(total)+','+str(ratio))
