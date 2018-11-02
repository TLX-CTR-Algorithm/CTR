# CTR预估

## 1 了解问题

### 1.1 问题描述

CTR预估问题是典型的2分类问题，其有数据量大，分类型特征特征值多，数据严重不均衡等特点。

### 1.2 数据描述

原始数据共包含 11 天的数据，其中 10 天为训练数据 train， 1 天为测试数据 test。

其特征共23维，均为类别型特征。

train 解压后文件有 5.6G， 样本数目较大， 项目参数调优时考虑进行下采样。

## 2 特征工程

### 2.1 特征探索

对train数据中各特征(及label)进行数据探索，尽量了解其物理意义，观察其与label的关系，思考可新建的特征，同时考虑对特征的编码方式。

#### **click**

> ```
> ------------click-------------
> 0    33563901
> 1     6865066
> Name: click, dtype: int64
> ```

click是数据集的label，显见其严重不均衡，需要对样本进行均衡处理，考虑对样本进行多次下采样（不放回采样），最后将几个样本集训练得出的分类器进行组合；或考虑进行上采样。

```
train_data.click.mean()
0.16980562476404604
```

训练集中点击率约为16.98%，该值可作为点击率的一个参考；

#### hour

> ```
> -------------hour-------------
> 14102209    447783
> 14102210    438270
> 14102813    432308
> 14102212    408650
> 14102814    387453
> 14102211    386757
> 14103004    347806
> 14102809    328576
> 14102213    323480
> 14102208    322803
> 14102808    291763
> ......
> Name: hour, Length: 240, dtype: int64
> ```

hour特征为日期型数据，格式为"YYMMDDHH"，因数据集中数据均为14年10月数据，故该特征中年可丢弃，从中抽取具体日期，具体时。

据此可构建特征：

* 分离出具体的“日期”
* 分离出“几时”；
* 根据“日期”衍生类别型变量“日”

- 根据“日期”衍生“是否工作日”以及“星期几”；

#### C1

> ```
> --------------C1--------------
> 1005    37140632
> 1002     2220812
> 1010      903457
> 1012      113512
> 1007       35304
> 1001        9463
> 1008        5787
> Name: C1, dtype: int64
> 
> -------------------------C1-------------------------
>       display_sum  click_rate
> C1                           
> 1001         9463    0.033393
> 1002      2220812    0.210731
> 1005     37140632    0.169331
> 1007        35304    0.039429
> 1008         5787    0.121652
> 1010       903457    0.095215
> 1012       113512    0.172493
> ```

C1为类别型特征，暂时无法猜测物理含义。其中各特征值的点击率差异较大，对标签有较大影响，应该是不错的特征，考虑保留。

#### banner_pos

> ```
> ----------banner_pos----------
> 0    29109590
> 1    11247282
> 7       43577
> 2       13001
> 4        7704
> 5        5778
> 3        2035
> Name: banner_pos, dtype: int64
> ```

该特征应为广告海报的显示位置，如“上”，“下”，“左”，“右”等，作为分类型特征，应该是个不错的特征，考虑保留。

#### site_id

> ```
> -----------site_id------------
> 85f751fd    14596137
> 1fbe01fe     6486150
> e151e245     2637747
> d9750ee7      963745
> 5b08c53b      913325
> 5b4d2eda      771360
> 856e6d3f      765891
> a7853007      461311
> b7e9786d      369099
> ......
> 1c31ac16           1
> 2aa30f4e           1
> Name: site_id, Length: 4737, dtype: int64
> ```

网址id，可能是广告投放的网址，考虑保留。

#### site_domain

> ```
> ---------site_domain----------
> c4e18dd6    15131739
> f3845767     6486150
> 7e091613     3325008
> 7687a86e     1290165
> 98572c79      996816
> 16a36ef3      855686
> 58a89a43      765891
> 9d54950b      375891
> ......
> 9bf9b346           1
> c266215a           1
> Name: site_domain, Length: 7745, dtype: int64
> ```

网址所属领域，对用户起到分群作用，考虑保留。

#### site_category

> ```
> --------site_category---------
> 50e219e0    16537234
> f028772b    12657073
> 28905ebd     7377208
> 3e814130     3050306
> f66779e6      252451
> 75fa27f6      160985
> 335d28a8      136463
> 76b2941d      104754
> c0dd3be3       42090
> ......
> 6432c423           2
> a72a0145           2
> Name: site_category, dtype: int64
> ```

网址类别，与site_domain属于不同的分类方法，考虑保留。

#### app_id

> ```
> ------------app_id------------
> ecad2386    25832830
> 92f5800b     1555283
> e2fcccd2     1129016
> febd1138      759098
> 9c13b419      757812
> 7358e05e      615635
> ......
> 6746fb41           1
> 7b2b2217           1
> Name: app_id, Length: 8552, dtype: int64
> ```

此列中特征值“ecad2386”的计数远大于其他值，可能为web识别标签，**考虑据此构建新特征“app_or_web"**.

同时主要到该特征中相异特征值数据较大，若直接采用one-hot编码会造成维度灾难，考虑进行均值插补。

#### app_domain

> ```
> ----------app_domain----------
> 7801e8d9    27237087
> 2347f47a     5240885
> ae637522     1881838
> 5c5a694b     1129228
> 82e27996      759125
> d9b5648e      713924
> ......
> 7366e108           1
> d1600859           1
> Name: app_domain, Length: 559, dtype: int64
> ```

处理方式同site_domain；

#### app_category

> ```
> ---------app_category---------
> 07d7df22    26165592
> 0f2161f8     9561058
> cef3e649     1731545
> 8ded1f7a     1467257
> f95efa07     1141673
> ......
> cba0e20d           1
> 52de74cf           1
> Name: app_category, dtype: int64
> ```

处理方式同site_domain；

#### device_id

> ```python
> ----------device_id-----------
> a99f214a    33358308
> 0f7c61dc       21356
> c357dbff       19667
> 936e92fb       13712
> afeffc18        9654
> 987552d1        4187
> 28dc8687        4101
> ......
> 5c0c9e31           1
> 5df83b0d           1
> Name: device_id, Length: 2686408, dtype: int64
> ```

该特征值中”a99f214a“计数占据该特征极大比例，考虑该特征值为未采集到数据，为缺失值。该特征可结合device_ip识别用户，作为user_id。

#### device_ip

> ```
> ----------device_ip-----------
> 6b9769f2    208701
> 431b3174    135322
> 2f323f36     88499
> af9205f9     87844
> 930ec31d     86996
> af62faf4     85802
> 009a7861     85382
> ......
> 060dd26f         1
> 9cac90f8         1
> Name: device_ip, Length: 6729486, dtype: int64
> ```

用户设备ip，若原始信息可取，则考虑根据ip分离处地址位置，然后进行聚类，形成新的类别变量，但这里信息不足。同时device_ip特征的相异特征值也相当多，直接采用one-hot编码会导致维度灾难，考虑均值编码。

#### device_model

> ```
> ---------device_model---------
> 8a4875bd    2455470
> 1f0bc64f    1424546
> d787e91b    1405169
> 76dc4769     767961
> be6db1d7     742913
> a0f5f879     652751
> ......
> e6df6670          1
> 2dca9f52          1
> Name: device_model, Length: 8251, dtype: int64
> ```

暂不清楚具体含义，作为一般分类型特征处理；

#### device_type

> ```
> ---------device_type----------
> 1    37304667
> 0     2220812
> 4      774272
> 5      129185
> 2          31
> Name: device_type, dtype: int64
> 
> -------------------------device_type-------------------------
>              display_sum  click_rate
> device_type                         
> 0                2220812    0.210731
> 1               37304667    0.169176
> 2                     31    0.064516
> 4                 774272    0.095444
> 5                 129185    0.093842
> ```

device_type中特征值0、1两部分，广告投放量占比较大，同时点击率也较高，说明投放选择的设备类型正确，这应该是不错的特征。

#### device_conn_type

```
-------device_conn_type-------
0    34886838
2     3317443
3     2181796
5       42890
Name: device_conn_type, dtype: int64

-------------------------device_conn_type-------------------------
                  display_sum  click_rate
device_conn_type                         
0                    34886838    0.181125
2                     3317443    0.135289
3                     2181796    0.044043
5                       42890    0.029611
```

可能上网的连接方式，比如移动上网，WIFI等，广告投放量与点击率正比，也是值得关注的特征。

#### C15、C16

```
-------------C15--------------
320     37708959
300      2337294
216       298794
728        74533
120         3069
1024        2560
480         2137
768         1621
Name: C15, dtype: int64

-------------C16--------------
50      38136554
250      1806334
36        298794
480       103365
90         74533
20          3069
768         2560
320         2137
1024        1621
Name: C16, dtype: int64
```

C15、C16结合起来分析，猜测可能为设备的显示尺寸，考虑将这两个特征拼接，新建特征"C15_C16"；

其余特征”C14“，”C17“，”C18“，”C19“，”C20“，”C21“一时难以分辨含义，作为一般分类型特征处理；

### 2.2 特征构建

特征构建部分，考虑抽取部分原始特征，并利用其进行拼接组合生成新的类别型特征，同时根据广告浏览者的唯一标识统计其访问的频率，访问的集中时间段计数，device_id、device_ip的计数。

#### 2.2.1 抽取原始特征

'C1', 'C14', 'C17', 'C18', 'C19','C21', 'app_category', 'app_domain',  'banner_pos','device_conn_type', 'device_id', 'device_ip', 'device_model','device_type', 'site_category', 'site_domain', ；

#### 2.2.2 新建分类型特征

'day', 'hour_n', 'weekday', 'app_or_web', 'C15_C16', 'pub_id','pub_domain', 'pub_category'

```python
# 生成日期，某天,10~31
data_df['day'] = np.round(data_df.hour %10000 / 100).astype('int')
# 生成时间，时，0~23
data_df['hour_n'] = np.round(data_df.hour % 100)
# 生成星期几，Mon， Tues
data_df['weekday'] = list(map(to_weekday, data_df.hour))[0]
# 生成app，web识别特征
data_df['app_or_web'] = 0  
data_df.loc[data_df.app_id.values == 'ecad2386', 'app_or_web'] = 1
# 将C15，C16拼接，构成使用设备的屏幕尺寸
data_df['C15_C16'] = np.add(data_df.C15.map(str), data_df.C16.map(str))  # 组合图形尺寸

# 广告投放设备不同，存在app以及site两种方式，此处将二者合并，去除对于原始特征
# 合并访问方式id
data_df['pub_id'] = np.where(data_df['site_id'].map(is_app),
                             data_df['app_id'],data_df['site_id'])
# 合并domain
data_df['pub_domain'] = np.where(data_df['site_id'].map(is_app),
                                 data_df['app_domain'],data_df['site_domain'])
# 合并category
data_df['pub_category']=np.where(data_df['site_id'].map(is_app),
                                 data_df['app_category'],data_df['site_category'])
```

#### 2.2.3 新建连续型特征

特征device_id中特征值'a99f214a'占其特征值数目的82.5%，怀疑其为缺失值，即该特征值代表未采集到device_id。对于特征device_id中非缺失值部分，以其特征值为用户唯一标识user_id；缺失值部分，结合device_ip定位用户。

```python
# 生成用户唯一标识user
def def_user(row):
    '''
    定位用户，作为用户id
    '''
    if row['device_id'] == 'a99f214a':
        user = 'ip-' + row['device_ip'] + '-' + row['device_model']
    else:
        user = 'id-' + row['device_id']

    return user
```

根据用户标识user，统计相关数据：

* 某个用户收到广告次数user_count
* 某个用户在某个时间内收到广告次数smooth_user_hour_count
* 某个device_id收到广告的次数device_id_count
* 某个device_ip收到广告的次数device_ip_count

```python
data_df['user_count'] = data_df['user'].map(user_cnt)
data_df['user_hour'] = data_df['user'] + '-' + data_df['hour'].map(str)
data_df['smooth_user_hour_count'] = data_df['user_hour'].map(user_hour_cnt)
data_df['device_id_count'] = data_df['device_id'].map(device_id_cnt)
data_df['device_ip_count'] = data_df['device_ip'].map(device_ip_cnt)
```



### 2.3 待选模型

CTR预估是二分类问题，所以模型考虑：

* FFM
* embedding + FCN+FTRL
* FFM + embedding + FCN+FTRL

所以需要生成FCN及FFM两种数据输入格式。

### 2.4 数据预处理

#### 2.4.1 数据均衡处理

##### 下采样

```python
def down_sampling(tr_path, label, outpath):
    '''
    数据下采样
    '''
    tr_data = pd.read_csv(os.path.join(tr_path, 'train.csv'))
    temp_0 = tr_data[label] == 0
    data_0 = tr_data[temp_0]
    temp_1 = tr_data[label] == 1
    data_1 = tr_data[temp_1]
    sampler = np.random.permutation(data_0.shape[0])[:data_1.shape[0]]
    data_0_ed = data_0.iloc[sampler, :]
    data_downsampled = pd.concat([data_1, data_0_ed], ignore_index=True)
    data_downsampled=data_downsampled.sort_index(by = 'hour')
    data_downsampled.to_csv(os.path.join(
        outpath, 'dnsp_train.csv'), index=None)
```



##### 上采样

```python
def up_sampling(tr_path, label, outpath):
    '''
    数据上采样
    '''
    tr_data = pd.read_csv(os.path.join(tr_path, 'train.csv'))
    temp_0 = tr_data[label] == 0
    data_0 = tr_data[temp_0]
    temp_1 = tr_data[label] == 1
    data_1 = tr_data[temp_1]
    sampler = np.random.randint(data_1.shape[0],size = len(data_0))
    data_1_ed = data_1.iloc[sampler, :]
    data_upsampled = pd.concat([data_1_ed, data_0], ignore_index=True)
    data_upsampled=data_upsampled.sort_index(by = 'hour')
    data_upsampled.to_csv(os.path.join(outpath, 'dnsp_train.csv'), index=None)
```



#### 2.4.2 连续型特征

每个连续型特征，以其第95百分位数为分界点，大于该值的特征值视为异常值，以第95百分位数插补。同时对于插补后的数据，利用max-min进行归一化。

```python
class ContinuousFeatureGenerator:
    """
    Normalize the integer features to [0, 1] by min-max normalization
    """

    def __init__(self, num_feature):  # 初始化，与传入特征长度一致的列表，用来存放最大最小值
        self.num_feature = num_feature
        self.min = [sys.maxsize] * num_feature
        self.max = [-sys.maxsize] * num_feature

    def build(self, datafile, continous_features):
        with open(datafile, 'r') as f:
            for line in f:
                features = line.rstrip('\n').split(',')  # 从文本中抽取数据，并根据分隔符分开
                for i in range(0, self.num_feature):  # 估计是除去首列id列，循环处理前面13个连续特征
                    val = features[continous_features[i]]  # 该行i对应的特征的值
                    if val != '':
                        val = int(val)  # 向下取整
                        # 若大于分割点，则以分割点为值，也就是去除异常值，超过95%为异常点
                        if val > continous_clip[i]:
                            val = continous_clip[i]
                        # 与该位置的值比较，取较小值，找出该位置的最小值
                        self.min[i] = min(self.min[i], val)
                        # 与该位置的值比较，取较大值，找出该位置的最大值
                        self.max[i] = max(self.max[i], val)

    def gen(self, idx, val):  # val为值，idx是连续型变量的位置，从0开始
        if val == '':
            return 0.0
        val = float(val)
        # 对应位置做最大最小化
        return (val - self.min[idx]) / (self.max[idx] - self.min[idx])
```



#### 2.4.3 分类型特征

分类型特征，设定分界点cutoff为30，对每个特征，其特征值计数小于cutoff的归为其他类。并按计数降序对特征值做排列，以序号作为其新特征值。

```python
class CategoryDictGenerator:
    """
    Generate dictionary for each of the categorical features
    """

    def __init__(self, num_feature):
        self.dicts = []
        self.num_feature = num_feature  # 类别型特征的数目
        for i in range(0, num_feature):  # 列表中元素依次为对应位置的特征及计数的字典
            self.dicts.append(collections.defaultdict(int))
	# 获取分类型特征的符合cutoff条件的特征值的名称，且是降序的

    def build(self, datafile, categorial_features, cutoff=0):  # categorial_features序列，cutoff切割点,传入
        with open(datafile, 'r') as f:
            for line in f:
                features = line.rstrip('\n').split(',')
                for i in range(0, self.num_feature):
                    if features[categorial_features[i]] != '':  # 对应位置的特征的值
                        # 该值对应字典的值加1，计数，字典里统计各种特征值的计数
                        self.dicts[i][features[categorial_features[i]]] += 1
        for i in range(0, self.num_feature):
            self.dicts[i] = filter(lambda x: x[1] >= cutoff,
                                   self.dicts[i].items())  # 根据分割点保留符合条件的特征值对，将新的字典放回

            self.dicts[i] = sorted(
                self.dicts[i], key=lambda x: (-x[1], x[0]))  # 字典按值降序排列，放回列表
            # 将特征值与对应计数分开成两个，对应的元组，前面是特征值，后面是计数
            vocabs, _ = list(zip(*self.dicts[i]))
            # 新字典，键是特征值，值是序号？放回列表
            self.dicts[i] = dict(zip(vocabs, range(1, len(vocabs) + 1)))
            self.dicts[i]['<unk>'] = 0  # 对应位置的特征的字典中添加特殊值的键，值为0

    def gen(self, idx, key):  # idx为特征对应的位置，key为特征值
        if key not in self.dicts[idx]:  # 获取序号
            res = self.dicts[idx]['<unk>']
        else:
            res = self.dicts[idx][key]
        return res

    def dicts_sizes(self):  # 返回每个特征的符合条件的特征值的个数，返回为列表
        return list(map(len, self.dicts))
```



## 3 模型构建

### 3.1 单模型

#### 3.1.1 FFM

##### 超参数设定



##### 模型结果分析

**添加均值编码前**

###### 降采样

> **表格，对比图**

###### 上采样

>  **表格，对比图**

下列相同处，与此处类似

**添加均值编码前**

###### 降采样



###### 上采样



#### 3.1.2 FCN + FTRL

##### 超参数设定



**添加均值编码前**

###### 降采样



###### 上采样



**添加均值编码前**

###### 降采样



###### 上采样



### 3.2 模型融合：FFM + FCN + FTRL

#### 3.2.1 模型结构图



#### 3.2.2 超参数设定



#### 3.2.3 模型结论

采用经过均值编码处理后的数据；

###### 降采样



###### 上采样



## 4 Kaggle成果



## 5 心得体会

