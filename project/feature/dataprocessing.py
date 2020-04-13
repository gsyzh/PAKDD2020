import numpy as np
import pandas as pd
import gc
import math
from tqdm import tqdm
import joblib
pd.options.display.max_columns = None
pd.options.display.max_rows = None

def read_from_local(file_name, chunk_size=500000):
    reader = pd.read_csv(file_name, header=0, iterator=True, encoding="utf-8")
    chunks = []
    loop = True
    while loop:
        try:
            chunk = reader.get_chunk(chunk_size)
            train_2018_11 = get_label(chunk)
            chunks.append(train_2018_11)
        except StopIteration:
            loop = False
            print("Iteration is stopped!")
    # 将块拼接为pandas dataFrame格式
    df_ac = pd.concat(chunks, ignore_index=True)
    return df_ac

def get_label(df):
    df = df[fea_list]
    df['dt'] = df['dt'].apply(lambda x: ''.join(str(x)[0:4] + '-' + str(x)[4:6] + '-' + str(x)[6:]))
    df['dt'] = pd.to_datetime(df['dt'])
    df = df.merge(tag[['serial_number', 'model', 'fault_time']], how='left', on=['serial_number', 'model'])
    df['diff_day'] = (df['fault_time'] - df['dt']).dt.days
    df['label'] = 0
    df.loc[(df['diff_day'] >= 0) & (df['diff_day'] <= 30), 'label'] = 1
    return df
test = pd.read_csv('../data/test_b.csv')
tag = pd.read_csv('../data/tag.csv')
test['dt'] = test['dt'].apply(lambda x:''.join(str(x)[0:4] +'-'+ str(x)[4:6]  +'-'+ str(x)[6:]))
test['dt'] = pd.to_datetime(test['dt'])
tag['fault_time'] = pd.to_datetime(tag['fault_time'])
#
###tag表里面有的硬盘同一天发生几种故障
tag['tag'] = tag['tag'].astype(str)
tag = tag.groupby(['serial_number','fault_time','model'])['tag'].apply(lambda x :'|'.join(x)).reset_index()
test = test.sort_values(['serial_number','dt'])
test = test.drop_duplicates().reset_index(drop=True)

###去掉全为空值和nunique为1的特征
drop_list = []
for i in tqdm([col for col in test.columns if col not in ['manufacturer', 'model']]):
    if (test[i].nunique() == 1) & (test[i].isnull().sum() == 0):
        drop_list.append(i)

df = pd.DataFrame()

df['fea'] = test.isnull().sum().index
df['isnull_sum'] = test.isnull().sum().values
fea_list = list(set(df.loc[df.isnull_sum != test.shape[0]]['fea']) - set(drop_list))
test = test[fea_list]
#

#提取jl.z
train_2018_7 = read_from_local('../data/disk_sample_smart_log_201807.csv')
# 7月只保留正样本
train_2018_7 = train_2018_7.loc[train_2018_7.label == 1]
joblib.dump(train_2018_7, '../user_data/train_2018_7.jl.z')
del train_2018_7


train_2018_6 = read_from_local('../data/disk_sample_smart_log_201806.csv')
joblib.dump(train_2018_6, '../user_data/train_2018_6.jl.z')
del train_2018_6

train_2018_5 = read_from_local('../data/disk_sample_smart_log_201805.csv')
joblib.dump(train_2018_5, '../user_data/train_2018_5.jl.z')
del train_2018_5

train_2018_4 = read_from_local('../data/disk_sample_smart_log_201804.csv')
joblib.dump(train_2018_4, '../user_data/train_2018_4.jl.z')
del train_2018_4

train_2018_3 = read_from_local('../data/disk_sample_smart_log_201803.csv')
joblib.dump(train_2018_3, '../user_data/train_2018_3.jl.z')
del train_2018_3

train_2018_2 = read_from_local('../data/disk_sample_smart_log_201802.csv')
joblib.dump(train_2018_2, '../user_data/train_2018_2.jl.z')
del train_2018_2

train_2018_1 = read_from_local('../data/disk_sample_smart_log_201801.csv')
joblib.dump(train_2018_1, '../user_data/train_2018_1.jl.z')
del train_2018_1

train_2017_7 = read_from_local('../data/disk_sample_smart_log_201707.csv')
joblib.dump(train_2017_7, '../user_data/train_2017_7.jl.z')
del train_2017_7

train_2017_8 = read_from_local('../data/disk_sample_smart_log_201707.csv')
joblib.dump(train_2017_8, '../user_data/train_2017_8.jl.z')
del train_2017_8

train_2017_9 = read_from_local('../data/disk_sample_smart_log_201709.csv')
joblib.dump(train_2017_9, '../user_data/train_2017_9.jl.z')
del train_2017_9

train_2017_10 = read_from_local('../data/disk_sample_smart_log_201710.csv')
joblib.dump(train_2017_10, '../user_data/train_2017_10.jl.z')
del train_2017_10

train_2017_11 = read_from_local('../data/disk_sample_smart_log_201711.csv')
joblib.dump(train_2017_11, '../user_data/train_2017_11.jl.z')
del train_2017_11

train_2017_12 = read_from_local('../data/disk_sample_smart_log_201712.csv')
joblib.dump(train_2017_12, '../user_data/train_2017_12.jl.z')
del train_2017_12

def get_serial(df):
    df =df.loc[df.model == 2]
    serialdf = df[['serial_number','dt','model']].sort_values('dt').drop_duplicates('serial_number')
    serialdf = serialdf.sort_values('dt').drop_duplicates(subset=['serial_number','model']).reset_index(drop=True)
    serialdf.columns = ['serial_number','dt_first','model']
    serialdf.dt_first = pd.to_datetime(serialdf.dt_first)
    df = serialdf
    return df

train_2018_7 = joblib.load('../user_data/train_2018_7.jl.z')
serial_2018_7 = get_serial(train_2018_7)
del train_2018_7


train_2018_6 = joblib.load('../user_data/train_2018_6.jl.z')
serial_2018_6 = get_serial(train_2018_6)
del train_2018_6


train_2018_5 = joblib.load('../user_data/train_2018_5.jl.z')
serial_2018_5 = get_serial(train_2018_5)
del train_2018_5


train_2018_4 = joblib.load('../user_data/train_2018_4.jl.z')
serial_2018_4 = get_serial(train_2018_4)
del train_2018_4


train_2018_3 = joblib.load('../user_data/train_2018_3.jl.z')
serial_2018_3 = get_serial(train_2018_3)
del train_2018_3


train_2018_2 = joblib.load('../user_data/train_2018_2.jl.z')
serial_2018_2 = get_serial(train_2018_2)
del train_2018_2

train_2018_1 = joblib.load('../user_data/train_2018_1.jl.z')
serial_2018_1 = get_serial(train_2018_1)
del train_2018_1


train_2017_7 =  joblib.load('../user_data/train_2017_7.jl.z')
serial_2017_7 = get_serial(train_2017_7)
del train_2017_7


train_2017_8 =  joblib.load('../user_data/train_2017_8.jl.z')
serial_2017_8 = get_serial(train_2017_8)
del train_2017_8


train_2017_9 =  joblib.load('../user_data/train_2017_9.jl.z')
serial_2017_9 = get_serial(train_2017_9)
del train_2017_9


train_2017_10 = joblib.load('../user_data/train_2017_10.jl.z')
serial_2017_10 = get_serial(train_2017_10)
del train_2017_10


train_2017_11 = joblib.load('../user_data/train_2017_11.jl.z')
serial_2017_11 = get_serial(train_2017_11)
del train_2017_11


train_2017_12 = joblib.load('../user_data/train_2017_12.jl.z')
serial_2017_12 = get_serial(train_2017_12)
del train_2017_12


serial_2018_8 =  get_serial(test)


serial = pd.concat((serial_2017_7,serial_2017_8,serial_2017_9,serial_2017_10,serial_2017_11,serial_2017_12),axis = 0)
serial = pd.concat((serial,serial_2018_1,serial_2018_2,serial_2018_3,serial_2018_4,serial_2018_5,serial_2018_6,serial_2018_7,serial_2018_8),axis = 0)
serial.columns = ['serial_number','dt','model']
serial =  get_serial(serial)
serial.to_csv("../user_data/serial.csv",index=False)