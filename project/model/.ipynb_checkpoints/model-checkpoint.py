#随机取样进行合并的做法
import numpy as np
import pandas as pd
import gc
from lightgbm.sklearn import LGBMClassifier
from tqdm import tqdm
import joblib
pd.options.display.max_columns = None
pd.options.display.max_rows = None


def lower_sample_data(df,path,n_clusters=10,percent=100):
    df = df.loc[df.model == 2]
    dffalse = df.loc[df.label == 0].reset_index(drop=True)
    dftrue = df.loc[df.label == 1].reset_index(drop=True)
    drop = ['smart_10raw','smart_12raw','smart_184raw','smart_187raw','smart_188raw','smart_189raw','smart_190raw','smart_191raw','smart_192raw','smart_193raw','smart_194raw','smart_195raw','smart_197raw','smart_198raw','smart_199raw','smart_1raw','smart_240raw','smart_241raw','smart_242raw','smart_3raw','smart_4raw','smart_5raw','smart_7raw','smart_9raw']
    drop_list =[]
    for i in tqdm([col for col in df.columns if col not in [drop,'label','serial_number','dt','model','fault_time','diff_day','manufacturer']]):
        drop_list.append(i)
    drop_list = list(set(drop_list)-set(drop))
    df2= dffalse[drop_list]
    df2 = df2.fillna(method='ffill')
    df2 = df2.fillna(method='bfill')
    cluster = KMeans(n_clusters, random_state = 0).fit(df2)
    dffalse['clul'] = cluster.labels_
    dffalse['clui'] = cluster.inertia_  
    cen = cluster.cluster_centers_
    cen1 = pd.DataFrame(cen,columns=df2.columns)
    dftemp = dffalse['clul']
    dfsize = (int)(df2.index.size)
    for i in tqdm(range(dfsize)):
        X=np.vstack([df2.iloc[i],cen1.iloc[(dftemp.iloc[i])]])
        dffalse.iloc[i,56]=pdist(X)
    dffalse = dffalse.sort_values(['clul','clui']).reset_index(drop=True)
    dffalsetrue = pd.concat([dffalse, dftrue]).reset_index(drop=True)
    joblib.dump(dffalsetrue, path)
    number = percent * (len(dftrue)) 
    size = (int)(len(dffalse)/number)
    df3 = dffalse.iloc[::size].reset_index(drop=True)
    df4 = pd.concat([df3, dftrue]).reset_index(drop=True)
    drop_over =[]
    for i in tqdm([col for col in df4.columns if col not in ['clul','clui']]):
        drop_over.append(i)
    df5 = df4[drop_over].reset_index(drop=True)
    return df5


train_2018_6 = joblib.load('../user_data/train_2018_6.jl.z')
train_2018_6=lower_sample_data(train_2018_6)
train_2018_6.to_csv("../user_data/train_2018_6.csv",index=False)
del train_2018_6

train_2018_5 = joblib.load('../user_data/train_2018_5.jl.z')
train_2018_5=lower_sample_data(train_2018_5)
train_2018_5.to_csv("../user_data/train_2018_5.csv",index=False)
del train_2018_5

train_2018_4 = joblib.load('../user_data/train_2018_4.jl.z')
train_2018_4=lower_sample_data(train_2018_4)
train_2018_4.to_csv("../user_data/train_2018_4.csv",index=False)
del train_2018_4

train_2018_3 = joblib.load('../user_data/train_2018_3.jl.z')
train_2018_3=lower_sample_data(train_2018_3)
train_2018_3.to_csv("../user_data/train_2018_3.csv",index=False)
del train_2018_3

train_2018_2 = joblib.load('../user_data/train_2018_2.jl.z')
train_2018_2=lower_sample_data(train_2018_2)
train_2018_2.to_csv("../user_data/train_2018_2.csv",index=False)
del train_2018_2

train_2018_1 = joblib.load('../user_data/train_2018_1.jl.z')
train_2018_1=lower_sample_data(train_2018_1)
train_2018_1.to_csv("../user_data/train_2018_1.csv",index=False)
del train_2018_1

train_2017_7 =  joblib.load('../user_data/train_2017_7.jl.z')
train_2017_7=lower_sample_data(train_2017_7)
train_2017_7.to_csv("../user_data/train_2017_7.csv",index=False)
del train_2017_7

train_2017_8 =  joblib.load('../user_data/train_2017_8.jl.z')
train_2017_8=lower_sample_data(train_2017_8)
train_2017_8.to_csv("../user_data/train_2017_8.csv",index=False)
del train_2017_8

train_2017_9 =  joblib.load('../user_data/train_2017_9.jl.z')
train_2017_9=lower_sample_data(train_2017_9)
train_2017_9.to_csv("../user_data/train_2017_9.csv",index=False)
del train_2017_9

train_2017_10 = joblib.load('../user_data/train_2017_10.jl.z')
train_2017_10=lower_sample_data(train_2017_10)
train_2017_10.to_csv("../user_data/train_2017_10.csv",index=False)
del train_2017_10

train_2017_11 = joblib.load('../user_data/train_2017_11.jl.z')
train_2017_11=lower_sample_data(train_2017_11)
train_2017_11.to_csv("../user_data/train_2017_11.csv",index=False)
del train_2017_11

train_2017_12 = joblib.load('../user_data/train_2017_12.jl.z')
train_2017_12=lower_sample_data(train_2017_12)
train_2017_12.to_csv("../user_data/train_2017_12.csv",index=False)
del train_2017_12

#生成新的文件
train_2018_4 = pd.read_csv('../user_data/train_2018_4.csv')
train_2018_5 = pd.read_csv('../user_data/train_2018_5.csv')
train_2018_6 = pd.read_csv('../user_data/train_2018_6.csv')
train_x = train_2018_4
train_x = train_x.append(train_2018_5).reset_index(drop=True)
train_x = train_x.append(train_2018_6).reset_index(drop=True)
train_x.to_csv("../user_data/train_x.csv",index=False)

train_2018_2 = pd.read_csv('../user_data/train_2018_2.csv')
train_2018_3 = pd.read_csv('../user_data/train_2018_3.csv')
val_x = train_2018_2
val_x = train_x.append(train_2018_3).reset_index(drop=True)
val_x.to_csv("../user_data/val_x.csv",index=False)

train_2017_7 = pd.read_csv('../user_data/train_2017_7.csv')
train_2017_8 = pd.read_csv('../user_data/train_2017_8.csv')
train_2017_9 = pd.read_csv('../user_data/train_2017_9.csv')
train_2017_10 = pd.read_csv('../user_data/train_2017_10.csv')
train_2017_11 = pd.read_csv('../user_data/train_2017_11.csv')
train_2017_12 = pd.read_csv('../user_data/train_2017_12.csv')
train_2018_1 = pd.read_csv('../user_data/train_2018_1.csv')
train_2018_2 = pd.read_csv('../user_data/train_2018_2.csv')
train_2018_3 = pd.read_csv('../user_data/train_2018_3.csv')
train_2018_4 = pd.read_csv('../user_data/train_2018_4.csv')
train_2018_5 = pd.read_csv('../user_data/train_2018_5.csv')
train_2018_6 = pd.read_csv('../user_data/train_2018_6.csv')
train_2018_7 = joblib.load('../user_data/train_2018_7.jl.z')
train_x = train_2017_7
train_x = train_x.append(train_2017_8).reset_index(drop=True)
train_x = train_x.append(train_2017_9).reset_index(drop=True)
train_x = train_x.append(train_2017_10).reset_index(drop=True)
train_x = train_x.append(train_2017_11).reset_index(drop=True)
train_x = train_x.append(train_2017_12).reset_index(drop=True)
train_x = train_x.append(train_2018_1).reset_index(drop=True)
train_x = train_x.append(train_2018_2).reset_index(drop=True)
train_x = train_x.append(train_2018_3).reset_index(drop=True)
train_x = train_x.append(train_2018_4).reset_index(drop=True)
train_x = train_x.append(train_2018_5).reset_index(drop=True)
train_x = train_x.append(train_2018_6).reset_index(drop=True)
train_x = train_x.append(train_2018_7).reset_index(drop=True)
train_x.to_csv("../user_data/train_df.csv",index=False)
#训练数据
#训练数据
test = pd.read_csv('../data/test_b.csv')
tag = pd.read_csv('../data/tag.csv')
tag = tag.loc[tag.model==2]
test['dt'] = test['dt'].apply(lambda x:''.join(str(x)[0:4] +'-'+ str(x)[4:6]  +'-'+ str(x)[6:]))
test['dt'] = pd.to_datetime(test['dt'])
tag['fault_time'] = pd.to_datetime(tag['fault_time'])

test = test.sort_values(['serial_number','dt'])
test = test.drop_duplicates().reset_index(drop=True)
sub = test[['manufacturer','model','serial_number','dt']]

drop_list =[]
for i in tqdm([col for col in test.columns if col not in ['manufacturer','model']]):
    if (test[i].nunique() == 1)&(test[i].isnull().sum() == 0):
        drop_list.append(i)

df= pd.DataFrame()
df['fea'] = test.isnull().sum().index
df['isnull_sum'] = test.isnull().sum().values
fea_list = list(set(df.loc[df.isnull_sum != test.shape[0]]['fea']) - set(drop_list))
test = test[fea_list]

serial = pd.read_csv('../user_data/serial.csv')
serial.columns = ['serial_number','dt_first','model']
serial.dt_first = pd.to_datetime(serial.dt_first)

tag['tag'] = tag['tag'].astype(str)
tag = tag.groupby(['serial_number','fault_time','model'])['tag'].apply(lambda x :'|'.join(x)).reset_index()
tag.columns = ['serial_number','fault_time_1','model','tag']
map_dict = dict(zip(tag['tag'].unique(), range(tag['tag'].nunique())))
tag['tag'] = tag['tag'].map(map_dict).fillna(-1).astype('int32')

###用到的特征
feature_name = [i for i in test.columns if i not in ['dt','manufacturer']] + ['days','days_1','days_2','tag']
train_x = pd.read_csv('../user_data/train_x.csv')
train_x = train_x.merge(serial,how = 'left',on = ['serial_number','model'])
###硬盘的使用时常
train_x['dt'] = pd.to_datetime(train_x['dt'],format = '%Y-%m-%d %H:%M:%S')
train_x['days'] = (train_x['dt'] - train_x['dt_first']).dt.days

train_x = train_x.merge(tag,how = 'left',on = ['serial_number','model'])
###当前时间与另一个model故障的时间差，
train_x['days_1'] = (train_x['dt'] - train_x['fault_time_1']).dt.days
train_x.loc[train_x.days_1 <= 0,'tag'] = None
train_x.loc[train_x.days_1 <= 0,'days_1'] = None

###同一硬盘第一次使用到开始故障的时间
train_x['days_2'] = (train_x['fault_time_1'] - train_x['dt_first']).dt.days
train_x.loc[train_x.fault_time_1 >= train_x.dt,'days_2'] = None

train_x['serial_number'] = train_x['serial_number'].apply(lambda x:int(x.split('_')[1]))
train_y = train_x.label.values
train_x = train_x[feature_name]
gc.collect()

val_x = pd.read_csv('../user_data/val_x.csv')
val_x = val_x.merge(serial,how = 'left',on = ['serial_number','model'])
val_x['dt'] = pd.to_datetime(val_x['dt'],format = '%Y-%m-%d %H:%M:%S')
val_x['dt_first'] = pd.to_datetime(val_x['dt_first'],format = '%Y-%m-%d %H:%M:%S')
val_x['days'] = (val_x['dt'] - val_x['dt_first']).dt.days


val_x = val_x.merge(tag,how = 'left',on = ['serial_number','model'])
val_x['days_1'] = (val_x['dt'] - val_x['fault_time_1']).dt.days
val_x.loc[val_x.days_1 <= 0,'tag'] = None
val_x.loc[val_x.days_1 <= 0,'days_1'] = None
val_x['days_2'] = (val_x['fault_time_1'] - val_x['dt_first']).dt.days
val_x.loc[val_x.fault_time_1 >= val_x.dt,'days_2'] = None

val_x['serial_number'] = val_x['serial_number'].apply(lambda x:int(x.split('_')[1]))
val_y = val_x.label.values
val_x = val_x[feature_name]
gc.collect()

gc.collect()
clf = LGBMClassifier(
    learning_rate=0.001,
    n_estimators=10000,
    num_leaves=125,
    max_depth=7,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=2019,
    bagging_fraction= 0.9,
    bagging_freq= 8,
    lambda_l1 = 0.5,
    lambda_l2 = 0,
    cat_smooth = 10, 
    is_unbalenced = 'True',
    metric=None
)   

print('************** training **************')
print(train_x.shape,val_x.shape)
clf.fit(
    train_x, train_y,
    eval_set=[(val_x, val_y)],
    eval_metric='auc',
    early_stopping_rounds=50,
    verbose=100
)

train_df = pd.read_csv('../user_data/train_df.csv')
train_df = train_df.merge(serial,how = 'left',on = ['serial_number','model'])
train_df['dt'] = pd.to_datetime(train_df['dt'],format = '%Y-%m-%d %H:%M:%S')
train_df['dt_first'] = pd.to_datetime(train_df['dt_first'],format = '%Y-%m-%d %H:%M:%S')
train_df['days'] = (train_df['dt'] - train_df['dt_first']).dt.days

train_df = train_df.merge(tag,how = 'left',on = ['serial_number','model'])
train_df['days_1'] = (train_df['dt'] - train_df['fault_time_1']).dt.days
train_df.loc[train_df.days_1 <= 0,'tag'] = None
train_df.loc[train_df.days_1 <= 0,'days_1'] = None
train_df['days_2'] = (train_df['fault_time_1'] - train_df['dt_first']).dt.days
train_df.loc[train_df.fault_time_1 >= train_df.dt,'days_2'] = None
train_df['serial_number'] = train_df['serial_number'].apply(lambda x:int(x.split('_')[1]))
labels = train_df.label.values
train_df = train_df[feature_name]
gc.collect()
test = test.merge(serial,how = 'left',on =  ['serial_number','model'])
test['days'] = (test['dt'] - test['dt_first']).dt.days
test = test.merge(tag,how = 'left',on = ['serial_number','model'])
test['days_1'] = (test['dt'] - test['fault_time_1']).dt.days
test.loc[test.days_1 <= 0,'tag'] = None
test.loc[test.days_1 <= 0,'days_1'] = None
test['days_2'] = (test['fault_time_1'] - test['dt_first']).dt.days
test.loc[test.fault_time_1 >= test.dt,'days_2'] = None
test['serial_number'] = test['serial_number'].apply(lambda x:int(x.split('_')[1]))
test_x = test[feature_name]
clf = LGBMClassifier(
    learning_rate=0.001,
    n_estimators=10000,
    num_leaves=125,
    max_depth=7,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=2019,
    bagging_fraction= 0.9,
    bagging_freq= 8,
    lambda_l1 = 0.5,
    lambda_l2 = 0,
    cat_smooth = 10, 
    is_unbalenced = 'True',
    metric=None
)
print('************** training **************')
print(train_df.shape,test_x.shape)
clf.fit(
    train_df, labels,
    eval_set=[(train_df, labels)],
    eval_metric='auc',
    early_stopping_rounds=10,
    verbose=100
)

joblib.dump(clf, 'user_data/clf.pkl')
print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')

test['p'] = clf.predict_proba(test_x)[:,1]
test['label'] = test['p'].rank()
test['label']= (test['label']>=test.shape[0] * 0.990).astype(int)
submit = test.loc[test.label == 1]
submit = submit.sort_values('p',ascending=False)
submit = submit.drop_duplicates(['serial_number','model'])
submit['serial_number'] = submit['serial_number'].astype(str)
submit['serial_number'] = ['disk_'+ x for x in submit['serial_number']]
submit = submit[0:90]
submit[['manufacturer','model','serial_number','dt']].to_csv("../prediction_result/predictions.csv",index=False,header = None)
submit.shape