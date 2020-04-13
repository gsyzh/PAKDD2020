import pandas as pd
import joblib
import glob,os
import zipfile
from lightgbm.sklearn import LGBMClassifier
pd.options.display.max_columns = None
pd.options.display.max_rows = None

clf=joblib.load('clf.pkl')
tag = pd.read_csv('tag.csv')
serial = pd.read_csv('serial.csv')
serial.columns = ['serial_number','dt_first','model']
serial.dt_first = pd.to_datetime(serial.dt_first)
feature_name = ['days', 'days_1', 'days_2', 'model', 'serial_number', 'smart_10_normalized', 'smart_10raw', 'smart_12_normalized', 'smart_12raw', 'smart_184_normalized', 'smart_184raw', 'smart_187_normalized', 'smart_187raw', 'smart_188_normalized', 'smart_188raw', 'smart_189_normalized', 'smart_189raw', 'smart_190_normalized', 'smart_190raw', 'smart_191_normalized', 'smart_191raw', 'smart_192_normalized', 'smart_192raw', 'smart_193_normalized', 'smart_193raw', 'smart_194_normalized', 'smart_194raw', 'smart_195_normalized', 'smart_195raw', 'smart_197_normalized', 'smart_197raw', 'smart_198_normalized', 'smart_198raw', 'smart_199_normalized', 'smart_199raw', 'smart_1_normalized', 'smart_1raw', 'smart_240_normalized', 'smart_240raw', 'smart_241_normalized', 'smart_241raw', 'smart_242_normalized', 'smart_242raw', 'smart_3_normalized', 'smart_3raw', 'smart_4_normalized', 'smart_4raw', 'smart_5_normalized', 'smart_5raw', 'smart_7_normalized', 'smart_7raw', 'smart_9_normalized', 'smart_9raw', 'tag']
feature_name = sorted(feature_name)
path = '/tcdata/disk_sample_smart_log_round2'
file = glob.glob(os.path.join(path, "*.csv"))
submit1 = pd.DataFrame()
for f in file:
    test1=pd.read_csv(f) 
    submit1 = submit1.append(test1,ignore_index=True)
test = submit1

test123 = test
test123 = test123.sort_values(['serial_number','dt'])
test123 = test123.drop_duplicates().reset_index(drop=True)
test1231 = test123.loc[test123.model==1]
test1232 = test123.loc[test123.model==2]
drop_list =[]
for i in [col for col in test1231.columns if col not in ['manufacturer','model']]:
    if (test1231[i].nunique() == 1)&(test1231[i].isnull().sum() == 0):
        drop_list.append(i)

df1= pd.DataFrame()
df1['fea'] = test1231.isnull().sum().index
df1['isnull_sum'] = test1231.isnull().sum().values
fea_list1 = list(set(df1.loc[df1.isnull_sum != test1231.shape[0]]['fea']) - set(drop_list))
print(fea_list1)
print(test.shape)

drop_list2 =[]
for i in [col for col in test1232.columns if col not in ['manufacturer','model']]:
    if (test1232[i].nunique() == 1)&(test1232[i].isnull().sum() == 0):
        drop_list.append(i)

df2= pd.DataFrame()
df2['fea'] = test1232.isnull().sum().index
df2['isnull_sum'] = test1232.isnull().sum().values
fea_list2 = list(set(df2.loc[df2.isnull_sum != test1232.shape[0]]['fea']) - set(drop_list))
print(fea_list2)
print(test.shape)
test = test.merge(serial,how = 'left',on =  ['serial_number','model'])
test['dt'] = test['dt'].apply(lambda x:''.join(str(x)[0:4] +'-'+ str(x)[4:6]  +'-'+ str(x)[6:]))
test['dt'] = pd.to_datetime(test['dt'])
sub = test[['manufacturer','model','serial_number','dt']]
print(test.dt[0:5])
test['serial_number'] = test['serial_number'].apply(lambda x:int(x.split('_')[1]))
feature_name = [ 'model', 'smart_10raw', 'smart_12raw', 'smart_184raw', 'smart_187raw', 'smart_188raw', 'smart_189raw', 'smart_190raw', 'smart_191raw', 'smart_192raw', 'smart_193raw', 'smart_194raw', 'smart_195raw', 'smart_197raw', 'smart_198raw', 'smart_199raw', 'smart_1raw', 'smart_240raw', 'smart_241raw', 'smart_242raw', 'smart_3raw', 'smart_4raw', 'smart_5raw', 'smart_7raw', 'smart_9raw']
feature_name = sorted(feature_name)
test_x = test[feature_name]
sub['p'] = clf.predict_proba(test_x)[:,1]
sub['label'] = sub['p'].rank()
sub['label']= (sub['label']>=sub.shape[0] * 0.9).astype(int)
list= ['2018-09-01','2018-09-02','2018-09-03','2018-09-04','2018-09-05','2018-09-06','2018-09-07','2018-09-08','2018-09-09','2018-09-10','2018-09-11','2018-09-12','2018-09-13','2018-09-14','2018-09-15','2018-09-16','2018-09-17','2018-09-18','2018-09-19','2018-09-20','2018-09-21','2018-09-22','2018-09-23','2018-09-24','2018-09-25','2018-09-26','2018-09-27','2018-09-28','2018-09-29','2018-09-30']
submit = pd.DataFrame()
for f in list:
    test11=sub.loc[sub.dt == f]
    test11=test11.loc[test11.label == 1]、
    test11 = test11.sort_values('p',ascending=False)
    test11 = test11.sort_values('p',ascending=False)
	test11 = test11[0:5]
    submit1 = submit.append(test11,ignore_index=True)
submit1[['manufacturer','model','serial_number','dt']].to_csv("predictions.csv",index=False,header = None)
def zipFile(filepath,outFullName):
    if not os.path.exists(filepath):
        print('not found')
    zip = zipfile.ZipFile(outFullName,"w",zipfile.ZIP_DEFLATED)
    zip.write(filepath, filepath)
    zip.close()

zipFile('predictions.csv','result.zip')
