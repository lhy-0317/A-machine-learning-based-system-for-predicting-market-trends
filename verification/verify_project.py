import os
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import sys, json, time, platform, hashlib, copy, importlib.util, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import sklearn
import joblib
import flask
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

P = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
torch.set_num_threads(2)
df = pd.read_csv(P/'data/market_forecast_dataset.csv')
meta = json.loads((P/'models/model_metadata.json').read_text(encoding='utf-8'))
cols = meta['feature_columns']; label_names=list(meta['label_map'])
y = df.target_trend.map(meta['label_map']).to_numpy()
X = df[cols].to_numpy(dtype=np.float32)
train, val = train_test_split(np.arange(len(df)), test_size=.2, random_state=2024, stratify=y)
scaler = joblib.load(P/'models/feature_scaler.pkl')
def model(dim):
    return nn.Sequential(nn.Linear(dim,128),nn.ReLU(),nn.Dropout(.2),nn.Linear(128,64),nn.ReLU(),nn.Linear(64,3))
m=model(13); m.load_state_dict(torch.load(P/'models/market_trend_model.pth',map_location='cpu',weights_only=True)); m.eval()
with torch.no_grad(): pred=m(torch.from_numpy(scaler.transform(X[val]))).argmax(1).numpy()
fresh = StandardScaler().fit(X[train])
result = {
 'verification_date':'2026-10-23',
 'environment':{'python':platform.python_version(),'torch':torch.__version__,'sklearn':sklearn.__version__,'pandas':pd.__version__,'numpy':np.__version__,'joblib':joblib.__version__,'flask':__import__('importlib.metadata',fromlist=['version']).version('flask'),'device':'CPU','threads':2},
 'dataset':{'rows':len(df),'columns':len(df.columns),'start':df.date.min(),'end':df.date.max(),'missing':int(df.isna().sum().sum()),'duplicates':int(df.duplicated().sum()),'date_duplicates':int(df.date.duplicated().sum()),'daily_continuous':bool((pd.to_datetime(df.date).diff().dropna()==pd.Timedelta(days=1)).all()),'class_counts':df.target_trend.value_counts().to_dict(),'future_return_ranges':df.groupby('target_trend').future_7d_return.agg(['min','max']).to_dict('index')},
 'split':{'train':len(train),'validation':len(val),'train_counts':{label_names[i]:int((y[train]==i).sum()) for i in range(3)},'validation_counts':{label_names[i]:int((y[val]==i).sum()) for i in range(3)}},
 'scaler_max_mean_difference':float(np.max(np.abs(fresh.mean_-scaler.mean_))),
 'saved_model':{'accuracy':accuracy_score(y[val],pred),'macro_f1':f1_score(y[val],pred,average='macro'),'report':classification_report(y[val],pred,labels=[0,1,2],target_names=label_names,output_dict=True),'confusion_matrix':confusion_matrix(y[val],pred).tolist(),'parameter_count':sum(p.numel() for p in m.parameters())},
 'majority_baseline':{'accuracy':accuracy_score(y[val],np.full(len(val),2)),'macro_f1':f1_score(y[val],np.full(len(val),2),average='macro')},
 'assets_sha256':{str(p.relative_to(P)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [P/'app.py',P/'train_model.py',P/'models/market_trend_model.pth',P/'models/feature_scaler.pkl',P/'models/training_history.json',P/'data/market_forecast_dataset.csv']}
}
# Import the original service in-process; no server or source changes.
spec=importlib.util.spec_from_file_location('verified_app',P/'app.py'); appmod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=appmod;spec.loader.exec_module(appmod)
appmod.app.config.update(TESTING=False,PROPAGATE_EXCEPTIONS=False)
appmod.app.logger.disabled=True
client=appmod.app.test_client(); payload=meta['default_payload'].copy()
tests=[]
def record(name,method,path,expected,**kw):
    try:
        r=getattr(client,method)(path,**kw)
        body=r.get_json(silent=True)
        tests.append({'name':name,'expected':expected,'status':r.status_code,'body':body if name not in ['index','metrics'] else ('html' if name=='index' else {'history_length':len(body['history'])})})
    except Exception as e:tests.append({'name':name,'exception':repr(e)})
for name,path in [('index','/'),('health','/health'),('status','/status'),('metrics','/metrics')]:record(name,'get',path,200)
record('single','post','/predict',200,json=payload)
record('batch','post','/predict',200,json=[payload,payload])
record('wrong_content_type','post','/predict',415,data='x',content_type='text/plain')
bad=payload.copy();bad.pop('open'); record('missing_open','post','/predict',400,json=bad)
bad=payload.copy();bad['open']='not-a-number';record('invalid_number','post','/predict',400,json=bad)
record('scalar_body','post','/predict',400,json=123)
record('non_object_item','post','/predict',400,json=[1])
record('empty_batch','post','/predict',400,json=[])
record('invalid_json','post','/predict',400,data='{',content_type='application/json')
for name,value,field in [('negative_volume',-1,'volume'),('string_nan','NaN','open'),('string_inf','Infinity','open'),('huge_number',1e100,'open')]:
    bad=payload.copy();bad[field]=value;record(name,'post','/predict',400,json=bad)
result['api_tests']=tests
# The latency is in-process CPU latency, not browser or network latency.
for _ in range(10):client.post('/predict',json=payload)
elapsed=[]
for _ in range(100):
    start=time.perf_counter();client.post('/predict',json=payload);elapsed.append((time.perf_counter()-start)*1000)
result['inprocess_latency_ms']={'n':100,'warmup':10,'mean':float(np.mean(elapsed)),'median':float(np.median(elapsed)),'p95':float(np.percentile(elapsed,95))}

# Supplementary controls: separate artifacts, fixed seeds and a copied checkpoint.
def experiment(name,features,idx_train,idx_val):
    torch.manual_seed(2024);np.random.seed(2024)
    a=df[features].to_numpy(np.float32);s=StandardScaler().fit(a[idx_train]);xt=torch.from_numpy(s.transform(a[idx_train])); xv=torch.from_numpy(s.transform(a[idx_val]));yt=torch.tensor(y[idx_train]);
    mod=model(len(features));opt=torch.optim.Adam(mod.parameters(),lr=1e-3,weight_decay=1e-4);lossfn=nn.CrossEntropyLoss();best=-1;state=None;best_epoch=0
    loader=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(xt,yt),batch_size=64,shuffle=True)
    for epoch in range(1,51):
        mod.train()
        for batch,labels in loader:
            opt.zero_grad();loss=lossfn(mod(batch),labels);loss.backward();opt.step()
        mod.eval()
        with torch.no_grad():p=mod(xv).argmax(1).numpy()
        acc=accuracy_score(y[idx_val],p)
        if acc>best:best=acc;state=copy.deepcopy(mod.state_dict());best_epoch=epoch
    mod.load_state_dict(state);mod.eval()
    with torch.no_grad():p=mod(xv).argmax(1).numpy()
    return {'name':name,'features':features,'n_train':len(idx_train),'n_validation':len(idx_val),'train_end':df.date.iloc[idx_train[-1]],'validation_start':df.date.iloc[idx_val[0]],'selected_epoch':best_epoch,'accuracy':accuracy_score(y[idx_val],p),'macro_f1':f1_score(y[idx_val],p,labels=[0,1,2],average='macro',zero_division=0),'report':classification_report(y[idx_val],p,labels=[0,1,2],target_names=label_names,output_dict=True,zero_division=0),'confusion_matrix':confusion_matrix(y[idx_val],p,labels=[0,1,2]).tolist()}
cut=int(len(df)*.8)
result['supplementary_experiments']=[experiment('13_features_random_split',cols,train,val),experiment('12_features_random_split',[c for c in cols if c!='future_7d_return'],train,val),experiment('12_features_time_split_gap7',[c for c in cols if c!='future_7d_return'],np.arange(cut-7),np.arange(cut,len(df)))]
result['history_summary']={ 'first':json.loads((P/'models/training_history.json').read_text())[0], 'last':json.loads((P/'models/training_history.json').read_text())[-1], 'best':max(json.loads((P/'models/training_history.json').read_text()),key=lambda v:v['val_acc'])}
def safe_json(x):
    if isinstance(x, dict):return {k:safe_json(v) for k,v in x.items()}
    if isinstance(x, list):return [safe_json(v) for v in x]
    if isinstance(x, float) and not np.isfinite(x):return str(x)
    return x
result=safe_json(result)
(OUT/'verification_results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k not in ['api_tests','assets_sha256']},ensure_ascii=False,indent=2))
print('API TESTS',[(x['name'],x.get('status')) for x in tests])
