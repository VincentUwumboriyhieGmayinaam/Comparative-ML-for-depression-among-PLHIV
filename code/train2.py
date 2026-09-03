import os
os.environ.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import pandas as pd, numpy as np, warnings, pickle, json, sys, time
warnings.filterwarnings('ignore')
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, RepeatedStratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (roc_auc_score,accuracy_score,precision_score,recall_score,f1_score,
    confusion_matrix,cohen_kappa_score,brier_score_loss)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import statsmodels.api as sm
CUT=int(sys.argv[1]); ONLY=sys.argv[2] if len(sys.argv)>2 else None
S=pickle.load(open(f'split_{CUT}.pkl','rb'))
Xtr_i,Xte_i,Xtr_s,Xte_s=S['Xtr_i'],S['Xte_i'],S['Xtr_s'],S['Xte_s']
Xtr_r,Xtr_rs,ytr_r,ytr,yte=S['Xtr_r'],S['Xtr_rs'],S['ytr_r'],S['ytr'],S['yte']
cv=StratifiedKFold(5,shuffle=True,random_state=42)
SP={
 'Logistic Regression':(LogisticRegression(max_iter=5000,random_state=42),{'C':[0.01,0.03,0.1,0.3,1,3,10]},True),
 'Decision Tree':(DecisionTreeClassifier(random_state=42),{'max_depth':[3,4,5,6,8,None],'min_samples_leaf':[3,5,10,15],'criterion':['gini','entropy']},False),
 'Random Forest':(RandomForestClassifier(random_state=42,n_jobs=1),{'n_estimators':[200,300,500],'max_depth':[4,6,8,None],'min_samples_leaf':[1,3,5],'max_features':['sqrt','log2']},False),
 'Gradient Boosting':(GradientBoostingClassifier(random_state=42),{'n_estimators':[100,200,300],'learning_rate':[0.01,0.05,0.1],'max_depth':[2,3,4],'subsample':[0.8,1.0]},False),
 'XGBoost':(XGBClassifier(eval_metric='logloss',random_state=42,verbosity=0,n_jobs=1),{'n_estimators':[100,200,300],'learning_rate':[0.01,0.05,0.1],'max_depth':[2,3,5],'subsample':[0.8,1.0]},False),
 'LightGBM':(LGBMClassifier(random_state=42,verbose=-1,n_jobs=1,force_col_wise=True),{'n_estimators':[100,200],'learning_rate':[0.05,0.1],'num_leaves':[7,15],'min_child_samples':[10,20]},False),
 'CatBoost':(CatBoostClassifier(random_seed=42,verbose=0,thread_count=1,allow_writing_files=False),{'iterations':[200],'learning_rate':[0.05,0.1],'depth':[3,4]},False),
 'Support Vector Machine':(SVC(probability=True,random_state=42),{'C':[0.1,1,10],'gamma':['scale',0.01,0.1]},True),
 'K-Nearest Neighbours':(KNeighborsClassifier(),{'n_neighbors':[5,7,11,15,21],'weights':['uniform','distance'],'p':[1,2]},True),
 'Naive Bayes':(GaussianNB(),{'var_smoothing':[1e-11,1e-9,1e-7,1e-5]},True),
}
RES=f'rows_{CUT}.jsonl'; PRB=f'prob_{CUT}.pkl'
done=set()
if os.path.exists(RES):
    done={json.loads(l)['Model'] for l in open(RES)}
probs=pickle.load(open(PRB,'rb')) if os.path.exists(PRB) else {}
for name,(est,grid,scaled) in SP.items():
    if name in done or (ONLY and name!=ONLY): continue
    t0=time.time()
    Xt = Xtr_rs if scaled else Xtr_r
    Xv = Xte_s if scaled else Xte_i
    Xf = Xtr_s if scaled else Xtr_i
    n_iter=min(12,int(np.prod([len(v) for v in grid.values()])))
    rs=RandomizedSearchCV(est,grid,n_iter=n_iter,scoring='roc_auc',cv=cv,random_state=42,n_jobs=1)
    rs.fit(Xt,ytr_r); m=rs.best_estimator_
    p=m.predict_proba(Xv)[:,1]; pred=(p>=0.5).astype(int)
    tn,fp,fn,tp=confusion_matrix(yte,pred).ravel()
    rcv=RepeatedStratifiedKFold(n_splits=5,n_repeats=4,random_state=7)
    cvs=cross_val_score(m,Xf,ytr,cv=rcv,scoring='roc_auc',n_jobs=1)
    lo=np.clip(p,1e-6,1-1e-6); lo=np.log(lo/(1-lo))
    try:
        r=sm.Logit(yte,sm.add_constant(lo)).fit(disp=0); cs,ci_=float(r.params[1]),float(r.params[0])
    except Exception: cs,ci_=float('nan'),float('nan')
    row=dict(Model=name,Accuracy=accuracy_score(yte,pred),Precision=precision_score(yte,pred,zero_division=0),
      Sensitivity=recall_score(yte,pred),Specificity=tn/(tn+fp),F1=f1_score(yte,pred,zero_division=0),
      AUC=roc_auc_score(yte,p),Kappa=cohen_kappa_score(yte,pred),Brier=brier_score_loss(yte,p),
      CV_AUC=float(cvs.mean()),CV_SD=float(cvs.std()),Cal_slope=cs,Cal_int=ci_,
      TP=int(tp),TN=int(tn),FP=int(fp),FN=int(fn),Params=str(rs.best_params_),secs=round(time.time()-t0,1))
    with open(RES,'a') as f: f.write(json.dumps(row)+'\n')
    probs[name]=p; pickle.dump(probs,open(PRB,'wb'))
    print('%-24s AUC %.3f | CV %.3f±%.3f | Acc %.3f | Sens %.3f | Spec %.3f | F1 %.3f | K %.3f | %.0fs'%(
        name,row['AUC'],row['CV_AUC'],row['CV_SD'],row['Accuracy'],row['Sensitivity'],row['Specificity'],row['F1'],row['Kappa'],row['secs']),flush=True)
print('ALLDONE')
