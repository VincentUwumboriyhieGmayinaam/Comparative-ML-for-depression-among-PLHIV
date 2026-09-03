import os; os.environ.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
import pandas as pd, numpy as np, pickle, json, warnings, time; warnings.filterwarnings('ignore')
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, RepeatedStratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (roc_auc_score,accuracy_score,precision_score,recall_score,f1_score,
    confusion_matrix,cohen_kappa_score,brier_score_loss)
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import statsmodels.api as sm
SEL=json.load(open('selected_features.json'))
S=pickle.load(open('split_10.pkl','rb'))
Xtr=S['Xtr_i'][SEL]; Xte=S['Xte_i'][SEL]; ytr=S['ytr']; yte=S['yte']
sc=StandardScaler().fit(Xtr); Xtr_s=pd.DataFrame(sc.transform(Xtr),columns=SEL,index=Xtr.index); Xte_s=pd.DataFrame(sc.transform(Xte),columns=SEL,index=Xte.index)
sm_=SMOTE(random_state=42,k_neighbors=5)
Xtr_r,ytr_r=sm_.fit_resample(Xtr,ytr); Xtr_rs,_=sm_.fit_resample(Xtr_s,ytr)
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
rows=[];probs={}
for name,(est,grid,scaled) in SP.items():
    Xt=Xtr_rs if scaled else Xtr_r; Xv=Xte_s if scaled else Xte; Xf=Xtr_s if scaled else Xtr
    n_iter=min(12,int(np.prod([len(v) for v in grid.values()])))
    rs=RandomizedSearchCV(est,grid,n_iter=n_iter,scoring='roc_auc',cv=cv,random_state=42,n_jobs=1).fit(Xt,ytr_r)
    m=rs.best_estimator_; p=m.predict_proba(Xv)[:,1]; pred=(p>=.5).astype(int)
    tn,fp,fn,tp=confusion_matrix(yte,pred).ravel()
    cvs=cross_val_score(m,Xf,ytr,cv=RepeatedStratifiedKFold(n_splits=5,n_repeats=4,random_state=7),scoring='roc_auc',n_jobs=1)
    lo=np.clip(p,1e-6,1-1e-6); lo=np.log(lo/(1-lo))
    try:
        r=sm.Logit(yte,sm.add_constant(lo)).fit(disp=0); cs=float(r.params[1])
    except Exception: cs=float('nan')
    rows.append(dict(Model=name,Accuracy=accuracy_score(yte,pred),Precision=precision_score(yte,pred,zero_division=0),
      Sensitivity=recall_score(yte,pred),Specificity=tn/(tn+fp),F1=f1_score(yte,pred,zero_division=0),
      AUC=roc_auc_score(yte,p),Kappa=cohen_kappa_score(yte,pred),Brier=brier_score_loss(yte,p),
      CV_AUC=float(cvs.mean()),CV_SD=float(cvs.std()),Cal_slope=cs,Params=str(rs.best_params_)))
    probs[name]=p
    print('%-24s CV %.3f±%.3f | AUC %.3f | Acc %.3f | Sens %.3f | Spec %.3f | F1 %.3f | K %.3f'%(
        name,cvs.mean(),cvs.std(),rows[-1]['AUC'],rows[-1]['Accuracy'],rows[-1]['Sensitivity'],rows[-1]['Specificity'],rows[-1]['F1'],rows[-1]['Kappa']),flush=True)
df=pd.DataFrame(rows).sort_values('CV_AUC',ascending=False)
df.to_csv('perf_reduced.csv',index=False); pickle.dump(probs,open('prob_reduced.pkl','wb'))
print('\n=== REDUCED FEATURE SET (33 features) ===')
print(df[['Model','CV_AUC','CV_SD','AUC','Accuracy','Sensitivity','Specificity','F1','Kappa','Brier','Cal_slope']].round(3).to_string(index=False))
