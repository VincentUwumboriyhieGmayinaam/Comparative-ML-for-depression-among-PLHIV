import os; os.environ.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
import pandas as pd, numpy as np, pickle, json, warnings, ast; warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.linewidth':.8,
 'axes.edgecolor':'#52514e','xtick.color':'#52514e','ytick.color':'#52514e','axes.labelcolor':'#0b0b0b',
 'text.color':'#0b0b0b','figure.dpi':300,'savefig.dpi':300,'savefig.bbox':'tight','axes.spines.top':False,'axes.spines.right':False})
BLUE,ORANGE,AQUA,VIOLET,RED,MUT,GRID,INK='#2a78d6','#eb6834','#1baf7a','#4a3aa7','#e34948','#52514e','#e6e5e0','#0b0b0b'
S=pickle.load(open('split_10.pkl','rb'))
Xtr,ytr,Xte,yte=S['Xtr_i'],S['ytr'],S['Xte_i'],S['yte']
Xall=pd.concat([Xtr,Xte]); yall=np.concatenate([ytr,yte])
sc=StandardScaler().fit(Xall); Xall_s=pd.DataFrame(sc.transform(Xall),columns=Xall.columns)
rows={json.loads(l)['Model']:json.loads(l) for l in open('rows_10.jsonl')}
pr=lambda n: ast.literal_eval(rows[n]['Params'])
TOP=[('Random Forest',RandomForestClassifier(random_state=42,n_jobs=1,**pr('Random Forest')),False,BLUE),
     ('CatBoost',CatBoostClassifier(random_seed=42,verbose=0,thread_count=1,allow_writing_files=False,**pr('CatBoost')),False,ORANGE),
     ('Gradient Boosting',GradientBoostingClassifier(random_state=42,**pr('Gradient Boosting')),False,AQUA),
     ('Logistic Regression',LogisticRegression(max_iter=5000,random_state=42,**pr('Logistic Regression')),True,VIOLET),
     ('XGBoost',XGBClassifier(eval_metric='logloss',random_state=42,verbosity=0,n_jobs=1,**pr('XGBoost')),False,RED)]
cv=StratifiedKFold(5,shuffle=True,random_state=42)
OOF={}
for name,est,scaled,col in TOP:
    X=Xall_s if scaled else Xall
    OOF[name]=cross_val_predict(est,X,yall,cv=cv,method='predict_proba')[:,1]
pickle.dump(OOF,open('oof_top5.pkl','wb'))
fig,axs=plt.subplots(1,2,figsize=(12.2,5.0),gridspec_kw={'width_ratios':[1,1],'wspace':0.22})
a=axs[0]
a.plot([0,1],[0,1],ls=(0,(4,3)),color='#a8a7a0',lw=1.2,label='Ideal (perfect calibration)',zorder=1)
stats=[]
for name,est,scaled,col in TOP:
    p=np.clip(OOF[name],1e-6,1-1e-6)
    pt,pp=calibration_curve(yall,p,n_bins=6,strategy='quantile')
    a.plot(pp,pt,'o-',ms=5.5,lw=1.9,color=col,label=name,zorder=3,markeredgecolor='white',markeredgewidth=.9)
    lo=np.log(p/(1-p)); r=sm.Logit(yall,sm.add_constant(lo)).fit(disp=0)
    stats.append((name,float(r.params[1]),float(r.params[0]),float(np.mean((p-yall)**2)),col))
a.set_xlabel('Predicted probability of depression'); a.set_ylabel('Observed proportion depressed')
a.set_xlim(-.02,1.0); a.set_ylim(-.02,1.0)
a.legend(frameon=False,fontsize=8,loc='upper left'); a.grid(color=GRID,lw=.7); a.set_axisbelow(True)
a.set_title('A  Calibration curves, out-of-fold predictions (n=264)',loc='left',fontsize=9.8,fontweight='bold',pad=10)
b=axs[1]
p=np.clip(OOF['Random Forest'],1e-6,1-1e-6)
bins=np.linspace(0,1,21)
b.hist(p[yall==0],bins=bins,color=BLUE,alpha=.85,label='Not depressed (n=164)',edgecolor='white',linewidth=.5)
b.hist(p[yall==1],bins=bins,color=ORANGE,alpha=.85,label='Depressed (n=100)',edgecolor='white',linewidth=.5,
       bottom=np.histogram(p[yall==0],bins=bins)[0])
b.set_xlabel('Predicted probability of depression (Random Forest)'); b.set_ylabel('Participants')
b.legend(frameon=False,fontsize=8); b.grid(axis='y',color=GRID,lw=.7); b.set_axisbelow(True)
b.set_title('B  Distribution of predicted risk by true outcome',loc='left',fontsize=9.8,fontweight='bold',pad=10)
txt='\n'.join([f'{n:<20s} slope {s:5.2f}   intercept {i:6.2f}   Brier {br:.3f}' for n,s,i,br,c in stats])
fig.text(0.02,-0.055,'Calibration statistics (out-of-fold):\n'+txt,fontsize=7.6,family='DejaVu Sans Mono',va='top')
plt.savefig('fig/Fig15_calibration_top5.png',bbox_inches='tight'); plt.close()
print('=== OUT-OF-FOLD CALIBRATION, TOP FIVE ===')
for n,s,i,br,c in stats: print('%-22s slope %5.2f  intercept %6.2f  Brier %.3f'%(n,s,i,br))
