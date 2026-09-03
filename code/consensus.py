import os; os.environ.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
import pandas as pd, numpy as np, pickle, json, warnings, time, ast; warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import shap
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.linewidth':.8,
 'axes.edgecolor':'#52514e','xtick.color':'#52514e','ytick.color':'#52514e','axes.labelcolor':'#0b0b0b',
 'text.color':'#0b0b0b','figure.dpi':300,'savefig.dpi':300,'savefig.bbox':'tight','axes.spines.top':False,'axes.spines.right':False})
BLUE,ORANGE,AQUA,YELLOW,VIOLET,RED,GREEN='#2a78d6','#eb6834','#1baf7a','#eda100','#4a3aa7','#e34948','#008300'
MUT,GRID,INK='#52514e','#e6e5e0','#0b0b0b'
S=pickle.load(open('split_10.pkl','rb'))
Xtr,ytr,Xte,yte=S['Xtr_i'],S['ytr'],S['Xte_i'],S['yte']
Xtr_r,ytr_r=S['Xtr_r'],S['ytr_r']; Xtr_rs=S['Xtr_rs']; Xtr_s,Xte_s=S['Xtr_s'],S['Xte_s']
rows={json.loads(l)['Model']:json.loads(l) for l in open('rows_10.jsonl')}
pr=lambda n: ast.literal_eval(rows[n]['Params'])
# Top five SHAP-tractable models by cross-validated AUC
TOP=[('Random Forest',RandomForestClassifier(random_state=42,n_jobs=1,**pr('Random Forest')),False),
     ('CatBoost',CatBoostClassifier(random_seed=42,verbose=0,thread_count=1,allow_writing_files=False,**pr('CatBoost')),False),
     ('Gradient Boosting',GradientBoostingClassifier(random_state=42,**pr('Gradient Boosting')),False),
     ('Logistic Regression',LogisticRegression(max_iter=5000,random_state=42,**pr('Logistic Regression')),True),
     ('XGBoost',XGBClassifier(eval_metric='logloss',random_state=42,verbosity=0,n_jobs=1,**pr('XGBoost')),False)]
disp=[c.replace('_',' ') for c in Xte.columns]
SV={}; MODELS={}
for name,est,scaled in TOP:
    Xt=Xtr_rs if scaled else Xtr_r; Xv=Xte_s if scaled else Xte
    m=est.fit(Xt,ytr_r); MODELS[name]=(m,Xv)
    t0=time.time()
    if scaled:
        ex=shap.LinearExplainer(m,Xt); sv=ex.shap_values(Xv)
    else:
        ex=shap.TreeExplainer(m); sv=ex.shap_values(Xv)
        if isinstance(sv,list): sv=sv[1]
        elif np.ndim(sv)==3: sv=sv[:,:,1]
    SV[name]=np.asarray(sv)
    print('%-22s SHAP %s in %.1fs'%(name,np.shape(sv),time.time()-t0),flush=True)
pickle.dump(SV,open('shap_top5.pkl','wb'))

# ---------- FIG 13: beeswarm panels for top five ----------
cmap=LinearSegmentedColormap.from_list('bo',[BLUE,'#b9b8b2',ORANGE])
fig=plt.figure(figsize=(15.5,9.4))
for i,(name,_,scaled) in enumerate(TOP):
    ax=fig.add_subplot(2,3,i+1); plt.sca(ax)
    Xv=MODELS[name][1].copy(); Xv.columns=disp
    shap.summary_plot(SV[name],Xv,max_display=12,show=False,cmap=cmap,plot_size=None,color_bar=(i==2))
    ax=plt.gca(); ax.set_xlabel('SHAP value',fontsize=8)
    ax.tick_params(labelsize=7.4)
    ax.set_title(f'{chr(65+i)}  {name}   (CV AUC {rows[name]["CV_AUC"]:.3f})',loc='left',fontsize=9.6,fontweight='bold',pad=8)
fig.suptitle('SHAP beeswarm plots for the five best-performing algorithms',x=0.02,y=1.0,ha='left',fontsize=12,fontweight='bold')
plt.tight_layout(); plt.savefig('fig/Fig13_shap_top5.png',bbox_inches='tight'); plt.close()

# ---------- consensus ranking ----------
R={}
for name in SV:
    mi=np.abs(SV[name]).mean(axis=0)
    R[name]=pd.Series(mi,index=disp).rank(ascending=False)
RD=pd.DataFrame(R)
RD['Mean rank']=RD.mean(axis=1)
RD['Top-10 count']=(RD[list(SV)]<=10).sum(axis=1)
RD=RD.sort_values('Mean rank')
RD.to_csv('shap_consensus.csv')
print('\n=== CROSS-MODEL SHAP CONSENSUS (top 15) ===')
print(RD.head(15).round(1).to_string())

# ---------- FIG 14: consensus ----------
top=RD.head(15).iloc[::-1]
fig,axs=plt.subplots(1,2,figsize=(12.6,5.8),gridspec_kw={'width_ratios':[1.15,1],'wspace':0.28})
a=axs[0]
seq=LinearSegmentedColormap.from_list('bl',['#1b5aa4','#2a78d6','#7fb0e8','#c7dcf5','#eef4fc','#f7f7f5'])
M=top[list(SV)].values
im=a.imshow(M,cmap=seq,vmin=1,vmax=30,aspect='auto')
a.set_xticks(range(len(SV))); a.set_xticklabels(list(SV),rotation=28,ha='right',fontsize=8.2)
a.set_yticks(range(len(top))); a.set_yticklabels(top.index,fontsize=8.4)
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        a.text(j,i,int(M[i,j]),ha='center',va='center',fontsize=7.8,color='white' if M[i,j]<=12 else INK)
a.set_xticks(np.arange(-.5,len(SV),1),minor=True); a.set_yticks(np.arange(-.5,len(top),1),minor=True)
a.grid(which='minor',color='white',lw=1.8); a.tick_params(which='minor',length=0); a.tick_params(length=0)
for s in a.spines.values(): s.set_visible(False)
cb=plt.colorbar(im,ax=a,fraction=.030,pad=.02); cb.outline.set_visible(False)
cb.set_label('SHAP importance rank within model',fontsize=8); cb.ax.tick_params(labelsize=7.4)
a.set_title('A  Feature rank by model',loc='left',fontsize=9.8,fontweight='bold',pad=10)
b=axs[1]
cols=[GREEN if c==5 else (AQUA if c==4 else (YELLOW if c==3 else '#c9c8c2')) for c in top['Top-10 count']]
b.barh(np.arange(len(top)),top['Mean rank'],height=.66,color=cols,edgecolor='white',linewidth=.8)
for i,(mr,c) in enumerate(zip(top['Mean rank'],top['Top-10 count'])):
    b.text(mr+0.35,i,f'{mr:.1f}   ({c}/5)',va='center',fontsize=7.8,color=INK)
b.set_yticks(np.arange(len(top))); b.set_yticklabels([]); b.set_xlabel('Mean SHAP importance rank across the five models')
b.set_xlim(0,max(top['Mean rank'])*1.35); b.invert_xaxis(); b.invert_xaxis()
b.grid(axis='x',color=GRID,lw=.7); b.set_axisbelow(True)
from matplotlib.patches import Patch
b.legend(handles=[Patch(facecolor=GREEN,label='in top 10 of all 5 models'),Patch(facecolor=AQUA,label='4 of 5'),
                  Patch(facecolor=YELLOW,label='3 of 5'),Patch(facecolor='#c9c8c2',label='≤2 of 5')],
         frameon=False,fontsize=7.8,loc='upper right',bbox_to_anchor=(1.0,0.42))
b.set_title('B  Consensus importance and cross-model agreement',loc='left',fontsize=9.8,fontweight='bold',pad=10)
fig.suptitle('Cross-model consensus on feature importance, five best-performing algorithms',x=0.02,y=1.02,ha='left',fontsize=11.5,fontweight='bold')
plt.savefig('fig/Fig14_consensus.png',bbox_inches='tight'); plt.close()
print('\nFigs 13-14 written')
