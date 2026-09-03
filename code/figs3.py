import os; os.environ.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
import pandas as pd, numpy as np, pickle, json, warnings; warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
import shap
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.linewidth':.8,
 'axes.edgecolor':'#52514e','xtick.color':'#52514e','ytick.color':'#52514e','axes.labelcolor':'#0b0b0b',
 'text.color':'#0b0b0b','figure.dpi':300,'savefig.dpi':300,'savefig.bbox':'tight',
 'axes.spines.top':False,'axes.spines.right':False})
BLUE,ORANGE,AQUA,RED,MUT,GRID,INK='#2a78d6','#eb6834','#1baf7a','#e34948','#52514e','#e6e5e0','#0b0b0b'
O='fig/'; CUT=10
S=pickle.load(open(f'split_{CUT}.pkl','rb'))
Xtr_r,ytr_r,Xte_i,yte,Xtr_i,ytr=S['Xtr_r'],S['ytr_r'],S['Xte_i'],S['yte'],S['Xtr_i'],S['ytr']
hp=json.load(open(f'hyper_{CUT}.json')) if os.path.exists(f'hyper_{CUT}.json') else {}
rows={json.loads(l)['Model']:json.loads(l) for l in open(f'rows_{CUT}.jsonl')}
import ast
bp=ast.literal_eval(rows['Random Forest']['Params'])
print('Random Forest best params:',bp)
rf=RandomForestClassifier(random_state=42,n_jobs=1,**bp).fit(Xtr_r,ytr_r)

# ---------- FIG 8: permutation importance ----------
pi=permutation_importance(rf,Xte_i,yte,n_repeats=50,random_state=42,scoring='roc_auc',n_jobs=1)
imp=pd.DataFrame({'f':Xte_i.columns,'m':pi.importances_mean,'s':pi.importances_std}).sort_values('m',ascending=False).head(20)
imp=imp.iloc[::-1]
fig,ax=plt.subplots(figsize=(7.4,6.0))
cols=[BLUE if v>0 else '#c9c8c2' for v in imp.m]
ax.barh(np.arange(len(imp)),imp.m,xerr=imp.s,height=.66,color=cols,edgecolor='white',linewidth=.7,
        error_kw=dict(ecolor=MUT,lw=.8,capsize=2))
ax.set_yticks(np.arange(len(imp))); ax.set_yticklabels([f.replace('_',' ') for f in imp.f],fontsize=8.2)
ax.set_xlabel('Decrease in test-set AUC when the feature is permuted (mean ± SD, 50 repeats)')
ax.axvline(0,color=MUT,lw=.9); ax.grid(axis='x',color=GRID,lw=.7); ax.set_axisbelow(True)
ax.set_title('Permutation importance — Random Forest (top 20 of 52 features)',loc='left',fontsize=9.5,fontweight='bold',pad=10)
plt.tight_layout(); plt.savefig(O+'Fig8_permutation.png'); plt.close()
imp.iloc[::-1].to_csv('perm_importance.csv',index=False)

# ---------- SHAP ----------
ex=shap.TreeExplainer(rf)
sv=ex.shap_values(Xte_i)
if isinstance(sv,list): sv1=sv[1]
elif sv.ndim==3: sv1=sv[:,:,1]
else: sv1=sv
print('SHAP matrix:',np.shape(sv1))
disp=[c.replace('_',' ') for c in Xte_i.columns]
Xd=Xte_i.copy(); Xd.columns=disp

# FIG 9 bar
plt.figure(figsize=(7.4,6.0))
shap.summary_plot(sv1,Xd,plot_type='bar',max_display=20,show=False,color=BLUE)
plt.gca().set_xlabel('Mean |SHAP value|  (mean absolute contribution to predicted log-odds)',fontsize=8.6)
plt.title('Global feature importance — SHAP, Random Forest',loc='left',fontsize=9.5,fontweight='bold',pad=10)
plt.tight_layout(); plt.savefig(O+'Fig9_shap_bar.png',dpi=300,bbox_inches='tight'); plt.close()

# FIG 10 beeswarm
cmap=LinearSegmentedColormap.from_list('bo',[BLUE,'#b9b8b2',ORANGE])
plt.figure(figsize=(7.8,6.2))
shap.summary_plot(sv1,Xd,max_display=20,show=False,cmap=cmap)
plt.gca().set_xlabel('SHAP value  (impact on predicted log-odds of depression)',fontsize=8.6)
plt.title('SHAP beeswarm — direction and distribution of feature effects',loc='left',fontsize=9.5,fontweight='bold',pad=10)
plt.tight_layout(); plt.savefig(O+'Fig10_shap_beeswarm.png',dpi=300,bbox_inches='tight'); plt.close()

# ---------- FIG 11: decision curve ----------
from sklearn.model_selection import cross_val_predict, StratifiedKFold
Xall=pd.concat([Xtr_i,Xte_i]); yall=np.concatenate([ytr,yte])
oof=cross_val_predict(RandomForestClassifier(random_state=42,n_jobs=1,**bp),Xall,yall,cv=StratifiedKFold(5,shuffle=True,random_state=42),method='predict_proba')[:,1]
N=len(yall); prev=yall.mean()
ths=np.linspace(0.05,0.75,60); nb=[];nba=[]
for t in ths:
    pred=oof>=t; tp=((pred)&(yall==1)).sum(); fp=((pred)&(yall==0)).sum()
    nb.append(tp/N-(fp/N)*(t/(1-t))); nba.append(prev-(1-prev)*(t/(1-t)))
fig,axs=plt.subplots(1,2,figsize=(10,4.2))
a=axs[0]
a.plot(ths,nb,color=BLUE,lw=2.2,label='Model-guided screening')
a.plot(ths,nba,color=ORANGE,lw=1.6,ls=(0,(5,3)),label='Screen everyone')
a.axhline(0,color=MUT,lw=1.2,label='Screen no one')
a.set_ylim(-0.15,max(nb)*1.15); a.set_xlim(0.05,0.75)
a.set_xlabel('Threshold probability'); a.set_ylabel('Net benefit')
a.legend(frameon=False,fontsize=8); a.grid(color=GRID,lw=.7); a.set_axisbelow(True)
a.set_title('A  Decision-curve analysis',loc='left',fontsize=9.5,fontweight='bold',pad=8)
b=axs[1]
tl=[0.20,0.25,0.30,0.35,0.40,0.50]; sens=[];frac=[]
for t in tl:
    pred=oof>=t; tp=((pred)&(yall==1)).sum(); fn=((~pred)&(yall==1)).sum()
    sens.append(tp/(tp+fn)); frac.append(pred.mean())
x=np.arange(len(tl)); w=.36
b.bar(x-w/2,[100*f for f in frac],w,color=ORANGE,label='% of attendees given the PHQ-9',edgecolor='white',linewidth=1)
b.bar(x+w/2,[100*s for s in sens],w,color=AQUA,label='% of true cases detected',edgecolor='white',linewidth=1)
for i,(f,s) in enumerate(zip(frac,sens)):
    b.text(i-w/2,100*f+1.5,f'{100*f:.0f}',ha='center',fontsize=7.6,color=INK)
    b.text(i+w/2,100*s+1.5,f'{100*s:.0f}',ha='center',fontsize=7.6,color=INK)
b.set_xticks(x); b.set_xticklabels([f'{t:.2f}' for t in tl]); b.set_xlabel('Risk threshold')
b.set_ylabel('Percent'); b.set_ylim(0,115); b.legend(frameon=False,fontsize=8,loc='lower left')
b.grid(axis='y',color=GRID,lw=.7); b.set_axisbelow(True)
b.set_title('B  Screening workload vs case detection',loc='left',fontsize=9.5,fontweight='bold',pad=8)
plt.tight_layout(); plt.savefig(O+'Fig11_decision_curve.png'); plt.close()
out=pd.DataFrame({'threshold':tl,'pct_screened':[100*f for f in frac],'sensitivity':sens})
out.to_csv('dca_table.csv',index=False)
print(out.round(3).to_string(index=False))
print('figs 8-11 done')
