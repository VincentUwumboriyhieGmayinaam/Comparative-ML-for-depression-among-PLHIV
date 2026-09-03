import os; os.environ.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
import pandas as pd, numpy as np, pickle, warnings; warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.linewidth':.8,
 'axes.edgecolor':'#52514e','xtick.color':'#52514e','ytick.color':'#52514e','axes.labelcolor':'#0b0b0b',
 'text.color':'#0b0b0b','figure.dpi':300,'savefig.dpi':300,'savefig.bbox':'tight','axes.spines.top':False,'axes.spines.right':False})
BLUE,ORANGE,GREEN,VIOLET,RED,MUT,GRID,INK='#2a78d6','#eb6834','#008300','#4a3aa7','#e34948','#52514e','#e6e5e0','#0b0b0b'
SV=pickle.load(open('shap_top5.pkl','rb'))
S=pickle.load(open('split_10.pkl','rb')); Xte=S['Xte_i']
disp=[c.replace('_',' ') for c in Xte.columns]
Xd=Xte.copy(); Xd.columns=disp
MODELS=['Random Forest','CatBoost','Gradient Boosting','Logistic Regression','XGBoost']
COLS=[BLUE,ORANGE,GREEN,VIOLET,RED]
FEATS=['Counselling availability','Healthcare quality','Sex','Comfort disclosing']
rng=np.random.default_rng(7)
fig,axs=plt.subplots(len(FEATS),len(MODELS),figsize=(15.2,10.6),sharex='row')
for r,feat in enumerate(FEATS):
    j=disp.index(feat); v=Xd[feat].values
    lo=min(np.nanmin(SV[m][:,j]) for m in MODELS); hi=max(np.nanmax(SV[m][:,j]) for m in MODELS)
    pad=(hi-lo)*0.16
    for c,mname in enumerate(MODELS):
        ax=axs[r,c]; sv=SV[mname][:,j]
        jit=rng.normal(0,0.035*(np.nanmax(v)-np.nanmin(v)+1e-9),len(v))
        ax.axhline(0,color=MUT,lw=.9,zorder=1)
        ax.scatter(v+jit,sv,s=26,color=COLS[c],alpha=.5,edgecolors='white',linewidths=.5,zorder=3)
        lv=np.unique(v[~np.isnan(v)])
        if len(lv)<=6:
            mm=[np.nanmean(sv[v==L]) for L in lv]
            ax.plot(lv,mm,'-o',color=INK,lw=1.7,ms=6,markerfacecolor='white',markeredgewidth=1.5,zorder=5)
            ax.set_xticks(lv); ax.set_xticklabels([f'{int(L)}' for L in lv],fontsize=8)
        else:
            z=np.polyfit(v,sv,1); xs=np.linspace(v.min(),v.max(),50)
            ax.plot(xs,np.polyval(z,xs),color=INK,lw=1.7,zorder=5)
        ax.set_ylim(lo-pad,hi+pad)
        ax.grid(color=GRID,lw=.6); ax.set_axisbelow(True); ax.tick_params(labelsize=7.6)
        if r==0: ax.set_title(mname,fontsize=9.6,fontweight='bold',pad=9,color=COLS[c])
        if c==0: ax.set_ylabel(f'SHAP value\n{feat}',fontsize=8.6)
        else: ax.tick_params(labelleft=False)
        if r==len(FEATS)-1: ax.set_xlabel('Feature value',fontsize=8.4)
        rho=np.corrcoef(v,sv)[0,1]
        ax.text(.03,.96,f'ρ = {rho:+.2f}',transform=ax.transAxes,fontsize=7.6,va='top',color=MUT)
fig.suptitle('SHAP dependence plots for the four consensus features, across the five best-performing algorithms',
             x=0.035,y=0.995,ha='left',fontsize=12.5,fontweight='bold')
fig.text(0.035,0.968,'Each point is one participant. The black line joins the mean SHAP value at each observed feature level. ρ is the correlation between feature value and SHAP value within that model.',
         fontsize=8.6,color=MUT,ha='left')
plt.tight_layout(rect=[0,0,1,0.958]); plt.savefig('fig/Fig17_shap_dependence.png',bbox_inches='tight'); plt.close()
print('Fig17 written')
for feat in FEATS:
    j=disp.index(feat)
    rr=[np.corrcoef(Xd[feat].values,SV[m][:,j])[0,1] for m in MODELS]
    print('  %-26s rho across models: %s  (mean %+.2f)'%(feat,', '.join(f'{x:+.2f}' for x in rr),np.mean(rr)))
