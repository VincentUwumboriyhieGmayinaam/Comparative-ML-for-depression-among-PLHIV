import os; os.environ.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
import pandas as pd, numpy as np, pickle, json, warnings; warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import spearmanr
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.linewidth':.8,
 'axes.edgecolor':'#52514e','xtick.color':'#52514e','ytick.color':'#52514e','axes.labelcolor':'#0b0b0b',
 'text.color':'#0b0b0b','figure.dpi':300,'savefig.dpi':300,'savefig.bbox':'tight','axes.spines.top':False,'axes.spines.right':False})
BLUE,ORANGE,AQUA,RED,MUT,GRID,INK='#2a78d6','#eb6834','#1baf7a','#e34948','#52514e','#e6e5e0','#0b0b0b'
CUT=10; S=pickle.load(open(f'split_{CUT}.pkl','rb'))
Xtr,ytr,Xte,yte=S['Xtr_i'],S['ytr'],S['Xte_i'],S['yte']
Xall=pd.concat([Xtr,Xte]); yall=np.concatenate([ytr,yte])
disp=[c.replace('_',' ') for c in Xall.columns]
L=[]
def P(*a):
    s=' '.join(str(x) for x in a); L.append(s); print(s)

# ---- Spearman correlation ----
corr=spearmanr(Xall).correlation
corr=(corr+corr.T)/2; np.fill_diagonal(corr,1.0)
corr=np.nan_to_num(corr,nan=0.0)
P('='*76); P('MULTICOLLINEARITY ASSESSMENT AND HIERARCHICAL FEATURE REDUCTION'); P('='*76)
iu=np.triu_indices_from(corr,k=1)
P('\nSpearman correlation among %d features:'%corr.shape[0])
P('  |rho| >= 0.9 : %d pairs'%np.sum(np.abs(corr[iu])>=0.9))
P('  |rho| >= 0.8 : %d pairs'%np.sum(np.abs(corr[iu])>=0.8))
P('  |rho| >= 0.7 : %d pairs'%np.sum(np.abs(corr[iu])>=0.7))
P('  max |rho| = %.3f ; mean |rho| = %.3f'%(np.abs(corr[iu]).max(),np.abs(corr[iu]).mean()))
hi=[(disp[i],disp[j],corr[i,j]) for i,j in zip(*iu) if abs(corr[i,j])>=0.6]
hi=sorted(hi,key=lambda t:-abs(t[2]))
P('\nFeature pairs with |rho| >= 0.6 (%d):'%len(hi))
for a,b,r in hi[:15]: P('   %-34s %-34s %+.3f'%(a,b,r))

# ---- Ward linkage ----
dist=1-np.abs(corr); np.fill_diagonal(dist,0); dist=np.clip(dist,0,None)
link=hierarchy.ward(squareform(dist,checks=False))
THR=0.7
cl=hierarchy.fcluster(link,THR,criterion='distance')
P('\nWard linkage on distance = 1 - |Spearman rho|; cut at %.1f'%THR)
P('  clusters formed: %d (from %d features)'%(len(set(cl)),len(cl)))

# representative per cluster = highest univariable |log-OR| z-statistic with the outcome
zs={}
for i,c in enumerate(Xall.columns):
    try:
        r=sm.Logit(yall,sm.add_constant(Xall[c].values.astype(float))).fit(disp=0)
        zs[c]=abs(r.tvalues[1])
    except Exception: zs[c]=0.0
groups={}
for i,c in enumerate(cl): groups.setdefault(c,[]).append(i)
sel=[]
for c,idx in sorted(groups.items()):
    best=max(idx,key=lambda i: zs[Xall.columns[i]])
    sel.append(best)
sel=sorted(sel)
SELF=[Xall.columns[i] for i in sel]
P('  features retained: %d'%len(SELF))
P('\nRetained feature set:')
for f in SELF: P('   - %s  (|z| = %.2f)'%(f.replace('_',' '),zs[f]))
json.dump(SELF,open('selected_features.json','w'),indent=1)

# ---- VIF on retained set ----
Xs=Xall[SELF].astype(float)
Xs=Xs.loc[:,Xs.std()>0]
vif=pd.DataFrame({'feature':Xs.columns,
    'VIF':[variance_inflation_factor(Xs.values,i) for i in range(Xs.shape[1])]}).sort_values('VIF',ascending=False)
P('\nVariance inflation factors on the retained set:')
P('  max VIF = %.2f ; features with VIF > 5: %d ; VIF > 10: %d'%(vif.VIF.max(),(vif.VIF>5).sum(),(vif.VIF>10).sum()))
for _,r in vif.head(10).iterrows(): P('   %-40s %6.2f'%(r.feature.replace('_',' '),r.VIF))
vif.to_csv('vif.csv',index=False)

# ---- FIG 12: correlation heatmap + dendrogram ----
dend=hierarchy.dendrogram(link,no_plot=True)
o=dend['leaves']
fig=plt.figure(figsize=(12.6,6.4))
gs=fig.add_gridspec(1,2,width_ratios=[1.05,1],wspace=0.16)
ax1=fig.add_subplot(gs[0,0])
div=LinearSegmentedColormap.from_list('dv',['#1b5aa4','#4d90dd','#a9c9ef','#f3f2ee','#f6b394','#eb6834','#8a3410'])
im=ax1.imshow(corr[np.ix_(o,o)],cmap=div,vmin=-1,vmax=1,aspect='auto')
ax1.set_xticks(range(len(o))); ax1.set_xticklabels([disp[i] for i in o],rotation=90,fontsize=4.6)
ax1.set_yticks(range(len(o))); ax1.set_yticklabels([disp[i] for i in o],fontsize=4.6)
ax1.tick_params(length=0)
for s in ax1.spines.values(): s.set_visible(False)
cb=plt.colorbar(im,ax=ax1,fraction=.036,pad=.02); cb.outline.set_visible(False)
cb.ax.tick_params(labelsize=7); cb.set_label('Spearman ρ',fontsize=8)
ax1.set_title('A  Correlation matrix, features ordered by hierarchical clustering',loc='left',fontsize=9.5,fontweight='bold',pad=10)
ax2=fig.add_subplot(gs[0,1])
hierarchy.set_link_color_palette([BLUE,ORANGE,AQUA,'#4a3aa7',RED,'#eda100','#008300'])
hierarchy.dendrogram(link,labels=disp,orientation='right',ax=ax2,color_threshold=THR,
                     above_threshold_color='#a8a7a0',leaf_font_size=4.8)
ax2.axvline(THR,color=RED,ls=(0,(4,3)),lw=1.3)
ax2.text(THR+0.03,len(disp)*9.5,f'cut at {THR}',color=RED,fontsize=7.6,va='top')
ax2.set_xlabel('Ward linkage distance  (1 − |Spearman ρ|)',fontsize=8.4)
ax2.tick_params(axis='y',length=0)
for s in ['top','right','left']: ax2.spines[s].set_visible(False)
ax2.set_title('B  Dendrogram and cut point defining the retained feature set',loc='left',fontsize=9.5,fontweight='bold',pad=10)
plt.savefig('fig/Fig12_collinearity.png',bbox_inches='tight'); plt.close()
open('cluster_log.txt','w').write('\n'.join(L))
print('\nFig12 written')
