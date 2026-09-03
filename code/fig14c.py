import os; os.environ.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import Patch
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.linewidth':.8,
 'axes.edgecolor':'#52514e','xtick.color':'#52514e','ytick.color':'#52514e','axes.labelcolor':'#0b0b0b',
 'text.color':'#0b0b0b','figure.dpi':300,'savefig.dpi':300,'savefig.bbox':'tight','axes.spines.top':False,'axes.spines.right':False})
BLUE,ORANGE,AQUA,YELLOW,VIOLET,RED,GREEN,MAG='#2a78d6','#eb6834','#1baf7a','#eda100','#4a3aa7','#e34948','#008300','#e87ba4'
MUT,GRID,INK,PALE='#52514e','#e6e5e0','#0b0b0b','#c9c8c2'
RD=pd.read_csv('shap_consensus.csv',index_col=0)
MODELS=['Random Forest','CatBoost','Gradient Boosting','Logistic Regression','XGBoost']
SHORT=['Random\nForest','CatBoost','Gradient\nBoosting','Logistic\nRegression','XGBoost']
top=RD.head(10)
HL=[BLUE,ORANGE,GREEN,VIOLET,AQUA,YELLOW,MAG,RED,'#8c8b85','#b0afa8']
fig,axs=plt.subplots(1,2,figsize=(14.6,6.2),gridspec_kw={'width_ratios':[1.42,1],'wspace':0.62})
a=axs[0]
a.axhspan(0.3,10.5,color='#f3f7fd',zorder=0)
x=np.arange(len(MODELS))
for i,(feat,row) in enumerate(top.iterrows()):
    r=[row[m] for m in MODELS]; five=row['Top-10 count']==5
    col=HL[i]; a.plot(x,r,'-o',color=col,lw=2.6 if five else 1.4,ms=6.5 if five else 4.6,
        alpha=1.0 if five else .5,markeredgecolor='white',markeredgewidth=1.1,zorder=6 if five else 3)
    a.text(-0.12,r[0],feat,ha='right',va='center',fontsize=8.3,
           color=col if five else MUT,fontweight='bold' if five else 'normal')
a.axhline(10.5,color=MUT,ls=(0,(4,3)),lw=1,zorder=2)
a.text(len(MODELS)-1+0.06,10.9,'top-10 boundary',fontsize=7.5,color=MUT,va='top',ha='right')
a.set_xticks(x); a.set_xticklabels(SHORT,fontsize=8.4)
a.set_xlim(-2.35,len(MODELS)-1+0.10)
a.set_ylim(0.3,30); a.invert_yaxis()
a.set_yticks([1,5,10,15,20,25,30])
a.set_ylabel('SHAP importance rank within model   (1 = most important of 52)',fontsize=8.8)
a.grid(axis='y',color=GRID,lw=.7); a.set_axisbelow(True)
a.spines['left'].set_visible(False); a.tick_params(axis='y',length=0)
a.set_title('A  Rank trajectory across the five models',loc='left',fontsize=10,fontweight='bold',pad=10)
b=axs[1]
t2=top.iloc[::-1]
cols=[GREEN if c==5 else (AQUA if c==4 else (YELLOW if c==3 else PALE)) for c in t2['Top-10 count']]
yp=np.arange(len(t2))
b.barh(yp,t2['Mean rank'],height=.66,color=cols,edgecolor='white',linewidth=1)
XMAX=max(t2['Mean rank'])*1.95
for i,(mr,c) in enumerate(zip(t2['Mean rank'],t2['Top-10 count'])):
    b.text(mr+0.45,i,f'{mr:.1f}',va='center',fontsize=8.2,color=INK,fontweight='bold')
    b.text(XMAX*0.985,i,f'{int(c)}/5',va='center',ha='right',fontsize=7.8,color=MUT)
b.text(XMAX*0.985,len(t2)-0.35,'models',va='bottom',ha='right',fontsize=7.4,color=MUT,style='italic')
b.set_yticks(yp); b.set_yticklabels(t2.index,fontsize=8.7)
b.set_xlabel('Mean SHAP importance rank across the five models',fontsize=8.8)
b.set_xlim(0,XMAX); b.set_xticks([0,5,10,15])
b.grid(axis='x',color=GRID,lw=.7); b.set_axisbelow(True)
b.spines['left'].set_visible(False); b.tick_params(axis='y',length=0)
b.set_title('B  Consensus importance and agreement',loc='left',fontsize=10,fontweight='bold',pad=10)
fig.legend(handles=[Patch(facecolor=GREEN,label='top 10 in all 5 models'),Patch(facecolor=AQUA,label='4 of 5'),
                    Patch(facecolor=YELLOW,label='3 of 5'),Patch(facecolor=PALE,label='2 of 5 or fewer')],
           frameon=False,fontsize=8.2,ncol=4,loc='lower center',bbox_to_anchor=(0.5,-0.055),handlelength=1.3)
fig.suptitle('Cross-model consensus on SHAP feature importance, five best-performing algorithms',
             x=0.035,y=1.02,ha='left',fontsize=12,fontweight='bold')
plt.savefig('fig/Fig14_consensus.png',bbox_inches='tight'); plt.close()
print('rebuilt')
