import os; os.environ.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
import numpy as np, pandas as pd, pickle, warnings; warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.linewidth':.8,
 'axes.edgecolor':'#52514e','xtick.color':'#52514e','ytick.color':'#52514e','axes.labelcolor':'#0b0b0b',
 'text.color':'#0b0b0b','figure.dpi':300,'savefig.dpi':300,'savefig.bbox':'tight','axes.spines.top':False,'axes.spines.right':False})
BLUE,ORANGE,AQUA,RED,MUT,GRID,INK='#2a78d6','#eb6834','#1baf7a','#e34948','#52514e','#e6e5e0','#0b0b0b'
OOF=pickle.load(open('oof_top5.pkl','rb'))
S=pickle.load(open('split_10.pkl','rb')); yall=np.concatenate([S['ytr'],S['yte']])
p=OOF['Random Forest']; o=np.argsort(p); ps=p[o]; ys=yall[o]
fig,axs=plt.subplots(1,2,figsize=(12.4,4.8),gridspec_kw={'width_ratios':[1.55,1],'wspace':0.22})
a=axs[0]
x=np.arange(len(ps))
a.vlines(x[ys==0],0,ps[ys==0],color=BLUE,lw=1.05,alpha=.85,label='Not depressed (PHQ-9 < 10)')
a.vlines(x[ys==1],0,ps[ys==1],color=ORANGE,lw=1.05,alpha=.9,label='Depressed (PHQ-9 ≥ 10)')
for t,ls in [(0.30,(0,(5,3))),(0.40,(0,(2,2)))]:
    a.axhline(t,color=MUT,ls=ls,lw=1.1)
    n_ref=(ps>=t).sum(); sens=((ps>=t)&(ys==1)).sum()/ys.sum()
    a.text(2,t+0.018,f'threshold {t:.2f} — screen {100*n_ref/len(ps):.0f}% of attendees, detect {100*sens:.0f}% of cases',
           fontsize=7.5,color=MUT)
a.set_xlim(-2,len(ps)+2); a.set_ylim(0,1.02)
a.set_xlabel('Participants, ranked by predicted probability (n=264)')
a.set_ylabel('Predicted probability of depression')
a.legend(frameon=False,fontsize=8,loc='upper left'); a.grid(axis='y',color=GRID,lw=.7); a.set_axisbelow(True)
a.set_title('A  Individual predicted risk, Random Forest (out-of-fold)',loc='left',fontsize=9.8,fontweight='bold',pad=10)
b=axs[1]
data=[p[yall==0],p[yall==1]]
bp=b.boxplot(data,vert=True,widths=.5,patch_artist=True,showfliers=False,
             medianprops=dict(color=INK,lw=1.6),whiskerprops=dict(color=MUT,lw=1),capprops=dict(color=MUT,lw=1),
             boxprops=dict(lw=.9,edgecolor=MUT))
for patch,c in zip(bp['boxes'],[BLUE,ORANGE]): patch.set_facecolor(c); patch.set_alpha(.55)
rng=np.random.default_rng(1)
for i,(d,c) in enumerate(zip(data,[BLUE,ORANGE])):
    b.scatter(np.full(len(d),i+1)+rng.normal(0,.055,len(d)),d,s=11,color=c,alpha=.6,edgecolors='white',linewidths=.35,zorder=3)
b.set_xticks([1,2]); b.set_xticklabels([f'Not depressed\n(n={(yall==0).sum()})',f'Depressed\n(n={(yall==1).sum()})'],fontsize=8.4)
b.set_ylabel('Predicted probability of depression'); b.set_ylim(0,1.02)
b.grid(axis='y',color=GRID,lw=.7); b.set_axisbelow(True)
med0,med1=np.median(p[yall==0]),np.median(p[yall==1])
b.text(1,med0+0.035,f'median {med0:.2f}',ha='center',fontsize=7.8,color=INK)
b.text(2,med1+0.035,f'median {med1:.2f}',ha='center',fontsize=7.8,color=INK)
b.set_title('B  Predicted risk separation by true outcome',loc='left',fontsize=9.8,fontweight='bold',pad=10)
plt.savefig('fig/Fig16_predicted_probabilities.png',bbox_inches='tight'); plt.close()
print('median predicted risk: not depressed %.3f | depressed %.3f'%(med0,med1))
for t in [0.20,0.30,0.40,0.50]:
    print('  threshold %.2f -> screen %.1f%%, detect %.1f%% of cases'%(t,100*(p>=t).mean(),100*((p>=t)&(yall==1)).sum()/yall.sum()))
