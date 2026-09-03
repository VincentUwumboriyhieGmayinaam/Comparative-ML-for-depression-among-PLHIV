import os; os.environ.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
import numpy as np, pickle, warnings; warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.linewidth':.8,
 'axes.edgecolor':'#52514e','xtick.color':'#52514e','ytick.color':'#52514e','axes.labelcolor':'#0b0b0b',
 'text.color':'#0b0b0b','figure.dpi':300,'savefig.dpi':300,'savefig.bbox':'tight','axes.spines.top':False,'axes.spines.right':False})
BLUE,ORANGE,AQUA,MUT,GRID,INK='#2a78d6','#eb6834','#1baf7a','#52514e','#e6e5e0','#0b0b0b'
OOF=pickle.load(open('oof_top5.pkl','rb')); p=OOF['Random Forest']
S=pickle.load(open('split_10.pkl','rb')); y=np.concatenate([S['ytr'],S['yte']])
N=len(y); prev=y.mean()
ths=np.linspace(0.05,0.75,60); nb=[];nba=[]
for t in ths:
    pred=p>=t; tp=((pred)&(y==1)).sum(); fp=((pred)&(y==0)).sum()
    nb.append(tp/N-(fp/N)*(t/(1-t))); nba.append(prev-(1-prev)*(t/(1-t)))
fig,axs=plt.subplots(1,2,figsize=(11.4,4.8),gridspec_kw={'wspace':0.24})
a=axs[0]
a.plot(ths,nb,color=BLUE,lw=2.4,label='Model-guided screening',zorder=4)
a.plot(ths,nba,color=ORANGE,lw=1.8,ls=(0,(5,3)),label='Screen everyone',zorder=3)
a.axhline(0,color=MUT,lw=1.2,label='Screen no one',zorder=2)
a.set_ylim(-0.15,max(nb)*1.18); a.set_xlim(0.05,0.75)
a.set_xlabel('Threshold probability'); a.set_ylabel('Net benefit')
a.legend(frameon=False,fontsize=8.2,loc='upper right'); a.grid(color=GRID,lw=.7); a.set_axisbelow(True)
a.set_title('A  Decision-curve analysis',loc='left',fontsize=10,fontweight='bold',pad=10)
b=axs[1]
tl=[0.20,0.25,0.30,0.35,0.40,0.50]; sens=[];frac=[]
for t in tl:
    pred=p>=t; tp=((pred)&(y==1)).sum(); fn=((~pred)&(y==1)).sum()
    sens.append(100*tp/(tp+fn)); frac.append(100*pred.mean())
x=np.arange(len(tl)); w=.37
b.bar(x-w/2,frac,w,color=ORANGE,label='Attendees given the PHQ-9',edgecolor='white',linewidth=1)
b.bar(x+w/2,sens,w,color=AQUA,label='True cases detected',edgecolor='white',linewidth=1)
for i,(f,s) in enumerate(zip(frac,sens)):
    b.text(i-w/2,f+2.0,f'{f:.0f}',ha='center',fontsize=8,color=INK)
    b.text(i+w/2,s+2.0,f'{s:.0f}',ha='center',fontsize=8,color=INK)
b.set_xticks(x); b.set_xticklabels([f'{t:.2f}' for t in tl]); b.set_xlabel('Risk threshold')
b.set_ylabel('Percent'); b.set_ylim(0,118); b.set_yticks([0,20,40,60,80,100])
b.grid(axis='y',color=GRID,lw=.7); b.set_axisbelow(True)
b.legend(frameon=False,fontsize=8.2,ncol=2,loc='lower center',bbox_to_anchor=(0.5,1.045),handlelength=1.3,columnspacing=1.8)
b.set_title('B  Screening workload vs case detection',loc='left',fontsize=10,fontweight='bold',pad=34)
plt.savefig('fig/Fig11_decision_curve.png',bbox_inches='tight'); plt.close()
print('Fig11 fixed')
