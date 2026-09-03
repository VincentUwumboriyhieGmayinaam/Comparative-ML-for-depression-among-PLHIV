import os; os.environ.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
import pandas as pd, numpy as np, pickle, json, warnings; warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.calibration import calibration_curve
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.linewidth':.8,
 'axes.edgecolor':'#52514e','xtick.color':'#52514e','ytick.color':'#52514e','axes.labelcolor':'#0b0b0b',
 'text.color':'#0b0b0b','figure.dpi':300,'savefig.dpi':300,'savefig.bbox':'tight',
 'axes.spines.top':False,'axes.spines.right':False})
BLUE,ORANGE,AQUA,YELLOW,MAGENTA,GREEN,VIOLET,RED='#2a78d6','#eb6834','#1baf7a','#eda100','#e87ba4','#008300','#4a3aa7','#e34948'
CATS=[BLUE,ORANGE,AQUA,YELLOW,MAGENTA,GREEN,VIOLET,RED,'#52514e','#a8a7a0']
GRID='#e6e5e0'; INK='#0b0b0b'; MUT='#52514e'; O='fig/'
CUT=10
df=pd.DataFrame([json.loads(l) for l in open(f'rows_{CUT}.jsonl')]).sort_values('CV_AUC',ascending=False).reset_index(drop=True)
probs=pickle.load(open(f'prob_{CUT}.pkl','rb')); S=pickle.load(open(f'split_{CUT}.pkl','rb')); yte=S['yte']
ORDER=list(df.Model)

# ---------- FIG 4: performance heatmap ----------
M=['Accuracy','Precision','Sensitivity','Specificity','F1','AUC','Kappa']
H=df.set_index('Model')[M]
seq=LinearSegmentedColormap.from_list('bl',['#f4f8fd','#cfe1f6','#8fbaea','#4d90dd','#2a78d6','#1b5aa4'])
fig,ax=plt.subplots(figsize=(7.6,4.3))
im=ax.imshow(H.values,cmap=seq,vmin=0.15,vmax=1.0,aspect='auto')
ax.set_xticks(range(len(M))); ax.set_xticklabels(M,fontsize=8.4)
ax.set_yticks(range(len(H))); ax.set_yticklabels(H.index,fontsize=8.4)
for i in range(H.shape[0]):
    for j in range(H.shape[1]):
        v=H.values[i,j]
        ax.text(j,i,f'{v:.3f}',ha='center',va='center',fontsize=7.6,
                color='white' if v>0.72 else INK)
ax.set_xticks(np.arange(-.5,len(M),1),minor=True); ax.set_yticks(np.arange(-.5,len(H),1),minor=True)
ax.grid(which='minor',color='white',lw=1.6); ax.tick_params(which='minor',length=0)
for s in ax.spines.values(): s.set_visible(False)
cb=plt.colorbar(im,ax=ax,fraction=.028,pad=.02); cb.outline.set_visible(False); cb.ax.tick_params(labelsize=7.5)
ax.set_title('Held-out test-set performance, ranked by cross-validated AUC',loc='left',fontsize=9.5,fontweight='bold',pad=10)
plt.tight_layout(); plt.savefig(O+'Fig4_performance_heatmap.png'); plt.close()

# ---------- FIG 5: ROC ----------
fig,axs=plt.subplots(1,2,figsize=(10,4.3))
a=axs[0]
a.plot([0,1],[0,1],ls=(0,(4,3)),color='#a8a7a0',lw=1)
for i,m in enumerate(ORDER):
    fpr,tpr,_=roc_curve(yte,probs[m]); A=auc(fpr,tpr)
    lw=2.0 if i<3 else 1.0; al=1.0 if i<3 else .55
    a.plot(fpr,tpr,lw=lw,alpha=al,color=CATS[i%len(CATS)],label=f'{m} ({A:.3f})')
a.set_xlabel('1 − specificity (false-positive rate)'); a.set_ylabel('Sensitivity (true-positive rate)')
a.legend(frameon=False,fontsize=7.1,loc='lower right'); a.grid(color=GRID,lw=.7); a.set_axisbelow(True)
a.set_xlim(-.02,1.02); a.set_ylim(-.02,1.02)
a.set_title('A  ROC curves, held-out test set (n=53)',loc='left',fontsize=9.5,fontweight='bold',pad=8)
b=axs[1]
o=df.sort_values('CV_AUC')
ypos=np.arange(len(o))
b.barh(ypos,o.CV_AUC,xerr=o.CV_SD,height=.62,color=[BLUE if v>=o.CV_AUC.max()-.005 else '#9dc0ea' for v in o.CV_AUC],
       error_kw=dict(ecolor=MUT,lw=.9,capsize=2.5),edgecolor='white',linewidth=.8)
for i,(v,s) in enumerate(zip(o.CV_AUC,o.CV_SD)): b.text(v+s+.006,i,f'{v:.3f}',va='center',fontsize=7.6,color=INK)
b.set_yticks(ypos); b.set_yticklabels(o.Model,fontsize=8.2); b.set_xlim(0.5,1.0)
b.set_xlabel('Cross-validated AUC (mean ± SD, 5-fold × 4 repeats)')
b.axvline(0.5,color='#a8a7a0',ls=(0,(4,3)),lw=1); b.grid(axis='x',color=GRID,lw=.7); b.set_axisbelow(True)
b.set_title('B  Internal-validation AUC with fold-to-fold variability',loc='left',fontsize=9.5,fontweight='bold',pad=8)
plt.tight_layout(); plt.savefig(O+'Fig5_roc.png'); plt.close()

# ---------- FIG 6: confusion matrices ----------
fig,axs=plt.subplots(2,5,figsize=(12.5,5.4))
for i,m in enumerate(ORDER):
    ax=axs[i//5,i%5]; cm=confusion_matrix(yte,(probs[m]>=.5).astype(int))
    ax.imshow(cm,cmap=seq,vmin=0,vmax=cm.max()*1.15)
    for r in range(2):
        for c in range(2):
            ax.text(c,r,cm[r,c],ha='center',va='center',fontsize=13,fontweight='bold',
                    color='white' if cm[r,c]>cm.max()*0.6 else INK)
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(['Pred 0','Pred 1'],fontsize=7.6); ax.set_yticklabels(['True 0','True 1'],fontsize=7.6)
    row=df[df.Model==m].iloc[0]
    ax.set_title(f'{m}\nAUC {row.AUC:.3f} · κ {row.Kappa:.3f}',fontsize=8.1,pad=6)
    for s in ax.spines.values(): s.set_visible(False)
    ax.tick_params(length=0)
plt.tight_layout(); plt.savefig(O+'Fig6_confusion.png'); plt.close()

# ---------- FIG 7: calibration ----------
fig,axs=plt.subplots(1,2,figsize=(9.8,4.1))
a=axs[0]; a.plot([0,1],[0,1],ls=(0,(4,3)),color='#a8a7a0',lw=1,label='Perfect calibration')
for i,m in enumerate(ORDER[:5]):
    try:
        pt,pp=calibration_curve(yte,probs[m],n_bins=5,strategy='quantile')
        a.plot(pp,pt,'o-',ms=5,lw=1.6,color=CATS[i],label=m)
    except Exception: pass
a.set_xlabel('Mean predicted probability'); a.set_ylabel('Observed proportion depressed')
a.legend(frameon=False,fontsize=7.4,loc='upper left'); a.grid(color=GRID,lw=.7); a.set_axisbelow(True)
a.set_title('A  Calibration curves (top five by CV-AUC)',loc='left',fontsize=9.5,fontweight='bold',pad=8)
b=axs[1]; o=df.sort_values('Cal_slope')
yp=np.arange(len(o))
cols=[GREEN if 0.8<=v<=1.25 else (YELLOW if 0.5<=v<1.6 else RED) for v in o.Cal_slope]
b.barh(yp,o.Cal_slope,height=.62,color=cols,edgecolor='white',linewidth=.8)
b.axvline(1.0,color=MUT,ls=(0,(4,3)),lw=1.2)
b.text(1.03,len(o)-0.4,'ideal = 1.00',fontsize=7.4,color=MUT)
for i,v in enumerate(o.Cal_slope): b.text(v+.03,i,f'{v:.2f}',va='center',fontsize=7.6,color=INK)
b.set_yticks(yp); b.set_yticklabels(o.Model,fontsize=8.2); b.set_xlabel('Calibration slope')
b.grid(axis='x',color=GRID,lw=.7); b.set_axisbelow(True)
b.set_title('B  Calibration slope by algorithm',loc='left',fontsize=9.5,fontweight='bold',pad=8)
plt.tight_layout(); plt.savefig(O+'Fig7_calibration.png'); plt.close()
print('figs 4-7 done')
