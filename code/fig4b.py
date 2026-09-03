import pandas as pd, numpy as np, json, warnings; warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.linewidth':.8,
 'axes.edgecolor':'#52514e','xtick.color':'#52514e','ytick.color':'#52514e','axes.labelcolor':'#0b0b0b',
 'text.color':'#0b0b0b','figure.dpi':300,'savefig.dpi':300,'savefig.bbox':'tight'})
INK='#0b0b0b'
CUT=10
df=pd.DataFrame([json.loads(l) for l in open(f'rows_{CUT}.jsonl')]).sort_values('CV_AUC',ascending=False).reset_index(drop=True)
GOOD=['Accuracy','Precision','Sensitivity','Specificity','F1','AUC','Kappa','CV_AUC']
LBL=['Accuracy','Precision','Sensitivity\n(Recall)','Specificity','F1 score','AUC','Cohen κ','CV AUC']
H=df.set_index('Model')[GOOD]
seq=LinearSegmentedColormap.from_list('bl',['#f4f8fd','#cfe1f6','#8fbaea','#4d90dd','#2a78d6','#1b5aa4'])
seqr=LinearSegmentedColormap.from_list('or',['#8a3410','#c4501f','#eb6834','#f39a72','#fbdccf','#fdf3ec'])
fig,axs=plt.subplots(1,2,figsize=(9.8,4.4),gridspec_kw={'width_ratios':[8,1.25],'wspace':0.06})
ax=axs[0]
im=ax.imshow(H.values,cmap=seq,vmin=0.15,vmax=1.0,aspect='auto')
ax.set_xticks(range(len(GOOD))); ax.set_xticklabels(LBL,fontsize=8.2)
ax.set_yticks(range(len(H))); ax.set_yticklabels(H.index,fontsize=8.5)
for i in range(H.shape[0]):
    for j in range(H.shape[1]):
        v=H.values[i,j]
        ax.text(j,i,f'{v:.3f}',ha='center',va='center',fontsize=7.7,color='white' if v>0.72 else INK)
ax.set_xticks(np.arange(-.5,len(GOOD),1),minor=True); ax.set_yticks(np.arange(-.5,len(H),1),minor=True)
ax.grid(which='minor',color='white',lw=1.8); ax.tick_params(which='minor',length=0); ax.tick_params(length=0)
for s in ax.spines.values(): s.set_visible(False)
ax.set_title('Higher is better',loc='left',fontsize=9,fontweight='bold',pad=8,color='#1b5aa4')
b=axs[1]
B=df.set_index('Model')[['Brier']]
imb=b.imshow(B.values,cmap=seqr,vmin=0.10,vmax=0.30,aspect='auto')
b.set_xticks([0]); b.set_xticklabels(['Brier\nscore'],fontsize=8.2)
b.set_yticks(range(len(B))); b.set_yticklabels([])
for i,v in enumerate(B.values.ravel()):
    b.text(0,i,f'{v:.3f}',ha='center',va='center',fontsize=7.7,color='white' if v<0.155 else INK)
b.set_xticks(np.arange(-.5,1,1),minor=True); b.set_yticks(np.arange(-.5,len(B),1),minor=True)
b.grid(which='minor',color='white',lw=1.8); b.tick_params(which='minor',length=0); b.tick_params(length=0)
for s in b.spines.values(): s.set_visible(False)
b.set_title('Lower is better',loc='left',fontsize=9,fontweight='bold',pad=8,color='#8a3410')
fig.suptitle('Comparative performance of ten machine learning algorithms, ranked by cross-validated AUC',
             x=0.055,y=1.035,ha='left',fontsize=10.5,fontweight='bold')
cb=fig.colorbar(im,ax=axs,fraction=.016,pad=.02); cb.outline.set_visible(False); cb.ax.tick_params(labelsize=7.4)
plt.savefig('fig/Fig4_performance_heatmap.png',bbox_inches='tight'); plt.close()
print('Fig4 regenerated with Brier score')
