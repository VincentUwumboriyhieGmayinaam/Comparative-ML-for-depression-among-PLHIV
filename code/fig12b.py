import os; os.environ.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
import pandas as pd, numpy as np, pickle, warnings; warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from scipy.stats import spearmanr
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.linewidth':.8,
 'axes.edgecolor':'#52514e','xtick.color':'#52514e','ytick.color':'#52514e','axes.labelcolor':'#0b0b0b',
 'text.color':'#0b0b0b','figure.dpi':300,'savefig.dpi':300,'savefig.bbox':'tight'})
BLUE,ORANGE,AQUA,VIOLET,RED,YELLOW,GREEN='#2a78d6','#eb6834','#1baf7a','#4a3aa7','#e34948','#eda100','#008300'
MUT,GRID,INK='#52514e','#e6e5e0','#0b0b0b'
S=pickle.load(open('split_10.pkl','rb'))
Xall=pd.concat([S['Xtr_i'],S['Xte_i']])
disp=[c.replace('_',' ') for c in Xall.columns]
corr=spearmanr(Xall).correlation; corr=(corr+corr.T)/2; np.fill_diagonal(corr,1.0); corr=np.nan_to_num(corr)
dist=1-np.abs(corr); np.fill_diagonal(dist,0); dist=np.clip(dist,0,None)
link=hierarchy.ward(squareform(dist,checks=False)); THR=0.7
dend=hierarchy.dendrogram(link,no_plot=True); o=dend['leaves']
lab=[disp[i] for i in o]; Cm=corr[np.ix_(o,o)]
div=LinearSegmentedColormap.from_list('dv',['#1b5aa4','#5c9ae2','#b9d4f2','#f4f3ef','#f7bda1','#eb6834','#8a3410'])

fig,ax=plt.subplots(figsize=(11.8,10.8))
im=ax.imshow(Cm,cmap=div,vmin=-1,vmax=1)
ax.set_xticks(range(len(lab))); ax.set_xticklabels(lab,rotation=90,fontsize=7.0)
ax.set_yticks(range(len(lab))); ax.set_yticklabels(lab,fontsize=7.0)
ax.tick_params(length=0)
for s in ax.spines.values(): s.set_visible(False)
ax.set_xticks(np.arange(-.5,len(lab),1),minor=True); ax.set_yticks(np.arange(-.5,len(lab),1),minor=True)
ax.grid(which='minor',color='white',lw=.45); ax.tick_params(which='minor',length=0)

def block(names,col,note,dx,dy):
    idx=[lab.index(n) for n in names if n in lab]
    if not idx: return
    i0,i1=min(idx),max(idx)
    ax.add_patch(Rectangle((i0-.5,i0-.5),i1-i0+1,i1-i0+1,fill=False,edgecolor=col,lw=2.6,zorder=6))
    ax.annotate(note,xy=((i0+i1)/2,i0-.5),xytext=((i0+i1)/2+dx,i0+dy),fontsize=9,color=col,fontweight='bold',
        ha='left',va='center',arrowprops=dict(arrowstyle='-',color=col,lw=1.3),zorder=7)

block(['Health information received','TB screened','TB test requested','Medication side effects','Comorbid illness','On TPT','Chills fever'],
      RED,'Tuberculosis-service and\nphysical-health block\nmax |ρ| = 0·90',9,4)
block(['Occupation Self Employed','Occupation Unemployed'],VIOLET,'Occupation indicators\nρ = −0·92',-19,-3)
cb=fig.colorbar(im,ax=ax,orientation='horizontal',fraction=.028,pad=.245,aspect=48)
cb.outline.set_visible(False); cb.ax.tick_params(labelsize=8.4); cb.set_label('Spearman rank correlation, ρ',fontsize=9.6)
ax.set_title('A   Spearman correlation matrix of the 52 candidate features, ordered by hierarchical clustering',
             loc='left',fontsize=12,fontweight='bold',pad=14)
plt.savefig('fig/Fig12A_correlation.png',bbox_inches='tight'); plt.close()

fig,ax=plt.subplots(figsize=(8.6,11.2))
hierarchy.set_link_color_palette([BLUE,ORANGE,AQUA,VIOLET,GREEN,YELLOW,'#e87ba4','#8a3410'])
hierarchy.dendrogram(link,labels=disp,orientation='right',ax=ax,color_threshold=THR,
                     above_threshold_color='#b0afa8',leaf_font_size=7.6)
ax.axvline(THR,color=RED,ls=(0,(5,3)),lw=1.7)
ax.text(THR+0.025,len(disp)*10-6,'cut at 0·7\n33 clusters retained',color=RED,fontsize=9.4,va='top',fontweight='bold')
ax.set_xlabel('Ward linkage distance   (1 − |Spearman ρ|)',fontsize=9.8)
ax.tick_params(axis='y',length=0,labelsize=7.6); ax.tick_params(axis='x',labelsize=8.6)
for s in ['top','right','left']: ax.spines[s].set_visible(False)
ax.grid(axis='x',color=GRID,lw=.7); ax.set_axisbelow(True)
ax.set_title('B   Ward-linkage dendrogram and the cut defining the retained feature set',
             loc='left',fontsize=12,fontweight='bold',pad=14)
plt.savefig('fig/Fig12B_dendrogram.png',bbox_inches='tight'); plt.close()
print('Fig12A and Fig12B rebuilt')
