import os; os.environ.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
import pandas as pd, numpy as np, pickle, json, warnings
warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.calibration import calibration_curve
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.linewidth':.8,
 'axes.edgecolor':'#52514e','xtick.color':'#52514e','ytick.color':'#52514e',
 'axes.labelcolor':'#0b0b0b','text.color':'#0b0b0b','figure.dpi':300,'savefig.dpi':300,
 'savefig.bbox':'tight','axes.spines.top':False,'axes.spines.right':False})
BLUE,ORANGE,AQUA,YELLOW,MAGENTA,GREEN,VIOLET,RED='#2a78d6','#eb6834','#1baf7a','#eda100','#e87ba4','#008300','#4a3aa7','#e34948'
CATS=[BLUE,ORANGE,AQUA,YELLOW,MAGENTA,GREEN,VIOLET,RED,'#52514e','#8c8b85']
GRID='#e6e5e0'; INK='#0b0b0b'; MUT='#52514e'
CUT=10
rows=[json.loads(l) for l in open(f'rows_{CUT}.jsonl')]
df=pd.DataFrame(rows).sort_values('CV_AUC',ascending=False).reset_index(drop=True)
probs=pickle.load(open(f'prob_{CUT}.pkl','rb'))
S=pickle.load(open(f'split_{CUT}.pkl','rb')); yte=S['yte']; ytr=S['ytr']
FEAT=json.load(open('features.json'))
O='fig/'

# ---------- FIG 1: analytic workflow ----------
fig,ax=plt.subplots(figsize=(9,5.6)); ax.axis('off'); ax.set_xlim(0,10); ax.set_ylim(0,10)
def box(x,y,w,h,t,fc,ec,fs=8.2,tc=INK):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.06,rounding_size=0.12",
        fc=fc,ec=ec,lw=1.1)); ax.text(x+w/2,y+h/2,t,ha='center',va='center',fontsize=fs,color=tc,linespacing=1.45)
def arr(x1,y1,x2,y2):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=11,lw=1.0,color=MUT,shrinkA=2,shrinkB=2))
box(0.2,8.5,2.6,1.2,"Source data\n264 PLHIV × 78 variables\nKadjebi District, Feb 2024","#eef4fc",BLUE)
box(3.4,8.5,3.0,1.2,"Outcome definition\nPHQ-9 (9 items, 0–27)\n≥10 = depressed","#eef4fc",BLUE)
box(7.0,8.5,2.8,1.2,"Feature engineering\none-hot + ordinal\n52 predictors","#eef4fc",BLUE)
arr(2.8,9.1,3.4,9.1); arr(6.4,9.1,7.0,9.1)
box(0.2,6.6,4.6,1.2,"Pre-processing\nduplicate check · >50% missing rule\nmedian imputation · z-standardisation","#fdf3ec",ORANGE)
box(5.2,6.6,4.6,1.2,"Stratified split  80 : 20\ntrain n=211 (80 events)\ntest n=53 (20 events)","#fdf3ec",ORANGE)
arr(4.8,7.2,5.2,7.2); arr(2.5,8.5,2.5,7.8)
box(0.2,4.7,4.6,1.2,"SMOTE — training fold only\n131:80 → 131:131\ntest set left untouched","#e9f7f1",AQUA)
box(5.2,4.7,4.6,1.2,"Hyperparameter tuning\nrandomised search, 5-fold CV\nscoring = ROC-AUC","#e9f7f1",AQUA)
arr(7.5,6.6,7.5,5.9); arr(4.8,5.3,5.2,5.3)
box(0.2,2.6,9.6,1.4,"Ten supervised classifiers\nLogistic Regression · Decision Tree · Random Forest · Gradient Boosting · XGBoost\nLightGBM · CatBoost · Support Vector Machine · k-Nearest Neighbours · Naïve Bayes","#f2f0fa",VIOLET)
arr(2.5,4.7,2.5,4.0)
box(0.2,0.5,4.6,1.6,"Evaluation\nheld-out test set + 5×4 repeated CV\naccuracy · precision · sensitivity\nspecificity · F1 · AUC · κ · Brier\ncalibration slope & intercept","#fdeeee",RED,7.8)
box(5.2,0.5,4.6,1.6,"Interpretation\npermutation importance\nSHAP (TreeExplainer)\nbar + beeswarm plots\ndecision-curve analysis","#fdeeee",RED,7.8)
arr(2.5,2.6,2.5,2.1); arr(7.5,2.6,7.5,2.1)
plt.savefig(O+'Fig1_workflow.png'); plt.close()

# ---------- FIG 2: PHQ-9 distribution ----------
D="/mnt/user-data/uploads/David Naboare/"
lab=pd.read_csv(D+'Depression in HIV Data.csv'); cod=pd.read_csv(D+'Data CSV.csv',keep_default_na=False,na_values=[''])
NINE=['Littleinterestorpleasureind_1','Feelingdowndepressedorhope_1','Troublefallingorstayingaslee_1',
'Feelingtiredorhavinglittlee_1','Poorappetiteorovereating_1','Feelingbadaboutyourselfor_1',
'Troubleconcentratingonthings_1','Movingorspeakingsoslowlytha_1','Thoughtsthatyouwouldbebette_1']
phq=cod[NINE].sum(axis=1); sex=lab['Whatisyoursex_'].astype(str).str.strip()
fig,axs=plt.subplots(1,2,figsize=(9.6,3.5),gridspec_kw={'width_ratios':[1.35,1]})
a=axs[0]
bins=np.arange(-0.5,23.5,1)
a.hist(phq[sex=='Female'],bins=bins,color=BLUE,alpha=.85,label='Women (n=200)',edgecolor='white',linewidth=.4)
a.hist(phq[sex=='Male'],bins=bins,color=ORANGE,alpha=.85,label='Men (n=64)',edgecolor='white',linewidth=.4,bottom=np.histogram(phq[sex=='Female'],bins=bins)[0])
for c,lb in [(5,'≥5  any symptoms'),(10,'≥10  moderate+')]:
    a.axvline(c-0.5,color=MUT,ls='--',lw=1)
    a.text(c-0.3,a.get_ylim()[1]*0.94,lb,fontsize=7.2,color=MUT,rotation=0,ha='left',va='top')
a.set_xlabel('PHQ-9 total score (9 items, range 0–27)'); a.set_ylabel('Participants')
a.legend(frameon=False,fontsize=8); a.grid(axis='y',color=GRID,lw=.7); a.set_axisbelow(True)
a.set_title('A  Score distribution',loc='left',fontsize=9.5,fontweight='bold',pad=8)
b=axs[1]
sev=pd.cut(phq,[-1,4,9,14,19,27],labels=['None/\nminimal','Mild','Moderate','Moderately\nsevere','Severe'])
ct=pd.crosstab(sev,sex)[['Female','Male']]
idx=np.arange(len(ct)); w=.62
b.bar(idx,ct['Female'],w,color=BLUE,label='Women',edgecolor='white',linewidth=1)
b.bar(idx,ct['Male'],w,bottom=ct['Female'],color=ORANGE,label='Men',edgecolor='white',linewidth=1)
for i,(f,m) in enumerate(zip(ct['Female'],ct['Male'])):
    b.text(i,f+m+1.6,f'{f+m}\n({100*(f+m)/264:.1f}%)',ha='center',fontsize=7.4,color=INK,linespacing=1.2)
b.set_xticks(idx); b.set_xticklabels(ct.index,fontsize=7.6); b.set_ylabel('Participants')
b.set_ylim(0,ct.sum(axis=1).max()*1.30); b.grid(axis='y',color=GRID,lw=.7); b.set_axisbelow(True)
b.legend(frameon=False,fontsize=8); b.set_title('B  Severity category',loc='left',fontsize=9.5,fontweight='bold',pad=8)
plt.tight_layout(); plt.savefig(O+'Fig2_phq_distribution.png'); plt.close()

# ---------- FIG 3: SMOTE ----------
pre,post=np.load(f'smote_{CUT}.npy')
fig,ax=plt.subplots(figsize=(5.4,3.1))
x=np.arange(2); w=.36
ax.bar(x-w/2,pre,w,color=BLUE,label='Before SMOTE',edgecolor='white',linewidth=1)
ax.bar(x+w/2,post,w,color=AQUA,label='After SMOTE',edgecolor='white',linewidth=1)
for i,(p,q) in enumerate(zip(pre,post)):
    ax.text(i-w/2,p+2,str(p),ha='center',fontsize=8,color=INK); ax.text(i+w/2,q+2,str(q),ha='center',fontsize=8,color=INK)
ax.set_xticks(x); ax.set_xticklabels(['Not depressed\n(PHQ-9 < 10)','Depressed\n(PHQ-9 ≥ 10)'],fontsize=8.2)
ax.set_ylabel('Training-set observations'); ax.set_ylim(0,max(post)*1.22)
ax.legend(frameon=False,fontsize=8); ax.grid(axis='y',color=GRID,lw=.7); ax.set_axisbelow(True)
plt.tight_layout(); plt.savefig(O+'Fig3_smote.png'); plt.close()
print('figs 1-3 done')
