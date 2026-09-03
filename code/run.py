import pandas as pd, numpy as np, warnings, json, pickle
warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, roc_curve, accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, cohen_kappa_score, brier_score_loss)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import statsmodels.api as sm

D="/mnt/user-data/uploads/David Naboare/"
lab=pd.read_csv(D+'Depression in HIV Data.csv'); cod=pd.read_csv(D+'Data CSV.csv',keep_default_na=False,na_values=[''])
NINE=['Littleinterestorpleasureind_1','Feelingdowndepressedorhope_1','Troublefallingorstayingaslee_1',
'Feelingtiredorhavinglittlee_1','Poorappetiteorovereating_1','Feelingbadaboutyourselfor_1',
'Troubleconcentratingonthings_1','Movingorspeakingsoslowlytha_1','Thoughtsthatyouwouldbebette_1']
phq=cod[NINE].sum(axis=1)
LOG=[]
def P(*a):
    s=' '.join(str(x) for x in a); LOG.append(s); print(s)

# ---------- FEATURE CONSTRUCTION ----------
st=lambda c: lab[c].astype(str).str.strip()
X=pd.DataFrame()
X['Age']=lab['Whatisyourageinyears']
X['Sex']=(st('Whatisyoursex_')=='Female').astype(int)
for src,name in [('Whatisyourethnicity_','Ethnicity'),('Whatisyourreligion_','Religion'),('Maristat','Marital_status'),
                 ('Educational_level','Education'),('Occupation2','Occupation'),('Year_Diagnosis','Years_since_diagnosis'),
                 ('Duration_on_ART','ART_duration')]:
    d=pd.get_dummies(st(src),prefix=name,drop_first=True).astype(int); X=pd.concat([X,d],axis=1)
X['WHO_stage_known']=(~st('StageofHIVclassification_').str.contains('Don')).astype(int)
X['CD4_known']=(~st('WhatisyourCD4count_').str.contains("Don't know")).astype(int)
X['TB_screened']=(st('Haveyoueverbeenscreenedorq_')=='Yes').astype(int)
X['TB_test_requested']=(st('Wereyoueveraskedtogofora_')=='Yes').astype(int)
X['On_TPT']=(st('HaveyoubeeninitiatedonTPT_')=='Yes').astype(int)
X['Chills_fever']=cod['Chillsandfever_']
PSY={'Doyouhaveanycoexistingphys_':'Comorbid_illness','Haveyouexperiencedanymedicat_':'Medication_side_effects',
'DoesHIVAIDSimpactyourdaily_1':'HIV_impact_daily_life','Arethereanyspecificphysical_1':'Physical_limitation',
'Howdoyouratetheoverallimpa_1':'Overall_HIV_impact','Doyouengageinsubstanceabuse_':'Substance_use','Doyouusetobacco_':'Tobacco_use',
'Haveyoureceivedinformationan_':'Health_information_received','Haveyouattendedanycounseling_':'Counselling_attended',
'Accesstohealthcareservices_1':'Healthcare_access','Qualityofhealthcareservices_1':'Healthcare_quality',
'Availabilityofcounselingandp_1':'Counselling_availability','Tailoredmentalhealthservices_1':'Tailored_MH_services',
'Positiveproviderpatientrelati_1':'Provider_relationship','Howcomfortabledoyoufeeldisc_1':'Comfort_disclosing',
'Doculturalbeliefsinfluenceyo_1':'Cultural_beliefs_influence','BS_1':'Belief_system','Arethereculturalnormssurroun_1':'Cultural_norms',
'Doyouthinkthereisastigma_1':'Perceived_stigma','Arethereculturalbeliefsorat_1':'Cultural_attitudes',
'Howacceptingisyourculturalc_1':'Community_acceptance'}
for c,n in PSY.items(): X[n]=cod[c]
ANX={'Doyoutendtoworryorfeelanx_1':'Worry_anxiety','Haveyouexperiencedfeelingsof_1':'Feelings_of_distress',
'Howoftendoyoufeellowinene_1':'Low_energy','Doyoufinditchallengingtoex_1':'Difficulty_expressing',
'Haveyouhadrecurrentthoughts_1':'Recurrent_thoughts'}
for c,n in ANX.items(): X[n]=cod[c]

P('='*80); P('MLPLHIV — MACHINE LEARNING PREDICTION OF DEPRESSION AMONG PLHIV'); P('='*80)
P('\n[1] DATA PREPARATION')
P('  Raw records loaded: %d ; raw columns: %d'%(len(cod),cod.shape[1]))
dup=cod.duplicated().sum(); P('  Exact duplicate rows detected: %d (retained - see Methods)'%dup)
miss=X.isna().mean()
drop=list(miss[miss>0.5].index)
P('  Features with >50%% missing removed: %d %s'%(len(drop),drop if drop else ''))
X=X.drop(columns=drop)
P('  Final feature matrix: %d observations x %d features'%X.shape)
P('  Missing values remaining: %d cells (%.2f%%) - imputed within CV folds'%(X.isna().sum().sum(),100*X.isna().mean().mean()))
FEATNAMES=list(X.columns)
json.dump(FEATNAMES,open('features.json','w'),indent=1)

for CUT,TAG in [(10,'primary'),(5,'sensitivity')]:
    y=(phq>=CUT).astype(int).values
    P('\n'+'='*80); P('TARGET: PHQ-9 >= %d  (%s analysis)'%(CUT,TAG)); P('='*80)
    P('  Depressed (1): %d (%.1f%%) | Not depressed (0): %d (%.1f%%)'%(y.sum(),100*y.mean(),(y==0).sum(),100*(1-y.mean())))
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.2,stratify=y,random_state=42)
    P('  Training set: %d (%d events) | Test set: %d (%d events)'%(len(ytr),ytr.sum(),len(yte),yte.sum()))
    imp=SimpleImputer(strategy='median').fit(Xtr)
    Xtr_i=pd.DataFrame(imp.transform(Xtr),columns=X.columns,index=Xtr.index)
    Xte_i=pd.DataFrame(imp.transform(Xte),columns=X.columns,index=Xte.index)
    sc=StandardScaler().fit(Xtr_i)
    Xtr_s=pd.DataFrame(sc.transform(Xtr_i),columns=X.columns,index=Xtr.index)
    Xte_s=pd.DataFrame(sc.transform(Xte_i),columns=X.columns,index=Xte.index)
    sm_=SMOTE(random_state=42,k_neighbors=5)
    Xtr_r,ytr_r=sm_.fit_resample(Xtr_i,ytr)
    Xtr_rs,_=sm_.fit_resample(Xtr_s,ytr)
    P('  SMOTE applied to training set only: %d -> %d (class counts %s -> %s)'%(
        len(ytr),len(ytr_r),np.bincount(ytr).tolist(),np.bincount(ytr_r).tolist()))
    np.save(f'smote_{CUT}.npy',np.array([np.bincount(ytr).tolist(),np.bincount(ytr_r).tolist()]))
    pickle.dump(dict(Xtr_i=Xtr_i,Xte_i=Xte_i,Xtr_s=Xtr_s,Xte_s=Xte_s,Xtr_r=Xtr_r,Xtr_rs=Xtr_rs,ytr_r=ytr_r,ytr=ytr,yte=yte),
                open(f'split_{CUT}.pkl','wb'))
open('log1.txt','w').write('\n'.join(LOG))
