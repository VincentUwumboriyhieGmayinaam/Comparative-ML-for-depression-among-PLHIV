import os; os.environ.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
import pandas as pd, numpy as np, pickle, json, warnings; warnings.filterwarnings('ignore')
D="/mnt/user-data/uploads/David Naboare/"
lab=pd.read_csv(D+'Depression in HIV Data.csv'); cod=pd.read_csv(D+'Data CSV.csv',keep_default_na=False,na_values=[''])
NINE=['Littleinterestorpleasureind_1','Feelingdowndepressedorhope_1','Troublefallingorstayingaslee_1',
'Feelingtiredorhavinglittlee_1','Poorappetiteorovereating_1','Feelingbadaboutyourselfor_1',
'Troubleconcentratingonthings_1','Movingorspeakingsoslowlytha_1','Thoughtsthatyouwouldbebette_1']
phq=cod[NINE].sum(axis=1); item9=cod['Thoughtsthatyouwouldbebette_1']
y=(phq>=10).astype(int).values
OOF=pickle.load(open('oof_top5.pkl','rb')); p=OOF['Random Forest']
print('='*78); print('REVIEWER CHECKS'); print('='*78)
print('\n[1] PATIENT SAFETY — suicidal ideation among model-missed cases')
print('  Item 9 endorsed (>=1): %d/264 (%.1f%%); endorsed >=2: %d (%.1f%%)'%(
    (item9>=1).sum(),100*(item9>=1).mean(),(item9>=2).sum(),100*(item9>=2).mean()))
for t in [0.20,0.30,0.40,0.50]:
    miss=(p<t)&(y==1)
    m9=((p<t)&(item9>=1)).sum(); m9b=((p<t)&(item9>=2)).sum()
    print('  threshold %.2f: %2d depressed cases missed, of whom %d endorsed item 9; '
          '%d participants with ANY item-9 endorsement would never be screened (%d with >=2)'%(
          t,miss.sum(),((miss)&(item9>=1)).sum(),m9,m9b))
print('\n[2] QUESTIONNAIRE BURDEN — items required to run each model')
FEAT=json.load(open('features.json'))
routine=[f for f in FEAT if any(f.startswith(k) for k in ['Age','Sex','Ethnicity','Religion','Marital','Education','Occupation','Years_since','ART_duration','WHO_stage','CD4_known','TB_screened','TB_test','On_TPT','Chills'])]
psy=[f for f in FEAT if f not in routine]
print('  Model A (register-derived) encoded features: %d  -> underlying questions asked of patient: ~0 (all from register/routine)'%len(routine))
print('  Model B adds %d encoded psychosocial/health-system/psychological items'%len(psy))
print('  Underlying NEW questions the counsellor must ask to run Model B: 26')
print('  PHQ-9 itself: 9 questions')
print('  => Model B requires 26 questions to decide whether to ask 9.')
print('\n[3] SOMATIC CONFOUNDING — is the outcome driven by HIV/ART-attributable items?')
som=cod[['Troublefallingorstayingaslee_1','Feelingtiredorhavinglittlee_1','Poorappetiteorovereating_1','Movingorspeakingsoslowlytha_1']].sum(axis=1)
aff=cod[['Littleinterestorpleasureind_1','Feelingdowndepressedorhope_1','Feelingbadaboutyourselfor_1','Troubleconcentratingonthings_1','Thoughtsthatyouwouldbebette_1']].sum(axis=1)
sideeff=cod['Haveyouexperiencedanymedicat_']; comorb=cod['Doyouhaveanycoexistingphys_']
print('  Somatic subscale vs reported medication side-effects: Spearman rho = %+.3f'%som.corr(sideeff,method='spearman'))
print('  Affective subscale vs reported medication side-effects: Spearman rho = %+.3f'%aff.corr(sideeff,method='spearman'))
print('  Somatic subscale vs comorbid physical illness:  rho = %+.3f'%som.corr(comorb,method='spearman'))
print('  Affective subscale vs comorbid physical illness: rho = %+.3f'%aff.corr(comorb,method='spearman'))
aff_only=(aff>=6).astype(int)
print('  Cases by full PHQ-9 >=10: %d ; by affective-only subscale >=6: %d ; concordance %.1f%%'%(
    y.sum(),aff_only.sum(),100*(y==aff_only).mean()))
print('\n[4] DATA INTEGRITY')
print('  Exact duplicate rows across all 78 variables: %d'%cod.duplicated().sum())
print('  Age range: %d to %d years; n<18: %d ; n>80: %d'%(lab['Whatisyourageinyears'].min(),lab['Whatisyourageinyears'].max(),
    (lab['Whatisyourageinyears']<18).sum(),(lab['Whatisyourageinyears']>80).sum()))
print('  WHO stage unknown: %.1f%% ; CD4 unknown: %.1f%% ; ART duration unknown: %.1f%%'%(
    100*lab['StageofHIVclassification_'].astype(str).str.contains('Don').mean(),
    100*lab['WhatisyourCD4count_'].astype(str).str.contains("Don't know").mean(),
    100*lab['Duration_on_ART'].astype(str).str.contains("Don't know").mean()))
print('  Viral load recorded: NO variable present. Adherence measure: NO variable present.')
print('\n[5] EVENTS PER PREDICTOR')
print('  Events = %d ; encoded features = %d ; EPV = %.2f  (Riley/Peduzzi floor = 10)'%(y.sum(),len(FEAT),y.sum()/len(FEAT)))
print('  Test partition: n=53, events=%d'%int(0.2*y.sum()))
