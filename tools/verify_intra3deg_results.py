import numpy as np,pandas as pd,json,sys
from pathlib import Path
from scipy.stats import pearsonr,spearmanr
r=Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).resolve().parents[1]/'results/usgs_intra3deg_validation'
b=np.load(r/'source_bundle/frozen_well_cell_predictions.npz');p=pd.read_csv(r/'pair_metrics.csv');g=pd.read_csv(r/'well_grid_timeseries.csv');o=g.pivot(index='month',columns='fine_cell',values='normalized_groundwater_anomaly');lookup={int(k):i for i,k in enumerate(b['cell_ids'])}
errors=[]; cancellation=[]
for row in p.itertuples():
 i,j=lookup[row.fine_cell_i],lookup[row.fine_cell_j]
 assert b['coarse_ids'][i]==b['coarse_ids'][j]==row.coarse_cell_id
 common=np.isin(np.array([str(t)[:7] for t in b['times']]),row.common_months.split(';'))
 x=b[row.model+'_corrected'][:,i].astype(float)-b[row.model+'_corrected'][:,j].astype(float)
 y=(o[row.fine_cell_i]-o[row.fine_cell_j]).values
 expected=(b['valid'][:,i]>0)&(b['valid'][:,j]>0)&np.isfinite(y)
 for model in ['M0','M1','M2','M3','M4']:
  for variant in ['raw','corrected']:
   expected &= np.isfinite(b[model+'_'+variant][:,i])&np.isfinite(b[model+'_'+variant][:,j])
 assert np.array_equal(common,expected) and common.sum()==row.n_common_months
 errors.append(abs(pearsonr(x[common],y[common]).statistic-row.pearson_r))
 assert abs(spearmanr(x[common],y[common]).statistic-row.spearman_rho)<1e-12
 raw=b[row.model+'_raw'][:,i].astype(float)-b[row.model+'_raw'][:,j].astype(float)
 cancellation.append(np.max(np.abs(x[common]-raw[common])))
s=pd.read_csv(r/'well_eligibility.csv')
assert max(errors)<1e-12
report={'independent_scipy_PCC_max_error':max(errors),'pair_model_rows_checked':len(errors),'same_coarse_membership_all':True,'max_raw_corrected_pair_difference_mm':max(cancellation),'well_ge24':int((s.n_months>=24).sum()),'ge24_nonconstant':int(((s.n_months>=24)&(s.sd>0)).sum()),'ge24_nonconstant_on_land':int(((s.n_months>=24)&(s.sd>0)&s.land).sum()),'excluded_pair_reasons':pd.read_csv(r/'excluded_pairs.csv').reason.value_counts().to_dict()}
(r/'independent_numerical_check.json').write_text(json.dumps(report,indent=2));print(report)
