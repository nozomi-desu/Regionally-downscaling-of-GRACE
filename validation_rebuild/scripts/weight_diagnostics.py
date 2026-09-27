from pathlib import Path
import numpy as np,pandas as pd,xarray as xr,json
from evaluation_core import aggregate,temporal,harmonics,spearman
R=Path(__file__).resolve().parents[2];O=R/'validation_rebuild';T=O/'tables';C=O/'cache'
base=xr.open_zarr(R/'data_processed/training_rescue_v2/grace_downscaling_training_dataset_mm_area.zarr',consolidated=True)
land=base.land_mask.values>0;rows=[]
for v in ['alpha','beta']:
 old=xr.open_dataset(R/f'data_driven_alpha_beta/weights/{v}_data_driven.nc')[v].values
 new=xr.open_dataset(C/f'rebuilt_weights/{v}_data_driven.nc')[v].values
 diff=float(np.nanmax(abs(new-old)));rows.append(dict(weight=v,max_absolute_difference=diff,identical_nan_mask=bool(np.array_equal(np.isnan(old),np.isnan(new)))))
pd.DataFrame(rows).to_csv(O/'audit/weight_reproduction.csv',index=False)
if any(r['max_absolute_difference']>2e-6 or not r['identical_nan_mask'] for r in rows):raise RuntimeError('Frozen weights not reproduced; stop related analyses')
a=xr.open_dataset(R/'data_driven_alpha_beta/weights/alpha_data_driven.nc');b=xr.open_dataset(C/'rebuilt_weights/beta_data_driven.nc');out=[]
for label,index in [('train',0),('test',2)]:
 ds=base.isel(time=np.where(base.split_index.values==index)[0]);mask=ds.valid_mask.values.astype(bool)&land
 w=aggregate(ds.input_wghm_twsa.values,mask,base.lat.values);j=aggregate(ds.target_jplm_twsa.values,mask,base.lat.values)
 met=temporal(w,j,min_months=24);wh=harmonics(w,ds.time.values);jh=harmonics(j,ds.time.values)
 fields=dict(PCC=met['PCC'],NRMSE=met['RMSE']/np.nanstd(j,axis=0),abs_trend_difference=abs(wh['trend']-jh['trend']),abs_annual_difference=abs(wh['annual']-jh['annual']))
 frame=pd.DataFrame(dict(alpha=a.alpha_coarse.values.ravel(),**{k:v.ravel() for k,v in fields.items()}))
 frame.to_csv(T/f'alpha_diagnostics_{label}_cells.csv',index=False)
 for k,v in fields.items():out.append(dict(period=label,diagnostic=k,Spearman=spearman(a.alpha_coarse.values,v),N=int((np.isfinite(a.alpha_coarse.values)&np.isfinite(v)).sum())))
pd.DataFrame(out).to_csv(T/'alpha_diagnostic_summary.csv',index=False)
loo=[]
for v in ['era5_twsa','gldas_noah_twsa','clm5_twsa']:
 x=b['beta_without_'+v].values;y=b.beta.values;g=land&np.isfinite(x)&np.isfinite(y)
 loo.append(dict(removed_source=v,MAE=float(np.mean(abs(x[g]-y[g]))),Spearman=spearman(x[g],y[g]),N=int(g.sum())))
pd.DataFrame(loo).to_csv(T/'beta_leave_one_source_out.csv',index=False)
coverage=xr.open_dataset(C/'rebuilt_weights/beta_source_count.nc');lat,lon=np.meshgrid(base.lat.values,base.lon.values,indexing='ij')
pd.DataFrame(dict(lat=lat[land],lon=lon[land],alpha=a.alpha.values[land],beta=b.beta.values[land],aridity=base.aridity_mask.values[land],human=base.human_activity_index.values[land],glacier=base.glacier_fraction.values[land],**{v:coverage[v].values[land] for v in coverage.data_vars})).to_csv(T/'weight_land_source_data.csv',index=False)
print('WEIGHT DIAGNOSTICS COMPLETE',rows,flush=True)
