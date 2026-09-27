from pathlib import Path
import json,sys,warnings
import numpy as np,pandas as pd,xarray as xr
from evaluation_core import *
R=Path(__file__).resolve().parents[2];O=R/'validation_rebuild';T=O/'tables';C=O/'cache'
assert json.loads((O/'audit/audit_gate.json').read_text())['passed']
warnings.filterwarnings('ignore',category=RuntimeWarning)
paths={f'M{i}':R/f'data_driven_alpha_beta/M{5 if i==4 else i}/inference/test.nc' for i in range(5)}
paths.update({m:R/f'validation_extension_20260908/baselines/{m}/test.nc' for m in ['MLR_DS','RF_DS']})
base=xr.open_zarr(R/'data_processed/training_rescue_v2/grace_downscaling_training_dataset_mm_area.zarr',consolidated=True)
test=base.isel(time=np.where(base.split_index.values==2)[0]);lat=base.lat.values;lon=base.lon.values;times=test.time.values
land=base.land_mask.values>0;mask=test.valid_mask.values.astype(bool)
data={};raw={}
for m,p in paths.items():
 d=xr.open_dataset(p);assert np.array_equal(d.time.values,times)
 data[m]=d.predicted_twsa_corrected.values.astype(float);raw[m]=d.predicted_twsa_raw.values.astype(float)
 mask &= d.valid_mask.values.astype(bool)&np.isfinite(data[m])&np.isfinite(raw[m]);d.close()
data['JPL']=test.target_jplm_twsa.values.astype(float);data['WGHM']=test.input_wghm_twsa.values.astype(float)
mask &= np.isfinite(data['JPL'])&np.isfinite(data['WGHM'])
np.save(C/'common_fine_mask.npy',mask)
coarse={m:aggregate(a,mask,lat) for m,a in data.items()};reference=coarse['JPL']
common=np.logical_and.reduce([np.isfinite(a) for a in coarse.values()])
coarse={m:np.where(common,a,np.nan) for m,a in coarse.items()}
maps={};summary=[];monthrows=[];harms={m:harmonics(a,times) for m,a in coarse.items()}
for m in data:
 print('COARSE',m,flush=True)
 met=temporal(coarse[m],reference);monthly_metrics=monthly(coarse[m],reference)
 row={'model':m,'variant':'corrected','n_months':36,'coarse_common_cells_all_months':int(common.all(axis=0).sum())}
 for k,v in monthly_metrics.items():row['coarse_'+k]=float(np.mean(v))
 for k,v in met.items():row['median_temporal_'+k]=float(np.nanmedian(v));maps[m+'_'+k]=(('coarse_lat','coarse_lon'),v)
 for k,v in harms[m].items():
  row[k+'_MAE']=float(np.nanmean(abs(v-harms['JPL'][k])));maps[m+'_'+k]=(('coarse_lat','coarse_lon'),v)
 for j,t in enumerate(times):monthrows.append(dict(model=m,variant='corrected',time=str(t)[:10],**{k:float(v[j]) for k,v in monthly_metrics.items()}))
 summary.append(row)
 if m in raw:
  r=aggregate(raw[m],mask,lat);r=np.where(common,r,np.nan);mm=monthly(r,reference)
  rr=dict(model=m,variant='raw',n_months=36,**{'coarse_'+k:float(np.mean(v)) for k,v in mm.items()});summary.append(rr)
  for j,t in enumerate(times):monthrows.append(dict(model=m,variant='raw',time=str(t)[:10],**{k:float(v[j]) for k,v in mm.items()}))
pd.DataFrame(summary).to_csv(T/'coarse_full_metrics.csv',index=False);pd.DataFrame(monthrows).to_csv(T/'coarse_monthly_metrics.csv',index=False)
xr.Dataset(maps,coords={'coarse_lat':base.coarse_lat,'coarse_lon':base.coarse_lon}).to_netcdf(C/'coarse_diagnostics.nc')
print('COARSE COMPLETE',flush=True)
alpha=xr.open_dataset(R/'data_driven_alpha_beta/weights/alpha_data_driven.nc').alpha.values
beta=xr.open_dataset(R/'data_driven_alpha_beta/weights/beta_data_driven.nc').beta.values
aq=np.nanquantile(alpha[land],[1/3,2/3]);bq=np.nanquantile(beta[land],[1/3,2/3])
ab=np.digitize(alpha,aq,right=True);bb=np.digitize(beta,bq,right=True)
(O/'audit/strata_and_spectrum_definition.json').write_text(json.dumps(dict(alpha_tertile_edges=aq.tolist(),beta_tertile_edges=bq.tolist(),ties='np.digitize right=True; values equal to threshold enter lower bin; beta=0 included in low stratum',population='all finite land cells in frozen weights; not selected external coverage',highpass='mask normalized Gaussian sigma=3 cells kernel=13, nearest edge',spectrum='demeaned field, zeros outside common mask, radial mean FFT; cycles per 0.5-degree cell; no km conversion; masked-window diagnostic',high_frequency='cycles/cell >= 1/6; HFI exploratory',HFI_denominator_floor='1e-6 times global median positive WGHM high-pass mean-square; absolute denominator cutoff'),indent=2))
hp={};spectra=[];hp_rows=[];snap={}
for m,a in data.items():
 print('SPATIAL',m,flush=True)
 h=highpass(a,mask);rms=np.sqrt(np.nanmean(h*h,axis=0));hp[m]=rms
 snap[m]=(('lat','lon'),np.where(mask[0],a[0],np.nan));snap[m+'_highpass']=(('lat','lon'),h[0]);snap[m+'_hp_rms']=(('lat','lon'),rms)
 hp_rows.append(dict(model=m,median_highpass_RMS=float(np.nanmedian(rms)),mean_highpass_RMS=float(np.nanmean(rms)),rho_hp_beta=spearman(rms[land],beta[land])))
 for label,g in [('global',land),('low_beta',land&(bb==0)),('mid_beta',land&(bb==1)),('high_beta',land&(bb==2))]:
  for j,t in enumerate(times):
   f,p=power_spectrum(a[j],mask[j]&g)
   spectra.extend(dict(model=m,stratum=label,time=str(t)[:10],wavenumber=float(x),power=float(y)) for x,y in zip(f,p))
np.savez_compressed(C/'highpass_rms.npz',**hp)
xr.Dataset(snap,coords={'lat':lat,'lon':lon}).to_netcdf(C/'spatial_snapshots.nc')
pd.DataFrame(hp_rows).to_csv(T/'highpass_metrics.csv',index=False);pd.DataFrame(spectra).to_csv(T/'spectral_monthly.csv',index=False)
den=hp['WGHM']**2-hp['JPL']**2;floor=1e-6*np.nanmedian(hp['WGHM'][hp['WGHM']>0]**2)
hfi=np.divide(hp['M4']**2-hp['JPL']**2,den,out=np.full_like(den,np.nan),where=abs(den)>floor)
np.save(C/'HFI_highpass_energy.npy',hfi)
pd.DataFrame([dict(definition='highpass energy inheritance (not spectral-band HFI)',rho_beta=spearman(hfi[land],beta[land]),N=int(np.isfinite(hfi[land]).sum()),denominator_floor=floor)]).to_csv(T/'HFI_summary.csv',index=False)
print('SPATIAL COMPLETE',flush=True)
# External SMAP: actual observation cube only, never saved evaluation maps.
s=xr.open_dataset(R/'data_processed/full_model_smap_test_smap_monthly_overlap.nc')
obs=s.smap_surface_sm.sel(time=times,lat=lat,lon=lon).values.astype(float)
external=(base.human_activity_index.values<=.5)&(base.glacier_fraction.values<=.05)&land
g=np.isfinite(obs)&mask&external
sm={m:temporal(a,obs,g)['PCC'] for m,a in data.items()}
support=np.logical_and.reduce([np.isfinite(sm[m]) for m in paths])
iy,ix=np.where(support);frame=pd.DataFrame(dict(lat=lat[iy],lon=lon[ix],n_months=g[:,iy,ix].sum(axis=0),alpha=alpha[iy,ix],beta=beta[iy,ix],alpha_bin=ab[iy,ix],beta_bin=bb[iy,ix],human=base.human_activity_index.values[iy,ix],glacier=base.glacier_fraction.values[iy,ix],aridity=base.aridity_mask.values[iy,ix],HFI=hfi[iy,ix]))
for m in sm:frame[m]=sm[m][iy,ix]
frame.to_csv(T/'smap_common_grid_correlations.csv',index=False)
pair=[]
for m in paths:
 if m=='M4':continue
 r,boot=paired(frame.M4.values,frame[m].values,frame.lat.values,frame.lon.values);r['comparison']='M4-'+m;pair.append(r);np.save(C/f'smap_bootstrap_M4_{m}.npy',boot)
pd.DataFrame(pair).to_csv(T/'smap_pairwise_global.csv',index=False)
pd.DataFrame([dict(model=m,N_grid=int(frame[m].notna().sum()),median_PCC=float(frame[m].median())) for m in sm]).to_csv(T/'smap_summary.csv',index=False)
joint=[]
for a in range(3):
 for b in range(3):
  f=frame[(frame.alpha_bin==a)&(frame.beta_bin==b)]
  for m in ['M0','M1','M2','M3']:
   r,boot=paired(f.M4.values,f[m].values,f.lat.values,f.lon.values);r.update(alpha_bin=a,beta_bin=b,comparison='M4-'+m);joint.append(r);np.save(C/f'joint_bootstrap_{a}_{b}_{m}.npy',boot)
pd.DataFrame(joint).to_csv(T/'alpha_beta_joint_gains.csv',index=False)
frame['rx']=np.floor((frame.lon+180)/5).astype(int);frame['ry']=np.floor((frame.lat+90)/5).astype(int)
regions=[]
for (rx,ry),f in frame.groupby(['rx','ry']):
 if len(f)<5:continue
 row=dict(region_id=f'{ry:02d}_{rx:02d}',lon_min=rx*5-180,lon_max=rx*5-175,lat_min=ry*5-90,lat_max=ry*5-85,n_valid_grids=len(f))
 for m in ['M0','M1','M2','M3']:row['delta_M4_'+m]=float((f.M4-f[m]).median())
 row['S_min']=min(row['delta_M4_'+m] for m in ['M0','M1','M2','M3'])
 for v in ['alpha','beta','human','glacier','aridity']:row['median_'+v]=float(f[v].median())
 regions.append(row)
reg=pd.DataFrame(regions).sort_values('S_min',ascending=False);reg['rank']=np.arange(1,len(reg)+1);reg.to_csv(T/'all_5deg_regions.csv',index=False)
selected=pd.concat([reg[reg.S_min>0].head(2),reg[reg.S_min<0].tail(1)]);selected.to_csv(O/'regional_cases/selected_cases.csv',index=False)
(O/'regional_cases/case_selection.md').write_text('Exploratory post-hoc selection AFTER all fixed regions: two highest S_min positive regions, plus most negative S_min region. Uniform 5-degree grid anchored at (-180,-90), >=5 common grids. No boundary changes. True ranks in selected_cases.csv. No substitute if two positive cases do not exist.\n')
case_series=[]
for _,r in selected.iterrows():
 q=(frame.lon>=r.lon_min)&(frame.lon<r.lon_max)&(frame.lat>=r.lat_min)&(frame.lat<r.lat_max);yy=iy[q];xx=ix[q]
 for m,a in {**data,'SMAP':obs}.items():
  v=np.where(g[:,yy,xx],a[:,yy,xx],np.nan);series=np.nanmean(v,axis=1);standard=(series-np.nanmean(series))/np.nanstd(series)
  for t,val,st in zip(times,series,standard):case_series.append(dict(region_id=r.region_id,model=m,time=str(t)[:10],mean_value=val,standardized=st))
pd.DataFrame(case_series).to_csv(T/'regional_case_timeseries.csv',index=False)
print('SMAP COMPLETE',len(frame),'grids',len(reg),'regions',flush=True)
# Groundwater: frozen 2026-09-08 observation and non-groundwater separation protocol.
sys.path.insert(0,str(R))
from scripts.eval.evaluate_independent_groundwater_gwsa import open_wghm_groundwater
w=pd.read_csv(R/'validation_extension_20260908/groundwater/well_monthly.csv',parse_dates=['time']);sites=w.groupby('site_id').agg(lat=('lat','median'),lon=('lon','median'))
slat=xr.DataArray(sites.lat.values,dims='site');slon=xr.DataArray(sites.lon.values,dims='site')
gy=np.abs(lat[:,None]-sites.lat.values).argmin(axis=0);gx=np.abs(lon[:,None]-sites.lon.values).argmin(axis=0)
well=w.pivot(index='time',columns='site_id',values='water_level_proxy_m').reindex(index=pd.DatetimeIndex(times),columns=sites.index).values
gw=open_wghm_groundwater(R/'data_raw/wghm/watergap22e_gswp3-era5_groundwstor_histsoc_monthly_1901_2022.nc','2004-01-01','2009-12-31').sel(time=times,lat=slat,lon=slon,method='nearest').values
non=data['WGHM'][:,gy,gx]-gw;wd={m:a[:,gy,gx]-non for m,a in data.items()};wg=np.isfinite(well)&np.isfinite(non)&mask[:,gy,gx]
for m in paths:wg &= np.isfinite(wd[m])
wm={m:temporal(a,well,wg)['PCC'] for m,a in wd.items()};ws=np.logical_and.reduce([np.isfinite(wm[m]) for m in paths]);wr=[]
for j in np.flatnonzero(ws):
 rec=dict(site_id=sites.index[j],lat=lat[gy[j]],lon=lon[gx[j]],n_months=int(wg[:,j].sum()))
 rec.update({m:wm[m][j] for m in wm});wr.append(rec)
wf=pd.DataFrame(wr);wf.to_csv(T/'groundwater_well_correlations.csv',index=False)
cells=wf.groupby(['lat','lon'],as_index=False)[list(wm)].median();cells.to_csv(T/'groundwater_common_grid_correlations.csv',index=False)
gp=[]
for m in paths:
 if m=='M4':continue
 r,boot=paired(cells.M4.values,cells[m].values,cells.lat.values,cells.lon.values);r['comparison']='M4-'+m;gp.append(r);np.save(C/f'groundwater_bootstrap_M4_{m}.npy',boot)
pd.DataFrame(gp).to_csv(T/'groundwater_pairwise_global.csv',index=False)
pd.DataFrame([dict(model=m,N_grid=len(cells),N_well=len(wf),median_PCC=float(cells[m].median())) for m in wm]).to_csv(T/'groundwater_summary.csv',index=False)
print('GROUNDWATER COMPLETE',len(wf),'wells',len(cells),'grids',flush=True)
full=pd.DataFrame(summary);full=full[full.variant=='corrected'].merge(pd.DataFrame(hp_rows),on='model').merge(pd.read_csv(T/'smap_summary.csv')[['model','median_PCC']].rename(columns={'median_PCC':'SMAP_median_PCC'}),on='model').merge(pd.read_csv(T/'groundwater_summary.csv')[['model','median_PCC']].rename(columns={'median_PCC':'groundwater_median_PCC'}),on='model')
full.to_csv(T/'baseline_full_metrics.csv',index=False)
(O/'logs/metrics_complete.json').write_text(json.dumps(dict(status='complete',N_smap=len(frame),N_regions=len(reg),N_positive_regions=int((reg.S_min>0).sum()),N_wells=len(wf),N_groundwater_grids=len(cells)),indent=2))
