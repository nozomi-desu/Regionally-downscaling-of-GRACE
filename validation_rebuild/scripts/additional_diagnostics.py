from pathlib import Path
import json
import numpy as np,pandas as pd,xarray as xr
from evaluation_core import power_spectrum,spearman
R=Path(__file__).resolve().parents[2];O=R/'validation_rebuild';T=O/'tables';C=O/'cache'
base=xr.open_dataset(R/'data_driven_alpha_beta/M5/inference/test.nc');times=base.time.values;mask=np.load(C/'common_fine_mask.npy');lat=base.lat.values;lon=base.lon.values
models=['JPL','WGHM','M0','M1','M2','M3','M4'];data={'JPL':base.target_jplm_twsa.values,'WGHM':base.input_wghm_twsa.values}
for i in range(5):data[f'M{i}']=xr.open_dataset(R/f'data_driven_alpha_beta/M{5 if i==4 else i}/inference/test.nc').predicted_twsa_corrected.values
regions=pd.read_csv(T/'all_5deg_regions.csv');frozen=[('Amazon',-75,-50,-15,5),('Sahara',0,30,15,30),('Indo_Gangetic',70,90,20,30),('Alaska',-150,-130,60,70)]
ss=[];ps=[];hf=[]
for name,west,east,south,north in frozen+[(str(r.region_id),r.lon_min,r.lon_max,r.lat_min,r.lat_max) for _,r in regions.iterrows()]:
 iy=np.flatnonzero((lat>=south)&(lat<north));ix=np.flatnonzero((lon>=west)&(lon<east));mm=mask[:,iy][:,:,ix]
 energies={}
 for model,a in data.items():
  z=a[:,iy][:,:,ix];area=np.cos(np.deg2rad(lat[iy]))[None,:,None]*mm;series=(np.where(mm,z,0)*area).sum((1,2))/np.maximum(area.sum((1,2)),1e-12);series[area.sum((1,2))==0]=np.nan
  for t,v in zip(times,series):ss.append(dict(region=name,model=model,time=str(t)[:10],mean_mm=v,standardized=(v-np.nanmean(series))/np.nanstd(series)))
  energy=[]
  for t,field,g in zip(times,z,mm):
   freq,power=power_spectrum(field,g);energy.append(float(np.nansum(power[freq>=1/6])) if np.isfinite(power).any() else np.nan)
   if name in [v[0] for v in frozen] or name in pd.read_csv(O/'regional_cases/selected_cases.csv').region_id.astype(str).values:
    ps.extend(dict(region=name,model=model,time=str(t)[:10],wavenumber=f,power=p) for f,p in zip(freq,power))
  energies[model]=np.asarray(energy)
 if name not in [v[0] for v in frozen]:
  row=regions[regions.region_id.astype(str)==name].iloc[0]
  for i,t in enumerate(times):hf.append(dict(region_id=name,time=str(t)[:10],median_beta=row.median_beta,P_M4=energies['M4'][i],P_JPL=energies['JPL'][i],P_WGHM=energies['WGHM'][i]))
pd.DataFrame(ss).to_csv(T/'predefined_and_fixed_region_timeseries.csv',index=False);pd.DataFrame(ps).to_csv(T/'regional_spectral_monthly.csv',index=False)
h=pd.DataFrame(hf);den=h.P_WGHM-h.P_JPL;floor=1e-6*h.loc[h.P_WGHM>0,'P_WGHM'].median();h['HFI']=np.where(abs(den)>floor,(h.P_M4-h.P_JPL)/den,np.nan);h.to_csv(T/'regional_spectral_HFI.csv',index=False)
s=h.groupby('region_id',as_index=False).agg(HFI=('HFI','median'),median_beta=('median_beta','median'));s.to_csv(T/'regional_spectral_HFI_summary.csv',index=False)
(O/'audit/spectral_HFI_definition.json').write_text(json.dumps(dict(definition='sum of radial-mean spectral powers for normalized wavenumber >=1/6 cycles per cell; fixed 5-degree boxes; monthly HFI then region temporal median; not continuous physical wavelength',denominator_floor=float(floor),rho_beta=spearman(s.HFI.values,s.median_beta.values),N_valid=int(s.HFI.notna().sum())),indent=2))
(O/'audit/method_code_discrepancies.md').open('a').write('\n7. New evaluator initial beta digitize used strict lower comparison at q33=0. Corrected to <= threshold (right=True), without weight/data changes; reran spectra, strata and dependent outputs. Full global SMAP and regional counts unchanged.\n8. Fixed 13-cell Gaussian highpass is explicit, mask-normalized; masked FFT has no physical km calibration. Spectral HFI now computed separately for fixed 5-degree boxes; spatial correlations remain exploratory.\n')
print('ADDITIONAL DIAGNOSTICS COMPLETE')
