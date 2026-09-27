"""Matched-data adaptations of Pascal 2022 MLR/RF downscaling; no test tuning."""
from pathlib import Path
import sys,json,hashlib,time
import numpy as np
import xarray as xr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import sklearn,joblib
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'validation_extension_20260908'; OUT.mkdir(exist_ok=True)
sys.path.insert(0,str(ROOT))
from scripts.eval.common import aggregate_coarse_mean_numpy as pool,repeat_coarse_to_fine as repeat
DYNAMIC=['input_wghm_twsa','input_era5_twsa','input_era5_cwsc']
STATIC=['aridity_mask','human_activity_index','glacier_fraction']
def features(d,i,coarse):
    mask=d.valid_mask.isel(time=i).values.astype(bool)
    weight=mask*d.cell_area_weight.values
    a=[d[k].isel(time=i).values for k in DYNAMIC]+[d[k].values for k in STATIC]
    if coarse:a=[pool(np.where(mask,x,0),weight,6)[0] for x in a]
    month=int(d.month_of_year.isel(time=i).values)
    a += [np.full(a[0].shape,np.sin(2*np.pi*month/12)),np.full(a[0].shape,np.cos(2*np.pi*month/12))]
    x=np.stack(a,axis=-1)
    valid=(pool(mask.astype(float),weight,6)[1] if coarse else mask)&np.isfinite(x).all(axis=-1)
    return x,valid,mask
def main():
    frozen=json.loads((ROOT/'data_driven_alpha_beta/evaluation/frozen_manifest.json').read_text())
    d=xr.open_zarr(frozen['shared']['dataset']['zarr_path'])
    assert d.attrs['canonical_water_storage_units']=='mm EWH'
    splits=d.split_index.values
    assert sum(splits==0)==156 and sum(splits==1)==22 and sum(splits==2)==36
    xs=[];ys=[]
    for i in np.flatnonzero(splits==0):
        x,v,_=features(d,i,True); y=d.target_jplm_twsa_coarse.isel(time=i).values
        v &= np.isfinite(y);xs.append(x[v]);ys.append(y[v])
    x=np.concatenate(xs);y=np.concatenate(ys);sc=StandardScaler().fit(x);x=sc.transform(x)
    print('TRAIN',x.shape,flush=True)
    info={'source_doi':'10.5194/hess-26-4169-2022','adaptation_not_exact_replication':True,'features':DYNAMIC+STATIC+['month_sin','month_cos'],'training_months':156,'training_samples':len(y),'seed':42,'sklearn':sklearn.__version__,'test_tuning':False}
    (OUT/'baseline_design.json').write_text(json.dumps(info,indent=2))
    for label,model in [('MLR_DS',LinearRegression()),('RF_DS',RandomForestRegressor(n_estimators=200,max_features='sqrt',min_samples_leaf=5,max_depth=20,random_state=42,n_jobs=8))]:
        folder=OUT/'baselines'/label;folder.mkdir(parents=True,exist_ok=True)
        if (folder/'complete.json').exists():continue
        t=time.time(); model.fit(x,y);joblib.dump({'scaler':sc,'model':model},folder/'fitted.joblib')
        val=[]
        for i in np.flatnonzero(splits==1):
            a,v,_=features(d,i,True); target=d.target_jplm_twsa_coarse.isel(time=i).values;v &= np.isfinite(target)
            val.extend((model.predict(sc.transform(a[v]))-target[v]).tolist())
        arrays=[];raws=[];closure=[]
        inds=np.flatnonzero(splits==2)
        for i in inds:
            a,v,mask=features(d,i,False); assert np.all(v[mask]),'Feature missingness differs from frozen land mask'
            raw=np.full(mask.shape,np.nan,dtype=np.float32);raw[v]=model.predict(sc.transform(a[v]))
            c,_=pool(np.where(v,raw,0),v*d.cell_area_weight.values,6)
            target=d.target_jplm_twsa_coarse.isel(time=i).values
            pred=np.where(v,raw+repeat(target-c,6),np.nan).astype('float32')
            check,ok=pool(np.nan_to_num(pred),v*d.cell_area_weight.values,6)
            closure.append(float(np.max(np.abs(check[ok]-target[ok]))))
            raws.append(raw);arrays.append(pred)
            print(label,str(d.time.values[i])[:10],flush=True)
        out=d[['target_jplm_twsa','target_jplm_twsa_coarse','input_wghm_twsa','valid_mask']].isel(time=inds).load()
        out['predicted_twsa']=(('time','lat','lon'),np.stack(arrays))
        out['predicted_twsa_raw']=(('time','lat','lon'),np.stack(raws))
        out['predicted_twsa_corrected']=out.predicted_twsa
        out.attrs.update(baseline=label,units='mm EWH',adaptation_not_exact_replication='true',default_prediction_variant='corrected',correction='full masked coslat additive correction; intrinsic baseline algorithm')
        out.to_netcdf(folder/'test.nc',encoding={k:{'zlib':True,'complevel':2} for k in out.data_vars})
        result={'validation_coarse_raw_rmse_mm':float(np.sqrt(np.mean(np.square(val)))),'max_corrected_closure_mm':max(closure),'elapsed_seconds':time.time()-t,'model_params':model.get_params()}
        (folder/'complete.json').write_text(json.dumps(result,indent=2));print(label,result,flush=True)
    print('BASELINES_COMPLETE',flush=True)
if __name__=='__main__':main()
