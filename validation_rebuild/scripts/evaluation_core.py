"""Shared float64 metrics. Axis 0 is time; no model-specific branches."""
import numpy as np
from scipy import stats, ndimage

def aggregate(a, mask, lat, factor=6):
 a=np.asarray(a,float); h,w=a.shape[-2:]
 if h%factor or w%factor: raise ValueError('Grid not divisible by factor')
 weights=np.broadcast_to(np.cos(np.deg2rad(lat))[:,None],a.shape)
 good=np.broadcast_to(mask,a.shape).astype(bool)&np.isfinite(a)
 weights=np.where(good,weights,0)
 shape=(*a.shape[:-2],h//factor,factor,w//factor,factor)
 den=weights.reshape(shape).sum(axis=(-3,-1))
 num=(np.where(good,a,0)*weights).reshape(shape).sum(axis=(-3,-1))
 return np.divide(num,den,out=np.full_like(num,np.nan),where=den>1e-6)

def temporal(a,b,good=None,min_months=24):
 a=np.asarray(a,float); b=np.asarray(b,float)
 g=np.isfinite(a)&np.isfinite(b)
 if good is not None:g &= good
 n=g.sum(axis=0); safe=np.maximum(n,1)
 am=np.where(g,a,0).sum(axis=0)/safe; bm=np.where(g,b,0).sum(axis=0)/safe
 ac=np.where(g,a-am,0);bc=np.where(g,b-bm,0);e=np.where(g,a-b,0)
 aa=(ac*ac).sum(axis=0);bb=(bc*bc).sum(axis=0)
 with np.errstate(divide='ignore',invalid='ignore'):
  vals={'RMSE':np.sqrt((e*e).sum(axis=0)/safe),'MAE':np.abs(e).sum(axis=0)/safe,'Bias':e.sum(axis=0)/safe,'PCC':(ac*bc).sum(axis=0)/np.sqrt(aa*bb),'NSE':1-(e*e).sum(axis=0)/bb}
 return {k:np.where((n>=min_months)&np.isfinite(v),v,np.nan) for k,v in vals.items()}

def monthly(a,b):
 e=a-b;axes=tuple(range(1,e.ndim))
 return {'RMSE':np.sqrt(np.nanmean(e*e,axis=axes)), 'MAE':np.nanmean(abs(e),axis=axes),'Bias':np.nanmean(e,axis=axes)}

def harmonics(a,times,min_months=24):
 """Joint intercept + trend + annual + semiannual OLS, actual calendar months."""
 t=(np.asarray(times).astype('datetime64[M]').astype(int)-np.asarray(times)[0].astype('datetime64[M]').astype(int))/12
 X=np.column_stack([np.ones(len(t)),t,np.sin(2*np.pi*t),np.cos(2*np.pi*t),np.sin(4*np.pi*t),np.cos(4*np.pi*t)])
 a=np.asarray(a,float); flat=a.reshape(len(t),-1);out=np.full((3,flat.shape[1]),np.nan)
 masks=np.isfinite(flat)
 # Group identical availability masks to avoid one least-squares call per grid.
 groups,inv=np.unique(masks.T,axis=0,return_inverse=True)
 for i,g in enumerate(groups):
  if g.sum()<max(6,min_months) or np.linalg.matrix_rank(X[g])<6:continue
  ids=np.flatnonzero(inv==i);coef=np.linalg.lstsq(X[g],flat[g][:,ids],rcond=None)[0]
  out[:,ids]=np.stack([coef[1],np.hypot(coef[2],coef[3]),np.hypot(coef[4],coef[5])])
 return dict(zip(['trend','annual','semiannual'],out.reshape(3,*a.shape[1:])))

def spearman(a,b):
 g=np.isfinite(a)&np.isfinite(b)
 return float(stats.spearmanr(np.asarray(a)[g],np.asarray(b)[g]).statistic) if g.sum()>2 else np.nan

def highpass(a,mask,sigma=3.,kernel=13):
 """Fixed 13-cell Gaussian, sigma 3 cells, mask-normalized; no tuning."""
 a=np.asarray(a,float);good=np.isfinite(a)&mask
 sig=(0,)*(a.ndim-2)+(sigma,sigma)
 w=ndimage.gaussian_filter(good.astype(float),sig,mode='nearest',truncate=(kernel//2)/sigma)
 s=ndimage.gaussian_filter(np.where(good,a,0),sig,mode='nearest',truncate=(kernel//2)/sigma)
 return np.where(good,a-s/np.maximum(w,1e-12),np.nan)

def power_spectrum(a,mask):
 """Masked demeaned rectangular FFT; zero outside support; normalized cycles/cell.
 Mask/window effects preclude interpreting this as an unqualified physical global spectrum.
 """
 a=np.asarray(a,float);g=mask&np.isfinite(a)
 if g.sum()<4:return np.arange(1,min(a.shape)//2+1)/min(a.shape),np.full(min(a.shape)//2,np.nan)
 z=np.where(g,a-np.mean(a[g]),0);p=abs(np.fft.fft2(z))**2/(a.size*g.sum())
 f=np.hypot(np.fft.fftfreq(a.shape[0])[:,None],np.fft.fftfreq(a.shape[1])[None,:]);edges=np.linspace(0,.5,min(a.shape)//2+1)
 idx=np.digitize(f.ravel(),edges)-1
 values=np.array([p.ravel()[idx==i].mean() if np.any(idx==i) else np.nan for i in range(len(edges)-1)])
 return (edges[:-1]+edges[1:])/2,values

def ecdf(a):
 x=np.sort(np.asarray(a)[np.isfinite(a)]);return x,np.arange(1,len(x)+1)/max(len(x),1)

def paired(a,b,lat,lon,resamples=2000,seed=42):
 d=np.asarray(a)-np.asarray(b);g=np.isfinite(d);d=d[g];lat=np.asarray(lat)[g];lon=np.asarray(lon)[g]
 blocks=np.floor((lat+90)/3).astype(int)*120+np.floor((lon+180)/3).astype(int)
 groups=[d[blocks==j] for j in np.unique(blocks)];rng=np.random.default_rng(seed)
 boot=np.full(resamples,np.nan)
 if len(groups)>=2:
  for i in range(resamples):boot[i]=np.median(np.concatenate([groups[j] for j in rng.integers(len(groups),size=len(groups))]))
 ci=np.nanpercentile(boot,[2.5,97.5]) if len(groups)>=2 else [np.nan,np.nan]
 return {'N_grid':len(d),'N_blocks':len(groups),'median_pair_difference':float(np.median(d)) if len(d) else np.nan,'CI_low':ci[0],'CI_high':ci[1],'positive_fraction':np.mean(d>0) if len(d) else np.nan,'negative_fraction':np.mean(d<0) if len(d) else np.nan},boot

def regional_statistics(values,lat,lon,min_grids=5):
 ids=np.floor((np.asarray(lat)+90)/5).astype(int)*72+np.floor((np.asarray(lon)+180)/5).astype(int)
 return {int(i):{'N':int(np.sum((ids==i)&np.isfinite(values))),'median':float(np.nanmedian(np.asarray(values)[ids==i]))} for i in np.unique(ids) if np.sum((ids==i)&np.isfinite(values))>=min_grids}
