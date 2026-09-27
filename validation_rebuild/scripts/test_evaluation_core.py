import numpy as np
from evaluation_core import aggregate,temporal,monthly,harmonics,paired,highpass
def run():
 lat=np.array([0.,60.]);a=np.array([[1.,3.],[5.,7.]])
 assert np.allclose(aggregate(a,np.ones_like(a,bool),lat,2),10/3)
 a=np.array([[[0.,0.]],[[0.,4.]]]);b=np.zeros_like(a)
 assert np.isclose(monthly(a,b)['RMSE'].mean(),np.sqrt(8)/2)
 assert not np.isclose(monthly(a,b)['RMSE'].mean(),np.sqrt(np.mean(a*a)))
 times=np.arange('2002-01','2007-01',dtype='datetime64[M]');keep=np.arange(60)%7!=0;times=times[keep]
 t=(times.astype(int)-times[0].astype(int))/12
 y=7+2*t+3*np.sin(2*np.pi*t)+4*np.cos(2*np.pi*t)+2*np.sin(4*np.pi*t)
 h=harmonics(y[:,None,None],times)
 assert np.allclose([h['trend'].item(),h['annual'].item(),h['semiannual'].item()],[2,5,2])
 assert np.allclose(temporal(y[:,None],y[:,None])['PCC'],1)
 a=np.array([0.,100.,101.]);b=np.array([0.,1.,100.])
 r,z=paired(a,b,np.array([0,4,8]),np.zeros(3),resamples=100)
 assert r['median_pair_difference']==1 and np.median(a)-np.median(b)==99
 assert np.array_equal(z,paired(a,b,np.array([0,4,8]),np.zeros(3),resamples=100)[1])
 f=np.full((2,20,20),10.);mask=np.ones_like(f,bool);mask[:,:,:3]=False
 assert np.allclose(highpass(f,mask)[mask],0)
 print('PASS: area weighting, mean monthly RMSE, missing-calendar harmonic recovery, identity PCC, paired median, deterministic block bootstrap, masked constant highpass')
if __name__=='__main__':run()
