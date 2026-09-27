"""Figures from newly recomputed source tables; all display limits shared by variable."""
from pathlib import Path
import sys,json,string,textwrap
import numpy as np,pandas as pd,xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm,SymLogNorm,PowerNorm
from matplotlib.ticker import FormatStrFormatter,MaxNLocator
from matplotlib.patches import Rectangle
from evaluation_core import ecdf,spearman
from audit_panel_alignment import require_matplotlib_panel_alignment
from qa_preview import save_preview
R=Path(__file__).resolve().parents[2];O=R/'validation_rebuild';T=O/'tables';C=O/'cache';Q=O/'audit/figure_qa';Q.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],'font.size':8,'axes.titlesize':8,'axes.labelsize':8,'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
colors={'JPL':'#242424','WGHM':'#3575A6','M0':'#888888','M1':'#8661A6','M2':'#E69F00','M3':'#168C8C','M4':'#AD3349','MLR_DS':'#847344','RF_DS':'#417345'}
names=list(colors);models=['M0','M1','M2','M3','M4'];allmods=models+['MLR_DS','RF_DS']
co=xr.open_dataset(C/'coarse_diagnostics.nc');sp=xr.open_dataset(C/'spatial_snapshots.nc');w=pd.read_csv(T/'weight_land_source_data.csv');sm=pd.read_csv(T/'smap_common_grid_correlations.csv');gw=pd.read_csv(T/'groundwater_common_grid_correlations.csv');reg=pd.read_csv(T/'all_5deg_regions.csv');bm=pd.read_csv(T/'baseline_full_metrics.csv').set_index('model');spec=pd.read_csv(T/'spectral_monthly.csv')
bd=xr.open_dataset(C/'rebuilt_weights/beta_data_driven.nc');coverage=xr.open_dataset(C/'rebuilt_weights/beta_source_count.nc')
base=xr.open_zarr(R/'data_processed/training_rescue_v2/grace_downscaling_training_dataset_mm_area.zarr',consolidated=True);land=base.land_mask.values>0
contracts=[];caption_rows=[];map_scales=[]
def layout(rows,cols):
 fig,ax=plt.subplots(rows,cols,figsize=(8.5,rows*2.5),squeeze=False)
 fig.subplots_adjust(left=.09,right=.96,bottom=.10,top=.94,wspace=.50,hspace=.72)
 return fig,ax.ravel()
def title(ax,t):ax.set_title(textwrap.fill(t,width=28),loc='left',pad=8)
def mapplot(ax,a,t,lim=None,diff=False,extent=(-180,180,-90,90),unit=''):
 a=np.asarray(a);cmap=plt.get_cmap('RdBu_r' if diff else 'viridis').copy();cmap.set_bad('#dddddd')
 if lim is None:lim=(float(np.nanmin(a)),float(np.nanmax(a)))
 norm=None;scale='linear';ticks=None;maxabs=max(abs(lim[0]),abs(lim[1]))
 if 'mm' in unit and maxabs>0:
  if diff:
   norm=SymLogNorm(linthresh=maxabs/1000,vmin=lim[0],vmax=lim[1]);scale='symlog; linear threshold=maxabs/1000';ticks=[-maxabs,0,maxabs]
  elif lim[0]==0:
   norm=PowerNorm(gamma=.25,vmin=0,vmax=lim[1]);scale='power gamma=0.25';ticks=[0,maxabs/100,maxabs]
 kwargs={'norm':norm} if norm is not None else {'vmin':lim[0],'vmax':lim[1]}
 im=ax.imshow(a,origin='lower',extent=extent,aspect='auto',cmap=cmap,interpolation='nearest',rasterized=True,**kwargs)
 map_scales.append(dict(panel_title=t,unit=unit,vmin=float(lim[0]),vmax=float(lim[1]),scale=scale,N_finite=int(np.isfinite(a).sum()),N_below=int((a<lim[0]).sum()),N_above=int((a>lim[1]).sum())))
 ax.set_xlim(extent[:2]);ax.set_ylim(extent[2:]);ax.set_xticks([-120,0,120] if extent[1]-extent[0]>100 else np.linspace(extent[0],extent[1],3));ax.set_yticks([-60,0,60] if extent[3]-extent[2]>100 else np.linspace(extent[2],extent[3],3));title(ax,t)
 cbax=ax.inset_axes([0,-.25,1,.055]);cb=plt.colorbar(im,cax=cbax,orientation='horizontal');cb.ax.tick_params(labelsize=8,pad=1)
 if ticks is not None:cb.set_ticks(ticks);cb.set_ticklabels([f'{x:,.0f}' if abs(x)>=100 else f'{x:.2g}' for x in ticks]);cb.ax.minorticks_off()
 cb.ax.set_xlabel(unit+(' (symlog)' if diff else ' (power 1/4)') if norm is not None else unit,labelpad=1)
 return im
def pointsmap(ax,frame,v,t,lim=(-1,1),extent=(-180,180,-90,90)):
 a=np.full((360,720),np.nan);iy=np.rint((frame.lat.values+89.75)/.5).astype(int);ix=np.rint((frame.lon.values+179.75)/.5).astype(int);a[iy,ix]=np.asarray(v)
 return mapplot(ax,a,t,lim,True,extent)
def curve(ax,frame,cols,t,xlabel='Temporal PCC'):
 for m in cols:
  x,y=ecdf(frame[m].values);ax.plot(x,y,label=m.replace('_','-'),color=colors[m],lw=1.8 if m=='M4' else 1)
 title(ax,t);ax.set_xlabel(xlabel);ax.set_ylabel('Cumulative fraction');ax.legend(fontsize=8,loc='best')
def density(ax,x,y,t,xlabel,ylabel):
 x=np.asarray(x);y=np.asarray(y);g=np.isfinite(x)&np.isfinite(y);x=x[g];y=y[g]
 ax.hexbin(x,y,gridsize=25,mincnt=1,cmap='cividis',rasterized=True)
 edges=np.quantile(x,np.linspace(0,1,11));xc=[];yc=[]
 for i in range(10):
  q=(x>=edges[i])&(x<=edges[i+1]);xc.append(np.median(x[q]));yc.append(np.median(y[q]))
 ax.plot(xc,yc,color='#AD3349',lw=1.3);title(ax,t+f' (rho={spearman(x,y):.2f})');ax.set_xlabel(xlabel);ax.set_ylabel(ylabel)
def spectral(ax,stratum='global',month=None,cols=names):
 f=spec[spec.stratum==stratum]
 if month is not None:f=f[f.time==month]
 if not np.any(np.isfinite(f.power)&(f.power>0)):
  title(ax,month or stratum.replace('_',' ').title());ax.text(.5,.5,'No eligible support\nunder frozen tertile rule',ha='center',va='center',transform=ax.transAxes);ax.set_xlim(0,1);ax.set_ylim(0,1);return
 for m in cols:
  p=f[f.model==m].groupby('wavenumber').power;med=p.median();ax.plot(med.index,med.values,color=colors[m],label=m.replace('_','-'),lw=1.5 if m=='M4' else .9)
  if month is None:ax.fill_between(med.index,p.quantile(.25),p.quantile(.75),color=colors[m],alpha=.08)
 ax.set_xscale('log');ax.set_yscale('log');ax.set_xlabel('Wavenumber (cycles/cell)');ax.set_ylabel('Power (mm²)');ax.legend(fontsize=8,ncol=2);title(ax,month or stratum.replace('_',' ').title())
def save(fig,axes,name,question,panels,sources,supp=False):
 handles={}
 for ax in axes:
  hh,ll=ax.get_legend_handles_labels()
  for h,l in zip(hh,ll):handles.setdefault(l,h)
  legend=ax.get_legend()
  if legend is not None:legend.remove()
 if handles:fig.legend(list(handles.values()),list(handles),loc='upper center',bbox_to_anchor=(.5,-.04),ncol=min(5,len(handles)),frameon=False)
 for i,ax in enumerate(axes):ax.text(-.17,1.12,string.ascii_lowercase[i],transform=ax.transAxes,fontweight='bold',fontsize=10,va='bottom')
 require_matplotlib_panel_alignment(fig,axes=list(axes),panel_ids=list(string.ascii_lowercase[:len(axes)]),json_out=Q/(name+'.alignment.json'),strict=True)
 out=O/('figures_supp' if supp else 'figures_main')/name
 fig.savefig(out.with_suffix('.pdf'),dpi=600,bbox_inches='tight',pad_inches=.12)
 fig.savefig(out.with_suffix('.svg'),dpi=600,bbox_inches='tight',pad_inches=.12)
 fig.savefig(out.with_suffix('.png'),dpi=600,bbox_inches='tight',pad_inches=.12)
 save_preview(fig,Q/(name+'.preview.png'));plt.close(fig)
 contracts.append(dict(figure=name,question=question,archetype='quantitative grid / maps plus paired summaries',panels=panels,sources=sources))
 caption_rows.append(dict(figure=name,panels=panels,sources=sources,supplementary=supp))
 print('RENDERED',name,flush=True)

# Fig 2: weights, non-construction diagnostics and source omission.
fig,ax=layout(4,3)
for i,v in enumerate(['alpha','beta']):
 a=np.full((360,720),np.nan);a[land]=w[v];mapplot(ax[i],a,'Frozen '+v,(0,1),unit='Weight')
for v in ['alpha','beta']:
 x,y=ecdf(w[v]);ax[2].plot(x,y,label=v)
title(ax[2],'Weight distributions');ax[2].set_xlabel('Weight');ax[2].set_ylabel('Cumulative fraction');ax[2].legend()
df=pd.read_csv(T/'alpha_diagnostics_train_cells.csv')
for a,v,t in zip(ax[3:7],['PCC','NRMSE','abs_trend_difference','abs_annual_difference'],['PCC','NRMSE','Trend difference (mm/year)','Annual difference (mm)']):density(a,df.alpha,df[v],'Training diagnostic','Alpha',t)
loo=pd.read_csv(T/'beta_leave_one_source_out.csv')
for i,v in enumerate(['era5_twsa','gldas_noah_twsa','clm5_twsa']):mapplot(ax[7+i],bd['beta_without_'+v]-bd.beta,'Without '+['ERA5','Noah','CLM5'][i],(-1,1),True,unit='Delta beta')
ax[10].plot(range(3),loo.MAE,'o');ax[10].set_xticks(range(3),['ERA5','Noah','CLM5']);ax[10].set_ylabel('Mean absolute change');title(ax[10],'Source omission sensitivity')
ax[11].plot(range(3),loo.Spearman,'o');ax[11].set_xticks(range(3),['ERA5','Noah','CLM5']);ax[11].set_ylabel('Spearman rho');title(ax[11],'Rank stability')
save(fig,ax,'Fig02_alpha_beta_validation','Do frozen credibility weights have interpretable structure?','a-b maps; c distributions; d-g alpha diagnostics; h-j omitted-source minus full beta; k-l robustness.',['weight_land_source_data.csv','alpha_diagnostics_train_cells.csv','beta_leave_one_source_out.csv','cache/rebuilt_weights'])

fig,ax=layout(4,3)
for i,v in enumerate(['coarse_RMSE','median_temporal_PCC','coarse_Bias','trend_MAE','annual_MAE']):
 ax[i].plot(range(7),bm.loc[allmods,v],'o',color='#555555');ax[i].set_xticks(range(7),[m.replace('_DS','') for m in allmods],rotation=45);title(ax[i],v.replace('_',' '))
 if v=='median_temporal_PCC':ax[i].yaxis.set_major_formatter(FormatStrFormatter('%.5f'));ax[i].yaxis.set_major_locator(MaxNLocator(4))
lim=(0,max(float(co[m+'_RMSE'].max()) for m in models))
for i,m in enumerate(models):mapplot(ax[5+i],co[m+'_RMSE'],m+' temporal RMSE',lim,unit='mm')
dl=max(float(abs(co['M4_RMSE']-co[m+'_RMSE']).max()) for m in ['M0','M1'])
for i,m in enumerate(['M0','M1']):mapplot(ax[10+i],co['M4_RMSE']-co[m+'_RMSE'],'M4 minus '+m,(-dl,dl),True,unit='Delta RMSE (mm)')
save(fig,ax,'Fig03_coarse_fidelity','How well is GRACE-resolvable variability preserved?','a-e common-support global summaries; f-j temporal RMSE maps; k-l paired RMSE maps.',['coarse_full_metrics.csv','coarse_monthly_metrics.csv','cache/coarse_diagnostics.nc'])

fig,ax=layout(3,4)
for row,v in enumerate(['trend','annual','semiannual']):
 lo=min(float(co[m+'_'+v].min()) for m in ['JPL','M4','WGHM']);hi=max(float(co[m+'_'+v].max()) for m in ['JPL','M4','WGHM']);limits=(-max(abs(lo),hi),max(abs(lo),hi)) if v=='trend' else (0,hi)
 for col,m in enumerate(['JPL','M4','WGHM']):mapplot(ax[row*4+col],co[m+'_'+v],m+' '+v,limits,v=='trend',unit='mm/year' if v=='trend' else 'mm')
 for m in models:
  x,y=ecdf(abs(co[m+'_'+v].values-co['JPL_'+v].values));ax[row*4+3].plot(x,y,color=colors[m],label=m)
 title(ax[row*4+3],'Absolute '+v+' error');ax[row*4+3].set_ylabel('Cumulative fraction');ax[row*4+3].legend()
save(fig,ax,'Fig04_temporal_structure','Which temporal components are preserved?','Rows: three-year diagnostic trend, annual amplitude, semiannual amplitude. Columns: JPL, M4, WGHM, absolute-error ECDFs.',['cache/coarse_diagnostics.nc'])

fig,ax=layout(2,3)
for a,s,mo in zip(ax,['global','global','global','low_beta','mid_beta','high_beta'],[None,'2020-01-01','2021-07-01',None,None,None]):spectral(a,s,mo,cols=['JPL','WGHM']+models)
save(fig,ax,'Fig05_scale_dependent_spectra','Does added variability depend on scale and beta?','a median with monthly IQR; b-c fixed January 2020 and July 2021; d-f frozen beta strata. Masked FFT in normalized wavenumber; not physical km spectrum.',['spectral_monthly.csv'])

fig,ax=layout(5,3);show=['JPL','WGHM','M0','M1','M4']
lim=max(float(abs(sp[m]).max()) for m in show);hl=max(float(abs(sp[m+'_highpass']).max()) for m in show);rl=max(float(sp[m+'_hp_rms'].max()) for m in show)
for i,m in enumerate(show):
 mapplot(ax[i*3],sp[m],m+' January 2020',(-lim,lim),True,unit='TWSA (mm)');mapplot(ax[i*3+1],sp[m+'_highpass'],'High-pass',(-hl,hl),True,unit='mm');mapplot(ax[i*3+2],sp[m+'_hp_rms'],'36-month RMS',(0,rl),unit='mm')
save(fig,ax,'Fig06_spatial_variability','How much spatial variability is added?','Rows JPL, WGHM, M0, M1, M4; columns TWSA, fixed Gaussian high-pass, temporal high-pass RMS.',['cache/spatial_snapshots.nc','highpass_metrics.csv'])

fig,ax=layout(3,3)
for i,m in enumerate(['M0','M1','M4']):pointsmap(ax[i],sm,sm[m],m+' vs SMAP')
curve(ax[3],sm,models,'Common-grid correlations')
for i,m in enumerate(models[:-1]):pointsmap(ax[4+i],sm,sm.M4-sm[m],'M4 minus '+m,(-.2,.2))
p=pd.read_csv(T/'smap_pairwise_global.csv').iloc[:4];ax[8].errorbar(p.median_pair_difference,np.arange(4),xerr=[p.median_pair_difference-p.CI_low,p.CI_high-p.median_pair_difference],fmt='o',color=colors['M4']);ax[8].set_yticks(range(4),p.comparison);ax[8].axvline(0,color='gray',lw=.7);title(ax[8],'Paired median and 95% interval');ax[8].set_xlabel('Delta PCC')
save(fig,ax,'Fig07_smap_validation','Do reconstructed signals agree with SMAP?','a-c PCC maps; d ECDF; e-h paired grid differences (display saturation at +/-0.2); i spatial block bootstrap.',['smap_common_grid_correlations.csv','smap_pairwise_global.csv'])

fig,ax=layout(2,3);j=pd.read_csv(T/'alpha_beta_joint_gains.csv');j=j[j.comparison=='M4-M1'];z=j.pivot(index='alpha_bin',columns='beta_bin',values='median_pair_difference').values;lim=float(np.nanmax(abs(z)));im=ax[0].imshow(z,origin='lower',cmap='RdBu_r',vmin=-lim,vmax=lim,aspect='auto');ax[0].set_xticks(range(3),['Low','Mid','High']);ax[0].set_yticks(range(3),['Low','Mid','High']);ax[0].set_xlabel('Beta tertile');ax[0].set_ylabel('Alpha tertile');title(ax[0],'M4-M1 paired SMAP gain')
for _,r in j.iterrows():ax[0].text(r.beta_bin,r.alpha_bin,f'{r.median_pair_difference:.4f}\nN={int(r.N_grid)}',ha='center',va='center',fontsize=8,color='black')
for a,v in zip(ax[1:3],['beta','alpha']):
 vals=[float((sm.loc[sm[v+'_bin']==i,'M4']-sm.loc[sm[v+'_bin']==i,'M1']).median()) for i in range(3)];a.plot(range(3),vals,'o-',color=colors['M4']);a.axhline(0,color='gray',lw=.7);a.set_xticks(range(3),['Low','Mid','High']);a.set_ylabel('Median delta PCC');title(a,v.title()+' tertiles')
density(ax[3],sm.beta,sm.M4-sm.M1,'External gain vs beta','Beta','Delta PCC');density(ax[4],sm.alpha,sm.M4-sm.M1,'External gain vs alpha','Alpha','Delta PCC')
ax[5].scatter(sm.beta,sm.HFI,c=sm.M4-sm.M1,cmap='RdBu_r',vmin=-.2,vmax=.2,s=3,rasterized=True,alpha=.5);ax[5].set_yscale('symlog',linthresh=1);ax[5].set_xlabel('Beta');ax[5].set_ylabel('HFI (symlog)');title(ax[5],f'High-pass inheritance; rho={spearman(sm.beta.values,sm.HFI.values):.2f}')
save(fig,ax,'Fig08_weight_gain_associations','Are external gains associated with credibility weights?','a joint tertiles with N; b-c marginal bins; d-e density with bin medians; f exploratory high-pass energy inheritance. Associations do not establish causality.',['alpha_beta_joint_gains.csv','smap_common_grid_correlations.csv','HFI_summary.csv'])

fig,ax=layout(2,3);rm=np.full((36,72),np.nan)
for _,r in reg.iterrows():rm[int((r.lat_min+90)/5),int((r.lon_min+180)/5)]=r.S_min
lim=float(abs(reg.S_min).max());mapplot(ax[0],rm,'All eligible fixed regions',(-lim,lim),True,unit='Minimum paired gain S')
ax[1].plot(reg['rank'],reg.S_min,color=colors['M4']);ax[1].axhline(0,color='gray',lw=.7);ax[1].set_xlabel('Rank');ax[1].set_ylabel('S');title(ax[1],'Complete regional ordering')
x,y=ecdf(reg.S_min);ax[2].plot(x,y);ax[2].axvline(0,color='gray',lw=.7);title(ax[2],'Regional ECDF');ax[2].set_xlabel('S')
for m in models[:-1]:x,y=ecdf(reg['delta_M4_'+m]);ax[3].plot(x,y,label=m,color=colors[m])
ax[3].legend();title(ax[3],'All four paired comparisons');ax[3].set_xlabel('Median regional delta PCC')
density(ax[4],reg.median_beta,reg.S_min,'Regional association','Median beta','S');density(ax[5],reg.median_alpha,reg.S_min,'Regional association','Median alpha','S')
save(fig,ax,'Fig09_complete_regional_gains','Where are gains and failures located?','a signed 5-degree map; b full ranking; c ECDF; d four paired distributions; e-f weight associations. All eligible regions included.',['all_5deg_regions.csv'])

fig,ax=layout(3,4);cases=pd.read_csv(O/'regional_cases/selected_cases.csv');ts=pd.read_csv(T/'regional_case_timeseries.csv')
for i,(_,r) in enumerate(cases.iterrows()):
 extent=(r.lon_min,r.lon_max,r.lat_min,r.lat_max);iy=(sp.lat>=r.lat_min)&(sp.lat<r.lat_max);ix=(sp.lon>=r.lon_min)&(sp.lon<r.lon_max)
 pointsmap(ax[i*4],sm,sm.M4-sm.M1,f'Rank {int(r["rank"])}; S={r.S_min:.3f}',(-.2,.2));ax[i*4].add_patch(Rectangle((r.lon_min,r.lat_min),5,5,fill=False,edgecolor='black',lw=1.5))
 a=sp.M4.values[np.ix_(iy,ix)];b=sp.M1.values[np.ix_(iy,ix)];lim=float(np.nanmax(abs(np.stack([a,b]))));mapplot(ax[i*4+1],a,'M4 local TWSA',(-lim,lim),True,extent,'mm');mapplot(ax[i*4+2],b,'M1 local TWSA',(-lim,lim),True,extent,'mm')
 for m in ['JPL','M0','M1','M4','SMAP']:
  f=ts[(ts.region_id.astype(str)==str(r.region_id))&(ts.model==m)];ax[i*4+3].plot(np.arange(len(f)),f.standardized,label=m,color=colors.get(m,'#49A7BA'),lw=1)
 title(ax[i*4+3],'Standardized regional series');ax[i*4+3].set_xlabel('Test month index');ax[i*4+3].legend(fontsize=8,ncol=2)
save(fig,ax,'Fig10_regional_cases','What do two positive cases and one failure look like?','Rows selected positive, positive, strongest negative; columns location/gain, M4 January 2020 TWSA, M1 TWSA, common-support standardized time series. Post-hoc exploratory cases; full ranks retained.',['regional_cases/selected_cases.csv','regional_case_timeseries.csv','cache/spatial_snapshots.nc'])

fig,ax=layout(2,3);wf=pd.read_csv(T/'groundwater_well_correlations.csv');ax[0].scatter(wf.lon,wf.lat,s=2,color='#555555',rasterized=True);title(ax[0],f'{len(wf)} wells; {len(gw)} grids');ax[0].set_xlabel('Longitude');ax[0].set_ylabel('Latitude')
for i,m in enumerate(['M0','M1','M4']):
 im=ax[1+i].scatter(gw.lon,gw.lat,c=gw[m],vmin=-1,vmax=1,cmap='RdBu_r',s=12,rasterized=True);ax[1+i].set_xlim(-125,-65);ax[1+i].set_ylim(24,50);title(ax[1+i],m+' groundwater PCC');cbax=ax[1+i].inset_axes([0,-.25,1,.055]);plt.colorbar(im,cax=cbax,orientation='horizontal',ticks=[-1,0,1])
curve(ax[4],gw,models,'Common-grid correlations');p=pd.read_csv(T/'groundwater_pairwise_global.csv').iloc[:4];ax[5].errorbar(p.median_pair_difference,np.arange(4),xerr=[p.median_pair_difference-p.CI_low,p.CI_high-p.median_pair_difference],fmt='o',color=colors['M4']);ax[5].set_yticks(range(4),p.comparison);ax[5].axvline(0,color='gray',lw=.7);title(ax[5],'Paired median and 95% interval');ax[5].set_xlabel('Delta PCC')
save(fig,ax,'Fig11_groundwater_validation','Is groundwater consistency improved?','a qualifying wells; b-d grid-median well PCC on shared [-1,1] scale; e complete ECDF; f paired spatial block bootstrap. WGHM non-groundwater subtraction is model dependent.',['groundwater_well_correlations.csv','groundwater_common_grid_correlations.csv','groundwater_pairwise_global.csv'])

fig,ax=layout(2,2);cols=['coarse_RMSE','annual_MAE','median_highpass_RMS','SMAP_median_PCC','groundwater_median_PCC'];v=bm.loc[allmods,cols];norm=(v-v.min())/(v.max()-v.min());ax[0].imshow(norm.values,aspect='auto',cmap='viridis',vmin=0,vmax=1);ax[0].set_yticks(range(7),allmods);ax[0].set_xticks(range(5),['RMSE','Annual','HP RMS','SMAP','GW'],rotation=30);title(ax[0],'Metric magnitude normalized [0,1]')
for a,y in zip(ax[1:3],['SMAP_median_PCC','median_highpass_RMS']):
 for m in allmods:a.scatter(bm.loc[m,'coarse_RMSE'],bm.loc[m,y],color=colors[m],label=m,s=24)
 a.set_xlabel('Coarse RMSE (mm)');a.set_ylabel(y.replace('_',' '));a.legend(ncol=2);title(a,'Joint benchmark')
spectral(ax[3],cols=['JPL','WGHM','M4','MLR_DS','RF_DS'])
save(fig,ax,'Fig12_baseline_benchmark','How do coarse fidelity, added variance and external skill trade off?','a visualization-only magnitude normalization (not overall skill ranking); b coarse fidelity vs SMAP; c coarse fidelity vs high-pass RMS; d median spectra with IQR.',['baseline_full_metrics.csv','spectral_monthly.csv'])

# Supplementary figures retain full evidence dimensions.
for num,titletext,variables in [(1,'source_coverage',list(coverage.data_vars)),(3,'beta_robustness',['beta_without_era5_twsa','beta_without_gldas_noah_twsa','beta_without_clm5_twsa'])]:
 fig,ax=layout(2,3) if num==1 else layout(1,len(variables));ds=coverage if num==1 else bd
 for a,v in zip(ax,variables):mapplot(a,np.where(land,ds[v],np.nan),v.replace('valid_months_','').replace('beta_without_',''),(0,float(ds[v].max()) if num==1 else 1),unit='Months / sources' if num==1 else 'Beta')
 if num==1:
  vals=coverage.source_count.values[land];ax[5].bar(range(4),[(vals==i).mean() for i in range(4)],color='#777777');title(ax[5],'Auxiliary-source availability');ax[5].set_xlabel('Number of sources');ax[5].set_ylabel('Land-cell fraction')
 save(fig,ax,f'FigS{num:02d}_{titletext}','Is auxiliary evidence coverage heterogeneous?',', '.join(variables),['cache/rebuilt_weights'],True)
fig,ax=layout(2,2);df=pd.read_csv(T/'alpha_diagnostics_test_cells.csv')
for a,v in zip(ax,['PCC','NRMSE','abs_trend_difference','abs_annual_difference']):density(a,df.alpha,df[v],'Held-out test diagnostic','Alpha',v)
save(fig,ax,'FigS02_alpha_test_diagnostics','Do non-construction relationships persist in test?', 'a-d test diagnostics.',['alpha_diagnostics_test_cells.csv'],True)
fig,ax=layout(1,2)
for i,m in enumerate(models):
 f=pd.read_csv(R/f'data_driven_alpha_beta/M{5 if i==4 else i}/training/metrics.csv');f.to_csv(T/f'training_{m}_source.csv',index=False)
 for a,token in zip(ax,['train','val']):
  col=token+'_loss_total';assert col in f.columns
  if col:a.plot(f['epoch'] if 'epoch' in f else np.arange(len(f)),f[col],color=colors[m],label=m);title(a,token.title()+' objective');a.set_xlabel('Epoch');a.legend()
save(fig,ax,'FigS04_training_trajectories','Were models trained with the same budget?','a training and b validation objectives; differing alpha/beta terms mean objectives are not equal physical error measures.',['training_M0_source.csv','training_M1_source.csv','training_M2_source.csv','training_M3_source.csv','training_M4_source.csv'],True)
fig,ax=layout(1,2);cm=pd.read_csv(T/'coarse_monthly_metrics.csv')
for a,v in zip(ax,['raw','corrected']):
 for m in models:f=cm[(cm.model==m)&(cm.variant==v)];a.plot(np.arange(len(f)),f.RMSE,color=colors[m],label=m)
 title(a,v.title()+' coarse RMSE');a.set_xlabel('Test month index');a.set_ylabel('RMSE (mm)');a.legend()
save(fig,ax,'FigS05_raw_corrected','What does frozen postprocessing change?','a raw; b corrected monthly spatial RMSE.',['coarse_monthly_metrics.csv'],True)
fig,ax=layout(2,3);lim=max(float(sp[m+'_hp_rms'].max()) for m in models)
for a,m in zip(ax,models):mapplot(a,sp[m+'_hp_rms'],m+' high-pass RMS',(0,lim),unit='mm')
for m in models:x,y=ecdf(sp[m+'_hp_rms'].values);ax[5].plot(x,y,color=colors[m],label=m)
title(ax[5],'Complete high-pass distributions');ax[5].legend();ax[5].set_xlabel('RMS (mm)')
save(fig,ax,'FigS06_complete_spatial_output','How do all five ablations differ spatially?','a-e complete high-pass RMS; f ECDF.',['cache/spatial_snapshots.nc'],True)
fig,ax=layout(1,3)
for a,v in zip(ax,['coarse_MAE','median_temporal_NSE','semiannual_MAE']):a.plot(range(7),bm.loc[allmods,v],'o');a.set_xticks(range(7),['M0','M1','M2','M3','M4','MLR','RF'],rotation=45);title(a,v.replace('_',' '))
save(fig,ax,'FigS07_secondary_coarse_metrics','Do secondary fidelity diagnostics agree?','a MAE; b median NSE; c semiannual amplitude MAE.',['coarse_full_metrics.csv'],True)
fig,ax=layout(2,3)
for a,m in zip(ax,allmods[:-3]+['MLR_DS','RF_DS']):pointsmap(a,sm,sm.M4-sm[m],'M4 minus '+m,(-.2,.2))
save(fig,ax,'FigS08_full_smap_pairs','Which external comparisons improve or deteriorate?','Complete paired difference maps, +/-0.2 display saturation; full untruncated values in source table.',['smap_common_grid_correlations.csv'],True)
fig,ax=layout(2,2)
for a,m in zip(ax,models[:-1]):a.scatter(reg['rank'],reg['delta_M4_'+m],s=6,color=colors[m],rasterized=True);a.axhline(0,color='gray',lw=.7);title(a,'M4 minus '+m);a.set_xlabel('Full S rank');a.set_ylabel('Regional delta PCC')
save(fig,ax,'FigS09_full_region_comparisons','Are regional gains shared across all ablations?','a-d every eligible region against each ablation.',['all_5deg_regions.csv'],True)
fig,ax=layout(2,3)
for a,s in zip(ax[:3],['global','low_beta','high_beta']):spectral(a,s,cols=names)
for a,v in zip(ax[3:],['aridity','human','glacier']):density(a,w[v],w.beta,'Weight context',v,'Beta')
save(fig,ax,'FigS11_full_spectrum_context','How does spatial structure vary with regional context?','a-c all-model spectra; d-f frozen regional covariates.',['spectral_monthly.csv','weight_land_source_data.csv'],True)
fig,ax=layout(1,2);ls=pd.read_csv(T/'loss_scale_checkpoint_batches.csv');terms=[v for v in ls if v.startswith('loss_') and v!='loss_total']
for v in terms:ax[0].plot(ls.batch,ls[v],label=v.replace('loss_',''),lw=.8)
ax[0].set_yscale('symlog');ax[0].set_xlabel('Fixed training batch index');title(ax[0],'Frozen checkpoint loss terms');ax[0].legend(fontsize=8,ncol=2)
ax[1].plot(ls.batch,ls.total,color=colors['M4']);title(ax[1],'Total weighted objective');ax[1].set_xlabel('Fixed training batch index');ax[1].set_ylabel('Objective (not RMSE)')
save(fig,ax,'FigS10_loss_scale_audit','What are the objective term magnitudes?','a raw loss terms; b total objective; 48 forward-only batches of 4 training tiles evenly sampled through dataset. No optimization.',['loss_scale_checkpoint_batches.csv'],True)
fig,ax=layout(1,3);jj=pd.read_csv(T/'alpha_beta_joint_gains.csv');limit=float(np.nanmax(abs(jj.median_pair_difference)))
for a,m in zip(ax,['M0','M2','M3']):
 z=jj[jj.comparison=='M4-'+m].pivot(index='alpha_bin',columns='beta_bin',values='median_pair_difference').values;a.imshow(z,origin='lower',aspect='auto',cmap='RdBu_r',vmin=-limit,vmax=limit);a.set_xticks(range(3),['Low','Mid','High']);a.set_yticks(range(3),['Low','Mid','High']);a.set_xlabel('Beta tertile');a.set_ylabel('Alpha tertile');title(a,'M4 minus '+m)
 for yy in range(3):
  for xx in range(3):a.text(xx,yy,f'{z[yy,xx]:.4f}',ha='center',va='center',fontsize=8)
save(fig,ax,'FigS12_complete_joint_strata','Are stratified patterns consistent across ablations?','a-c complete joint alpha-beta gains against M0/M2/M3; M1 in main Fig8.',['alpha_beta_joint_gains.csv'],True)
fig,ax=layout(2,2);series=pd.read_csv(T/'predefined_and_fixed_region_timeseries.csv')
for a,r in zip(ax,['Amazon','Sahara','Indo_Gangetic','Alaska']):
 for m in ['JPL','WGHM']+models:
  f=series[(series.region==r)&(series.model==m)];a.plot(np.arange(len(f)),f.mean_mm,color=colors[m],label=m,lw=1)
 title(a,r.replace('_',' '));a.set_xlabel('Test month index');a.set_ylabel('Area-weighted TWSA (mm)');a.legend(fontsize=8,ncol=2)
save(fig,ax,'FigS13_predefined_regions','How do signals behave in predefined geographic contexts?','a Amazon; b Sahara; c Indo-Gangetic; d Alaska. Bounds declared before performance calculation; all four retained, including missing-support behavior.',['predefined_and_fixed_region_timeseries.csv','audit/predefined_regions.md'],True)
fig,ax=layout(2,3);cols=['MLR_DS','RF_DS','M4'];lim=max(float(abs(sp[m]).max()) for m in cols);hpmax=max(float(sp[m+'_hp_rms'].max()) for m in cols)
for i,m in enumerate(cols):mapplot(ax[i],sp[m],m+' January 2020',(-lim,lim),True,unit='mm');mapplot(ax[3+i],sp[m+'_hp_rms'],m+' high-pass RMS',(0,hpmax),unit='mm')
save(fig,ax,'FigS14_baseline_maps','How do traditional baselines allocate spatial variability?','a-c fixed-month TWSA; d-f high-pass RMS for MLR/RF/M4.',['cache/spatial_snapshots.nc'],True)
rspec=pd.read_csv(T/'regional_spectral_monthly.csv')
for ci,(_,r) in enumerate(cases.iterrows()):
 fig,ax=layout(3,4);iy=np.flatnonzero((sp.lat>=r.lat_min)&(sp.lat<r.lat_max));ix=np.flatnonzero((sp.lon>=r.lon_min)&(sp.lon<r.lon_max));extent=(r.lon_min,r.lon_max,r.lat_min,r.lat_max)
 alpha_grid=np.full((360,720),np.nan);alpha_grid[land]=w.alpha
 for a,v,n in [(ax[0],alpha_grid,'Alpha'),(ax[1],bd.beta.values,'Beta')]:mapplot(a,v[np.ix_(iy,ix)],n,(0,1),extent=extent,unit='Weight')
 limits=max(float(np.nanmax(abs(sp[m].values[np.ix_(iy,ix)]))) for m in ['JPL','WGHM','M0','M1','M4'])
 for a,m in zip(ax[2:7],['JPL','WGHM','M0','M1','M4']):mapplot(a,sp[m].values[np.ix_(iy,ix)],m+' January 2020',(-limits,limits),True,extent,'mm')
 f=sm[(sm.lon>=r.lon_min)&(sm.lon<r.lon_max)&(sm.lat>=r.lat_min)&(sm.lat<r.lat_max)]
 for a,values,n,lim in [(ax[7],f.M4,'M4 SMAP PCC',(-1,1)),(ax[8],f.M4-f.M1,'M4-M1 SMAP gain',(-.2,.2))]:
  arr=np.full((10,10),np.nan);yy=np.floor((f.lat-r.lat_min)/.5).astype(int);xx=np.floor((f.lon-r.lon_min)/.5).astype(int);arr[yy,xx]=values;mapplot(a,arr,n,lim,True,extent,'PCC / delta PCC')
 for m in ['JPL','M0','M1','M4','SMAP']:
  f=ts[(ts.region_id.astype(str)==str(r.region_id))&(ts.model==m)];ax[9].plot(np.arange(len(f)),f.standardized,label=m,color=colors.get(m,'#49A7BA'),lw=1)
 title(ax[9],'Standardized regional series');ax[9].set_xlabel('Test month');ax[9].legend(ncol=2)
 for m in ['JPL','WGHM','M0','M1','M4']:
  p=rspec[(rspec.region.astype(str)==str(r.region_id))&(rspec.model==m)].groupby('wavenumber').power.median();p=p[p.index>=1/min(len(iy),len(ix))];ax[10].plot(p.index,p,color=colors[m],label=m)
 ax[10].set_xscale('log');ax[10].set_yscale('log');title(ax[10],'Median regional spectra');ax[10].set_xlabel('Cycles/cell');ax[10].set_ylabel('Power (mm²)');ax[10].legend(ncol=2)
 ax[11].plot(range(4),[r['delta_M4_'+m] for m in models[:-1]],'o',color=colors['M4']);ax[11].axhline(0,color='gray',lw=.7);ax[11].set_xticks(range(4),models[:-1]);title(ax[11],f'Rank {int(r["rank"])}; S={r.S_min:.4f}');ax[11].set_ylabel('Paired median delta PCC')
 save(fig,ax,f'FigS{15+ci:02d}_regional_case_{ci+1}','What supports or limits each exploratory regional case?','a-b frozen alpha/beta; c-g JPL/WGHM/M0/M1/M4; h SMAP correlation; i paired gain; j standardized series; k median spectra; l all four paired gains and true rank. Location in main Fig10.',['regional_cases/selected_cases.csv','regional_case_timeseries.csv','regional_spectral_monthly.csv','smap_common_grid_correlations.csv','cache/spatial_snapshots.nc'],True)
(O/'audit/figure_contracts.json').write_text(json.dumps(contracts,indent=2));(O/'audit/figure_caption_metadata.json').write_text(json.dumps(caption_rows,indent=2))
pd.DataFrame(map_scales).to_csv(O/'audit/map_display_scales.csv',index=False)
