"""Source-preserving redesign requested 2026-09-17. No model recomputation."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, SymLogNorm
from matplotlib.patches import Rectangle, FancyBboxPatch
import cartopy.crs as ccrs
import cartopy.feature as cf
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

R=Path(__file__).resolve().parents[1]
O=R/'figure_redesign_20260917';O.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':12,
 'axes.labelsize':10,'xtick.labelsize':9,'ytick.labelsize':9,'pdf.fonttype':42,'svg.fonttype':'none'})
PC=ccrs.PlateCarree()
LAND='#f3f1e9';WATER='#e5eff5';INK='#243747';MUTED='#566777'
coast=cf.NaturalEarthFeature('physical','coastline','50m',facecolor='none')
borders=cf.NaturalEarthFeature('cultural','admin_0_boundary_lines_land','50m',facecolor='none')
lakes=cf.NaturalEarthFeature('physical','lakes','50m',facecolor=WATER)
states=cf.NaturalEarthFeature('cultural','admin_1_states_provinces_lines','50m',facecolor='none')

def background(ax,extent,states_on=False,detail=True):
    ax.set_extent(extent,crs=PC);ax.set_facecolor(WATER)
    ax.add_feature(cf.LAND.with_scale('50m' if detail else '110m'),facecolor=LAND,zorder=0)
    ax.add_feature(lakes,edgecolor='#a8bcc8',linewidth=.35,zorder=1)
    ax.add_feature(coast,edgecolor='#728997',linewidth=.55,zorder=4)
    ax.add_feature(borders,edgecolor='#82939a',linewidth=.45,zorder=4)
    if states_on: ax.add_feature(states,edgecolor='#a4adb0',linewidth=.32,zorder=3)
    ax.spines['geo'].set_edgecolor('#aab8c2');ax.spines['geo'].set_linewidth(.65)

def ticks(ax,x,y,size=9):
    ax.set_xticks(x,crs=PC);ax.set_yticks(y,crs=PC)
    ax.xaxis.set_major_formatter(LongitudeFormatter(number_format='.0f'))
    ax.yaxis.set_major_formatter(LatitudeFormatter(number_format='.0f'))
    ax.tick_params(length=2.5,pad=3,labelsize=size,color='#8696a0')

def save(fig,name):
    for suffix in ['png','pdf','svg']:
        fig.savefig(O/f'{name}.{suffix}',dpi=600,facecolor='white')
    fig.savefig(O/f'{name}_preview.png',dpi=140,facecolor='white')
    plt.close(fig)
    print('SAVED',name,flush=True)

gw=pd.read_csv(R/'tables/groundwater_common_grid_correlations.csv')
wf=pd.read_csv(R/'tables/groundwater_well_correlations.csv')
fig=plt.figure(figsize=(12,8.6),facecolor='white')
fig.text(.055,.955,'Groundwater observation coverage and model agreement',fontsize=18,weight='bold',color=INK)
fig.text(.055,.919,'2020–2022  |  763 wells aggregated to 122 common 0.5° cells',fontsize=11,color=MUTED)
extent=[-126,-66,23,51]
axes=[]
for i,(name,rect) in enumerate(zip(['Well coverage','M0','M1','M4'],[[.055,.52,.415,.34],[.54,.52,.415,.34],[.055,.115,.415,.34],[.54,.115,.415,.34]])):
    ax=fig.add_axes(rect,projection=PC);axes.append(ax);background(ax,extent,True)
    ticks(ax,[-120,-100,-80],[25,35,45]);ax.gridlines(xlocs=[-120,-100,-80],ylocs=[25,35,45],linewidth=.4,color='#b7c5cd',alpha=.5,zorder=1)
    ax.set_title(f'{"abcd"[i]}   {name}',loc='left',weight='bold',color=INK,pad=10)
    if i==0:
        ax.scatter(wf.lon,wf.lat,s=13,color='#273d4f',alpha=.65,edgecolor='white',linewidth=.25,transform=PC,zorder=6,rasterized=True)
        for txt,lon,lat in [('UNITED STATES',-104,38.5),('CANADA',-108,49),('MEXICO',-106,26),('Atlantic\nOcean',-72,29),('Pacific\nOcean',-123,29)]:
            ax.text(lon,lat,txt,transform=PC,ha='center',va='center',fontsize=8,color='#617c8b',zorder=5)
    else:
        # Exact 0.5-degree footprints, not enlarged point markers.
        values=np.full((360,720),np.nan)
        iy=np.rint((gw.lat+89.75)/.5).astype(int);ix=np.rint((gw.lon+179.75)/.5).astype(int)
        values[iy,ix]=gw[name]
        im=ax.pcolormesh(np.linspace(-180,180,721),np.linspace(-90,90,361),np.ma.masked_invalid(values),cmap='RdBu_r',norm=Normalize(-1,1),transform=PC,zorder=5,rasterized=True)
cax=fig.add_axes([.31,.078,.42,.017]);cb=fig.colorbar(im,cax=cax,orientation='horizontal',ticks=[-1,-.5,0,.5,1]);cb.set_label('Groundwater Pearson correlation (PCC)',labelpad=4);cb.outline.set_linewidth(.5)
fig.text(.055,.012,'Background: Natural Earth. Model maps share the same extent and color scale; blank areas have no eligible observations.',fontsize=8,color=MUTED)
save(fig,'Fig11_groundwater_maps_revised')

sp=xr.open_dataset(R/'cache/spatial_snapshots.nc')
w=pd.read_csv(R/'tables/weight_land_source_data.csv')
sm=pd.read_csv(R/'tables/smap_common_grid_correlations.csv')
cases=pd.read_csv(R/'regional_cases/selected_cases.csv').sort_values('rank')
fig=plt.figure(figsize=(16,14),facecolor='white')
fig.text(.035,.975,'Three regions, eight complementary diagnostics',fontsize=22,weight='bold',color=INK)
fig.text(.035,.949,'Location first; credibility, storage fields and external agreement within each region',fontsize=12,color=MUTED)
world=fig.add_axes([.035,.725,.625,.20],projection=ccrs.Robinson())
world.set_global();world.set_facecolor(WATER)
world.add_feature(cf.LAND,facecolor=LAND);world.add_feature(cf.COASTLINE,edgecolor='#8a9ca8',linewidth=.45)
world.add_feature(cf.BORDERS,edgecolor='#bbc5ca',linewidth=.35)
world.spines['geo'].set_edgecolor('#b6c3cc')
colors=['#217c8b','#bf7132','#815687']
names=['Caspian region','China–Mongolia border','Inland Australia']
coords=['45–50°E  /  45–50°N','100–105°E  /  40–45°N','135–140°E  /  25–20°S']
offsets=[(-18,13),(10,14),(3,-19)]
for k,(_,r) in enumerate(cases.iterrows()):
    lon=(r.lon_min+r.lon_max)/2;lat=(r.lat_min+r.lat_max)/2
    world.add_patch(Rectangle((r.lon_min,r.lat_min),5,5,fill=False,edgecolor=colors[k],lw=2,transform=PC,zorder=6))
    dx,dy=offsets[k]
    xy=world.projection.transform_point(lon,lat,PC);xytext=world.projection.transform_point(lon+dx,lat+dy,PC)
    world.annotate(str(k+1),xy=xy,xytext=xytext,ha='center',va='center',fontsize=12,weight='bold',color='white',bbox=dict(boxstyle='circle,pad=.3',fc=colors[k],ec='white',lw=1),arrowprops=dict(arrowstyle='-',color=colors[k],lw=1.3),zorder=8)
fig.text(.70,.90,'READING THE REGIONAL GROUPS',fontsize=11,weight='bold',color=INK)
fig.text(.70,.878,'α / β   Frozen numerical / structural credibility\nTWSA   January 2020; five fields per region\nPCC     M4 vs SMAP; 2020–2022',fontsize=10,color=MUTED,linespacing=1.8,va='top')
for rect,cmap,norm,label,ts in [([.705,.795,.23,.009],'viridis',Normalize(0,1),'Credibility weight',[0,.5,1]),([.705,.748,.23,.009],'RdBu_r',Normalize(-1,1),'SMAP PCC',[-1,0,1])]:
    cb=fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),cax=fig.add_axes(rect),orientation='horizontal',ticks=ts)
    cb.ax.tick_params(labelsize=8,length=2,pad=2);cb.set_label(label,fontsize=9,labelpad=2);cb.outline.set_linewidth(.4)

audit=[]
for k,(_,r) in enumerate(cases.iterrows()):
    left=.035+k*.322;cw=.286;bottom=.075;top=.695
    fig.add_artist(FancyBboxPatch((left-.007,bottom-.008),cw+.014,top-bottom+.008,boxstyle='round,pad=.006,rounding_size=.009',transform=fig.transFigure,facecolor='#f8fafb',edgecolor='#dbe3e8',linewidth=.8,zorder=-2))
    fig.text(left,.674,f'{k+1}  {names[k]}',fontsize=16,weight='bold',color=colors[k])
    fig.text(left,.650,coords[k],fontsize=10,color=MUTED)
    fig.text(left,.629,f'Rank {int(r["rank"])} of 245  •  {int(r.n_valid_grids)} common SMAP cells',fontsize=9,color=MUTED)
    extent=[r.lon_min,r.lon_max,r.lat_min,r.lat_max]
    cut=sp.sel(lon=slice(r.lon_min,r.lon_max),lat=slice(r.lat_min,r.lat_max))
    wfilt=w[w.lon.between(r.lon_min,r.lon_max)&w.lat.between(r.lat_min,r.lat_max)]
    sf=sm[sm.lon.between(r.lon_min,r.lon_max)&sm.lat.between(r.lat_min,r.lat_max)]
    lo=min(float(cut[m].min()) for m in ['JPL','WGHM','M0','M1','M4']);hi=max(float(cut[m].max()) for m in ['JPL','WGHM','M0','M1','M4'])
    lim=max(abs(lo),abs(hi));norm=SymLogNorm(linthresh=lim/1000,vmin=-lim,vmax=lim)
    for j,key in enumerate(['alpha','beta','JPL','WGHM','M0','M1','M4','PCC']):
        row=j//2;col=j%2
        ax=fig.add_axes([left+col*.151,.487-row*.127,.133,.103],projection=PC)
        background(ax,extent)
        ax.set_aspect('auto')
        if key in ['alpha','beta','PCC']:
            f=sf if key=='PCC' else wfilt
            a=np.full((10,10),np.nan)
            iy=np.floor((f.lat-r.lat_min)/.5).astype(int);ix=np.floor((f.lon-r.lon_min)/.5).astype(int)
            a[iy,ix]=f.M4 if key=='PCC' else f[key]
            nm=Normalize(-1,1) if key=='PCC' else Normalize(0,1);cm='RdBu_r' if key=='PCC' else 'viridis'
        else: a=cut[key].values;nm=norm;cm='RdBu_r'
        ax.pcolormesh(np.linspace(extent[0],extent[1],11),np.linspace(extent[2],extent[3],11),np.ma.masked_invalid(a),cmap=cm,norm=nm,transform=PC,zorder=2,rasterized=True)
        label={'alpha':'α  Numerical credibility','beta':'β  Structural credibility','PCC':'M4 × SMAP  |  PCC'}.get(key,f'{key}  |  TWSA')
        ax.set_title(label,fontsize=10,loc='left',pad=5,color=INK,weight='bold' if key=='M4' else 'normal')
        ticks(ax,[extent[0],extent[1]],[extent[2],extent[3]],7)
        ax.tick_params(labelbottom=(row==3),labelleft=(col==0))
    cax=fig.add_axes([left+.015,.082,cw-.030,.008])
    cb=fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap='RdBu_r'),cax=cax,orientation='horizontal',ticks=[-lim,0,lim])
    cb.ax.set_xticklabels([f'{-lim:,.0f}','0',f'{lim:,.0f}']);cb.ax.tick_params(labelsize=8,pad=2,length=2)
    cb.set_label('TWSA (mm)  •  one shared scale for the five fields',fontsize=8,labelpad=2);cb.outline.set_linewidth(.4)
    audit.append({'region':names[k],'rank':int(r['rank']),'n_smap':len(sf),'TWSA_limits':[-lim,lim],'TWSA_scale':'symlog','linear_threshold':lim/1000,'panels':['alpha','beta','JPL','WGHM','M0','M1','M4','PCC']})
fig.text(.035,.022,'Exploratory cases selected from the complete regional ranking. Missing data remain unfilled; geographic lines are reference overlays.',fontsize=9,color=MUTED)
fig.text(.035,.007,'TWSA uses a symmetric-log scale (linear threshold = maximum absolute value / 1,000); scales differ between regions. Background: Natural Earth.',fontsize=9,color=MUTED)
save(fig,'FigS15_S17_regional_diagnostics_combined')
(O/'design_audit.json').write_text(json.dumps({'Fig11':{'panels':['a well coverage','b M0 PCC','c M1 PCC','d M4 PCC'],'wells':len(wf),'cells':len(gw),'extent':[-126,-66,23,51],'PCC_limits':[-1,1]},'combined':audit},indent=2),encoding='utf-8')
(O/'图注与文件说明.md').write_text('''# 图 11：地下水观测覆盖与模型一致性
a：763 口共同有效井的分布；b–d：M0、M1、M4 的地下水分量匹配 PCC，共 122 个 0.5° 网格。各候选 TWSA 统一扣除 WGHM 非地下水分量，先逐井计算 PCC，再在网格内取中位数；每井至少 24 个共同有效月份。四幅地图采用相同范围，模型图共用 −1 至 1 色标。海岸线、国界、州界和湖泊来自 Natural Earth；空白处无合格观测。原图 e、f 已删除。

# 合并图 S15–S17：三个区域的地理位置与诊断
上方全球定位图以 1–3 标记三个区域，与下方同色区域组对应。每组从上至下展示数值可信度 α、结构可信度 β，JPL/WGHM、M0/M1、M4 的 2020 年 1 月 TWSA，以及 M4 与 SMAP 的测试期 PCC，完整保留各原图 a–h，移除 i–l。可信度及 PCC 分别全图共用 [0,1]、[−1,1] 线性色标；每个区域的五幅 TWSA 场共用完整范围的对称对数色标，线性阈值为该区域最大绝对值的 1/1000，不同区域不共用 TWSA 色限。自然地理背景来自 Natural Earth。案例为事后探索，保留原始排名与共同样本数。

输出包含 600 dpi PNG、PDF、SVG 及预览图。源结果、旧图和论文文件均未覆盖。
''',encoding='utf-8')
