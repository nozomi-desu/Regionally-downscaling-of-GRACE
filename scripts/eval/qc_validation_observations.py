"""Observation-only QC, independent of candidate predictions."""
from pathlib import Path
import json,collections,datetime
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]/'validation_extension_20260908'
def wells(output_dir=None):
    """Apply the frozen observation QC; optionally write to an isolated audit directory."""
    output_dir=Path(output_dir) if output_dir is not None else ROOT/'groundwater'
    output_dir.mkdir(parents=True,exist_ok=True)
    manifests=json.loads((ROOT/'groundwater/download_manifest.json').read_text())
    assert all(x['complete'] for x in manifests),'Do not silently evaluate incomplete spatial download'
    rows=[];seen=set();reject=collections.Counter();quals=collections.Counter()
    for p in sorted((ROOT/'groundwater/pages').glob('*.json')):
        for f in json.loads(p.read_text()).get('features',[]):
            if f['id'] in seen:continue
            seen.add(f['id']);s=f['properties'];q=s.get('qualifier') or []
            quals.update(map(str,q))
            if s.get('parameter_code')!='72019':reject['parameter']+=1;continue
            if s.get('approval_status')!='Approved':reject['not_approved']+=1;continue
            if 'Static' not in q:reject['not_explicitly_static']+=1;continue
            if set(q)&{'Pumping','NoMeasurement','Dry','Obstructed','GWSWAffected','ForeignSubstance','Flowing','Above','Below','Frozen','GWTideAffected'}:
                reject['affected_or_censored']+=1;continue
            unit=s.get('unit_of_measure');factor={'ft':.3048,'m':1}.get(unit)
            if factor is None:reject['unit']+=1;continue
            try:
                v=float(s['value'])*factor;lon,lat=f['geometry']['coordinates'][:2]
                when=datetime.datetime.fromisoformat(s['time'].replace('Z','+00:00')).astimezone(datetime.timezone.utc)
                assert np.isfinite(v) and 2020<=when.year<=2022
            except Exception:reject['invalid']+=1;continue
            rows.append({'site_id':s['monitoring_location_id'],'time':when.strftime('%Y-%m-01'),'water_level_proxy_m':-v,'lon':lon,'lat':lat})
    df=pd.DataFrame(rows).groupby(['site_id','time'],as_index=False).agg(water_level_proxy_m=('water_level_proxy_m','median'),lon=('lon','median'),lat=('lat','median'))
    df.to_csv(output_dir/'well_monthly.csv',index=False)
    counts=df.groupby('site_id').size()
    report={'unique_measurements':len(seen),'retained_measurements':len(rows),'retained_wells':len(counts),'wells_at_least_24_months':int((counts>=24).sum()),'wells_at_least_12_months':int((counts>=12).sum()),'rejections':dict(reject),'qualifiers':dict(quals),'qc':'Approved and explicitly Static; ft/m only; median within site-month; no candidate-dependent exclusions'}
    (output_dir/'qc.json').write_text(json.dumps(report,indent=2));print('WELLS',report,flush=True)
def gnss():
    rows=json.loads((ROOT/'gnss/download_manifest.json').read_text());monthly=[];audits=[]
    steps=collections.defaultdict(set)
    for l in (ROOT/'sources/steps.txt').read_text().splitlines():
        a=l.split()
        if len(a)<2:continue
        try:dt=datetime.datetime.strptime(a[1],'%y%b%d')
        except ValueError:continue
        if 2020<=dt.year<=2022:steps[a[0]].add(dt.strftime('%Y-%m-%d'))
    for row in rows:
        audit={k:row[k] for k in ['station','lat','lon','block']};station=row['station']
        if 'error' in row:audit['exclude']='download_failed';audits.append(audit);continue
        df=pd.read_csv(ROOT/row['path'],sep=r'\s+')
        required=['__MJD','____up(m)','sig_u(m)','__uu_ntal','__uu_ntol']
        if not set(required)<=set(df.columns):audit['exclude']='unknown_schema';audits.append(audit);continue
        df['date']=pd.to_datetime(df['__MJD'],unit='D',origin='1858-11-17')
        df=df[df.date.between('2020-01-01','2022-12-31')].copy()
        good=np.isfinite(df[required]).all(axis=1)&df['sig_u(m)'].between(0,.020,inclusive='right')
        df=df[good].drop_duplicates('__MJD').copy()
        # No MASC, HYDL or lake predictions removed; corrections remain model-dependent.
        df['up_ntal_ntol_removed_mm']=(df['____up(m)']-df['__uu_ntal']-df['__uu_ntol'])*1000
        df['month']=df.date.dt.strftime('%Y-%m-01')
        m=df.groupby('month').agg(up_mm=('up_ntal_ntol_removed_mm','median'),days=('__MJD','size'))
        m=m[m.days>=15];audit['valid_months']=len(m);audit['test_steps']=sorted(steps[station])
        audit['eligible_for_forward_model']=len(m)>=24 and not steps[station]
        audit['exclude']='' if audit['eligible_for_forward_model'] else ('listed_test_step' if steps[station] else 'insufficient_months')
        for date,s in m.iterrows():monthly.append({'station':station,'time':date,'up_mm':s.up_mm,'days':int(s.days),'lat':row['lat'],'lon':row['lon'],'eligible':audit['eligible_for_forward_model']})
        audits.append(audit)
    pd.DataFrame(monthly).to_csv(ROOT/'gnss/monthly_qc.csv',index=False)
    (ROOT/'gnss/qc.json').write_text(json.dumps({'stations':audits,'eligible':sum(x.get('eligible_for_forward_model',False) for x in audits),'candidate_performance_computed':False,'remaining':'Reference-frame provenance, loading units cross-check, elastic forward operator and near-field convergence must pass before scientific validation.'},indent=2))
    print('GNSS selected',len(rows),'eligible',sum(x.get('eligible_for_forward_model',False) for x in audits),flush=True)
if __name__=='__main__':wells();gnss()
