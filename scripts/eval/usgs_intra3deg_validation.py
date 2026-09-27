"""Independent intra-coarse-cell validation; frozen inputs only, no model fitting.

Run from the project root with --refresh-remote to refresh the hash-verified
source bundle, or without it to reproduce from the previously exported bundle.
"""
from pathlib import Path
import argparse
import hashlib
import itertools
import json
import logging
import subprocess
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'validation_rebuild/scripts'))
from evaluation_core import temporal
from scripts.eval.extract_intra3deg_sources import sha

MODELS = [f'M{i}' for i in range(5)]
B = 2000
SEED = 42


def write_json(path, obj):
    """Write portable, human-readable provenance."""
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=lambda x: x.item() if isinstance(x, np.generic) else str(x)), encoding='utf-8')


def audit_observations(out, source):
    """Reproduce existing QC from every manifest-hashed raw API page, without infill."""
    from scripts.eval import qc_validation_observations as qc
    folder = ROOT / 'validation_extension_20260908/groundwater'
    assert sha(folder / 'well_monthly.csv') == source['well_monthly_sha256']
    manifest = json.loads((folder / 'download_manifest.json').read_text())
    expected = {}; raw_sites = set(); units = set(); parameters = set()
    for region in manifest:
        assert region['complete'], 'Incomplete original download'
        for page in region['pages']:
            name = page['path'].replace('\\', '/').split('/')[-1]
            expected[name] = page['sha256']
    assert set(expected) == {p.name for p in (folder / 'pages').glob('*.json')}
    for name, h in expected.items():
        p = folder / 'pages' / name
        assert sha(p) == h, f'Raw USGS page changed: {name}'
        for f in json.loads(p.read_text())['features']:
            props = f['properties']; raw_sites.add(props['monitoring_location_id'])
            units.add(props.get('unit_of_measure')); parameters.add(props.get('parameter_code'))
    assert parameters == {'72019'}, parameters
    qc.wells(output_dir=out / 'observation_qc')
    original = pd.read_csv(folder / 'well_monthly.csv').sort_values(['site_id', 'time']).reset_index(drop=True)
    rebuilt = pd.read_csv(out / 'observation_qc/well_monthly.csv').sort_values(['site_id', 'time']).reset_index(drop=True)
    pd.testing.assert_frame_equal(original, rebuilt, check_exact=False, rtol=1e-13, atol=1e-13)
    report = json.loads((out / 'observation_qc/qc.json').read_text())
    report.update(raw_wells=len(raw_sites), raw_pages=len(expected), raw_page_sha256=expected,
                  parameter_code='72019', original_field='value', raw_units=sorted(units),
                  original_sign='Depth to water below land surface; larger values mean lower water level',
                  authoritative_definition='https://help.waterdata.usgs.gov/parameter_cd?group_cd=PHY',
                  conversion='ft * 0.3048; m unchanged; then negate', sign_flip_in_existing_qc=True,
                  additional_sign_flip=False, interpolation=False, qc_reproduction='matched all monthly records',
                  version='USGS OGC field measurements snapshot 2026-09-08',
                  qc_code_sha256=sha(ROOT / 'scripts/eval/qc_validation_observations.py'))
    write_json(out / 'usgs_sign_and_qc_audit.json', report)
    original['time'] = pd.to_datetime(original.time)
    logging.info('USGS sign: 72019 depth, %s; existing QC already negates; no second flip', units)
    return original, report


def grid_observations(w, bundle, out):
    """Standardize eligible wells over test months, then take monthly grid medians."""
    times = pd.DatetimeIndex(bundle['times']); lat, lon = bundle['lat'], bundle['lon']
    assert np.array_equal(times, pd.date_range('2020-01-01', '2022-12-01', freq='MS'))
    assert w.time.isin(times).all() and not w.duplicated(['site_id', 'time']).any()
    sites = w.groupby('site_id').agg(lat=('lat', 'median'), lon=('lon', 'median'),
                                     n_months=('water_level_proxy_m', 'count'),
                                     sd=('water_level_proxy_m', lambda x: x.std(ddof=0)))
    y = abs(lat[:, None] - sites.lat.values).argmin(0)
    x = abs(lon[:, None] - sites.lon.values).argmin(0)
    sites['fine_cell'] = y * len(lon) + x
    mapping = {int(k): i for i, k in enumerate(bundle['cell_ids'])}
    positions = np.array([mapping[int(k)] for k in sites.fine_cell])
    sites['coarse_cell_id'] = bundle['coarse_ids'][positions]
    sites['land'] = bundle['land'][positions] > 0
    sites['inside_cell'] = (abs(sites.lat.values - lat[y]) <= .25) & (abs(sites.lon.values - lon[x]) <= .25)
    sites['eligible'] = (sites.n_months >= 24) & (sites.sd > 0) & sites.land & sites.inside_cell
    sites['grid_lat'] = lat[y]; sites['grid_lon'] = lon[x]
    sites.to_csv(out / 'well_eligibility.csv', index=True)
    eligible = sites[sites.eligible]
    z = w[w.site_id.isin(eligible.index)].copy()
    g = z.groupby('site_id').water_level_proxy_m
    z['normalized_groundwater_anomaly'] = (z.water_level_proxy_m - g.transform('mean')) / g.transform(lambda v: v.std(ddof=0))
    z = z.merge(eligible[['fine_cell', 'coarse_cell_id', 'grid_lat', 'grid_lon']], left_on='site_id', right_index=True)
    z.to_csv(out / 'well_normalized_timeseries.csv', index=False)
    grouped = z.groupby(['fine_cell', 'coarse_cell_id', 'grid_lat', 'grid_lon', 'time'], as_index=False).agg(
        normalized_groundwater_anomaly=('normalized_groundwater_anomaly', 'median'),
        n_wells_contributing=('site_id', 'nunique'))
    cell = eligible.groupby('fine_cell').agg(n_wells=('grid_lat', 'size'), lat=('grid_lat', 'first'),
        lon=('grid_lon', 'first'), coarse_cell_id=('coarse_cell_id', 'first'))
    # Well eligibility remains >=24 for all N_common sensitivity analyses.
    obs = grouped.pivot(index='time', columns='fine_cell', values='normalized_groundwater_anomaly').reindex(index=times, columns=cell.index)
    counts = grouped.pivot(index='time', columns='fine_cell', values='n_wells_contributing').reindex(index=times, columns=cell.index).fillna(0)
    rows = []
    for cid, s in cell.iterrows():
        for t in times:
            rows.append(dict(fine_cell=int(cid), coarse_cell_id=int(s.coarse_cell_id), lat=s.lat, lon=s.lon,
                month=str(t.date()), normalized_groundwater_anomaly=obs.loc[t, cid],
                n_wells_contributing=int(counts.loc[t, cid]), n_wells_total=int(s.n_wells)))
    pd.DataFrame(rows).to_csv(out / 'well_grid_timeseries.csv', index=False)
    cell.to_csv(out / 'well_grid_metadata.csv')
    return cell, obs.values, sites, positions


def pair_statistics(observed, predictions, good, minimum):
    """One finite-month intersection across all models; joint zero-variance exclusion."""
    common = good & np.isfinite(observed) & np.isfinite(predictions).all(axis=0)
    n = int(common.sum())
    if n < minimum:
        return None, common, 'insufficient_common_months'
    a = observed[common]; v = predictions[:, common]
    if np.std(a) == 0 or np.any(np.std(v, axis=1) == 0):
        return None, common, 'constant_difference_in_observation_or_model'
    r = temporal(v.T, a[:, None], min_months=minimum)['PCC']
    rho = temporal(np.array([rankdata(q) for q in v]).T, rankdata(a)[:, None], min_months=minimum)['PCC']
    sign = (np.sign(v) == np.sign(a)).mean(axis=1)
    return (r, rho, sign), common, ''


def compute_pairs(cell, obs, bundle, out):
    """Enumerate deterministic within-membership pairs; preserve all sensitivities."""
    mapping = {int(k): i for i, k in enumerate(bundle['cell_ids'])}
    pos = np.array([mapping[int(k)] for k in cell.index]); ix = {int(k): j for j, k in enumerate(cell.index)}
    masks = bundle['valid'][:, pos] > 0
    variants = {v: np.stack([bundle[f'{m}_{v}'][:, pos] for m in MODELS]).astype(float) for v in ['corrected', 'raw']}
    rows = []; excluded = []
    for coarse, cells in cell.groupby('coarse_cell_id'):
        ordered = cells.sort_values(['lat', 'lon']).index
        for ci, cj in itertools.combinations(ordered, 2):
            i, j = ix[int(ci)], ix[int(cj)]; observed = obs[:, i] - obs[:, j]
            good = masks[:, i] & masks[:, j]
            # Use the same raw/corrected support, permitting an exact paired sensitivity.
            good &= np.logical_and.reduce([np.isfinite(v[:, :, i]).all(0) & np.isfinite(v[:, :, j]).all(0) for v in variants.values()])
            results = {v: pair_statistics(observed, a[:, :, i] - a[:, :, j], good, 18) for v, a in variants.items()}
            invalid = [reason for _, _, reason in results.values() if reason]
            if invalid:
                excluded.append(dict(coarse_cell_id=int(coarse), fine_cell_i=int(ci), fine_cell_j=int(cj), reason=';'.join(sorted(set(invalid)))))
                continue
            for variant, (metrics, common, _) in results.items():
                n = int(common.sum())
                bi, bj = bundle['beta'][pos[i]], bundle['beta'][pos[j]]
                for m, model in enumerate(MODELS):
                    rows.append(dict(coarse_cell_id=int(coarse), fine_cell_i=int(ci), fine_cell_j=int(cj),
                        lat_i=cell.loc[ci, 'lat'], lon_i=cell.loc[ci, 'lon'], lat_j=cell.loc[cj, 'lat'], lon_j=cell.loc[cj, 'lon'],
                        n_common_months=n, common_months=';'.join(str(t)[:7] for t in bundle['times'][common]),
                        model=model, pearson_r=metrics[0][m], spearman_rho=metrics[1][m], sign_agreement=metrics[2][m],
                        obs_zero_months=int((observed[common] == 0).sum()),
                        beta_i=float(bi), beta_j=float(bj), beta_pair=float((bi + bj) / 2), variant=variant))
    allpairs = pd.DataFrame(rows)
    assert len(allpairs), 'No pairs meet even the supplementary 18-month threshold'
    pd.DataFrame(excluded, columns=['coarse_cell_id', 'fine_cell_i', 'fine_cell_j', 'reason']).to_csv(out / 'excluded_pairs.csv', index=False)
    allpairs.to_csv(out / 'pair_metrics_all_thresholds.csv', index=False)
    main = allpairs[(allpairs.variant == 'corrected') & (allpairs.n_common_months >= 24)]
    main.to_csv(out / 'pair_metrics.csv', index=False)
    return allpairs


def coarse_summary(pairs):
    """Give each coarse cell one median per model, regardless of pair density."""
    records = []
    for (c, m), p in pairs.groupby(['coarse_cell_id', 'model']):
        records.append(dict(coarse_cell_id=int(c), model=m, n_fine_cells=len(set(p.fine_cell_i) | set(p.fine_cell_j)),
            n_pairs=len(p), median_PCC=p.pearson_r.median(), median_Spearman=p.spearman_rho.median(),
            median_sign_agreement=p.sign_agreement.median()))
    df = pd.DataFrame(records)
    if df.empty:
        return df
    for metric in ['PCC', 'Spearman', 'sign_agreement']:
        wide = df.pivot(index='coarse_cell_id', columns='model', values='median_' + metric)
        assert wide.notna().all().all() and list(wide.columns) == MODELS
        for base in MODELS[:-1]:
            df[f'{metric}_M4-{base}_gain'] = df.coarse_cell_id.map(wide.M4 - wide[base])
    return df


def bootstrap_summary(coarse, threshold, variant, block=3):
    """Paired cluster bootstrap with equal coarse-cell weight, B=2000 and seed=42."""
    if coarse.empty:
        return []
    ids = np.sort(coarse.coarse_cell_id.unique()); rng = np.random.default_rng(SEED)
    if block == 3:
        draws = [r for r in rng.integers(len(ids), size=(B, len(ids)))]
        nblocks = len(ids)
    else:
        # Adjacent 2x2 original coarse blocks, aligned to the same array origin.
        y, x = np.divmod(ids, 120); groups = (y // 2) * 60 + x // 2
        units = [np.flatnonzero(groups == k) for k in np.unique(groups)]; nblocks = len(units)
        draws = [np.concatenate([units[k] for k in d]) for d in rng.integers(nblocks, size=(B, nblocks))]
    n_pairs = int(coarse[coarse.model == 'M0'].n_pairs.sum()); rows = []
    def add(metric, comparison, stat, values):
        fun = np.median if stat == 'median' else lambda v: np.mean(v > 0)
        boot = np.array([fun(values[d]) for d in draws])
        lo, hi = np.percentile(boot, [2.5, 97.5]) if nblocks >= 2 else (np.nan, np.nan)
        rows.append(dict(threshold=threshold, variant=variant, block_degrees=block, metric=metric,
            comparison=comparison, statistic=stat, estimate=fun(values), lower_95=lo, upper_95=hi,
            bootstrap_B=B, seed=SEED, n_coarse_cells=len(ids), n_blocks=nblocks, n_pairs=n_pairs))
    for metric in ['PCC', 'Spearman', 'sign_agreement']:
        w = coarse.pivot(index='coarse_cell_id', columns='model', values='median_' + metric).reindex(ids)
        for m in MODELS:
            add(metric, m, 'median', w[m].values)
        for base in MODELS[:-1]:
            d = (w.M4 - w[base]).values
            add(metric, 'M4-' + base, 'median', d)
            add(metric, 'M4-' + base, 'improvement_fraction', d)
    return rows


def beta_diagnostic(pairs, out):
    """Exploratory pair-level beta association, clustered by original coarse cells."""
    p = pairs.query("variant == 'corrected' and n_common_months >= 24")
    wide = p.pivot(index=['coarse_cell_id', 'fine_cell_i', 'fine_cell_j'], columns='model', values='pearson_r')
    b = p[p.model == 'M0'].set_index(['coarse_cell_id', 'fine_cell_i', 'fine_cell_j']).beta_pair.reindex(wide.index)
    table = pd.DataFrame({'beta_pair': b, 'gain': wide.M4 - wide.M0}).reset_index().dropna()
    if table.coarse_cell_id.nunique() < 10 or table.beta_pair.nunique() < 3:
        write_json(out / 'beta_diagnostic.json', {'status': 'insufficient independent coarse-cell support or beta variation'})
        return
    ids = np.sort(table.coarse_cell_id.unique()); units = [table[table.coarse_cell_id == c] for c in ids]
    edges = table.beta_pair.quantile([1/3, 2/3]).values
    if edges[0] == edges[1]:
        write_json(out / 'beta_diagnostic.json', {'status': 'tied beta tertile boundaries; no forced tertiles'})
        return
    table['tertile'] = np.digitize(table.beta_pair, edges, right=True)
    table.to_csv(out / 'beta_pair_diagnostics.csv', index=False)
    rng = np.random.default_rng(SEED); samples = []
    for _ in range(B):
        d = pd.concat([units[k] for k in rng.integers(len(ids), size=len(ids))])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            rho = spearmanr(d.beta_pair, d.gain).statistic
        g = np.digitize(d.beta_pair, edges, right=True)
        samples.append([rho] + [d.loc[g == k, 'gain'].median() for k in range(3)])
    samples = np.array(samples); records = []
    estimates = [spearmanr(table.beta_pair, table.gain).statistic] + [table[table.tertile == k].gain.median() for k in range(3)]
    for j, name in enumerate(['Spearman_beta_gain', 'low', 'middle', 'high']):
        lo, hi = np.nanpercentile(samples[:, j], [2.5, 97.5])
        records.append(dict(statistic=name, estimate=estimates[j], lower_95=lo, upper_95=hi,
            n_coarse_cells=len(ids), n_pairs=len(table) if j == 0 else int((table.tertile == j-1).sum()),
            bootstrap_B=B, seed=SEED))
    pd.DataFrame(records).to_csv(out / 'beta_bootstrap_summary.csv', index=False)
    write_json(out / 'beta_diagnostic.json', dict(status='exploratory', edges=edges.tolist(),
        definition='arithmetic mean of beta_i and beta_j; pair-weighted association; whole-coarse-cell bootstrap'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=ROOT / 'results/usgs_intra3deg_validation')
    p.add_argument('--refresh-remote', action='store_true')
    p.add_argument('--remote-host', default='codex-linux')
    p.add_argument('--no-figures', action='store_true')
    a = p.parse_args(); out = a.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', handlers=[logging.FileHandler(out / 'run.log', mode='a', encoding='utf-8'), logging.StreamHandler()])
    logging.getLogger('fontTools').setLevel(logging.WARNING)
    try:
        run(out, a)
    except Exception:
        logging.exception('Validation stopped; no scientific gate bypassed')
        raise


def run(out, args):
    """Execute source gates, primary analysis, robustness and automatic reporting."""
    bundle_dir = out / 'source_bundle'; bundle_dir.mkdir(exist_ok=True)
    if args.refresh_remote:
        remote = '/home/user02/grace_remote_train'
        subprocess.run(['scp', str(ROOT / 'scripts/eval/extract_intra3deg_sources.py'), f'{args.remote_host}:{remote}/scripts/eval/extract_intra3deg_sources.py'], check=True)
        subprocess.run(['ssh', args.remote_host, '/home/user02/miniconda3/envs/grace/bin/python', f'{remote}/scripts/eval/extract_intra3deg_sources.py', '--root', remote, '--output', f'{remote}/results/usgs_intra3deg_validation/source_bundle'], check=True)
        for name in ['source_manifest.json', 'frozen_well_cell_predictions.npz']:
            subprocess.run(['scp', f'{args.remote_host}:{remote}/results/usgs_intra3deg_validation/source_bundle/{name}', str(bundle_dir / name)], check=True)
    source = json.loads((bundle_dir / 'source_manifest.json').read_text())
    assert sha(bundle_dir / 'frozen_well_cell_predictions.npz') == source['bundle_sha256']
    bundle = np.load(bundle_dir / 'frozen_well_cell_predictions.npz')
    logging.info('Frozen prediction/checkpoint provenance verified; paper M4 = project M5')
    w, qc = audit_observations(out, source)
    cell, obs, sites, _ = grid_observations(w, bundle, out)
    pairs = compute_pairs(cell, obs, bundle, out)
    summaries = []; counts = []
    for v in ['corrected', 'raw']:
        for threshold in [18, 24, 30]:
            selected = pairs[(pairs.variant == v) & (pairs.n_common_months >= threshold)]
            coarse = coarse_summary(selected)
            coarse.to_csv(out / f'coarse_cell_summary_{v}_{threshold}.csv', index=False)
            if v == 'corrected' and threshold == 24:
                coarse.to_csv(out / 'coarse_cell_summary.csv', index=False)
            summaries.extend(bootstrap_summary(coarse, threshold, v))
            if threshold == 24:
                summaries.extend(bootstrap_summary(coarse, threshold, v, block=6))
            used = set(selected.fine_cell_i) | set(selected.fine_cell_j)
            counts.append(dict(variant=v, threshold=threshold, n_coarse_cells=selected.coarse_cell_id.nunique(),
                n_pairs=len(selected) // 5, n_fine_cells=len(used),
                n_wells=int(sites[sites.eligible & sites.fine_cell.isin(used)].shape[0])))
    stats = pd.DataFrame(summaries); stats.to_csv(out / 'bootstrap_summary.csv', index=False)
    pd.DataFrame(counts).to_csv(out / 'sample_counts.csv', index=False)
    main_counts = next(c for c in counts if c['variant'] == 'corrected' and c['threshold'] == 24)
    assert main_counts['n_pairs'] > 0, 'No primary pairs; do not relax the primary threshold'
    if main_counts['n_coarse_cells'] < 10:
        logging.warning('Primary support <10 coarse cells; precision and geographic generalization limited')
    beta_diagnostic(pairs, out)
    subprocess.run([sys.executable, str(ROOT/'tools/verify_intra3deg_results.py'), str(out)], check=True)
    manifest = dict(status='computed', run_utc=pd.Timestamp.now(tz='UTC').isoformat(), source=source,
        usgs_qc=qc, test_month_list=[str(t)[:7] for t in bundle['times']],
        primary_variant='corrected', sensitivity_variant='raw from the identical NetCDF and checkpoint',
        protocol=dict(well_min_months=24, pair_min_months=24, sensitivities=[18,24,30], B=B, seed=SEED,
            standardization_ddof=0, grid_aggregation='median', primary_bootstrap_unit='original 3-degree coarse cell',
            tie_policy='exact sign equality, including zeros', pair_orientation='ascending latitude then longitude',
            zero_variance='exclude pair jointly across all models and both variants; recorded in excluded_pairs.csv',
            common_support='intersection across M0-M4, raw/corrected, original valid mask and both observed cells'),
        sample_counts=dict(raw_wells=qc['raw_wells'], qc_wells=len(sites), wells_ge24=int((sites.n_months>=24).sum()),
            eligible_wells=int(sites.eligible.sum()), well_supported_cells=len(cell),
            candidate_coarse_cells=int((cell.groupby('coarse_cell_id').size()>=2).sum()), primary=main_counts),
        code_sha256={str(p.relative_to(ROOT)): sha(p) for p in [Path(__file__), ROOT/'scripts/eval/plot_usgs_intra3deg.py', ROOT/'scripts/eval/qc_validation_observations.py', ROOT/'validation_rebuild/scripts/evaluation_core.py', ROOT/'tools/verify_intra3deg_results.py']},
        environment=dict(python=sys.version, numpy=np.__version__, pandas=pd.__version__))
    write_json(out / 'validation_manifest.json', manifest)
    from scripts.eval.plot_usgs_intra3deg import write_report, render
    write_report(out, manifest)
    if not args.no_figures:
        render(out)
    outputs = {str(p.relative_to(out)): sha(p) for p in out.rglob('*') if p.is_file() and p.name not in ['output_hashes.json','run.log']}
    write_json(out / 'output_hashes.json', outputs)
    logging.info('COMPLETE %s', main_counts)


if __name__ == '__main__':
    main()
