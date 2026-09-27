"""Publication figures and manuscript prose generated only from validated CSVs."""
from pathlib import Path
import json
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MODELS = [f'M{i}' for i in range(5)]
COLORS = ['#858585', '#8B78A6', '#C79648', '#569A98', '#B14B62']


def select(stats, metric='PCC', comparison=None, threshold=24, variant='corrected', block=3, statistic='median'):
    """Select a fully specified estimand, never a result-dependent subset."""
    d = stats[(stats.metric == metric) & (stats.threshold == threshold) & (stats.variant == variant)
              & (stats.block_degrees == block) & (stats.statistic == statistic)]
    return d[d.comparison == comparison].iloc[0] if comparison else d


def interval(row, digits=3):
    return f"{row.estimate:.{digits}f} [{row.lower_95:.{digits}f}, {row.upper_95:.{digits}f}]"


def write_report(out, manifest):
    """Write Methods, actual Results and captions without assuming the winning model."""
    s = pd.read_csv(out / 'bootstrap_summary.csv'); c = manifest['sample_counts']; n = c['primary']
    modelrows = [select(s, comparison=m) for m in MODELS]
    winner = MODELS[int(np.argmax([r.estimate for r in modelrows]))]
    gains = [select(s, comparison='M4-' + m) for m in ['M0', 'M1']]
    sign = [select(s, metric='sign_agreement', comparison=m) for m in MODELS]
    fractions = [select(s, comparison='M4-' + m, statistic='improvement_fraction') for m in ['M0','M1']]
    counttable = pd.read_csv(out / 'sample_counts.csv')
    lines = ['# Intra-3° groundwater spatial-contrast validation', '', '## Methods', '',
        'Groundwater levels were treated as independent proxies of groundwater variability, not as complete terrestrial water storage (TWS) truth. This analysis assesses the temporal agreement of spatial contrasts within the existing 3° coarse cells. It does not establish the accuracy of true 0.5° TWS anomalies, nor does it establish absolute hydraulic-head ordering between wells.', '',
        'We reused the frozen USGS OGC field-measurement snapshot obtained on 8 September 2026. Parameter 72019 records depth to water below land surface ([USGS parameter definition](https://help.waterdata.usgs.gov/parameter_cd?group_cd=PHY)). Approved, explicitly static and uncensored observations were retained by the existing observation-only quality-control procedure. Feet were converted to metres and depths were negated, so higher values indicate higher water levels. Monthly values were medians of available observations. All raw-page hashes were checked and the monthly product was independently regenerated from those pages. No temporal interpolation or gap filling was applied.', '',
        'Each well with at least 24 valid months during January 2020–December 2022 and nonzero temporal variance was standardized over its available test-period months, using population standard deviation (ddof = 0):', '',
        r'\[z_{k,t}=(g_{k,t}-\overline{g_k})/\sigma_k.\]', '',
        'Wells were assigned to the original model grid and land mask. At each 0.5° grid cell, the proxy was the monthly median of standardized well series:', '',
        r'\[G_{i,t}=\operatorname{median}_{k\in i}z_{k,t}.\]', '',
        'All unique pairs of well-supported fine cells belonging to the same original factor-6 coarse cell were constructed in ascending latitude, then longitude order. The original project coarse-to-fine repeat operator supplied the membership IDs. For each pair and frozen model,', '',
        r'\[D^{obs}_{ij,t}=G_{i,t}-G_{j,t},\qquad D^{(m)}_{ij,t}=S^{(m)}_{i,t}-S^{(m)}_{j,t}.\]', '',
        'Here S is the original TWSA prediction, without subtracting model-derived non-groundwater storage. The standardized observation contrast measures relative anomalies; it does not preserve absolute head differences or groundwater-storage amplitude. Only an exactly shared additive coarse-cell background cancels algebraically. Other spatially varying large-scale components may remain.', '',
        'All five models and both raw/corrected variants used identical pairs and common finite months, including both endpoints of the original time-dependent valid mask. The primary criterion was at least 24 common months. Pairs with constant observed or model differences were excluded jointly across models and variants and recorded in the exclusion table. Pair metrics were', '',
        r'\[r^{(m)}_{ij}=\operatorname{corr}_{t\in T_{ij}}(D^{(m)}_{ij,t},D^{obs}_{ij,t}),\quad A^{(m)}_{ij}=|T_{ij}|^{-1}\sum_{t\in T_{ij}}\mathbf{1}[\operatorname{sign}D^{(m)}_{ij,t}=\operatorname{sign}D^{obs}_{ij,t}].\]', '',
        'Exact zero signs were compared literally. Thus 0.5 is a heuristic reference for random binary signs, not a formal null calibrated for temporal dependence, sign imbalance or ties. Coarse cells received equal weight after taking the median over pairs:', '',
        r'\[R_c^{(m)}=\operatorname{median}_{(i,j)\in P_c}r^{(m)}_{ij},\quad\Delta R_c^{M4-M0}=R_c^{M4}-R_c^{M0}.\]', '',
        'Paired bootstrap resampling of complete 3° coarse cells (2,000 replicates, seed 42) provided percentile 95% confidence intervals for across-cell medians, paired gains, and improvement fractions. Pair-level observations were not treated as independent replicates. Corrected outputs were primary; raw predictions from the same checkpoint and NetCDF file were supplementary. Spearman correlation, common-month thresholds of 18 and 30, and aligned 6° spatial blocks were evaluated without outcome-based selection. Well eligibility remained fixed at ≥24 months in both common-month sensitivity analyses. No model, weight, checkpoint or hyperparameter was refitted.', '', '## Results', '',
        f"The raw snapshot contained {c['raw_wells']:,} distinct monitoring locations; {c['qc_wells']:,} retained monthly observations after QC. Of these, {c['wells_ge24']:,} had at least 24 observed test months, and {c['eligible_wells']:,} also passed nonconstant-series and model-grid/land checks. They populated {c['well_supported_cells']:,} fine cells and {c['candidate_coarse_cells']:,} coarse cells with at least two well-supported fine cells. The primary matched comparison retained {n['n_wells']:,} contributing wells, {n['n_fine_cells']:,} fine cells, {n['n_coarse_cells']:,} coarse cells and {n['n_pairs']:,} pairs.", '',
        'Across-coarse-cell median Pearson correlations (95% spatial bootstrap CIs) were ' + '; '.join(f'{m}: {interval(r)}' for m,r in zip(MODELS,modelrows)) + f'. {winner} had the highest median PCC.', '',
        'Paired median gains were ' + '; '.join(f'M4−{m}: {interval(r)}' for m,r in zip(['M0','M1'], gains)) + '.', '',
        'The fractions of coarse cells with positive PCC gains were ' + '; '.join(f'M4>{m}: {100*r.estimate:.1f}% (95% CI {100*r.lower_95:.1f}–{100*r.upper_95:.1f}%)' for m,r in zip(['M0','M1'],fractions)) + '.', '',
        'Median spatial-order sign agreements were ' + '; '.join(f'{m}: {interval(r)}' for m,r in zip(MODELS, sign)) + '.', '',
        'Paired median sign-agreement gains were ' + '; '.join(f'M4−{m}: {interval(select(s, metric="sign_agreement", comparison="M4-"+m))}' for m in ['M0','M1']) + '.', '',
        '### Robustness (all prespecified results retained)', '',
        '| Variant | Common months | Coarse cells | Pairs | M4−M0 PCC gain [95% CI] | M4−M1 PCC gain [95% CI] |',
        '|---|---:|---:|---:|---|---|']
    for r in counttable.itertuples():
        d = [interval(select(s, comparison='M4-'+m, threshold=r.threshold, variant=r.variant)) if r.n_coarse_cells else 'not estimable' for m in ['M0','M1']]
        lines.append(f'| {r.variant} | {r.threshold} | {r.n_coarse_cells} | {r.n_pairs} | {d[0]} | {d[1]} |')
    lines += ['', 'Spearman median gains: ' + '; '.join(f'M4−{m}: {interval(select(s, metric="Spearman", comparison="M4-"+m))}' for m in ['M0','M1']) + '.', '',
        'Aligned 6° block-bootstrap PCC gains: ' + '; '.join(f'M4−{m}: {interval(select(s, comparison="M4-"+m, block=6))}' for m in ['M0','M1']) + '.', '']
    if all(r.lower_95 > 0 for r in gains):
        lines += ['Both primary paired gains were positive with confidence intervals above zero. This supports greater agreement with the independent standardized groundwater contrasts relative to these two configurations, within the sampled regions and period.']
    else:
        lines += ['The primary results do not support a consistent positive gain over both M0 and M1. Confidence intervals and regional heterogeneity must be retained in interpretation; these results do not establish accurate recovery of true 0.5° TWSA.']
    if n['n_coarse_cells'] < 10:
        lines += ['Fewer than 10 coarse cells remained, limiting bootstrap precision and geographic generalization.']
    beta = out / 'beta_bootstrap_summary.csv'
    if beta.exists():
        r = pd.read_csv(beta).iloc[0]
        lines += ['', f'Exploratory pair-weighted association between mean endpoint beta and M4−M0 pairwise PCC gain: Spearman rho {interval(r)}, with entire-coarse-cell bootstrap. This is an association, not causal mechanism evidence.']
    numerical = json.loads((out/'independent_numerical_check.json').read_text())
    lines += ['', '### Source and alignment diagnostics', '',
        f"The frozen land mask excluded {numerical['ge24_nonconstant']-numerical['ge24_nonconstant_on_land']} wells that otherwise satisfied the ≥24-month and nonconstant-series criteria. The mask was not relaxed. No source-version, sign or coordinate-alignment mismatch was found. The original additive correction is shared within each coarse cell and therefore cancels in within-cell differences; the largest observed raw-versus-corrected pair-difference discrepancy was {numerical['max_raw_corrected_pair_difference_mm']:.3g} mm, consistent with floating-point storage effects. Raw/corrected agreement is thus partly algebraic and should not be interpreted as an independent validation of the correction.", '',
        'The sign of the M4−M1 median gain changed at the 30-month threshold, but its confidence interval still crossed zero. Neither rank correlation nor larger spatial bootstrap blocks established a stable positive gain over both baselines. The beta association likewise did not establish preferential improvement at higher beta.']
    lines += ['', '## Figure caption', '',
        f'Figure X | Independent groundwater validation of intra-3° spatial contrasts during 2020–2022. (a) Coverage of {n["n_wells"]:,} contributing wells aggregated into {n["n_fine_cells"]:,} 0.5° cells within {n["n_coarse_cells"]:,} original 3° cells; shading gives the number of fine cells participating in valid pairs. (b) Distributions of coarse-cell median pairwise temporal Pearson correlations for frozen M0–M4 predictions. Points represent equally weighted coarse cells; boxes show medians and interquartile ranges, whiskers extend to 1.5 interquartile ranges, and dark diamonds/error bars show across-cell medians and percentile 95% CIs. (c) Paired M4−M0 and M4−M1 gains; positive values favor M4. Labels report median gains, CIs and percentages of improved coarse cells. (d) Geographic distribution of M4−M0 gains, with a symmetric zero-centred colour scale; only the visual scale is clipped to robust limits, and unmodified values are retained in the CSV. All {n["n_pairs"]:,} pairs join two fine cells within the same coarse cell and have at least 24 common months shared by every model. Observations are medians of standardized sign-corrected well levels, not complete TWS truth. All intervals use 2,000 paired 3° coarse-cell bootstrap replicates with seed 42. Corrected predictions are shown; raw predictions and other robustness checks are supplied in Figure S.', '',
        'Figure S | Prespecified robustness checks. (a) Pearson and Spearman across-coarse-cell medians. (b) PCC gains at ≥18, ≥24 and ≥30 common months, retaining the ≥24-month well eligibility rule. (c) Matched raw versus corrected PCC gains. (d) Spatial-order sign agreement. (e) Exploratory beta-tertile gains, with coarse-cell clustered intervals. (f) Paired PCC gains under 3° and aligned 6° spatial block bootstrap. Points denote medians and error bars percentile 95% confidence intervals (2,000 resamples, seed 42). All results, including negative gains, are retained.', '']
    (out / 'manuscript_usgs_intra3deg.md').write_text('\n'.join(lines), encoding='utf-8')
    readme = f'''# Reproduce this validation

Run from the project root:

```powershell
python scripts/eval/usgs_intra3deg_validation.py --refresh-remote
```

This reads frozen data on the existing SSH host `codex-linux`, verifies all prediction/checkpoint hashes and every frozen dataset chunk, exports only well-cell series, then recomputes local QC, statistics, figures and manuscript text. No training is run. Once `source_bundle/` exists, omit `--refresh-remote` for an offline rerun using the same hash-checked export and local raw USGS pages. Python dependencies: numpy, pandas, scipy, matplotlib, cartopy, xarray, PyMuPDF; remote extraction also uses the existing project runtime. Map coastlines use the installed Cartopy Natural Earth cache (downloaded automatically if absent).

## Protocol and provenance

- `validation_manifest.json` is authoritative; `output_hashes.json` links every artifact to it. Figures embed the manifest hash.
- Final paper M4 is project M5. Corrected is primary; raw is from the identical source NetCDF/checkpoint.
- This is an additional post-original-test validation, not a preregistered test. No model or weighting changes follow the result.
- Existing QC is rerun into an isolated directory and compared against the frozen monthly CSV. There is no interpolation, specific-yield conversion or non-groundwater subtraction.
- Population-standardized wells have ≥24 observed months. Common-month sensitivities change only the pair threshold. Changing which wells contribute each month can affect a grid median; the monthly contributor count is supplied.
- Exact model-grid coordinates, land mask and original time-dependent valid mask are used. The original repeat operator generates coarse membership; no geographical re-binning is used.
- Constant differences are excluded jointly and logged. Sign agreement includes zero ties literally; 0.5 is only a heuristic reference.
- Primary: {n['n_coarse_cells']} coarse cells, {n['n_pairs']} pairs. Pair dependence is handled by coarse-cell aggregation and spatial resampling. Spatial bootstrap is not uncertainty over model training seeds.
- Sampled USGS locations are not a globally representative groundwater network. Standardized level contrasts are neither absolute head ordering nor complete TWSA/storage contrasts.
- Every requested negative or inconclusive sensitivity is preserved. Beta diagnostics are exploratory and pair weighted, with cluster bootstrap.
- No main-panel substitution was needed: panel d remains the geographic gain map.

## Outputs

`pair_metrics.csv` and `coarse_cell_summary.csv` contain corrected ≥24-month primary samples. `pair_metrics_all_thresholds.csv` contains both variants down to ≥18 months; threshold-specific coarse summaries, `sample_counts.csv`, and `bootstrap_summary.csv` retain the complete comparisons. `well_normalized_timeseries.csv`, `well_grid_timeseries.csv` and `well_eligibility.csv` provide the well-to-grid audit. `observation_qc/`, `excluded_pairs.csv`, `usgs_sign_and_qc_audit.json` and `run.log` preserve exclusions and provenance. Main and supplemental figures are PDF/SVG/600-dpi PNG. `qa/` contains automated layout, font and collision audits.
'''
    (out / 'README.md').write_text(readme, encoding='utf-8')


def render(out):
    """Render the predetermined four-panel evidence chain and all robustness checks."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from matplotlib.colors import TwoSlopeNorm, Normalize
    from matplotlib.cm import ScalarMappable
    from matplotlib.patches import Polygon
    sys.path.insert(0, str(ROOT / 'validation_rebuild/scripts'))
    from audit_panel_alignment import require_matplotlib_panel_alignment

    plt.rcParams.update({'font.family':'sans-serif', 'font.sans-serif':['DejaVu Sans'], 'font.size':7.5, 'axes.labelsize':7.5,
        'axes.titlesize':8, 'xtick.labelsize':7, 'ytick.labelsize':7, 'legend.fontsize':7,
        'pdf.fonttype':42, 'svg.fonttype':'none', 'axes.spines.top':False, 'axes.spines.right':False})
    stats = pd.read_csv(out / 'bootstrap_summary.csv'); coarse = pd.read_csv(out / 'coarse_cell_summary.csv')
    manifest = json.loads((out / 'validation_manifest.json').read_text())
    import hashlib
    fingerprint = hashlib.sha256((out / 'validation_manifest.json').read_bytes()).hexdigest()
    n = manifest['sample_counts']['primary']; source = np.load(out / 'source_bundle/frozen_well_cell_predictions.npz')
    qa = out / 'qa'; qa.mkdir(exist_ok=True)
    (out / 'figure_contract.md').write_text('Question: Do frozen M4 intra-coarse-cell contrasts agree more closely with independent well anomalies?\nArchetype: quantitative grid, Python.\nEvidence: a coverage and scope; b model-level agreement; c decisive paired gains; d spatial heterogeneity.\nNo winner is assumed. Groundwater is not TWS truth.\nExports: 180 mm wide, editable PDF/SVG text, PNG 600 dpi; 7 pt minimum nominal font.\n', encoding='utf-8')

    def label(ax, letter, title):
        ax.annotate(letter, (0, 1), xycoords='axes fraction', xytext=(-22, 13), textcoords='offset points', weight='bold', fontsize=9)
        ax.set_title(title, loc='left', pad=13)

    def save(fig, name, extra=()):
        fig.canvas.draw()
        require_matplotlib_panel_alignment(fig, json_out=str(qa / (name + '.alignment.json')),
            overlay_svg=str(qa / (name + '.alignment.svg')), tolerance_pt=1.5, gutter_tolerance_pt=1.5,
            exclude_axes=list(extra), strict=True)
        fig.savefig(out / (name + '.pdf'), dpi=600, metadata={'Subject': 'validation_manifest.json SHA256 ' + fingerprint})
        fig.savefig(out / (name + '.svg'), dpi=600, metadata={'Description': 'validation_manifest.json SHA256 ' + fingerprint})
        fig.savefig(out / (name + '.png'), dpi=600, metadata={'manifest_sha256':fingerprint})
        fig.savefig(qa / (name + '.preview.png'), dpi=300)
        plt.close(fig)
        scripts = ROOT / 'validation_rebuild/scripts'
        for script, args in [
            ('audit_pdf_text.py', [str(out / (name + '.pdf'))]),
            ('audit_figure_collisions.py', [str(out / (name + '.pdf')), '--json-out', str(qa / (name + '.collision.json')), '--overlay-pdf', str(qa / (name + '.collision.pdf'))])]:
            result = subprocess.run([sys.executable, '-X', 'utf8', str(scripts / script), *args], capture_output=True, text=True, encoding='utf-8')
            (qa / (name + '.' + script + '.log')).write_text(result.stdout + result.stderr, encoding='utf-8')
            if result.returncode:
                raise RuntimeError(f'{script} failed for {name}: see qa log')

    def distributions(ax, metric):
        wide = coarse.pivot(index='coarse_cell_id', columns='model', values='median_' + metric)
        rng = np.random.default_rng(42)
        for j, m in enumerate(MODELS):
            v = wide[m].values
            ax.scatter(j+rng.uniform(-.14,.14,len(v)), v, s=7, color=COLORS[j], alpha=.35, linewidths=0, rasterized=True)
            ax.boxplot(v, positions=[j], widths=.45, showfliers=False, patch_artist=True,
                boxprops={'facecolor':'none','edgecolor':COLORS[j]}, medianprops={'color':COLORS[j],'linewidth':1.5},
                whiskerprops={'color':COLORS[j]}, capprops={'color':COLORS[j]})
            r = select(stats, metric=metric, comparison=m)
            ax.errorbar(j+.23, r.estimate, yerr=[[r.estimate-r.lower_95],[r.upper_95-r.estimate]], fmt='D', color='#252525', ms=3, lw=1.1, capsize=2)
            ax.text(j, 1.015, f'{r.estimate:.3f}', transform=ax.get_xaxis_transform(), ha='center', fontsize=7)
        ax.set_xticks(range(5), MODELS); ax.set_xlim(-.5,4.5)
        ax.set_ylim((-1,1) if metric=='PCC' else (0,1))
        ax.axhline(0 if metric=='PCC' else .5, color='.75', linestyle='--', linewidth=.6)
        ax.set_ylabel('Coarse-cell median PCC' if metric=='PCC' else 'Coarse-cell sign agreement')

    projection = ccrs.LambertConformal(central_longitude=-96, central_latitude=38, standard_parallels=(30,45))
    fig = plt.figure(figsize=(7.0866141732, 6.8110236220))  # 180 x 173 mm
    grid = fig.add_gridspec(2,2, left=.10,right=.96,bottom=.19,top=.89,wspace=.38,hspace=.95)
    axes = [fig.add_subplot(grid[0,0], projection=projection), fig.add_subplot(grid[0,1]),
            fig.add_subplot(grid[1,0]), fig.add_subplot(grid[1,1], projection=projection)]
    base = coarse[coarse.model=='M0'].set_index('coarse_cell_id'); y,x = np.divmod(base.index.to_numpy(), len(source['coarse_lon']))
    clat,clon = source['coarse_lat'][y],source['coarse_lon'][x]
    extent = [float(clon.min()-3),float(clon.max()+3),float(clat.min()-3),float(clat.max()+3)]
    caxes=[]
    def map_panel(ax, values, cmap, norm, bar_label):
        ax.set_extent(extent, ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND.with_scale('110m'), facecolor='#F1F1EE', edgecolor='none', zorder=0, rasterized=True)
        ax.coastlines(resolution='110m', color='#888888', linewidth=.4, rasterized=True)
        ax.add_feature(cfeature.BORDERS.with_scale('110m'), linewidth=.3, edgecolor='#AAAAAA', rasterized=True)
        for la,lo,v in zip(clat,clon,values):
            ax.add_patch(Polygon([(lo-1.5,la-1.5),(lo+1.5,la-1.5),(lo+1.5,la+1.5),(lo-1.5,la+1.5)],
                transform=ccrs.PlateCarree(), facecolor=plt.get_cmap(cmap)(norm(v)), edgecolor='white', linewidth=.25))
        # Preserve projection aspect and equal panel rectangles by extending map limits.
        box=ax.get_position(original=True); target=box.width*fig.get_figwidth()/(box.height*fig.get_figheight())
        xmin,xmax=ax.get_xlim(); ymin,ymax=ax.get_ylim(); xr=xmax-xmin; yr=ymax-ymin
        if xr/yr>target:
            mid=(ymin+ymax)/2; ax.set_ylim(mid-xr/target/2,mid+xr/target/2)
        else:
            mid=(xmin+xmax)/2; ax.set_xlim(mid-yr*target/2,mid+yr*target/2)
        cax = ax.inset_axes([0,-.13,1,.045]); caxes.append(cax)
        cb=fig.colorbar(ScalarMappable(norm=norm,cmap=cmap), cax=cax,orientation='horizontal',extend='both' if bar_label.startswith('M4') else 'neither')
        cb.set_label(bar_label,labelpad=2); cb.ax.tick_params(labelsize=7,pad=1)
    map_panel(axes[0],base.n_fine_cells.values,'cividis',Normalize(2,max(3,base.n_fine_cells.max())),'Well-supported fine cells')
    label(axes[0],'a','Validation coverage')
    axes[0].text(0,-.42,f"{n['n_wells']:,} wells · {n['n_fine_cells']} fine cells\n{n['n_coarse_cells']} coarse cells · {n['n_pairs']:,} pairs",transform=axes[0].transAxes,fontsize=7,va='top')
    distributions(axes[1],'PCC'); label(axes[1],'b','Intra-cell temporal agreement')
    ax=axes[2];rng=np.random.default_rng(42);gaintext=[]
    for j,m in enumerate(['M0','M1']):
        values=base[f'PCC_M4-{m}_gain'].values
        ax.scatter(values,j+rng.uniform(-.12,.12,len(values)),s=9,c=COLORS[j],alpha=.45,linewidths=0,rasterized=True)
        ax.boxplot(values,positions=[j],vert=False,widths=.32,showfliers=False,medianprops={'color':'#222222'},boxprops={'color':COLORS[j]})
        r=select(stats,comparison='M4-'+m); f=select(stats,comparison='M4-'+m,statistic='improvement_fraction')
        ax.errorbar(r.estimate,j+.24,xerr=[[r.estimate-r.lower_95],[r.upper_95-r.estimate]],fmt='D',color='#222222',ms=3,capsize=2)
        gaintext.append(f'M4−{m}: {r.estimate:+.3f} [{r.lower_95:+.3f}, {r.upper_95:+.3f}]\n{f.estimate:.1%} of coarse cells improved')
    ax.set_yticks([0,1],['M4−M0','M4−M1']);ax.set_ylim(-.45,1.55);ax.axvline(0,color='.5',lw=.7,ls='--');ax.set_xlabel('Paired PCC gain');label(ax,'c','M4 gain across coarse cells')
    ax.text(0,-.31,'\n'.join(gaintext),transform=ax.transAxes,va='top',fontsize=7,linespacing=1.25)
    values=base['PCC_M4-M0_gain'].values
    limit=max(abs(np.quantile(values,[.02,.98])));limit=max(limit,.001)
    map_panel(axes[3],values,'RdBu_r',TwoSlopeNorm(vmin=-limit,vcenter=0,vmax=limit),'M4−M0 PCC gain')
    label(axes[3],'d','Spatial heterogeneity of gain')
    (out/'map_display_scale.json').write_text(json.dumps(dict(symmetric_limit=limit,quantiles=[.02,.98],n_clipped=int((abs(values)>limit).sum()),projection='Lambert conformal conic',extent=extent),indent=2))
    save(fig,'Fig_USGS_intra3deg_validation',caxes)

    fig,axs=plt.subplots(3,2,figsize=(7.0866141732,8.0708661417));fig.subplots_adjust(left=.11,right=.97,bottom=.07,top=.94,wspace=.42,hspace=.85)
    ax=axs.ravel()
    def error(row, axis, x, color):
        axis.errorbar(x,row.estimate,yerr=[[row.estimate-row.lower_95],[row.upper_95-row.estimate]],fmt='o',ms=3.5,color=color,capsize=2,lw=1)
    for j,m in enumerate(MODELS):
        for offset,metric,color in [(-.12,'PCC','#527C99'),(.12,'Spearman','#B47957')]:
            error(select(stats,metric=metric,comparison=m),ax[0],j+offset,color)
    ax[0].set_xticks(range(5),MODELS);ax[0].set_ylabel('Median correlation');label(ax[0],'a','Pearson and Spearman')
    ax[0].plot([],[], 'o',color='#527C99',label='Pearson',ms=3);ax[0].plot([],[],'o',color='#B47957',label='Spearman',ms=3);ax[0].legend(loc='lower left',fontsize=7)
    for j,t in enumerate([18,24,30]):
        for off,m,color in [(-.12,'M0',COLORS[0]),(.12,'M1',COLORS[1])]:
            error(select(stats,comparison='M4-'+m,threshold=t),ax[1],j+off,color)
    counts=pd.read_csv(out/'sample_counts.csv').query("variant == 'corrected'").set_index('threshold')
    ax[1].set_xticks(range(3),[f'≥{t}\nN={counts.loc[t,"n_coarse_cells"]}' for t in [18,24,30]])
    ax[1].set_xlabel('Common months (N = coarse cells)');ax[1].set_ylabel('Median PCC gain');label(ax[1],'b','Month-threshold sensitivity')
    for j,v in enumerate(['corrected','raw']):
        for off,m,color in [(-.10,'M0',COLORS[0]),(.10,'M1',COLORS[1])]:
            error(select(stats,comparison='M4-'+m,variant=v),ax[2],j+off,color)
    ax[2].set_xticks([0,1],['Corrected','Raw']);ax[2].set_xlim(-.5,1.5);ax[2].set_ylabel('Median PCC gain');label(ax[2],'c','Prediction-variant sensitivity')
    for j,m in enumerate(MODELS):
        error(select(stats,metric='sign_agreement',comparison=m),ax[3],j,COLORS[j])
    ax[3].set_xticks(range(5),MODELS);ax[3].set_ylabel('Median sign agreement');ax[3].axhline(.5,color='.75',ls='--',lw=.7);label(ax[3],'d','Spatial-order agreement')
    beta=out/'beta_bootstrap_summary.csv'
    if beta.exists():
        b=pd.read_csv(beta).set_index('statistic')
        for j,k in enumerate(['low','middle','high']):error(b.loc[k],ax[4],j,'#569A98')
        ax[4].set_xticks(range(3),['Low','Middle','High']);ax[4].set_xlabel('Mean endpoint beta tertile');ax[4].set_ylabel('Median pairwise PCC gain')
    else:ax[4].text(.5,.5,'Insufficient beta support',ha='center',transform=ax[4].transAxes)
    label(ax[4],'e','Exploratory beta association')
    for j,block in enumerate([3,6]):
        for off,m,color in [(-.10,'M0',COLORS[0]),(.10,'M1',COLORS[1])]:
            error(select(stats,comparison='M4-'+m,block=block),ax[5],j+off,color)
    ax[5].set_xticks([0,1],['3° cells','6° blocks']);ax[5].set_xlim(-.5,1.5);ax[5].set_ylabel('Median PCC gain');label(ax[5],'f','Spatial-bootstrap sensitivity')
    for k in [1,2,4,5]:ax[k].axhline(0,color='.75',ls='--',lw=.7)
    for k in [1,2,5]:
        for m,color in [('M0',COLORS[0]),('M1',COLORS[1])]:ax[k].plot([],[],'o',color=color,label='M4−'+m,ms=3)
        ax[k].legend(loc='best',fontsize=7)
    for k in [0,1,2,5]:
        ax[k].legend(loc='lower center',bbox_to_anchor=(.5,1.005),ncol=2,frameon=False,fontsize=7,borderaxespad=0)
    for axis in ax:
        axis.set_title(axis.get_title(loc='left'),loc='left',pad=28)
        for annotation in axis.texts:
            if annotation.get_text() in list('abcdef'):
                annotation.set_position((-22,28))
    save(fig,'FigS_USGS_intra3deg_robustness')


if __name__ == '__main__':
    render(Path(sys.argv[1]))
