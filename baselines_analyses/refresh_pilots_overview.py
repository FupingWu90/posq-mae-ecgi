"""Refresh the test-cohort fields of results/pilots_overview.json (the source of Supplementary Table tab:pilots) from the
current per-recording CSVs (generated/percase_candidate/<ID>_vc16_<split>_<layout>.csv), so that the table follows
the de-duplication rule of 2026-10-21 (one internal test recording per content key).  Fields: ati/atik (internal AT,
template / k-means), ate/atek (external AT), rti/rte (RT, template), hspi/hspe (HSP CC, template).  Rows without CSVs
(the frozen v1 run) are left unchanged and listed.  Aggregates only."""
import os, sys, json, time
V2 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(V2, 'training_sweeps')); sys.path.insert(0, os.path.join(V2, 'bsp_reconstruction'))
from pilots import percase_means
G = os.path.join(V2, 'baselines_analyses', 'generated', 'percase_candidate'); P = os.path.join(V2, 'results', 'pilots_overview.json')
rows = json.load(open(P)); n_upd = 0; untouched = []
for r in rows:
    pid = r['id']; changed = False
    for split, lay, fields in (('test_internal', 'template', {'ati': 'AT_mae', 'rti': 'RT_mae', 'hspi': 'hsp_cc'}), ('test_internal', 'kmeans', {'atik': 'AT_mae'}),
                               ('test_external', 'template', {'ate': 'AT_mae', 'rte': 'RT_mae', 'hspe': 'hsp_cc'}), ('test_external', 'kmeans', {'atek': 'AT_mae'})):
        pm = percase_means(os.path.join(G, f'{pid}_vc16_{split}_{lay}.csv'))
        if not pm:
            continue
        for f, k in fields.items():
            if pm.get(k) is not None and r.get(f) is not None and abs(float(pm[k]) - float(r[f])) > 1e-9:
                r[f] = float(pm[k]); changed = True
    if changed:
        r['test_refreshed'] = time.strftime('%Y-%m-%d'); n_upd += 1
    else:
        untouched.append(pid)
json.dump(rows, open(P, 'w'), indent=1)
print(f'pilots_overview.json: {n_upd} rows refreshed; unchanged: {untouched}')
