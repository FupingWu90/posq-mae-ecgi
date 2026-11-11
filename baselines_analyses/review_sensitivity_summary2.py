"""Summarise the batch-3 reviewer reruns (2026-10-23 afternoon; baselines_analyses/sources_rerun_v2.py --noise / --svd_k / --model_denoise)
into two tables of Supplementary Note 21 and a JSON of aggregates.  Aggregates only are printed.
  tab_sens_noise_v2.tex    noise injected on the active channels (white 20/10/5 dB, EMG-band 10 dB, baseline wander 0 dB):
                           AT / RT MAE of the VC-only inverse, the full-vest inverse and the completion (N = 16) against the
                           commercial reference, mean paired change vs the clean evaluation; BSP CC of the completion vs the
                           clean held-out field
  tab_sens_denoise_v2.tex  denoising variants of the full-vest field: SVD rank 8/16/32, denoise (rank 16) then complete,
                           the completion model as a denoiser (cross-completion of two halves; all electrodes visible and
                           every position queried), against the raw full-vest inverse and the 16-electrode completion
"""
import os, sys, json
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from review_sensitivity_summary import load, paired, stars, COH, V2_GENERATED, assert_v2

NOISE = [('base', 'none (paper evaluation)'), ('noise_white20', 'white, SNR $20$\\,dB'), ('noise_white10', 'white, SNR $10$\\,dB'),
         ('noise_white5', 'white, SNR $5$\\,dB'), ('noise_emg10', 'EMG band ($5$--$45$\\,Hz), SNR $10$\\,dB'), ('noise_bw0', 'baseline wander ($0.2$--$0.8$\\,Hz), SNR $0$\\,dB')]
DEN = [('base', 'src2_allvalid', 'full-vest inverse, raw (reference row)'), ('svd8', 'src5_svd', 'full vest, SVD rank $8$'),
       ('svd16', 'src5_svd', 'full vest, SVD rank $16$'), ('svd32', 'src5_svd', 'full vest, SVD rank $32$'),
       ('mdn', 'src6_mdn', 'full vest re-synthesised by PosQ-MAE (cross-completion of two halves)'),
       ('mdn_qall', 'src6_mdn', 'full vest re-synthesised by PosQ-MAE (all electrodes visible, every position queried)'),
       ('base', 'src4_recon_valid', 'PosQ-MAE from $16$ electrodes (paper)'),
       ('svd16in', 'src4_recon_valid', 'PosQ-MAE from $16$ electrodes of the rank-$16$ field'),
       ('svd16in', 'src1_vc', 'VC-only inverse from $16$ electrodes of the rank-$16$ field')]


def main():
    out = {}
    base = {sp: load('base', sp) for sp, _ in COH}
    # ---------------- noise table
    L = [r'\begin{tabular}{llccccccc}', r'\toprule',
         r' & & \multicolumn{3}{c}{AT MAE (ms)} & \multicolumn{3}{c}{RT MAE (ms)} & BSP CC \\', r'\cmidrule(lr){3-5}\cmidrule(lr){6-8}',
         r'Cohort & Added noise & VC-only & full vest & PosQ-MAE & VC-only & full vest & PosQ-MAE & PosQ-MAE \\']
    for sp, lab in COH:
        L.append(r'\midrule'); first = True
        for tag, name in NOISE:
            df = load(tag, sp)
            if df is None:
                continue
            cells = []
            for met in ('AT', 'RT'):
                for s in ('src1_vc', 'src2_allvalid', 'src4_recon_valid'):
                    sub = df[(df.source == s) & (df.ref == 'commercial')]
                    if len(sub) == 0:
                        cells.append('--'); continue
                    mean = float(np.nanmean(sub[f'{met}_mae'])); cell = f'{mean:.1f}'
                    if tag != 'base' and base[sp] is not None:
                        md, p, n = paired(sub, base[sp][(base[sp].source == s) & (base[sp].ref == 'commercial')], f'{met}_mae')
                        if np.isfinite(md):
                            cell += f' (${md:+.2f}{stars(p)}$)'
                    cells.append(cell); out[f'{lab}|{tag}|{s}|{met}'] = mean
            s4 = df[(df.source == 'src4_recon_valid') & (df.ref == 'commercial')]
            bcc = float(np.nanmean(s4.bsp_cc)) if len(s4) else float('nan'); out[f'{lab}|{tag}|bsp_cc'] = bcc
            cells.append(f'{bcc:.3f}' if np.isfinite(bcc) else '--')
            L.append(f"{lab if first else ''} & {name} & " + ' & '.join(cells) + r' \\'); first = False
    L += [r'\bottomrule', r'\end{tabular}']
    open(assert_v2(os.path.join(V2_GENERATED, 'tab_sens_noise_v2.tex')), 'w').write('\n'.join(L) + '\n')
    # ---------------- denoising table
    L = [r'\begin{tabular}{lcccc}', r'\toprule', r' & \multicolumn{2}{c}{internal} & \multicolumn{2}{c}{external} \\', r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}',
         r'Field passed to the inverse & AT MAE & RT MAE & AT MAE & RT MAE \\', r'\midrule']
    for tag, src, name in DEN:
        cells = []; have = False
        for sp, lab in COH:
            df = load(tag, sp)
            sub = df[(df.source == src) & (df.ref == 'commercial')] if df is not None else None
            if sub is None or len(sub) == 0:
                cells += ['--', '--']; continue
            have = True
            for met in ('AT', 'RT'):
                mean = float(np.nanmean(sub[f'{met}_mae'])); cell = f'{mean:.1f}'
                if not (tag == 'base' and src == 'src2_allvalid') and base[sp] is not None:
                    md, p, n = paired(sub, base[sp][(base[sp].source == 'src2_allvalid') & (base[sp].ref == 'commercial')], f'{met}_mae')
                    if np.isfinite(md):
                        cell += f' (${md:+.2f}{stars(p)}$)'
                cells.append(cell); out[f'{lab}|denoise|{tag}|{src}|{met}'] = mean
        if have:
            L.append(f'{name} & ' + ' & '.join(cells) + r' \\')
    L += [r'\bottomrule', r'\end{tabular}']
    open(assert_v2(os.path.join(V2_GENERATED, 'tab_sens_denoise_v2.tex')), 'w').write('\n'.join(L) + '\n')
    json.dump(out, open(assert_v2(os.path.join(V2_GENERATED, 'review_sensitivity2_v2.json')), 'w'), indent=1)
    for k, v in out.items():
        print(f'{k:55s} {v:7.3f}')
    print('wrote tab_sens_noise_v2.tex, tab_sens_denoise_v2.tex, review_sensitivity2_v2.json')


if __name__ == '__main__':
    main()
