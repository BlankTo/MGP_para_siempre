import json
import sys
from pathlib import Path

sys.path[0] = str(Path(__file__).resolve().parents[1])

import numpy as np

from config import DEFAULT
from mgp.analysis import capacity, cli, data, groups, single, timing
from mgp.analysis.conditions import labels_table, noise_table, select
from mgp.analysis.datasets import datasets_table, training_table
from mgp.analysis.stats import STRATS, design_rows
from mgp.progress import log

STUDIES = ('study1', 'study2', 'study3', 'study4', 'study5', 'study6', 'study7', 'study8')
INPUTS = {study: DEFAULT for study in STUDIES}
WHERE = DEFAULT
REFERENCE = DEFAULT
OUTPUT = DEFAULT
GROUP_TARGET = DEFAULT
TIME = DEFAULT
NOVELTY_POOLS = DEFAULT
CAPACITY_POOLS = DEFAULT
FINAL_ARCHIVE_MAX_SIZE = DEFAULT
OPTIONS = ('GROUP_TARGET', 'TIME', 'NOVELTY_POOLS', 'CAPACITY_POOLS', 'FINAL_ARCHIVE_MAX_SIZE')


def span(values):
    v = [x for x in values if x is not None and np.isfinite(x)]
    return [float(min(v)), float(max(v))] if v else None


def single_numbers(t, st, best, ratio, rows):
    row = rows.index
    ba = t['BA']['values']
    out = {'g1_numerical_reference': st['G1']['ref_num'], 'g1_numerical_standard': st['G1']['all_num'], 'g1_symbolic_reference': st['G1']['ref_sym'],
           'g1_symbolic_standard': st['G1']['all_sym'], 'numerical_standard_N1_G2_N2': {k: st[k]['all_num'] for k in ('N1', 'G2', 'N2')},
           'numerical_standard_N3_N4_N12': {k: st[k]['all_num'] for k in ('N3', 'N4', 'N12')},
           'im_component_sets': [ba[0]['cooccurrence'], ba[row(('im', 0.5))]['cooccurrence']],
           'im_high_order': [ba[0]['highpower'], ba[row(('im', 0.5))]['highpower']],
           'revisit_composites': [ba[0]['composite'], ba[row(('strategy', 'revisit_ic'))]['composite']],
           'recovery_reference_vs_revisit': [t['recovery']['values'][0]['num'], t['recovery']['values'][row(('strategy', 'revisit_ic'))]['num']],
           'best_recovery': best['num'], 'best_symbolic': best['sym_pct'], 'best_bba_composite': best['BBA_composite']}
    im = [f'{s}, IM 0.5' for s in ('Standard', 'Revisit IC')]
    out['ratio_N4_constructive_im'] = {'components': span(ratio['nguyen_4|component|Constructive, IM 0.5']),
                                       'component_sets': span(ratio['nguyen_4|cooccurrence|Constructive, IM 0.5'])}
    out['ratio_N4_others_max'] = {'components': max(max(ratio[f'nguyen_4|component|{k}']) for k in im + [f'{s}, IM 0' for s in ('Standard', 'Revisit IC')]),
                                  'component_sets': max(max(ratio[f'nguyen_4|cooccurrence|{k}']) for k in im + [f'{s}, IM 0' for s in ('Standard', 'Revisit IC')])}
    return out


def group_numbers(lists, s7, growth, growth_im, s7growth):
    at = lambda res, s, an, g, i: res['values'][f'{s}|{an}|{g}'][i]
    last = len(growth['n']) - 1
    out = {'targets_recovered_by_groups': sorted({x for g in lists for x in g['num'] + g['sym']}),
           'n12_groups_5000': {r['strategy']: [r['groups_num'], r['groups_sym'], r['groups']] for r in s7 if r['target'] == 'N12'},
           'components_and_sets_after_last': span([at(growth, s, 'BBA', g, last) for s in STRATS for g in ('component', 'cooccurrence')]
                                                  + [at(gi, s, 'BBA', g, last) for gi in growth_im.values() for s in STRATS for g in ('component', 'cooccurrence')]),
           'composites_after_last': {s: at(growth, s, 'BBA', 'composite', last) for s in STRATS},
           'composites_by_im_after_last': {im: {s: at(gi, s, 'BBA', 'composite', last) for s in STRATS} for im, gi in growth_im.items()},
           'composites_single_derivation': {s: at(growth, s, 'BBA', 'composite', 0) for s in STRATS}}
    l5 = len(s7growth['n']) - 1
    out['composites_5000'] = {an: {s: s7growth['values'][f'{s}|{an}|composite'][l5] for s in STRATS} for an in ('BA', 'BBA', 'FBBA')}
    return out


def novelty_numbers(stats):
    return {s: {'decline_per_tenfold': v['decline_per_tenfold'], 'new_last100': v['last100'], 'evaluated_per_derivation': v['evaluated_mean']}
            for s, v in stats.items()}


def capacity_numbers(vals):
    K = [str(k) for k in vals['ks']] + ['inf']
    i = {k: n for n, k in enumerate(K)}
    v = lambda s, an, g, part, k: vals[f'{s}|{an}|{g}|{part}'][i[k]]
    return {'union_K4_BBA': {g: span([v(s, 'BBA', g, 'union', '4') for s in STRATS]) for g in ('component', 'composite')},
            'union_K4_BA': {g: span([v(s, 'BA', g, 'union', '4') for s in STRATS]) for g in ('component', 'composite')},
            'one_K4_BBA_component': span([v(s, 'BBA', 'component', 'one', '4') for s in STRATS]),
            'union_K32_minus_unbounded': {an: {g: [v(s, an, g, 'union', '32') - v(s, an, g, 'union', 'inf') for s in STRATS]
                                               for g in ('component', 'cooccurrence')} for an in ('BA', 'BBA')},
            'fbba_union_composite_K4_vs_unbounded': {s: [v(s, 'FBBA', 'composite', 'union', '4'), v(s, 'FBBA', 'composite', 'union', 'inf')] for s in STRATS},
            'dinf_BBA_component': {k: span([v(s, 'BBA', 'component', 'dinf', k) for s in STRATS]) for k in ('4', '64')},
            'dinf_BA_component_max_up_to_64': max(v(s, 'BA', 'component', 'dinf', k) for s in STRATS for k in K[:K.index('64') + 1])}


def equal_cpu(res):
    B = np.array(res['budgets_s'])
    i100 = int(np.argmin(abs(np.log10(B) - 2)))
    out = {'composites_100s': {s: res[f'{s}|composite'][i100] for s in STRATS}, 'configurations_100s': {s: res[f'{s}|composite|config'][i100] for s in STRATS}}
    by = res['by_config']['10s']
    out['im_gain_10s'] = {}
    for s in STRATS:
        a = [c['composite'] for c in by if c['cfg'][0] == s and c['cfg'][1] == 0.5]
        b = [c['composite'] for c in by if c['cfg'][0] == s and c['cfg'][1] == 0.0]
        out['im_gain_10s'][s] = max(a) - max(b) if a and b else None
    out['max_composites'] = {}
    for s in STRATS:
        cp = res[f'{s}|composite']
        j = max(range(len(cp)), key=lambda k: -1 if cp[k] is None else cp[k])
        out['max_composites'][s] = {'value': cp[j], 'seconds': float(B[j]), 'minutes': float(B[j] / 60)}
    return out


def noise_numbers(out4):
    one = out4['all']['1']
    return {'identical_sigma_1': one['identical'], 'overlap_sigma_1': one['overlap'],
            'built_num_by_level': {k: [v['built'], v['num']] for k, v in out4['all'].items()}}


def labels_numbers(out5):
    return {'built_pure_noise_vs_noise_free': [out5['pure_noise']['built'], out5['clean']['built']],
            'fbba_composite_noise_free_to_pure_noise': [out5['clean']['FBBA_composite'], out5['pure_noise']['FBBA_composite']],
            'bba_composite_noise_free_to_inverted': [out5['clean']['BBA_composite'], out5['inverted']['BBA_composite']],
            'bba_component_noise_free_to_inverted': [out5['clean']['BBA_component'], out5['inverted']['BBA_component']]}


def training_numbers(t8):
    ref = t8['values'][0]
    rest = t8['values'][1:]
    return {'max_recovery_change': max(abs(r[k] - ref[k]) for r in rest for k in ('num', 'sym')),
            'max_component_change': max(abs(r[k] - ref[k]) for r in rest for k in ('BA', 'BBA', 'FBBA')), 'values': t8['values']}


def datasets_numbers(s3):
    return {r['target']: {'recovery_effects': r['rec_effect'], 'coverage_effects': r['cov_effect'], 'configs': r['configs']} for r in s3}


def section(N, name, fn):
    try:
        N[name] = fn()
    except Exception as exc:
        log('numbers', 'skip', section=name, error=type(exc).__name__, detail=repr(str(exc))[:120])


def compute(paths, ref, filters, option):
    target, time, final = option['GROUP_TARGET'], option['TIME'], option['FINAL_ARCHIVE_MAX_SIZE']
    N, lists = {}, None
    p = paths.get
    if p('study1'):
        S1 = single.load(p('study1'), filters)
        rows = design_rows(S1, ref)
        T = single.tables(S1, rows, ref)
        lists = single.group_lists(S1, rows, ref)
        section(N, 'single', lambda: single_numbers(T, single.standard_targets(S1, ref), single.best_configurations(S1), single.ratio_coverage(S1), rows))
        del S1
    if p('study5'):
        section(N, 'labels', lambda: labels_numbers(labels_table(data.where(data.study5(p('study5')), filters))[0]))
    if p('study4'):
        section(N, 'noise', lambda: noise_numbers(noise_table(select(data.study4(p('study4')), filters))[0]))
    if p('study8'):
        section(N, 'training', lambda: training_numbers(training_table(data.where(data.study8(p('study8')), filters))))
    if p('study3'):
        section(N, 'datasets', lambda: datasets_numbers(datasets_table(data.where(data.study3(p('study3')), filters))))
    if p('study6'):
        section(N, 'timing', lambda: timing.numbers(timing.frame(p('study6'), target, time, filters), ref))
    if p('study2') and p('study7') and lists is not None:
        def g():
            R2, _ = groups.frame(p('study2'), target, filters)
            growth = groups.group_growth(p('study2'), target, R2=R2, final=final)
            gim = {str(im): groups.group_growth(p('study2'), target, R2=R2, where=lambda R, im=im: (R.im == im).to_numpy(), final=final)
                   for im in data.levels(R2, 'im')}
            X7 = data.study7(p('study7'))
            return group_numbers(lists, groups.recovery7(X7), growth, gim, groups.coverage_growth7(X7, target, final=final, path7=p('study7')))
        section(N, 'groups', g)
    if p('study2'):
        section(N, 'novelty', lambda: novelty_numbers(groups.novelty(p('study2'), target, option['NOVELTY_POOLS'])[1]))
    if p('study2') and p('study6'):
        section(N, 'equal_cpu', lambda: equal_cpu(groups.group_time(p('study2'), p('study6'), target, filters, time, final=final)))
    if p('study2'):
        section(N, 'capacity', lambda: capacity_numbers(capacity.values(p('study2'), target, option['CAPACITY_POOLS'], filters)))
    return N


def main():
    args = cli.arguments('numbers', globals(), STUDIES, OPTIONS)
    N = compute(args.paths, args.ref, args.filters, args.option)
    N['inputs'] = {k: cli.name_of(v) for k, v in args.paths.items()}
    first = [args.paths[s] for s in STUDIES if s in args.paths]
    cli.write_text(cli.output(args, 'numbers' + cli.suffix(args), *first), '.json', json.dumps(N, indent=1, default=float))


if __name__ == '__main__':
    main()
