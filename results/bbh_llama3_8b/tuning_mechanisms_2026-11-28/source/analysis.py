"""Summarize completed mechanism experiments; Python standard library only."""
import argparse
import csv
import hashlib
import json
import math
import statistics as st
from collections import defaultdict
from pathlib import Path

from design import build_jobs

KEYS = ('stage', 'task', 'shots', 'demo_seed', 'ratio', 'method', 'budget', 'updates', 'param_mode', 'optimizer_seed')
ACCURACY = ('accuracy_pp', 'pred_true_pp', 'true_class_accuracy_pp', 'false_class_accuracy_pp')
DIFFERENCES = ACCURACY + ('gradient_MiB', 'optimizer_state_MiB', 'trainable_numel', 'peak_allocated_GiB', 'adaptation_seconds')


def write_table(folder, name, rows, first_fields):
    fields = list(dict.fromkeys([*first_fields, *(k for row in rows for k in row)]))
    with (folder / name).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: round(v, 8) if isinstance(v, float) else v for k, v in row.items()})


def describe(values):
    n = len(values)
    sd = st.stdev(values) if n > 1 else None
    return dict(n_demo_seeds=n, mean=st.mean(values), sd_across_demo_seeds=sd,
                se_across_demo_seeds=sd / math.sqrt(n) if sd is not None else None)


def scaled(metrics, key, divisor):
    v = metrics.get(key)
    return v / divisor if v is not None else None


def flatten_snapshots(prefix, value):
    if isinstance(value, dict):
        for k, v in value.items():
            yield from flatten_snapshots(f'{prefix}_{k}', v)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        yield prefix, value


def read_runs(root, jobs):
    runs, absolute, validation, profiles, training = {}, [], [], [], []
    for job in jobs:
        path = root / 'runs' / job['id']
        if not (path / 'DONE.json').is_file():
            continue
        stamp = json.loads((path / 'DONE.json').read_text())
        if stamp['job'] != job:
            raise ValueError(f"Completed job does not match this plan in {path}")
        for name in ('metrics.json', 'predictions.jsonl', 'validation.json'):
            if hashlib.sha256((path/name).read_bytes()).hexdigest() != stamp['files'][name]:
                raise ValueError(f"Unexpected file checksum for {path/name}")
        m = json.loads((path / 'metrics.json').read_text())
        rows = [json.loads(row) for row in (path / 'predictions.jsonl').read_text().splitlines() if row.strip()]
        if len(rows) != 200 or [r['row_id'] for r in rows] != job['test_ids']:
            raise ValueError(f"Unexpected test rows or order in {path}")
        if m['n_correct'] != sum(row['exact_match'] for row in rows):
            raise ValueError(f"Scores and metrics disagree in {path}")
        info = {k: job[k] for k in KEYS}
        info['id'] = job['id']
        acc = dict(n_questions=len(rows), n_correct=m['n_correct'], accuracy_pp=100*m['n_correct']/len(rows),
                   invalid_n=sum(r['valid_format'] is False for r in rows))
        if job['task'] == 'boolean_expressions':
            ntrue = sum(r['prediction'] == 'True' for r in rows)
            acc.update(n_pred_true=ntrue, pred_true_pp=100*ntrue/len(rows))
            for label in ('True', 'False'):
                items = [r for r in rows if r['gold'] == label]
                acc[f'n_gold_{label.lower()}'] = len(items)
                acc[f'{label.lower()}_class_accuracy_pp'] = 100*sum(r['exact_match'] for r in items)/len(items)
        acc.update(gradient_MiB=scaled(m, 'gradient_bytes', 2**20),
                   optimizer_state_MiB=scaled(m, 'optimizer_state_bytes', 2**20),
                   trainable_numel=m.get('trainable_numel'), peak_allocated_GiB=scaled(m, 'peak_allocated_bytes', 2**30),
                   adaptation_seconds=m.get('adaptation_seconds'))
        absolute.append({**info, **acc})
        checkpoints = json.loads((path / 'validation.json').read_text())
        for item in checkpoints:
            validation.append({**info, **item})
        for step, loss in enumerate(m.get('loss_per_step', []), 1):
            training.append({**info, 'step': step, 'training_nll': loss})
        profile = {k:v for k,v in m.items() if isinstance(v, (int, float)) and
                   (k.endswith(('_bytes', '_seconds', '_s', '_numel')) or k in ('n_demo_tokens','n_retained_demo_tokens','n_instruction_tokens'))}
        profile.update(flatten_snapshots('snapshot', m.get('memory_snapshots', {})))
        profiles.append({**info, **profile})
        signature = tuple(job[k] for k in KEYS)
        if signature in runs:
            raise ValueError(f'Duplicate condition {signature}')
        runs[signature] = dict(j=job, scores=acc, val=checkpoints,
                               cache_bytes=m.get('retained_cache_bytes'), gold=tuple(r['gold'] for r in rows))
    return runs, absolute, validation, profiles, training


def historical_comparisons(folder, runs):
    """Compare earlier saved endpoints by ordered test and prompt identity."""
    anchors = Path(__file__).resolve().parent / 'data' / 'historical_controls.csv'
    comparisons = []
    if anchors.exists():
        with anchors.open(newline='') as f:
            table = list(csv.DictReader(f))
        def hash_ids(ids):
            return hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest()
        for run in runs.values():
            job, scores = run['j'], run['scores']
            if job['stage'] == 'main':
                continue
            for hist in table:
                if hist['task'] != job['task'] or int(hist['shots']) != job['shots'] or int(hist['demo_seed']) != job['demo_seed']:
                    continue
                is_icl = hist['method'] == 'icl' and int(hist['updates']) == 0
                is_exact = (job['stage'] == 'controls' and job['method'] == hist['method']
                            and float(hist['budget']) == job['budget'] and int(hist['updates']) == job['updates']
                            and int(hist['optimizer_seed']) == job['optimizer_seed'])
                if not (is_exact or is_icl):
                    continue
                if hist['demo_ids_sha256'] != hash_ids(job['demo_ids']) or hist['test_ids_sha256'] != hash_ids(job['test_ids']):
                    raise ValueError(f"Historical IDs differ for {job['id']}")
                if int(hist['n_scored']) != scores['n_questions']:
                    raise ValueError(f"Historical test size mismatch for {job['id']}")
                if is_exact and int(hist['retained_cache_bytes']) != run['cache_bytes']:
                    raise ValueError(f"Mismatch in retained bytes with historical control for {job['id']}")
                historical_acc = 100*int(hist['n_correct'])/int(hist['n_scored'])
                historical_true = hist['pred_true'].strip()
                comparisons.append(dict(id=job['id'], **{k:job[k] for k in KEYS},
                                  reference='prior matched tuned' if is_exact else 'prior Standard ICL',
                                  old_accuracy_pp=historical_acc, new_accuracy_pp=scores['accuracy_pp'],
                                  delta_pp=scores['accuracy_pp']-historical_acc,
                                  old_pred_true=historical_true, new_pred_true=scores.get('n_pred_true')))
    write_table(folder, 'historical_checks.csv', comparisons, ('id','reference','old_accuracy_pp','new_accuracy_pp','delta_pp'))


def condition_means(out, absolute, profiles):
    """Collapse optimizer replicates before comparing scores or costs across demonstration seeds."""
    groups = defaultdict(lambda: defaultdict(list))
    group_fields = ('stage','task','shots','ratio','method','budget','updates','param_mode')
    profiles_by_id = {r['id']:r for r in profiles}
    for row in absolute:
        fullrow = {**profiles_by_id[row['id']], **row}
        groups[tuple(row[k] for k in group_fields)][row['demo_seed']].append(fullrow)
    result = []
    for cond, seed_rows in groups.items():
        info = dict(zip(group_fields,cond))
        expected_opts = {0,1} if info['stage']=='main' and info['updates'] else {0}
        metrics = defaultdict(list)
        for block in seed_rows.values():
            if {r['optimizer_seed'] for r in block} != expected_opts:
                continue
            for field in block[0].keys() - set(KEYS) - {'id'}:
                values = [r.get(field) for r in block]
                if all(isinstance(x,(float,int)) and not isinstance(x,bool) for x in values):
                    metrics[field].append(st.mean(values))
        for metric in sorted(metrics):
            result.append(dict(**info, metric=metric, **describe(metrics[metric])))
    write_table(out, 'condition_summary.csv', result, (*group_fields,'metric','n_demo_seeds','mean','sd_across_demo_seeds','se_across_demo_seeds'))


def validation_dynamics(out, runs):
    rows = []
    for run in runs.values():
        trace, job = run['val'], run['j']
        if not trace:
            continue
        first = trace[0]
        if first['step'] != 0:
            raise ValueError(f"Missing initial validation reading for {job['id']}")
        best = min(trace,key=lambda e: (e['nll'],e['step']))
        decline = next((e['step'] for e in trace if e['accuracy'] < first['accuracy']),None)
        rows.append(dict(id=job['id'], **{k:job[k] for k in KEYS}, n_validation=first['n'],
                         initial_validation_accuracy_pp=100*first['accuracy'], final_validation_accuracy_pp=100*trace[-1]['accuracy'],
                         first_sampled_step_below_initial_validation_accuracy=decline,
                         minimum_validation_nll=best['nll'], earliest_observed_minimum_nll_step=best['step']))
    write_table(out,'validation_dynamics.csv',rows,('id', *KEYS))


def run_all(root):
    jobs = build_jobs()
    runs, absolute, validation, profiles, training = read_runs(root, jobs)
    out = root / 'analysis'
    out.mkdir(parents=True, exist_ok=True)
    common = ('id', *KEYS)
    write_table(out, 'absolute_by_run.csv', absolute, (*common, *ACCURACY))
    write_table(out, 'resource_profiles.csv', profiles, common)
    write_table(out, 'validation_checkpoints.csv', validation, (*common, 'step','n', 'nll', 'accuracy'))
    write_table(out, 'training_loss.csv', training, (*common, 'step', 'training_nll'))

    condition_means(out, absolute, profiles)
    historical_comparisons(out, runs)
    validation_dynamics(out, runs)

    by_seed, summary, selections, associations = [], [], [], []
    id_columns = ('contrast','task','shots','ratio','method','budget','metric')

    def lookup(stage, task, k, seed, ratio, method, budget, updates, mode, opt):
        return runs.get((stage,task,k,seed,ratio,method,budget,updates,mode,opt))

    def aggregate(name, task, k, ratio, method, budget, opt_seeds, comparison, allow_demos=False, equal_bytes=False):
        """Each term is (coefficient, run); pair rows first, optimizer seeds second, demo seeds last."""
        label = dict(contrast=name, task=task, shots=k, ratio=ratio, method=method, budget=budget)
        samples = defaultdict(list)
        for seed in range(3):
            observations = defaultdict(list)
            for opt in opt_seeds:
                terms = comparison(seed, opt)
                if any(run is None for _,run in terms):
                    continue
                ref = terms[0][1]
                for _,run in terms[1:]:
                    if ref['j']['test_ids'] != run['j']['test_ids'] or ref['gold'] != run['gold']:
                        raise ValueError(f'Mismatched test in {name}')
                    if not allow_demos and ref['j']['demo_ids'] != run['j']['demo_ids']:
                        raise ValueError(f'Mismatched ordered demonstrations in {name}')
                    if equal_bytes and (ref['cache_bytes'] is None or ref['cache_bytes'] != run['cache_bytes']):
                        raise ValueError(f'Mismatched retained-cache budget in {name}')
                for field in DIFFERENCES:
                    if all(run['scores'].get(field) is not None for _,run in terms):
                        observations[field].append(sum(c*run['scores'][field] for c,run in terms))
            for metric, values in observations.items():
                if len(values) == len(opt_seeds):
                    delta = st.mean(values)
                    by_seed.append(dict(**label, metric=metric, demo_seed=seed, n_optimizer_blocks=len(values), delta=delta))
                    samples[metric].append(delta)
        for metric, values in samples.items():
            summary.append(dict(**label, metric=metric, **describe(values)))

    boolean = 'boolean_expressions'
    tasks = (boolean, 'date_understanding')
    def main(seed,opt,k,r,method):
        return lookup('main',boolean,k,seed,r,method,1.0 if method != 'answer_attention' else .25,
                      0 if method=='icl' else 16,'both',0 if method=='icl' else opt)

    def tuning_terms(seed,opt,k,r,method):
        return [(1,main(seed,opt,k,r,method)),(-1,main(seed,opt,k,r,'icl'))]

    def difference(plus,minus):
        return plus+[(c*-1,run) for c,run in minus]

    for method, budget in [('full',1.0), ('answer_attention',.25)]:
        for ratio in (.2,.5,.8):
            for k in (10,20):
                aggregate('E1 intervention minus full ICL',boolean,k,ratio,method,budget,(0,1),
                          lambda s,o: tuning_terms(s,o,k,ratio,method))
                if ratio != .5:
                    aggregate('E1 skew minus balanced, change relative to ICL',boolean,k,ratio,method,budget,(0,1),
                              lambda s,o: difference(tuning_terms(s,o,k,ratio,method),tuning_terms(s,o,k,.5,method)),True)
            aggregate('E1 20 minus 10 shots, change relative to ICL',boolean,'20-10',ratio,method,budget,(0,1),
                      lambda s,o: difference(tuning_terms(s,o,20,ratio,method),tuning_terms(s,o,10,ratio,method)),True)
            if ratio != .5:
                def shot_skew(seed,opt):
                    def shot(r):
                        return difference(tuning_terms(seed,opt,20,r,method),tuning_terms(seed,opt,10,r,method))
                    return difference(shot(ratio),shot(.5))
                aggregate('E1 shot-by-skew interaction relative to ICL',boolean,'20-10',ratio,method,budget,
                          (0,1), shot_skew, True)
    for k in (10,20):
        for ratio in (.2,.5,.8):
            aggregate('E1 25% attention tuned minus full tuned',boolean,k,ratio,'answer_attention',.25,(0,1),
                      lambda s,o: [(1,main(s,o,k,ratio,'answer_attention')),(-1,main(s,o,k,ratio,'full'))])
    for task in tasks:
        for k in (10,20):
            def full(seed):
                return lookup('controls',task,k,seed,None,'full',1.0,16,'both',0)
            for mode in ('key','value'):
                aggregate(f'E3 {mode}-only minus joint KV',task,k,None,mode,1,(0,),
                          lambda s,o: [(1,lookup('params',task,k,s,None,'full',1.0,16,mode,0)),(-1,full(s))],
                          equal_bytes=True)
            if k == 10:
                def horizon(seed, steps):
                    return full(seed) if steps == 16 else lookup('steps',task,10,seed,None,'full',1.0,steps,'both',0)
                for length in (4,8,32):
                    aggregate(f'E2 {length}-step schedule minus 16-step schedule',task,k,None,'full',1,(0,),
                              lambda s,o: [(1,horizon(s,length)),(-1,full(s))],equal_bytes=True)
                for s in range(3):
                    available = [horizon(s,t) for t in (4,8,16,32)]
                    if any(r is None for r in available):
                        continue
                    if any(r['j']['val_ids']!=available[0]['j']['val_ids'] for r in available):
                        raise ValueError('Validation IDs differ across training horizons')
                    candidates = []
                    for r in available:
                        endpoint = next((e for e in r['val'] if e['step']==r['j']['updates']),None)
                        if endpoint is not None:
                            candidates.append((r['j']['updates'], endpoint['nll'], r['scores']['accuracy_pp']))
                    if len(candidates)==4:
                        length, nll, testacc = min(candidates, key=lambda x:(x[1],x[0]))
                        reference = full(s)['scores']['accuracy_pp']
                        selections.append(dict(task=task,demo_seed=s,selected_horizon=length,validation_nll=nll,
                                               selected_test_accuracy_pp=testacc,fixed16_test_accuracy_pp=reference,
                                               delta_pp=testacc-reference))
                        # Spearman correlation across four paired, prespecified horizons; average ranks for ties.
                        nlls=[c[1] for c in candidates]; accs=[c[2] for c in candidates]
                        def ranks(xs):
                            return [sum(z<v for z in xs)+(sum(z==v for z in xs)+1)/2 for v in xs]
                        x,y=ranks(nlls),ranks(accs)
                        associations.append(dict(task=task,demo_seed=s,n_horizons=4,
                             spearman_nll_test_accuracy=st.correlation(x,y) if min(st.pvariance(x),st.pvariance(y))>0 else None))
            if k==20:
                for budget in (.25,.5,.75):
                    aggregate('E5 random minus answer attention',task,k,None,'random',budget,(0,),
                              lambda s,o: [(1,lookup('random',task,k,s,None,'random',budget,16,'both',0)),
                                           (-1,lookup('controls',task,k,s,None,'answer_attention',budget,16,'both',0))],
                              equal_bytes=True)
    write_table(out,'paired_by_seed.csv',by_seed,(*id_columns,'demo_seed','n_optimizer_blocks','delta'))
    write_table(out,'paired_summary.csv',summary,(*id_columns,'n_demo_seeds','mean','sd_across_demo_seeds','se_across_demo_seeds'))
    write_table(out,'validation_horizon_selection.csv',selections,('task','demo_seed','selected_horizon','validation_nll','selected_test_accuracy_pp','fixed16_test_accuracy_pp','delta_pp'))
    write_table(out,'validation_horizon_association.csv',associations,('task','demo_seed','n_horizons','spearman_nll_test_accuracy'))
    print(f'{len(runs)}/{len(jobs)} planned runs have a completion marker. {len(summary)} paired summary rows in {out}.')
    print('Read ANALYSIS.md with the tables. Missing or empty contrasts await complete matched blocks.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('results'))
    run_all(parser.parse_args().root)
