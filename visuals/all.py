import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path[0] = str(Path(__file__).resolve().parents[1])

from mgp.progress import duration, log

HERE = Path(__file__).resolve().parent
SCRIPTS = (('res_ba', ('study1',)), ('res_group_recovery', ('study1',)), ('res_ratio_coverage', ('study1',)), ('res_datasets', ('study3',)),
           ('res_training', ('study8',)), ('res_labels', ('study5',)), ('res_noise_search', ('study4',)), ('res_timing', ('study6',)),
           ('res_group_n', ('study2',)), ('res_group_novelty', ('study2',)), ('res_group_time', ('study2', 'study6')),
           ('res_group_capacity_single', ('study2',)), ('res_group_capacity_final', ('study2',)), ('res_group_recovery_5000', ('study7',)),
           ('res_group_n5000', ('study7',)), ('res_workspace_grid', ('workspace',)))


def main():
    p = argparse.ArgumentParser(description='all visuals')
    for s in ('study1', 'study2', 'study3', 'study4', 'study5', 'study6', 'study7', 'study8', 'workspace'):
        p.add_argument(f'--{s}', default=None)
    p.add_argument('--out', default=None)
    args = p.parse_args()
    failed = []
    started = time.perf_counter()
    for i, (script, inputs) in enumerate(SCRIPTS, 1):
        cmd = [sys.executable, str(HERE / f'{script}.py')]
        for s in inputs:
            if getattr(args, s):
                cmd += ['--input' if len(inputs) == 1 else f'--{s}', getattr(args, s)]
        if args.out:
            cmd += ['--out', args.out]
        log('visuals', 'run', script=script, step=f'{i}/{len(SCRIPTS)}')
        if subprocess.call(cmd) != 0:
            failed.append(script)
    log('visuals', 'end', failed=','.join(failed) or 'none', elapsed=duration(time.perf_counter() - started))
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
