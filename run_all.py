import argparse
import subprocess
import sys
import time
from pathlib import Path

from mgp.progress import duration, log

HERE = Path(__file__).resolve().parent
ORDER = ('workspace', 'study8', 'study5', 'study4', 'study6', 'study3', 'study7', 'study1', 'study2')


def main():
    p = argparse.ArgumentParser(description='all studies in sequence, then all visuals')
    p.add_argument('--workers', type=int, default=None)
    p.add_argument('--memory-mb', type=int, default=None)
    p.add_argument('--only', default=None)
    p.add_argument('--no-visuals', action='store_true')
    args = p.parse_args()
    studies = ORDER if not args.only else tuple(s for s in ORDER if s in args.only.split(','))
    failed = []
    started = time.perf_counter()
    for i, study in enumerate(studies, 1):
        cmd = [sys.executable, str(HERE / f'{study}.py')]
        if args.workers:
            cmd += ['--workers', str(args.workers)]
        if args.memory_mb:
            cmd += ['--memory-mb', str(args.memory_mb)]
        log('studies', 'run', study=study, step=f'{i}/{len(studies)}')
        if subprocess.call(cmd) != 0:
            failed.append(study)
    if not args.no_visuals and not failed:
        if subprocess.call([sys.executable, str(HERE / 'visuals' / 'all.py')]) != 0:
            failed.append('visuals')
    log('run_all', 'end', failed=','.join(failed) or 'none', elapsed=duration(time.perf_counter() - started))
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
