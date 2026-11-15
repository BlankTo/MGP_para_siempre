import argparse
import hashlib
import json
import warnings
from pathlib import Path

import config
from config import DEFAULT
from mgp.analysis.data import resolve
from mgp.progress import log

SHORT = {'GROUP_TARGET': 'target', 'NOVELTY_POOLS': 'pools', 'TIME': 'time', 'DIVERSITY': 'div', 'FINAL_ARCHIVE_MAX_SIZE': 'final',
         'CAPACITY_COLUMNS': 'columns'}
LISTS = ('NOVELTY_POOLS', 'CAPACITY_POOLS', 'CAPACITY_COLUMNS')


def value_of(text):
    for kind in (int, float):
        try:
            return kind(text)
        except ValueError:
            pass
    return text


def parse_pairs(items):
    out = {}
    for item in items or ():
        for part in item.split(';'):
            if not part.strip():
                continue
            key, _, values = part.partition('=')
            out[key.strip()] = [value_of(v.strip()) for v in values.split(',')]
    return out


def option(name):
    def parse(text):
        if text.lower() == 'none':
            return None
        if name == 'DIVERSITY':
            return text.lower() in ('1', 'true', 'yes')
        if name in LISTS:
            return tuple(value_of(x.strip()) for x in text.split(','))
        return value_of(text)
    return parse


def reference(base, text=None):
    ref = dict(base)
    for k, v in parse_pairs([text.replace(',', ';')] if text else []).items():
        if k not in ref:
            raise KeyError(f'unknown reference factor {k}')
        ref[k] = float(v[0]) if k in ('im', 'esel', 'esur') else int(v[0]) if k == 'size' else v[0]
    return ref


def arguments(visual, scope, inputs, options=()):
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    p = argparse.ArgumentParser(description=f'{visual}: inputs {", ".join(inputs)}')
    single = len(inputs) == 1
    defaults = {inputs[0]: config.value(scope, 'INPUT')} if single else scope.get('INPUTS', {})
    for study in inputs:
        default = defaults.get(study, DEFAULT)
        p.add_argument('--input' if single else f'--{study}', dest=study, default=config.INPUT if default is DEFAULT else default)
    p.add_argument('--where', action='append', default=[])
    p.add_argument('--reference', default=None)
    p.add_argument('--out', default=None)
    for name in options:
        p.add_argument('--' + name.lower().replace('_', '-'), dest=name, type=option(name), default=config.value(scope, name))
    args = p.parse_args()
    args.visual = visual
    args.paths = {}
    for study in inputs:
        try:
            args.paths[study] = resolve(study, getattr(args, study))
        except FileNotFoundError:
            if single:
                raise
            log(visual, 'skip', input=study, reason='no_execution')
    args.filters = {k: list(v) for k, v in config.value(scope, 'WHERE').items()}
    args.filters.update(parse_pairs(args.where))
    args.ref = reference(config.value(scope, 'REFERENCE'), args.reference)
    args.out = Path(args.out) if args.out else Path(config.value(scope, 'OUTPUT'))
    args.option = {name: getattr(args, name) for name in options}
    return args


def name_of(path):
    return Path(path).stem


def text(v):
    if isinstance(v, bool):
        return ''
    if isinstance(v, (tuple, list)):
        return '-'.join(map(str, v))
    return str(v).lower()


def suffix(args, names=None):
    names = args.option if names is None else names
    return ''.join(f'_{SHORT.get(n, n.lower())}{text(args.option[n])}' for n in names if args.option[n] != getattr(config, n))


def output(args, visual, *paths):
    names = [name_of(p) for p in paths if p is not None]
    tail = names[0] + ''.join('+' + n.rsplit('__', 1)[-1] for n in names[1:])
    if args.filters:
        tail += '__' + hashlib.sha256(json.dumps(args.filters, sort_keys=True, default=str).encode()).hexdigest()[:6]
    if args.ref != config.REFERENCE:
        tail += '__ref' + hashlib.sha256(json.dumps(args.ref, sort_keys=True).encode()).hexdigest()[:6]
    args.out.mkdir(parents=True, exist_ok=True)
    return args.out / f'{visual}__{tail}'


def write_text(base, extension, content):
    path = Path(str(base) + extension)
    path.write_text(content, encoding='utf-8')
    log('write', path)
    return path


def thousands(n):
    return f'{int(n):,}'.replace(',', '{,}')
