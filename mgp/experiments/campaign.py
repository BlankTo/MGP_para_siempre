import hashlib
import json
import os
import threading
import time
import traceback
from copy import deepcopy
from itertools import product

import config
from mgp.archives.codecs import ARCHIVE_FIELDS, pack_identity_sets, pack_metrics
from mgp.engine.operator_sequence import operator_statistics, reset_operator_statistics
from mgp.engine.protocol import stream_seed
from mgp.experiments.jobs import run_job
from mgp.experiments.plan import cases as make_cases, configurations as make_configurations, job_for
from mgp.experiments.runner import plain

THREAD_VARIABLES = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')
WORKER_WATCHDOG_SECONDS = 1800.0
_WORKER = {}
_CACHE_CLEARS = 0
RUNS_SCHEMA = ('''CREATE TABLE IF NOT EXISTS runs (run_id TEXT PRIMARY KEY, case_id TEXT NOT NULL, pool INTEGER NOT NULL, repetition INTEGER NOT NULL,
    restart_index INTEGER NOT NULL, status TEXT NOT NULL, built INTEGER, num INTEGER, sym REAL, train_rmse REAL, test_rmse REAL, merges INTEGER,
    termination TEXT, cpu_seconds REAL, dataset_id TEXT, metrics BLOB, archive_members BLOB, cost TEXT, error TEXT);
CREATE INDEX IF NOT EXISTS runs_case ON runs(case_id, pool, repetition);
CREATE TABLE IF NOT EXISTS datasets (dataset_id TEXT PRIMARY KEY, audit TEXT NOT NULL);''')


def memory_mb():
    try:
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD), ('PeakWorkingSetSize', ctypes.c_size_t),
                        ('WorkingSetSize', ctypes.c_size_t), ('QuotaPeakPagedPoolUsage', ctypes.c_size_t),
                        ('QuotaPagedPoolUsage', ctypes.c_size_t), ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t),
                        ('QuotaNonPagedPoolUsage', ctypes.c_size_t), ('PagefileUsage', ctypes.c_size_t), ('PeakPagefileUsage', ctypes.c_size_t)]
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        kernel32, psapi = (ctypes.WinDLL('kernel32'), ctypes.WinDLL('psapi'))
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = (wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD)
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        if psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            return (counters.WorkingSetSize / 2 ** 20, counters.PeakWorkingSetSize / 2 ** 20)
    except Exception:
        pass
    return (None, None)


class DerivationTimeLimit(Exception):
    pass


class MemorySampler:

    def __init__(self, interval=0.02, time_limit=None):
        self.interval, self.peak, self._stop = (interval, 0.0, threading.Event())
        self.time_limit, self.fired = (time_limit, False)
        self._target = threading.get_ident()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        import ctypes
        started = time.perf_counter()
        while not self._stop.is_set():
            current, _ = memory_mb()
            if current is not None:
                self.peak = max(self.peak, current)
            if self.time_limit and (not self.fired) and (time.perf_counter() - started > self.time_limit):
                self.fired = True
                ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_ulong(self._target), ctypes.py_object(DerivationTimeLimit))
            self._stop.wait(self.interval)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()
        return False


def limit_process_memory(limit_mb):
    try:
        import ctypes
        from ctypes import wintypes

        class Basic(ctypes.Structure):
            _fields_ = [('PerProcessUserTimeLimit', ctypes.c_int64), ('PerJobUserTimeLimit', ctypes.c_int64), ('LimitFlags', wintypes.DWORD),
                        ('MinimumWorkingSetSize', ctypes.c_size_t), ('MaximumWorkingSetSize', ctypes.c_size_t),
                        ('ActiveProcessLimit', wintypes.DWORD), ('Affinity', ctypes.c_size_t), ('PriorityClass', wintypes.DWORD),
                        ('SchedulingClass', wintypes.DWORD)]

        class IoCounters(ctypes.Structure):
            _fields_ = [(name, ctypes.c_uint64) for name in ('ReadOperationCount', 'WriteOperationCount', 'OtherOperationCount',
                                                             'ReadTransferCount', 'WriteTransferCount', 'OtherTransferCount')]

        class Extended(ctypes.Structure):
            _fields_ = [('BasicLimitInformation', Basic), ('IoInfo', IoCounters), ('ProcessMemoryLimit', ctypes.c_size_t),
                        ('JobMemoryLimit', ctypes.c_size_t), ('PeakProcessMemoryUsed', ctypes.c_size_t), ('PeakJobMemoryUsed', ctypes.c_size_t)]
        kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        kernel32.CreateJobObjectW.argtypes = (ctypes.c_void_p, wintypes.LPCWSTR)
        kernel32.SetInformationJobObject.argtypes = (wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD)
        kernel32.AssignProcessToJobObject.argtypes = (wintypes.HANDLE, wintypes.HANDLE)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        job = kernel32.CreateJobObjectW(None, None)
        info = Extended()
        info.BasicLimitInformation.LimitFlags = 256
        info.ProcessMemoryLimit = int(limit_mb) * 2 ** 20
        if not kernel32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info)):
            return False
        _WORKER['job'] = job
        return bool(kernel32.AssignProcessToJobObject(job, kernel32.GetCurrentProcess()))
    except Exception:
        return False


def clear_caches():
    global _CACHE_CLEARS
    import gc
    import sys
    for name, module in list(sys.modules.items()):
        if module is None or not (name == 'mgp' or name.startswith('mgp.')):
            continue
        for value in list(vars(module).values()):
            if callable(getattr(value, 'cache_clear', None)) and hasattr(value, 'cache_info'):
                value.cache_clear()
    try:
        from sympy.core.cache import clear_cache
        clear_cache()
    except ImportError:
        pass
    _CACHE_CLEARS += 1
    if _CACHE_CLEARS % 32 == 0:
        gc.collect()
    else:
        gc.collect(0)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), default=str).encode()).hexdigest()


GRID = ('targets', 'strategies', 'internal_merge', 'sizes', 'ratios', 'esel', 'esur', 'merge_limit', 'datasets', 'noises')


def grid(scope):
    return {**{name: config.value(scope, name.upper()) for name in GRID}, 'seed_stream': scope['SEED_STREAM']}


def settings(scope):
    return {**config.value(scope, 'SETTINGS'), 'revisit_ic_variants': config.value(scope, 'REVISIT_IC_VARIANTS'),
            'revisit_ic_pairs': config.value(scope, 'REVISIT_IC_PAIRS'), 'revisit_ic_budget': config.value(scope, 'REVISIT_IC_BUDGET'),
            'single_archive_max_size': config.value(scope, 'SINGLE_ARCHIVE_MAX_SIZE')}


def campaign(engine_settings=None):
    opts = deepcopy(config.CAMPAIGN)
    opts['settings'].update(engine_settings or {})
    return opts


def selected(levels):
    return [dict(total_objects=s, operator_variable_ratio=r, esel=e, esur=u, merge_strategy=strategy, internal_merge_probability=im)
            for strategy, im, s, r, e, u in product(levels['strategies'], levels['internal_merge'], levels['sizes'], levels['ratios'], levels['esel'],
                                                    levels['esur'])]


def options(levels, engine_settings=None, configurations=None):
    opts = campaign(engine_settings)
    opts.update(kind='confirmation', seed_family=levels['seed_stream'], targets=list(levels['targets']), datasets_per_target=levels['datasets'],
                noise_levels=list(levels['noises']), selected_configurations=configurations or selected(levels))
    opts['settings']['max_merge'] = levels['merge_limit']
    return opts


def job_options(levels, engine_settings=None):
    opts = options(levels, engine_settings)
    return {key: opts[key] for key in ('version', 'master_seed', 'test_stream', 'settings')}


def cells(levels, engine_settings=None, dataset_stream=None, configurations=None):
    opts = options(levels, engine_settings, configurations)
    configs, skipped = make_configurations(opts)
    if skipped:
        raise ValueError(f'skipped configurations {skipped}')
    for case in make_cases(opts, configs):
        if dataset_stream:
            case['dataset_seed'] = stream_seed(opts['master_seed'], case['target'], dataset_stream, case['dataset_replica'])
        yield case


def make_task(case, study, pool, index, job_opts):
    return {'run_id': digest([study, case['case_id'], pool, index])[:32], 'study': study, 'case': case, 'pool': pool, 'index': index,
            'restart_index': (pool - 1) * 10000 + index, 'options': job_opts, 'target': case['target'], 'dataset_replica': case['dataset_replica'],
            'noise_level': case['noise_level']}


def init_worker(memory_mb_limit):
    os.environ.update(dict.fromkeys(THREAD_VARIABLES, '1'))
    _WORKER['datasets'] = {}
    _WORKER['memory_limited'] = limit_process_memory(memory_mb_limit) if memory_mb_limit else False


def _execute_once(task):
    before = time.perf_counter()
    cpu = time.process_time()
    dataset_rows = []
    identity_sets = {}
    try:
        reset_operator_statistics()
        clear_caches()
        job = job_for(task['case'], task['restart_index'])
        with MemorySampler(time_limit=WORKER_WATCHDOG_SECONDS) as monitor:
            [(idx, metrics, detail)] = run_job(task['options'], job, _WORKER['datasets'],
                                               dataset_sink=lambda dataset_id, audit, payload: dataset_rows.append((dataset_id, audit)),
                                               archive_sink=None, archive_fields=ARCHIVE_FIELDS, observation_mode='archive_only',
                                               dataset_payload=False, identity_sink=lambda name, keys: identity_sets.update({name: keys}))
        pack_started = time.perf_counter()
        pack_cpu_started = time.process_time()
        packed_archives = pack_identity_sets(identity_sets)
        archive_pack_seconds = time.perf_counter() - pack_started
        archive_pack_cpu_seconds = time.process_time() - pack_cpu_started
        current, peak = memory_mb()
        return dict(run_id=task['run_id'], study=task['study'], case_id=task['case']['case_id'], target=task['target'],
                    dataset_replica=task['dataset_replica'], noise_level=task['noise_level'], pool=task['pool'], repetition=task['index'],
                    restart_index=task['restart_index'], status='ok', metrics=metrics, archive_members=packed_archives,
                    dataset_rows=dataset_rows,
                    cost={'task_wall_seconds': time.perf_counter() - before, 'task_cpu_seconds': time.process_time() - cpu,
                          'archive_pack_seconds': archive_pack_seconds, 'archive_pack_cpu_seconds': archive_pack_cpu_seconds,
                          'archive_pack_bytes': len(packed_archives), 'archive_member_count': sum(map(len, identity_sets.values())),
                          **operator_statistics(), 'peak_memory_mb': max(peak or 0, monitor.peak), 'memory_mb': current,
                          'memory_limited': _WORKER['memory_limited']}, error=None)
    except Exception as exc:
        return dict(run_id=task['run_id'], study=task['study'], case_id=task['case']['case_id'], target=task['target'],
                    dataset_replica=task['dataset_replica'], noise_level=task['noise_level'], pool=task['pool'], repetition=task['index'],
                    restart_index=task['restart_index'], status='error', metrics=None, archive_members=None, dataset_rows=dataset_rows,
                    cost={'task_wall_seconds': time.perf_counter() - before, 'task_cpu_seconds': time.process_time() - cpu,
                          **operator_statistics()},
                    error=f'{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=5)}\noperator_statistics={operator_statistics()}')


def execute(task):
    original_result = _execute_once(task)
    if original_result['status'] != 'error' or not (original_result.get('error') or '').startswith('HeuristicGCDFailed:'):
        return original_result
    from sympy.polys import rings
    from sympy.polys.polyerrors import HeuristicGCDFailed
    normal_gcd = rings.PolyElement._gcd_ZZ
    fallback_count = 0

    def verified_exact_gcd(f, g):
        nonlocal fallback_count
        try:
            return normal_gcd(f, g)
        except HeuristicGCDFailed:
            h, a, b = f.ring.dmp_inner_gcd(f, g)
            if h * a != f or h * b != g:
                raise ArithmeticError('exact GCD did not reconstruct both polynomials')
            fallback_count += 1
            return h, a, b

    try:
        rings.PolyElement._gcd_ZZ = verified_exact_gcd
        replay = _execute_once(task)
    finally:
        rings.PolyElement._gcd_ZZ = normal_gcd
    if replay['status'] != 'ok':
        replay['error'] = (replay.get('error') or 'recovery replay failed') + '\ninitial attempt: ' + original_result['error'].splitlines()[0]
        replay['cost']['gcd_exact_recovery_attempted'] = True
        replay['cost']['gcd_exact_fallback_count'] = fallback_count
        return replay
    if fallback_count < 1:
        original_result['error'] += '\nrecovery rejected: replay succeeded without the exact GCD fallback'
        return original_result
    replay['cost'].update(recovery_kind='exact_dense_polynomial_gcd_after_heuristic_gcd_failure', gcd_exact_fallback_count=fallback_count,
                          recovery_initial_error=original_result['error'].splitlines()[0],
                          recovery_initial_attempt_wall_seconds=original_result['cost']['task_wall_seconds'],
                          recovery_initial_attempt_cpu_seconds=original_result['cost']['task_cpu_seconds'])
    replay['cost']['task_wall_seconds'] += original_result['cost']['task_wall_seconds']
    replay['cost']['task_cpu_seconds'] += original_result['cost']['task_cpu_seconds']
    return replay


def store(con, result):
    for dataset_id, audit in result['dataset_rows']:
        metadata = json.dumps(audit, sort_keys=True, separators=(',', ':'), allow_nan=True)
        existing = con.execute('SELECT audit FROM datasets WHERE dataset_id=?', (dataset_id,)).fetchone()
        if existing is not None and existing[0] != metadata:
            raise RuntimeError('dataset manifest collision or non-deterministic generation: ' + dataset_id)
        con.execute('INSERT OR IGNORE INTO datasets VALUES (?,?)', (dataset_id, metadata))
    m = result['metrics']
    packed = pack_metrics(m) if m is not None else None
    cost = dict(result['cost'])
    if packed is not None:
        cost.update(metrics_pack_bytes=len(packed))
    get = (lambda k: plain(m.get(k))) if m is not None else (lambda k: None)
    con.execute('INSERT OR REPLACE INTO runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                (result['run_id'], result['case_id'], result['pool'], result['repetition'], result['restart_index'], result['status'],
                 get('evaluated_target_ever'), get('test_success'), get('symbolic_recovery'), get('train_rmse'), get('test_rmse'), get('merges'),
                 get('termination'), cost.get('task_cpu_seconds'), get('dataset_id'), packed, result['archive_members'], json.dumps(cost),
                 result['error']))
    return result['status'] == 'ok'
