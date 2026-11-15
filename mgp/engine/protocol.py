import hashlib

import numpy as np

from mgp.engine.fitness import readonly_target, rmse

PROTOCOL_VERSION = 2


def current_protocol(config):
    return config.get('protocol_version', 1) >= PROTOCOL_VERSION


def stream_seed(master, target, purpose, repetition=0):
    if isinstance(master, bool) or not isinstance(master, int) or master < 0:
        raise ValueError('stream seeds must be nonnegative integers')
    text = f'mgp-v27|{master}|{target}|{purpose}|{repetition}'
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], 'big') % 2 ** 63


def noisy_labels(clean, config, target_name, repetition):
    level = config.get('noise_level', 0.0)
    if not np.isfinite(level) or level < 0:
        raise ValueError('noise_level must be finite and nonnegative')
    master = config.get('noise_seed', 1)
    noise_repetition = 0 if config.get('fixed_noise_across_repetitions', False) else repetition
    seed = stream_seed(config.get('dataset_seed', 1) if master is None else master, target_name, 'noise', noise_repetition)
    finite = np.isfinite(clean)
    scale = rmse(np.zeros(int(finite.sum())), clean[finite])
    labels = np.array(clean, dtype=float, copy=True)
    if level:
        z = np.random.default_rng(seed).standard_normal(int(finite.sum()))
        labels[finite] += level * scale * z
    return (readonly_target(labels), seed)
