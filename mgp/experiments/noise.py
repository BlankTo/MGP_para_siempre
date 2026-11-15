from mgp.archives.diagnostics import noise_audit
from mgp.engine.problem import random_dataset
from mgp.engine.protocol import noisy_labels, stream_seed
from mgp.engine.targets import TARGETS
from mgp.experiments.jobs import job_settings
from mgp.experiments.plan import job_for
from mgp.experiments.plan import signature

POLICY = 'independent_config_restart_paired_intensities_v2'


def replica_seed(task):
    case = task['case']
    return stream_seed(task['options']['master_seed'], case['target'],
                       f"{POLICY}|{case['config_id']}|dataset={case['dataset_replica']}", task['restart_index'])


def paired_dataset(task):
    job = job_for(task['case'], task['restart_index'])
    settings = job_settings(task['options'], job)
    variables, clean = random_dataset(TARGETS[task['target']][0], settings)
    noise_master = replica_seed(task)
    labels, actual_noise_seed = noisy_labels(clean, {**settings, 'noise_seed': noise_master, 'fixed_noise_across_repetitions': True},
                                             task['target'], 0)
    test = random_dataset(TARGETS[task['target']][0], settings, 'test')
    audit = noise_audit(variables, clean, labels, test, settings, actual_noise_seed)
    audit.update(noise_protocol=POLICY, config_id=task['case']['config_id'], restart_index=task['restart_index'], noise_master=noise_master,
                 pairing_key=signature([task['target'], task['case']['config_id'], task['case']['dataset_replica'],
                                        task['restart_index']])[:24])
    return job, settings, (variables, clean, labels, test, audit)
