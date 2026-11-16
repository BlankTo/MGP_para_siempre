import math
from copy import copy, deepcopy

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import config
from config import DEFAULT
from mgp.archives.best_building_blocks import BestBuildingBlockArchive
from mgp.archives.codecs import ARCHIVE_FIELDS
from mgp.archives.observer import ArchiveOnlyObserver
from mgp.engine.derivation import Derivation
from mgp.engine.fitness import object_fitness
from mgp.engine.objects import IncompleteObject, OperatorObject
from mgp.engine.problem import make_workspace
from mgp.engine.symbolic import symbolic_data
from mgp.engine.targets import TARGETS
from mgp.experiments import campaign as C
from mgp.experiments.jobs import job_settings, make_job_dataset
from mgp.experiments.plan import job_for, paired_seed
from mgp.progress import log

STUDY = 'viewer'
SEED_STREAM = 'campaign2026_study1'
TARGET = 'nguyen_1'
STRATEGY = 'standard'
INTERNAL_MERGE = 0.0
SIZE = 32
RATIO = '1:1'
ESEL = 0.0
ESUR = 0.0
REPETITION = 1
MERGE_LIMIT = DEFAULT
REVISIT_IC_VARIANTS = DEFAULT
REVISIT_IC_PAIRS = DEFAULT
REVISIT_IC_BUDGET = DEFAULT
CONSTRUCTIVE_MERGES = DEFAULT
COMPLETE_INTO_HOLE = DEFAULT
SINGLE_ARCHIVE_MAX_SIZE = DEFAULT
ARCHIVE_CONSTANTS = DEFAULT
SETTINGS = DEFAULT
ROWS = 15
NAMES = {'standard': 'Standard', 'revisit_ic': 'Revisit IC', 'constructive': 'Constructive'}
COLORS = {'semantic': '#a5d8ff', 'operator': '#dee2e6', 'incomplete': '#fcc2d7', 'best': '#16a34a', 'last': '#eab308', 'other': '#000000',
          'line': '#94a3b8', 'rule': '#cbd5e1', 'ink': '#1e293b', 'muted': '#64748b'}
ENDS = {'single_object': 'one object remains', 'max_merge': 'MERGE limit reached', 'empty_workspace': 'the workspace is empty',
        'no_valid_merge': 'no valid MERGE remains', 'no_legal_merge': 'no legal MERGE remains'}


def task(repetition=None):
    levels = dict(targets=(TARGET,), strategies=(STRATEGY,), internal_merge=(INTERNAL_MERGE,), sizes=(SIZE,), ratios=(RATIO,), esel=(ESEL,), esur=(ESUR,),
                  merge_limit=config.value(globals(), 'MERGE_LIMIT'), datasets=1, noises=(0.0,), seed_stream=SEED_STREAM)
    engine = C.settings(globals())
    cases = list(C.cells(levels, engine))
    if len(cases) != 1:
        raise ValueError(f'expected one configuration, got {len(cases)}')
    return C.make_task(cases[0], STUDY, 1, REPETITION if repetition is None else repetition, C.job_options(levels, engine))


def fitness(node, labels):
    f = object_fitness(node, labels)
    return f if math.isfinite(f) else math.inf


def label(node):
    if isinstance(node, IncompleteObject):
        return f'{label(node.semantic)} [{node.head.expression}] ?'
    if isinstance(node, OperatorObject):
        return node.expression
    data = symbolic_data(node)
    return data[1] or data[0]


def shorten(text, width):
    return text if len(text) <= width else text[:width - 1] + '…'


def ranked(entries):
    return [(e['equivalence_key'], e['canonical_expression'] or e['equivalence_key'], e['score'])
            for e in sorted(entries, key=lambda e: (-e['score'], e['equivalence_key']))]


class Run:

    def __init__(self, task):
        C.clear_caches()
        options = task['options']
        job = job_for(task['case'], task['restart_index'])
        self.settings = job_settings(options, job)
        variables, clean, self.labels, _, _ = make_job_dataset(options, job, self.settings)
        index = job['restart_index']
        self.seeds = paired_seed(options, job, 'workspace', index), paired_seed(options, job, 'derivation', index)
        workspace = make_workspace(variables, self.settings, self.seeds[0])
        self.observer = ArchiveOnlyObserver(self.settings, variables, self.labels, clean)
        self.observer.include_similarity_archive = 'credit_by_depth' in ARCHIVE_FIELDS
        self.derivation = Derivation(self.labels, self.settings, self.seeds[1], clean, candidate_observer=self.observer.on_candidate,
                                     choice_observer=self.observer.on_choice)
        self.iterator = self.derivation.steps(workspace)
        self.done = False
        self.step()

    def step(self):
        if self.done:
            return
        self.workspace = next(self.iterator)
        self.observer.observe(self.derivation)
        if self.derivation.termination_reason:
            self.finish()

    def finish(self):
        for _ in self.iterator:
            raise RuntimeError('the derivation continued after its termination')
        archive = self.derivation.building_block_archive
        if archive is not None:
            archive.finalize_deferred()
            if self.settings.get('single_archive_max_size') is not None:
                archive.limit(self.settings['single_archive_max_size'])
        self.done = True
        d = self.derivation
        log('viewer', 'end', merges=d.merge_count, termination=d.termination_reason, train_rmse=f'{d.best_fitness:.6g}',
            transition_sequence_hash=self.observer.trajectory_digest.hexdigest())

    def ba(self):
        objects = self.derivation.archive.objects
        order = sorted(range(len(objects)), key=lambda i: (fitness(objects[i], self.labels), -i))
        return [(symbolic_data(objects[i])[0], label(objects[i]), fitness(objects[i], self.labels)) for i in order]

    def bba(self):
        archive = self.derivation.building_block_archive
        if archive is None:
            return []
        if not self.done:
            archive = copy(archive)
            archive.entries = deepcopy(archive.entries)
            archive.finalize_deferred()
        return ranked(archive.entries)

    def fbba(self):
        d = self.derivation
        archive = d.best_bb_archive
        if archive is None and not self.done and d.best is not None and d.config.get('best_bb_archive_enabled') and d.archive_mask['best_bb']:
            archive = BestBuildingBlockArchive(d.target, d.config)
            archive.build(d.best)
            if d.config.get('single_archive_max_size') is not None:
                archive.limit(d.config['single_archive_max_size'])
        return [] if archive is None else ranked(archive.entries)


def kind(node):
    if isinstance(node, IncompleteObject):
        return 'incomplete'
    if isinstance(node, OperatorObject):
        return 'operator'
    return 'semantic'


def layout(workspace):
    nodes, pending, leaves = [], [], 0
    for node in reversed(workspace):
        pending.append((node, None))
    while pending:
        node, parent = pending.pop()
        k = len(nodes)
        nodes.append({'node': node, 'parent': parent, 'children': [], 'x': 0.0, 'y': 0, 'leaves': 0})
        if parent is not None:
            nodes[parent]['children'].append(k)
        if node.children:
            pending.extend((child, k) for child in reversed(node.children))
        else:
            nodes[k].update(x=float(leaves), leaves=1)
            leaves += 1
    for node in reversed(nodes):
        if node['children']:
            kids = [nodes[c] for c in node['children']]
            node.update(x=sum(c['x'] for c in kids) / len(kids), y=max(c['y'] for c in kids) + 1, leaves=sum(c['leaves'] for c in kids))
    return nodes, leaves


def borders(tree, best, last):
    out = [COLORS['last']] if tree is last else []
    if tree is best:
        out.append(COLORS['best'])
    return out or [COLORS['other']]


def cell(ax, x, y, text, fill, rings, size):
    for i, color in enumerate(rings):
        ax.text(x, y, text, ha='center', va='center', fontsize=size, family='monospace', zorder=5 - i, color=COLORS['ink'] if i == 0 else 'none',
                bbox={'boxstyle': f'round,pad={0.3 + 0.35 * i:.2f}', 'facecolor': fill if i == 0 else 'none', 'edgecolor': color,
                      'linewidth': 0.7 if color == COLORS['other'] else 1.8})


class Viewer:

    def __init__(self, repetition):
        self.repetition, self.held = repetition, False
        self.run = start(repetition)
        self.figure = plt.figure(figsize=(16, 9.5))
        if self.figure.canvas.manager is not None:
            self.figure.canvas.manager.set_window_title('MGP derivation')
        self.trees = self.figure.add_axes((0.015, 0.47, 0.97, 0.51))
        self.panels = [self.figure.add_axes((0.015 + i * 0.33, 0.165, 0.31, 0.245)) for i in range(3)]
        for y in (0.455, 0.145):
            self.figure.add_artist(Line2D((0.015, 0.985), (y, y), transform=self.figure.transFigure, color=COLORS['rule'], linewidth=1.0))
        self.header = self.figure.text(0.015, 0.128, '', fontsize=10, va='top', color=COLORS['ink'])
        self.status = self.figure.text(0.015, 0.093, '', fontsize=10, va='top', color=COLORS['ink'])
        self.best = self.figure.text(0.015, 0.058, '', fontsize=10, va='top', color=COLORS['ink'])
        self.figure.text(0.015, 0.012, 'SPACE: next MERGE (at the end: new derivation)    R: new derivation', fontsize=9, va='bottom', color=COLORS['muted'])
        fills = [Patch(facecolor=COLORS[k], edgecolor=COLORS['other'], linewidth=0.7, label=k) for k in ('semantic', 'operator', 'incomplete')]
        edges = [Patch(facecolor='white', edgecolor=COLORS[k], linewidth=0.7 if k == 'other' else 1.8, label=text)
                 for k, text in (('best', 'best so far'), ('last', 'last created'), ('other', 'other'))]
        handles = [h for pair in zip(fills, edges) for h in pair]
        self.figure.legend(handles=handles, loc='upper right', bbox_to_anchor=(0.985, 0.135), ncol=3, frameon=False, fontsize=9, handlelength=1.6,
                           columnspacing=1.4)
        self.figure.canvas.mpl_connect('key_press_event', self.press)
        self.figure.canvas.mpl_connect('key_release_event', self.release)
        self.draw()

    def press(self, event):
        if event.key == 'r':
            self.restart()
            return
        if event.key not in (' ', 'space') or self.held:
            return
        self.held = True
        if self.run.done:
            self.restart()
            return
        self.run.step()
        self.draw()

    def restart(self):
        self.repetition += 1
        self.run = start(self.repetition)
        self.draw()

    def release(self, event):
        if event.key in (' ', 'space'):
            self.held = False

    def draw(self):
        r, d, s = self.run, self.run.derivation, self.run.settings
        self.header.set_text(f"{TARGET} = {TARGETS[TARGET][1]}   |   {NAMES[STRATEGY]}, {s['operator_count']} operators + {s['variable_atom_count']} "
                             f"variables, p_IM = {s['internal_merge_probability']:g}, e_sel = {s['esel']:g}, e_sur = {s['esur']:g}   |   repetition {self.repetition}")
        state = 'finished: ' + ENDS.get(d.termination_reason, d.termination_reason) if r.done else 'running'
        self.status.set_text(f'MERGE {d.merge_count}   |   attempts {d.attempt_count}   |   objects {len(r.workspace)}   |   last MERGE '
                             f'{d.last_merge_kind or "none"}   |   {state}')
        self.best.set_text('best so far: none' if d.best is None else f'best so far: {shorten(label(d.best), 90)}   (training RMSE {d.best_fitness:.4g})')
        self.draw_trees()
        self.draw_panel(self.panels[0], 'BA', r.ba(), 'RMSE', 'lowest training error')
        self.draw_panel(self.panels[1], 'BBA', r.bba(), 'credit', 'final credit' if r.done else 'credit if the derivation ended now')
        self.draw_panel(self.panels[2], 'FBBA', r.fbba(), 'credit', 'from the final best' if r.done else 'from the current best, final at the end')
        self.figure.canvas.draw_idle()

    def draw_trees(self):
        ax, r = self.trees, self.run
        ax.clear()
        ax.set_axis_off()
        nodes, width = layout(r.workspace)
        d = r.derivation
        best = d.best.structure if d.best is not None else None
        last = d.last_merge[2].structure if d.last_merge is not None else None
        height = max(n['y'] for n in nodes) if nodes else 0
        extent = ax.get_window_extent()
        size = max(3.5, min(8.0, extent.height / self.figure.dpi * 72 / (height + 1.4) / 2.0))
        chars = extent.width / self.figure.dpi * 72 / (max(width, 1) + 0.6) / (0.62 * size)
        for n in nodes:
            if n['parent'] is not None:
                p = nodes[n['parent']]
                ax.plot([n['x'], p['x']], [n['y'], p['y']], color=COLORS['line'], linewidth=0.9, zorder=1)
        for n in nodes:
            node = n['node']
            cell(ax, n['x'], n['y'], shorten(label(node), max(2, int(chars * n['leaves']))), COLORS[kind(node)], borders(node.structure, best, last),
                 size=size)
        ax.set_xlim(-0.8, max(width, 1) - 0.2)
        ax.set_ylim(-0.7, height + 0.7)

    def draw_panel(self, ax, name, rows, value, rule):
        ax.clear()
        ax.set_axis_off()
        ax.set_title(f'{name}: {len(rows)} entr{"y" if len(rows) == 1 else "ies"} ({rule})', loc='left', fontsize=10, color=COLORS['ink'])
        if not rows:
            ax.text(0, 1, 'empty', transform=ax.transAxes, va='top', fontsize=9, color=COLORS['muted'])
            return
        width = 44
        lines = [f'{"":>3} {"expression":<{width}} {value:>11}']
        lines += [f'{i:>3} {shorten(text, width):<{width}} {v:>11.5g}' for i, (_, text, v) in enumerate(rows[:ROWS], 1)]
        if len(rows) > ROWS:
            lines.append(f'    … {len(rows) - ROWS} more')
        ax.text(0, 1, '\n'.join(lines), transform=ax.transAxes, va='top', ha='left', fontsize=8, family='monospace', color=COLORS['ink'])


def start(repetition):
    run = Run(task(repetition))
    s = run.settings
    log('viewer', 'start', target=TARGET, strategy=STRATEGY, operators=s['operator_count'], variables=s['variable_atom_count'], im=INTERNAL_MERGE,
        esel=ESEL, esur=ESUR, repetition=repetition, workspace_seed=run.seeds[0], derivation_seed=run.seeds[1])
    return run


def main():
    plt.rcParams['keymap.home'] = [key for key in plt.rcParams['keymap.home'] if key != 'r']
    viewer = Viewer(REPETITION)
    plt.show()
    return viewer


if __name__ == '__main__':
    main()
