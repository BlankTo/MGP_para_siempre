import time

import config


def log(*parts, **values):
    print(time.strftime('%Y-%m-%d %H:%M:%S'), *parts, *(f'{k}={v}' for k, v in values.items()), flush=True)


def duration(seconds):
    if seconds is None:
        return 'na'
    s = int(round(seconds))
    if s < 60:
        return f'{s}s'
    m, s = divmod(s, 60)
    if m < 60:
        return f'{m}m{s:02d}s'
    h, m = divmod(m, 60)
    if h < 48:
        return f'{h}h{m:02d}m'
    d, h = divmod(h, 24)
    return f'{d}d{h:02d}h'


class Progress:

    def __init__(self, label, total, every=None, seconds=None):
        self.label, self.total = label, total
        self.every = max(1, int(every or config.PROGRESS_EVERY))
        self.seconds = config.PROGRESS_SECONDS if seconds is None else seconds
        self.n = 0
        self.started = self.last = time.perf_counter()

    def step(self, k=1, **values):
        before, self.n = self.n, self.n + k
        now = time.perf_counter()
        if self.n // self.every > before // self.every or now - self.last >= self.seconds:
            self.last = now
            self.report(now, **values)

    def report(self, now, **values):
        elapsed = now - self.started
        eta = elapsed / self.n * (self.total - self.n) if self.n and self.total else None
        pct = f'{100 * self.n / self.total:.1f}' if self.total else 'na'
        log(self.label, 'progress', done=f'{self.n}/{self.total}', pct=pct, elapsed=duration(elapsed), eta=duration(eta), **values)

    def end(self, **values):
        log(self.label, 'end', done=f'{self.n}/{self.total}', elapsed=duration(time.perf_counter() - self.started), **values)
