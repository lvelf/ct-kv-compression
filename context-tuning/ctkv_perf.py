"""Phase profiler for CT-KV adaptation: wall time and CUDA memory per phase.

Phases are flat (never nested). Each is bracketed by torch.cuda.synchronize(), so its time is the GPU-complete
wall time; the synchronizations are part of the measured adaptation time. Peak memory is reset at each phase
start, so `peak` is the max allocated inside the phase and `transient` = peak − allocated at phase start
(activations / workspaces the phase needs on top of what is already resident); `resident_at_start` = max
allocated when the phase starts (weights + KV + grads + optimizer state + anything else kept alive).
"""
import time
from collections import defaultdict
from contextlib import contextmanager

import torch


def tensor_bytes(ts):
    return int(sum(t.numel() * t.element_size() for t in ts if t is not None))


class PhaseProfiler:
    def __init__(self):
        self.time = defaultdict(float)
        self.calls = defaultdict(int)
        self.peak = defaultdict(int)
        self.transient = defaultdict(int)
        self.resident = defaultdict(int)
        self.step_s = []
        self.static = {}
        self.overall_peak = 0
        self.overall_reserved = 0

    @contextmanager
    def __call__(self, name):
        torch.cuda.synchronize()
        a0 = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        t = time.perf_counter()
        try:
            yield
        finally:
            torch.cuda.synchronize()
            self.time[name] += time.perf_counter() - t
            self.calls[name] += 1
            pk = torch.cuda.max_memory_allocated()
            self.peak[name] = max(self.peak[name], pk)
            self.transient[name] = max(self.transient[name], pk - a0)
            self.resident[name] = max(self.resident[name], a0)
            self.overall_peak = max(self.overall_peak, pk)
            self.overall_reserved = max(self.overall_reserved, torch.cuda.max_memory_reserved())

    def note(self, **kw):
        self.static.update({k: int(v) for k, v in kw.items()})

    def summary(self, adaptation_s=None):
        phases = {k: dict(s=self.time[k], calls=self.calls[k], peak_bytes=self.peak[k],
                          transient_bytes=self.transient[k], resident_at_start_bytes=self.resident[k]) for k in self.time}
        out = dict(phases=phases, static_bytes=self.static, overall_peak_bytes=self.overall_peak,
                   step_s=self.step_s, reserved_peak_bytes=int(self.overall_reserved))
        if adaptation_s is not None:
            covered = sum(v["s"] for k, v in phases.items() if k != "eval")
            out["adaptation_s"] = adaptation_s
            out["unattributed_s"] = adaptation_s - covered
        return out
