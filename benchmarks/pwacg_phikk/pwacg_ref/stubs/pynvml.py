"""Import stub for `pynvml`, imported at module scope by PWACG's generated fit
object and used only inside its `Control.run` (multi-GPU bookkeeping).  The
reference harness times `PWAFunc` directly, so the import is dead on the timed
path; anything that actually asks for device memory through it raises instead of
returning a made-up figure.
"""


def _unavailable(*_a, **_kw):
    raise RuntimeError(
        "pynvml is stubbed in this benchmark: it is imported by PWACG's "
        "generated module but used only by Control.run; measure device memory "
        "with tabpwa.monitor instead")


def nvmlInit(*a, **kw):
    _unavailable()


def nvmlDeviceGetCount(*a, **kw):
    _unavailable()


def nvmlDeviceGetHandleByIndex(*a, **kw):
    _unavailable()


def nvmlDeviceGetMemoryInfo(*a, **kw):
    _unavailable()
