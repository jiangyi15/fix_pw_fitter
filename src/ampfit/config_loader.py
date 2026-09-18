"""Config: raw YAML loader.

``Config`` owns only the raw input (``dic``, ``_config_path``,
``backend_spec``).  The interpreted physics (decay tree, index/tables, base
kernel config) lives on :class:`~ampfit.base_model.BaseModel`; build it with
``build_amplitude_model(config)`` (a model) or ``BaseModel(config.dic)``.
"""
import yaml
import numpy as np  # used by the __main__ dev block

from .base_model import (  # noqa: F401 (re-exported)
    BaseModel, row_block_factors, _projection_duplicate,
)
from .decay_tree import (  # noqa: F401 (Particle/Decay/... re-exported)
    DecayTree, Particle, Decay, DecayChain, DecayGroup,
)


def load_config(filename):
    if isinstance(filename, dict):
        return filename
    with open(filename) as f:
        ret = yaml.safe_load(f)
    return ret


class RawConfig:
    """Raw YAML loader: ``dic`` / ``_config_path`` / ``backend_spec`` only."""

    def __init__(self, filename):
        self.dic = load_config(filename)
        self._config_path = filename if isinstance(filename, str) else ""
        # optional ``config: {backend: ...}`` (or a top-level ``backend``)
        # selecting the compute backend for this config; ``None`` = default
        self.backend_spec = (self.dic.get("config") or {}).get("backend",
                                                              self.dic.get("backend"))


def Config(filename):
    """Legacy alias: ``Config(path)`` returns the amplitude model.

    Kept for callers that used ``Config`` as the entry point; the interpreted
    physics / kernel config all live on the returned model.  Prefer
    ``build_amplitude_model(...)`` (model) or ``RawConfig(...)`` (raw input).
    """
    from .amp_model import build_amplitude_model
    return build_amplitude_model(filename)


if __name__=="__main__":
    a = Config("config_angle.yml")
    c = a.build_all_index()
    from ampfit.numpy_kernel import NumpyKernel
    kernel  = NumpyKernel(c)
    ck = a.get_ck_map()
    # print(c)
    n_events = 7
    n = kernel._compute(
    {"ck": np.random.random(len(ck)) + 1j*np.random.random(len(ck)),
     "m0": np.random.random(len(a.m0_phys_name)) + 2,
     "g0": np.random.random(len(a.g0_phys_name)) + 0.01,
     "scalar": [0.6, 0.01, 0.506, 0.01, 0.9, 0.2],
    },
    {
     "mass": np.random.random((n_events, 2*3*8)),
     "q": np.random.random((n_events, 3*3*8)),
     "angle": np.random.random((n_events, 3*8, 3)),
     "frac": np.random.random((n_events, )),
     "time": np.random.random((n_events, )),
     "bkg": np.random.random((n_events, )),
     "weight": np.ones_like(np.random.random((n_events, ))),
    }
    )
    l = kernel._compute(
    {"ck": np.random.random(len(ck)) + 1j*np.random.random(len(ck)),
     "m0": np.random.random(len(a.m0_phys_name)) + 2,
     "g0": np.random.random(len(a.g0_phys_name)) + 0.01,
     "scalar": [0.6, 0.01, 0.506, 0.01, 0.9, 0.2],
    },
    {
     "mass": np.random.random((n_events, 2*3*8)),
     "q": np.random.random((n_events, 3*3*8)),
     "angle": np.random.random((n_events, 3*8, 3)),
     "frac": np.random.random((n_events, )),
     "time": np.random.random((n_events, )),
     "bkg": np.random.random((n_events, )),
     "weight": np.ones_like(np.random.random((n_events, ))),
    },
    norm=1.0
    )

    params = {"ck": np.random.random(len(ck)) + 1j*np.random.random(len(ck)),
     "m0": np.random.random(len(a.m0_phys_name)) + 2,
     "g0": np.random.random(len(a.g0_phys_name)) + 0.01,
     "scalar": [0.6, 0.01, 0.506, 0.01, 0.9, 0.2],
    }
    data =     {
     "mass": np.random.random((n_events, 2*3*8)),
     "q": np.random.random((n_events, 3*3*8)),
     "angle": np.random.random((n_events, 3*8, 3)),
     "frac": np.random.random((n_events, )),
     "time": np.random.random((n_events, )),
     "bkg": np.random.random((n_events, )),
     "weight": np.ones_like(np.random.random((n_events, ))),
    }


    l = kernel._compute(params,
    data,
    norm=1.0
    )
    l2 = [kernel._compute(params,
    {k: v[i:i+1] for k, v in data.items()},
    norm=1.0
    ) for i in range(n_events)]
    print(l[0], sum(i[0] for i in l2))
    print(l[2], [i[2] for i in l2])
