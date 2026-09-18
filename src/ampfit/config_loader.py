"""Config: raw YAML loader + the interpreted BaseModel it builds.

``Config`` owns the raw input (``dic``, ``_config_path``, ``backend_spec``)
and builds ``self.base_model`` (a :class:`~ampfit.base_model.BaseModel`) —
the decay tree, the index/tables and the base kernel config.  Attribute
access for that interpreted physics is delegated to the BaseModel, so
callers can keep using ``cfg.full_decay`` / ``cfg.build_all_index()`` / ...
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


class Config:
    """Raw YAML loader that builds the interpreted :class:`BaseModel`.

    Owns the raw input (``dic`` / ``_config_path`` / ``backend_spec``) and
    builds ``self.base_model``.  Interpreted-physics attributes are delegated
    to that BaseModel for backward compatibility.
    """

    def __init__(self, filename):
        self.dic = load_config(filename)
        self._config_path = filename if isinstance(filename, str) else ""
        # optional ``config: {backend: ...}`` (or a top-level ``backend``)
        # selecting the compute backend for this config; ``None`` = default
        self.backend_spec = (self.dic.get("config") or {}).get("backend",
                                                              self.dic.get("backend"))
        # the interpreted physical model (decay tree, index/tables, base kc)
        self.base_model = BaseModel(self.dic, self._config_path)

    def build_base_model(self):
        """The interpreted :class:`BaseModel` (built at construction)."""
        return self.base_model

    def __getattr__(self, name):
        # Only called when normal lookup fails: delegate the interpreted
        # physics to the BaseModel facade.
        try:
            base = object.__getattribute__(self, "base_model")
        except AttributeError:
            raise AttributeError(name) from None
        return getattr(base, name)


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
