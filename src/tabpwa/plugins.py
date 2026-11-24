"""Config-declared plugin imports: ``plugin: [module, ...]``.

A plugin is a Python MODULE whose import has the side effect the fit needs —
registering backends (`tabpwa_mpi`), amplitude models (`@register_amplitude_model`)
or particle models (`@register_model`).  A config names the modules it needs::

    plugin:
      - tabpwa_mpi
      - my_lab.models

and :class:`tabpwa.fitter.Fitter` imports them right after loading the raw
config, BEFORE it builds the amplitude model or resolves the backend — so no
entry-point machinery, no `import` in the calling script, and the dependency
stays visible in the config that needs it.

A config is trusted input: importing a plugin executes its code, exactly as a
Python entry point would.  Import is idempotent (`sys.modules`), and a failure
names the plugin and where it was declared.
"""

from __future__ import annotations

import importlib


def load_plugins(spec, where=""):
    """Import every module named by *spec* (a name, a list, or ``None``).

    Returns the names imported.  Raises :class:`ImportError` naming the plugin
    (and *where* it was declared) when one cannot be imported.
    """
    if spec is None:
        return []
    if isinstance(spec, str):
        names = [spec]
    elif isinstance(spec, (list, tuple)):
        names = list(spec)
    else:
        raise TypeError(
            f"plugin{'' if not where else ' (' + where + ')'}: expected a module "
            f"name or a list of names, got {type(spec).__name__} {spec!r}")
    loaded = []
    for name in names:
        name = str(name).strip()
        if not name:
            continue
        try:
            importlib.import_module(name)
        except Exception as exc:
            origin = f" ({where})" if where else ""
            raise ImportError(
                f"plugin {name!r}{origin} could not be imported: "
                f"{type(exc).__name__}: {exc}") from exc
        loaded.append(name)
    return loaded
