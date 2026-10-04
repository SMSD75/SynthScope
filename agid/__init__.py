"""Deprecated alias of :mod:`synthscope` (the project was renamed from AGID to SynthScope).

``import agid`` keeps working and returns the very same module object, so ``agid.load(...)``,
``from agid.api import Detector`` and ``python -m agid`` behave exactly as before.
"""
import sys

import synthscope
from synthscope import api, cli, hub

sys.modules[__name__] = synthscope
for _name, _mod in (("api", api), ("cli", cli), ("hub", hub)):
    sys.modules[f"{__name__}.{_name}"] = _mod
