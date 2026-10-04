"""Backward compatibility for the pre-rename ``AGID_*`` environment variables.

The project was renamed from AGID to SynthScope; every setting is now ``SYNTHSCOPE_<NAME>``. Old launch scripts that still
export ``AGID_<NAME>`` keep working: importing this module copies each such variable to its new name unless the new
name is already set. Dependency-free and idempotent, so any entry point can ``import _synthscope_env`` first.
"""
import os

LEGACY_PREFIX = "AGID_"
PREFIX = "SYNTHSCOPE_"


def apply_legacy_env() -> None:
    for key, value in list(os.environ.items()):
        if key.startswith(LEGACY_PREFIX):
            os.environ.setdefault(PREFIX + key[len(LEGACY_PREFIX):], value)


apply_legacy_env()
