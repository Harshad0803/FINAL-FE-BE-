"""Compatibility wrapper for legacy imports.

Older code and tests import the module as ``preprocessing``. The current
implementation lives in ``preprocessing_new.py``; re-export its public API here
so existing imports continue to work without changing callers.
"""

from preprocessing_new import *  # noqa: F401,F403
