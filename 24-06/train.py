"""Compatibility wrapper for legacy imports.

Older tests import ``train`` directly. Re-export the current training helpers
from the newer module namespace so the test suite can continue to import them.
"""

from train_new import *  # noqa: F401,F403
