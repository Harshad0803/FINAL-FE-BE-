"""Compatibility shim for the evaluation module.

The parity tests import this file both from the project root and from the
legacy reference repository path. The current implementation lives in
``evaluate_new.py``; re-export its public API here so the tests can exercise
it without depending on an external checkout.
"""

from evaluate_new import *  # noqa: F401,F403
