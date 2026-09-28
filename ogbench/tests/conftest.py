"""Test configuration: put the package root on sys.path, pin JAX matmul
precision so float32 checks are not affected by TF32, headless MuJoCo."""

import os
import sys

os.environ.setdefault('JAX_DEFAULT_MATMUL_PRECISION', 'highest')
os.environ.setdefault('MUJOCO_GL', 'egl')

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
