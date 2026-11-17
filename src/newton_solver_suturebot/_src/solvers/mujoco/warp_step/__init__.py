# SPDX-License-Identifier: Apache-2.0
"""Rigid MuJoCo Warp step of the needle and tool model.

``RigidStep`` advances a MuJoCo Warp ``Data`` as ``mujoco_warp.step`` does, with one
launch per body-tree pass (``tree``), a constraint solve on a kept context with fused
launches (``constraint_solver``) and a pool that serves the temporary arrays of the
other stages (``scratch``), so that the step can be captured in a conditional CUDA graph.
The fused kernels share one launch form (``block``). ``companion_data`` gives a second
``Data`` on which a step that repeats from one state keeps the stages no repetition changes.
"""

from .step import RigidStep, companion_data

__all__ = ["RigidStep", "companion_data"]
