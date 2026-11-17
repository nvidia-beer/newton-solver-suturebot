"""Exact experimental backend eligibility; native remains the default."""

from .subcycled_mujoco import SubcycledMuJoCo


def is_gpu_backend(solver):
    from .subcycled_mujoco_warp import SubcycledMuJoCoWarp

    return type(solver) is SubcycledMuJoCoWarp and getattr(solver.step, "__func__", None) is SubcycledMuJoCoWarp.step


def supported_subcycled(solver):
    return (
        type(solver) is SubcycledMuJoCo
        and getattr(solver.step, "__func__", None) is SubcycledMuJoCo.step
        and solver.use_mujoco_cpu
    ) or is_gpu_backend(solver)
