# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Body-tree passes of MuJoCo Warp's smooth dynamics with every tree level in one launch.

``com_pos``, ``crb`` and ``rne`` of ``mujoco_warp._src.smooth`` accumulate subtree
centers of mass, composite inertias and internal forces from the leaves of the
body tree to its root, one kernel launch per tree level. Every launch is a node
with a fixed cost in a replayed CUDA graph. ``TreePasses`` writes the same arrays
with one launch per accumulation and launches the other kernels of the three
functions as MuJoCo Warp does.

A fused kernel gives every world one CUDA block (``block``). Each thread of the
block strides over the bodies of a level, and a block barrier after a level keeps
a body from adding its sum to its parent before all of its children have added
to it.

The fused kernels repeat the arithmetic of the MuJoCo Warp 3.17.0 kernels they stand
for (``mujoco_warp/_src/smooth.py``, Apache-2.0). On the CPU one thread per world
visits the bodies in the order of the per-level launches, so the results equal
MuJoCo Warp's bit for bit. On CUDA siblings add to their parent in an unspecified
order, in these kernels as in the per-level launches.
"""

import warp as wp
from mujoco_warp._src.smooth import _M, _cdof, _cinert, _qfrc_bias, _rne_cacc_forward, _rne_cacc_world, _rne_cfrc
from mujoco_warp._src.types import DeterminismType, vec10

from .block import BLOCK_DIM_MAX, block_barrier, block_dim


@wp.kernel(module="unique", enable_backward=False)
def _subtree_com_fused(
    # Model:
    nbody: int,
    body_parentid: wp.array[int],
    body_mass: wp.array2d[float],
    body_subtreemass: wp.array2d[float],
    # Data in:
    xipos_in: wp.array2d[wp.vec3],
    # In:
    body_levels: wp.array[int],
    body_level_start: wp.array[int],
    nlevel: int,
    # Data out:
    subtree_com_out: wp.array2d[wp.vec3],
):
    """_subtree_com_init, every level of _subtree_com_acc and _subtree_div, one block per world."""
    worldid, tid = wp.tid()
    BLOCK_DIM = wp.block_dim()

    for bodyid in range(tid, nbody, BLOCK_DIM):
        subtree_com_out[worldid, bodyid] = xipos_in[worldid, bodyid] * body_mass[worldid % body_mass.shape[0], bodyid]
    block_barrier()

    # deepest level first; level 0 holds only the world body, which has no parent to add to
    for i in range(1, nlevel):
        level = nlevel - i
        for nodeid in range(body_level_start[level] + tid, body_level_start[level + 1], BLOCK_DIM):
            bodyid = body_levels[nodeid]
            pid = body_parentid[bodyid]
            wp.atomic_add(subtree_com_out, worldid, pid, subtree_com_out[worldid, bodyid])
        block_barrier()

    for bodyid in range(tid, nbody, BLOCK_DIM):
        mass = body_subtreemass[worldid % body_subtreemass.shape[0], bodyid]
        if mass != 0.0:
            subtree_com_out[worldid, bodyid] = subtree_com_out[worldid, bodyid] / mass


@wp.kernel(module="unique", enable_backward=False)
def _crb_fused(
    # Model:
    nbody: int,
    body_parentid: wp.array[int],
    # Data in:
    cinert_in: wp.array2d[vec10],
    # In:
    body_levels: wp.array[int],
    body_level_start: wp.array[int],
    nlevel: int,
    # Data out:
    crb_out: wp.array2d[vec10],
):
    """Copy of cinert and _crb_accumulate for every tree level, one block per world."""
    worldid, tid = wp.tid()
    BLOCK_DIM = wp.block_dim()

    for bodyid in range(tid, nbody, BLOCK_DIM):
        crb_out[worldid, bodyid] = cinert_in[worldid, bodyid]
    block_barrier()

    # deepest level first; the bodies of level 1 are the children of the world body, which
    # accumulates no composite inertia
    for i in range(1, nlevel - 1):
        level = nlevel - i
        for nodeid in range(body_level_start[level] + tid, body_level_start[level + 1], BLOCK_DIM):
            bodyid = body_levels[nodeid]
            pid = body_parentid[bodyid]
            wp.atomic_add(crb_out, worldid, pid, crb_out[worldid, bodyid])
        block_barrier()


@wp.kernel(module="unique", enable_backward=False)
def _cfrc_backward_fused(
    # Model:
    body_parentid: wp.array[int],
    # In:
    body_levels: wp.array[int],
    body_level_start: wp.array[int],
    nlevel: int,
    # Data out:
    cfrc_int_out: wp.array2d[wp.spatial_vector],
):
    """_cfrc_backward for every tree level, one block per world."""
    worldid, tid = wp.tid()
    BLOCK_DIM = wp.block_dim()

    # deepest level first; level 0 holds only the world body, which has no parent to add to
    for i in range(1, nlevel):
        level = nlevel - i
        for nodeid in range(body_level_start[level] + tid, body_level_start[level + 1], BLOCK_DIM):
            bodyid = body_levels[nodeid]
            pid = body_parentid[bodyid]
            wp.atomic_add(cfrc_int_out[worldid], pid, cfrc_int_out[worldid, bodyid])
        block_barrier()


class TreePasses:
    """``com_pos``, ``crb`` and ``rne`` of MuJoCo Warp for one Model and Data, one launch per tree pass.

    The level tables and the block size depend on the body tree only. They are built
    here, outside any graph capture: ``body_levels`` holds the body ids of all tree
    levels one after the other and ``body_level_start`` the offset of every level in
    it, followed by the body count. Every launch is on the device of ``data``, whatever
    the default device of the caller is.

    Args:
        model: MuJoCo Warp ``Model``.
        data: MuJoCo Warp ``Data`` of that model; the methods read and write its arrays.

    Raises:
        ValueError: If the model asks for deterministic atomics, which only the
            per-level launches provide, or if a tree level holds more than 256 bodies.
    """

    def __init__(self, model, data):
        if model.opt.deterministic & DeterminismType.ATOMICS:
            raise ValueError("Fused tree passes do not provide deterministic atomics")
        width = max(level.size for level in model.body_tree)
        if width > BLOCK_DIM_MAX:
            raise ValueError(f"A tree level of {width} bodies exceeds the {BLOCK_DIM_MAX} of a fused tree pass")
        self.model = model
        self.data = data
        self.device = data.subtree_com.device
        self.nlevel = len(model.body_tree)
        self.block_dim = block_dim(width)
        self.body_levels = wp.empty(model.nbody, dtype=int, device=self.device)
        starts = [0]
        for level in model.body_tree:
            wp.copy(self.body_levels, level, dest_offset=starts[-1], count=level.size)
            starts.append(starts[-1] + level.size)
        self.body_level_start = wp.array(starts, dtype=int, device=self.device)

    def com_pos(self):
        """Write ``subtree_com``, ``cinert`` and ``cdof`` as ``smooth.com_pos`` does."""
        m, d = self.model, self.data
        with wp.ScopedDevice(self.device):
            wp.launch_tiled(
                _subtree_com_fused,
                dim=d.nworld,
                inputs=[
                    m.nbody,
                    m.body_parentid,
                    m.body_mass,
                    m.body_subtreemass,
                    d.xipos,
                    self.body_levels,
                    self.body_level_start,
                    self.nlevel,
                ],
                outputs=[d.subtree_com],
                block_dim=self.block_dim,
            )
            wp.launch(
                _cinert,
                dim=(d.nworld, m.nbody),
                inputs=[m.body_rootid, m.body_mass, m.body_inertia, d.xipos, d.ximat, d.subtree_com],
                outputs=[d.cinert],
            )
            wp.launch(
                _cdof,
                dim=(d.nworld, m.njnt),
                inputs=[
                    m.body_rootid,
                    m.jnt_type,
                    m.jnt_dofadr,
                    m.jnt_bodyid,
                    d.xmat,
                    d.xanchor,
                    d.xaxis,
                    d.subtree_com,
                ],
                outputs=[d.cdof],
            )

    def crb(self):
        """Write ``crb`` and ``M`` as ``smooth.crb`` does."""
        m, d = self.model, self.data
        with wp.ScopedDevice(self.device):
            wp.launch_tiled(
                _crb_fused,
                dim=d.nworld,
                inputs=[m.nbody, m.body_parentid, d.cinert, self.body_levels, self.body_level_start, self.nlevel],
                outputs=[d.crb],
                block_dim=self.block_dim,
            )
            d.M.zero_()
            wp.launch(
                _M,
                dim=(d.nworld, m.nv),
                inputs=[m.dof_bodyid, m.dof_parentid, m.dof_armature, m.M_rownnz, m.M_rowadr, d.cdof, d.crb],
                outputs=[d.M],
            )

    def rne(self):
        """Write ``cacc``, ``cfrc_int`` and ``qfrc_bias`` as ``smooth.rne`` does, without the joint accelerations."""
        m, d = self.model, self.data
        with wp.ScopedDevice(self.device):
            _rne_cacc_world(m, d)
            _rne_cacc_forward(m, d, flg_acc=False)
            _rne_cfrc(m, d)
            wp.launch_tiled(
                _cfrc_backward_fused,
                dim=d.nworld,
                inputs=[m.body_parentid, self.body_levels, self.body_level_start, self.nlevel],
                outputs=[d.cfrc_int],
                block_dim=self.block_dim,
            )
            wp.launch(
                _qfrc_bias, dim=[d.nworld, m.nv], inputs=[m.dof_bodyid, d.cdof, d.cfrc_int], outputs=[d.qfrc_bias]
            )
