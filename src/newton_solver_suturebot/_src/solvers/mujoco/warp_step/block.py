# SPDX-License-Identifier: Apache-2.0
"""One CUDA block per world: the launch form of the fused kernels.

A fused kernel runs several stages that MuJoCo Warp launches one after the other. It is
launched with ``wp.launch_tiled(kernel, dim=nworld, block_dim=block_dim(width), ...)``,
which gives every world one block of ``block_dim`` threads; ``wp.tid()`` returns the world
and the thread of its block. The threads stride over the items of a stage
(``range(tid, count, wp.block_dim())``), and ``block_barrier()`` after a stage keeps a
thread from reading what another thread of its world has not written yet. With any other
launch a block does not hold exactly the threads of one world, and the barrier orders
nothing. On the CPU a block is one thread, which runs the stages in their order.
"""

import warp as wp

# Most threads of a block. The widest stage takes one thread per item, as the launch it
# stands for does, and 256 threads is the block of a default Warp launch.
BLOCK_DIM_MAX = 256
# Threads of one CUDA warp; a block is made of whole warps.
_WARP = 32


def block_dim(width):
    """Threads per world of a fused launch whose widest stage has ``width`` items."""
    return _WARP * ((width + _WARP - 1) // _WARP)


@wp.func_native(snippet="WP_TILE_SYNC();")
def block_barrier(): ...
