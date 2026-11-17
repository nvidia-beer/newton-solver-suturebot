# SPDX-License-Identifier: Apache-2.0
"""Shared pieces of the tests of an example's start-up check (examples/_startup.py)."""

import io
import json
from contextlib import contextmanager, redirect_stdout
from itertools import count
from types import SimpleNamespace
from unittest.mock import patch

import warp as wp

from newton_solver_suturebot._src.utils import capture

_serial = count(1)


@contextmanager
def marked_on_every_device():
    """Take the start-up mark on a CPU device as well, where the package takes none.

    The check of an example then runs on the CPU too: it sees no CUDA graph there, and the
    loads of the CPU builds of the kernel modules. On the CPU no start-up frame runs, so the
    kernel modules of the physics load with the first frame of the first scene of the process:
    a test runs one frame of another scene first (``prime``), as the eager start-up frame of a
    CUDA scene does.
    """
    finish = capture.finish_startup
    with patch.object(capture, "finish_startup", lambda begun, device: finish(begun, SimpleNamespace(is_cuda=True))):
        yield


def prime(example):
    """One frame of a scene that is not the one under test: the kernel modules of a frame are loaded."""
    with redirect_stdout(io.StringIO()):
        example.step()
        example.render()


def launch_late_kernel(device):
    """Launch a kernel of a module that the process has not loaded: one kernel module load, here and now."""
    value = float(next(_serial))

    @wp.kernel(enable_backward=False, module="unique")
    def late_startup_test_kernel(values: wp.array[float]):
        values[wp.tid()] = value

    wp.launch(late_startup_test_kernel, dim=1, inputs=[wp.zeros(1, dtype=float, device=device)], device=device)


def run_script(example, frames=None):
    """The frames of a ``--test`` run: step and render for the frames the script has left, or for ``frames`` frames."""
    script = example.script
    count = script.frames - script.frame if frames is None else frames
    with redirect_stdout(io.StringIO()):
        for _ in range(count):
            example.step()
            example.render()
    if frames is None and not script.complete:
        raise AssertionError(f"step() did not play the script: frame {script.frame} of {script.frames}")


def final_line(example):
    """Run ``test_final`` and return the line of numbers it prints last."""
    output = io.StringIO()
    with redirect_stdout(output):
        example.test_final()
    return json.loads(output.getvalue().strip().splitlines()[-1])
