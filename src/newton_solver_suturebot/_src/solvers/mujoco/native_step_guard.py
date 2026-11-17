# SPDX-License-Identifier: Apache-2.0
"""Reject native automatic resets before a rigid trial can be exported.

Warning slots are diagnostic state, not integration state. MuJoCo resets the
bad-Q counter to one during automatic reset, so comparing cumulative counts
cannot detect repeated resets. Temporarily clearing only three bad-Q counters
provides a fresh sentinel, then restores their cumulative diagnostic values.
No model option, tolerance, force, coordinate or integration budget changes.
"""

import math

from ..coupled.interface.insertion import NeedleStepTooLarge


class NativeRigidStepRejected(NeedleStepTooLarge):
    """Native integration failed; the enclosing transaction must refine/retry."""


def snapshot_native_warnings(data):
    # The native binding exposes strided views of the same mjWarningStat
    # storage. Avoid constructing a Python record proxy for every slot.
    warnings = data.warning
    return tuple(zip(warnings.number.tolist(), warnings.lastinfo.tolist(), strict=True))


def restore_native_warnings(data, snapshot):
    warnings = data.warning
    if len(snapshot) != len(warnings.number):
        raise ValueError("Native warning snapshot has the wrong number of records")
    if snapshot:
        numbers, lastinfo = zip(*snapshot, strict=True)
        warnings.number[:] = numbers
        warnings.lastinfo[:] = lastinfo


class NativeMicrostepGuard:
    """Small host-only guard owned by one native solver and its ordered calls."""

    def __init__(self, mujoco, data):
        self._data = data
        self.indices = tuple(
            int(getattr(mujoco.mjtWarning, name)) for name in ("mjWARN_BADQPOS", "mjWARN_BADQVEL", "mjWARN_BADQACC")
        )
        self.names = ("QPOS", "QVEL", "QACC")
        # mj_resetData/mj_setState modify this MjData's contents in place.
        # Keep data alive and retain its native array views, never raw pointers.
        self._numbers = data.warning.number
        self._lastinfo = data.warning.lastinfo

    @property
    def data(self):
        """The MjData owning these views is fixed for this guard's lifetime."""
        return self._data

    def begin(self, dt):
        data = self.data
        old = tuple((int(self._numbers[index]), int(self._lastinfo[index])) for index in self.indices)
        start = float(data.time)
        for index in self.indices:
            self._numbers[index] = 0
        return start, float(dt), old

    def check(self, token, *, integrated=True):
        start, dt, _ = token
        data = self.data
        for index, name in zip(self.indices, self.names, strict=True):
            if self._numbers[index]:
                raise NativeRigidStepRejected(
                    f"Native {name} warning at DOF {self._lastinfo[index]}; reject rigid trial starting at {start:.17g}s"
                )
        expected = start + dt if integrated else start
        actual = float(data.time)
        slack = 16 * math.ulp(max(1.0, abs(start), abs(expected)))
        if not math.isfinite(actual) or abs(actual - expected) > slack:
            raise NativeRigidStepRejected(
                f"Native clock changed unexpectedly: {start:.17g}s -> {actual:.17g}s, expected {expected:.17g}s"
            )

    def finish(self, token):
        # Preserve cumulative warning diagnostics, including their old lastinfo
        # when no new warning was emitted. Failed outer attempts restore the
        # complete warning snapshot separately, while rejection counts persist.
        _, _, old = token
        for index, (number, lastinfo) in zip(self.indices, old, strict=True):
            new = int(self._numbers[index])
            self._numbers[index] = number + new
            if not new:
                self._lastinfo[index] = lastinfo
