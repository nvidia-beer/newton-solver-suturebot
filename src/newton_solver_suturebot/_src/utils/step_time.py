# SPDX-License-Identifier: Apache-2.0
"""The time step of a solver step, held on the device.

A CUDA graph stores scalar launch arguments by value: a kernel that takes the time
step as a number keeps the time step of the capture in every replay. Kernels of a
step therefore read the time step from the arrays of a :class:`StepTime`, which the
host writes before the launch. A graph captured once then serves every time step.

The host forms each value in Python double precision, exactly as it did when the
value was a launch argument, and stores it in the type of the former kernel
parameter. A kernel reads element 0 into that type; its arithmetic is unchanged.
"""

import warp as wp


class StepTime:
    """Device arrays that hold the time step ``dt`` [s] of the step being solved.

    One object serves one step: the solver that owns the step creates it, and the
    solvers and force laws it drives read the same arrays. The arrays are created
    once and never replaced, so a captured graph keeps addressing them.

    ``set`` is the only writer of the time step. It compares with the value written
    last and writes only a different one. A write during a CUDA graph capture would
    be recorded in the graph with the value of the capture, so ``set`` raises there:
    write the time step in the host-side preparation, before the capture and before
    every launch. Code that issues launches reading these arrays calls ``set`` with
    the time step it was given; inside a capture or after the preparation that is a
    comparison.

    Args:
        device: Device of the arrays.

    Attributes:
        seconds: ``dt`` [s], one ``float32`` (kernel parameters ``float``/``wp.float32``).
        seconds_f64: ``dt`` [s], one ``float64``.
        writes: Number of times a changed value has been written.
    """

    def __init__(self, device):
        self.device = wp.get_device(device)
        self.seconds = wp.zeros(1, dtype=wp.float32, device=self.device)
        self.seconds_f64 = wp.zeros(1, dtype=wp.float64, device=self.device)
        # Key of a host expression of dt -> [its one-element array, the expression, its parameters].
        self._derived = {}
        self._value = None
        self.writes = 0

    @property
    def value(self):
        """The time step [s] the arrays hold, as written last.

        ``None`` before the first write and after a write that did not complete.
        """
        return self._value

    def derived(self, key, expression, dtype=wp.float32, *, parameters=None):
        """One element of ``dtype`` that holds ``expression(dt)``, a value the host derives from ``dt``.

        ``expression`` takes the time step as a Python float and returns the number that
        the host passed as a launch argument before; ``dtype`` is the type of that kernel
        parameter. ``key`` names the value: its array is created at the first request for
        the key, outside capture, and kept for the lifetime of this object.

        ``parameters`` are the host values other than the time step that ``expression``
        uses, as one comparable object. A request with other parameters than the array
        was filled with takes the new expression and refills the same array. That is a
        write, so it raises during a CUDA graph capture: request the value in the
        host-side preparation after changing a parameter.
        """
        entry = self._derived.get(key)
        if entry is None:
            if self.device.is_capturing:
                raise RuntimeError("Request a value derived from the time step before CUDA graph capture")
            # Formed before the array exists: an expression that raises leaves no entry behind.
            value = None if self._value is None else expression(self._value)
            array = wp.zeros(1, dtype=dtype, device=self.device)
            if value is not None:
                array.fill_(value)
            entry = self._derived[key] = [array, expression, parameters]
        elif entry[0].dtype != dtype:
            raise ValueError(f"The value derived from the time step as {key!r} is held as another type")
        elif entry[2] != parameters:
            if self.device.is_capturing:
                raise RuntimeError(
                    f"The value derived from the time step as {key!r} changed during a CUDA graph capture; "
                    "request it before the capture"
                )
            value = None if self._value is None else expression(self._value)
            entry[1], entry[2] = expression, parameters
            if value is not None:
                entry[0].fill_(value)
        return entry[0]

    def fraction(self, parts):
        """One ``float32`` that holds ``dt / parts`` [s], for a step subdivided into equal parts."""
        if type(parts) is not int or parts < 1:
            raise ValueError("A step is divided into a positive integer number of parts")
        return self.derived(("fraction", parts), lambda dt: dt / parts)

    def set(self, dt):
        """Make the arrays hold ``dt`` [s]; nothing is written when they already do.

        Every value is formed on the host before the first write. If an expression
        raises, for example for a time step of zero, the arrays and ``value`` stay as
        they were.

        Raises:
            RuntimeError: If ``dt`` differs from the value held and the device is
                capturing a CUDA graph.
        """
        dt = float(dt)
        if dt == self._value:
            return
        if self.device.is_capturing:
            raise RuntimeError(
                f"The time step changed from {self._value} s to {dt} s during a CUDA graph capture; "
                "write it before the capture"
            )
        values = [(entry[0], entry[1](dt)) for entry in self._derived.values()]
        # Unknown until the last array is written: a write that is interrupted between
        # two arrays is repeated by the next set, whatever value that asks for.
        self._value = None
        self.seconds.fill_(dt)
        self.seconds_f64.fill_(dt)
        for array, value in values:
            array.fill_(value)
        self._value = dt
        self.writes += 1
