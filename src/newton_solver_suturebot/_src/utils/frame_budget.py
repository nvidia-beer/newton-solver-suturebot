# SPDX-License-Identifier: Apache-2.0
"""How long one more substep of a coupled frame can take, measured on the host clock.

A scene advances a frame in substeps: accepted coupled steps, each with the rejected attempts
before it. A driver that has a time budget for one call may stop between two attempts (between
two substeps, or between a rejected attempt and its retry) and go on with the same frame in its
next call; it cannot stop inside an attempt. Before it starts an attempt it therefore asks
whether a substep that uses up its trials still fits. :class:`SubstepBound` gives the length
of such a substep, for the worst case that the scene has measured since it was reset::

    bound = outside + loops * max(trial_limit * per_trial, loop)

``outside`` is the longest time a substep took outside its trial loops, and ``loop`` the
longest time a trial loop took (of a substep with several loops: their mean). ``per_trial``
is the time the trial loops took per trial, a mean over every substep measured, and
``trial_limit`` the largest number of trials one loop evaluates: a loop is taken as no less
than ``trial_limit`` mean trials, which stands for a loop that uses up its trials before one
has been measured. ``loops`` is the number of trial loops the last substep ran: one for an
accepted attempt whose device loop converged, one more for a handover to the host-controlled
loop and for every rejected attempt. Such substeps come in rows (a needle entry or exit, a
tip that leaves the tissue), so the next substep is taken to run as many loops as the last
one did, each to the trial limit. Nothing is read from a device: the scene takes the host
clock where a substep starts and ends, and the coupler counts its trial loops and takes the
clock around them.

What follows the scene's step in a call of the driver (the observation of the accepted state,
the drawing) is the driver's to take off the deadline it gives, by the same rule: as long as
the longest it has measured, not as long as the last one.

An attempt can still take longer than the bound: one that is slower than everything measured
before it (it is in the bound from then on), and one that runs more loops than the substep
before it (the first handover to the host-controlled loop of a row). The driver cannot split
an attempt, so such an attempt ends after the time the driver planned for.

A maximum does not come down. One substep that took longer than every substep before it, for
whatever reason, stays in the bound until the scene is reset, and so does one long
observation or drawing in the deadline. It enters with its own excess and no more: ``loop``
is the time of a loop, not a time per trial that the trial limit would multiply, and the
mean trial moves by the share that the excess has in all the trials measured. What it costs
is calls, not time between two pictures: fewer attempts start as a further attempt of a
call, at worst none, and every call still runs one.

What in the bound is a choice, and not a maximum that the run has measured:

- ``trial_limit * per_trial``, the least a loop is taken to be. It is made of a mean, so a
  loop of ``trial_limit`` trials can take longer (it is ``loop`` from then on).
- ``loops``: the next substep is taken to repeat the loops of the last one, a prediction.
- No maximum is forgotten before the reset, as said above.

Measured in the window of the grasp example (2026-11-17, ``docs/usage.md``): after the first
attempt of the run the bound, the longest observation and the longest drawing were together
over the budget of 66.7 ms, so every call ran one attempt and none started a second one.

That is the unit ``attempt``. With the unit ``host-trial`` a driver may also stop inside an
attempt, at one kind of place: between two host-controlled trials, the trials that a loop
runs under the control of the host at a needle surface event. The attempt then stays open,
and the next call goes on with the trial that was not started. :class:`HostTrialBudget` says
whether one more such trial, with what may follow it without another question, still fits. A
device trial loop stays one piece under either unit.
"""

import math
import time


class SubstepBound:
    """Bound [s] for one more substep, from the substeps measured since the scene was reset.

    Attributes:
        substeps: Substeps measured: a count, which the bound does not read.
        trials: Trials they evaluated, rejected attempts included.
        loop_seconds: Host-clock time [s] of their trial loops.
        last_loops: Trial loops of the substep measured last.
        longest_loop: Longest time [s] of a trial loop; of a substep with several loops, the
            mean of its loops.
        longest_outside: Longest time [s] of a substep outside its trial loops.
    """

    def __init__(self):
        self.substeps = 0
        self.trials = 0
        self.loop_seconds = 0.0
        self.last_loops = 1
        self.longest_loop = 0.0
        self.longest_outside = 0.0

    def add(self, trials, loops, loop_seconds, seconds):
        """Count one substep; a stretch of time without a trial is not a substep and is left out.

        ``seconds`` is its host-clock time [s], from its start to the start of the next one or to
        the end of its frame.
        """
        if trials < 1:
            return
        self.substeps += 1
        self.trials += int(trials)
        self.loop_seconds += float(loop_seconds)
        self.last_loops = max(int(loops), 1)
        self.longest_loop = max(self.longest_loop, float(loop_seconds) / self.last_loops)
        self.longest_outside = max(self.longest_outside, float(seconds) - float(loop_seconds))

    @property
    def per_trial(self):
        """Mean time of a trial [s], or None before the first measured substep."""
        return self.loop_seconds / self.trials if self.trials else None

    def bound(self, trial_limit):
        """Longest time [s] of a substep with the loops of the last one, each run to ``trial_limit`` trials; infinite before any measurement.

        A loop is taken as the longest loop measured, and as no less than ``trial_limit`` mean trials.
        """
        if not self.trials:
            return math.inf
        return self.longest_outside + self.last_loops * max(trial_limit * self.per_trial, self.longest_loop)


class HostTrialBudget:
    """Admits the host-controlled trials of one call of a scene's step against the deadline of that call.

    The scene decides by its own rule whether an attempt starts in a call
    (``_attempt_fits``). Inside an attempt the coupler asks this object before every
    host-controlled trial (``admit("host")``) and before the checks and the commit of the
    attempt (``admit("end")``, never refused). A refused trial is the first thing the next
    call runs: the attempt stays open in between.

    A host-controlled trial can be the last of its loop. The checks and the commit of the
    attempt, the scene's work after an acceptance and, in the last substep of a frame, the end
    of the frame then follow without a question. The bound of a trial is therefore the
    longest host-controlled trial plus the longest such tail that this object has measured,
    both on the host clock, since the scene was reset. Before both have been measured a
    trial is admitted only as the first piece of a call.

    The first piece of a call that goes on with an open attempt is always admitted: a call
    makes progress. Without a deadline everything is admitted. Nothing is read from a device.

    Attributes:
        deadline: Deadline of the call in progress on the clock, or None.
        longest_host: Longest host-controlled trial measured since the reset, or None. A trial
            lasts from its admission to the next question, to :meth:`close`, or to the end of
            the call.
        longest_tail: Longest time from the ``"end"`` of an attempt to :meth:`close`, or None.
        admitted: Pieces admitted since the reset, by kind.
        refused: Pieces refused since the reset, by kind.
    """

    def __init__(self, clock=time.perf_counter):
        self._clock = clock
        self.deadline = None
        self.longest_host = None
        self.longest_tail = None
        self.admitted = {}
        self.refused = {}
        self._first = False
        self._timed = None  # (kind, start) of the host trial or the tail in progress

    def begin_call(self, deadline, resumed=False):
        """Start a call.

        Args:
            deadline: A time of the clock, or None for a call that finishes its frame.
            resumed: Whether the call goes on with an attempt that the call before left open.
                Its first piece is then admitted whatever the deadline says.
        """
        self.deadline = deadline
        self._first = bool(resumed)
        self._timed = None

    def bound(self):
        """Bound of one host-controlled trial with what may follow it without a question, or None."""
        if self.longest_host is None or self.longest_tail is None:
            return None
        return self.longest_host + self.longest_tail

    def admit(self, kind):
        """Whether a piece may start now: ``"host"`` (one host-controlled trial) or ``"end"`` (never refused)."""
        now = self._clock()
        self._measure(now)
        if self.deadline is None or self._first or kind != "host":
            admitted = True
        else:
            bound = self.bound()
            admitted = bound is not None and now + bound <= self.deadline
        counter = self.admitted if admitted else self.refused
        counter[kind] = counter.get(kind, 0) + 1
        if admitted:
            self._first = False
            self._timed = (kind, now)
        return admitted

    def close(self):
        """The piece in progress has ended: after a rejected attempt, after the work that follows an
        acceptance when another substep follows, and after the end of the frame."""
        self._measure(self._clock())

    def _measure(self, now):
        timed, self._timed = self._timed, None
        if timed is None:
            return
        kind, started = timed
        name = "longest_host" if kind == "host" else "longest_tail"
        longest = getattr(self, name)
        if longest is None or now - started > longest:
            setattr(self, name, now - started)
