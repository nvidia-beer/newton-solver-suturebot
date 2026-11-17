# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Eyelets of a thread along the track that a needle has actually cut through the tissue.

The needle/tissue interaction records where the needle tip went: one row per
material point, a quarter millimetre apart, in sections (one per entry of the
tip into material: a passage). The thread that hangs on the needle's tail must
later run along that track. :class:`NeedleTrackEyelets` makes eyelets of the
thread/tissue coupling from those rows while the needle goes in, one at a time
and long before the tail arrives, each written in place into the coupling's
fixed reserve. The coupling threads an eyelet by itself when the tail has gone
past it (``ThreadTissueCoupling.follow_tail``).

Everything here is host bookkeeping on the interaction's own host lists.
Nothing is read from the device.
"""

import numpy as np


class NeedleTrackEyelets:
    """Turns the rows of a needle's track into eyelets of a thread/tissue coupling, one group per passage.

    Call :meth:`advance` once after every accepted step, before the thread's step is launched.
    Of the rows that the interaction has recorded since the last call:

    - the first row of a section becomes an eyelet: the hole where the thread goes in;
    - a later row becomes one when its rest position is at least ``spacing`` from the last
      eyelet of its section;
    - when the interaction reports an ``exit`` of the tip, the last row of the section that the
      tip was laying becomes one, the hole where the thread comes out, unless it is nearer than
      half a spacing to the last eyelet of that section: that eyelet then serves, and no row is
      written twice. This presumes that the tip leaves a passage at the far end of the rows it
      laid; a tip that is withdrawn through its entry leaves an eyelet at its deepest row, which
      no tail passes and which therefore carries no force.

    The eyelets of one section are one track of the coupling, in the order in which the tip laid
    them, which is the order in which the tail passes them. Eyelets of one passage are then
    between half a spacing and one spacing and a row apart.

    The coupling's reserve is fixed. An eyelet that finds no slot or no track left is not made;
    the coupling counts it (``refused``), and the spacing goes on from it as if it had been made.

    Args:
        coupling: The ``ThreadTissueCoupling``. Its tetrahedra must be those of the interaction,
            in the same order.
        scene: Object whose ``interaction`` is the needle/tissue interaction (``track_count``,
            ``track_rows``, ``events``), or None before there is one. It is read at every call.
        spacing: Rest distance between the eyelets of one passage [m].

    Attributes:
        passages: The slots of each track of the coupling that was made here, by track index,
            in the order the tail passes them.
        rows: Number of track rows taken so far.
    """

    def __init__(self, coupling, scene, spacing):
        if not np.isfinite(spacing) or spacing <= 0:
            raise ValueError("The eyelet spacing must be finite and positive")
        self.coupling, self.scene, self.spacing = coupling, scene, float(spacing)
        self.passages = []
        self.rows = 0
        self._source = None
        # Per section: its track in the coupling, the rest position of its last eyelet, and its last row
        # (tetrahedron, weights, rest position). Then the section of the row taken last.
        self._track, self._eyelet_rest, self._last_row = {}, {}, {}
        self._section = None

    @property
    def refused(self):
        """Number of eyelets that found no slot or no track left in the coupling's reserve."""
        return self.coupling.refused_points

    def _make(self, section, tetrahedron, weights, rest):
        track = self._track.get(section, self.coupling.track_count)
        slot = self.coupling.add_track_point(track, tetrahedron, weights)
        self._eyelet_rest[section] = rest
        if slot < 0:
            return
        self._track[section] = track
        while len(self.passages) <= track:
            self.passages.append([])
        self.passages[track].append(int(slot))

    def advance(self):
        """Take the rows recorded since the last call and the exit of this step; makes at most a few eyelets.

        With nothing new this is one comparison of two counts. Raises if the scene's interaction
        is no longer the one the rows taken so far came from.
        """
        interaction = getattr(self.scene, "interaction", None)
        if self.rows and interaction is not self._source:
            raise RuntimeError("The needle/tissue interaction changed under the eyelets made from its track")
        if interaction is None:
            return
        count, events = interaction.track_count, interaction.events
        if count == self.rows and not events:
            return
        if count > self.rows:
            self._source = interaction
            tetrahedra, weights, rests, sections = interaction.track_rows(self.rows)
            for tetrahedron, weight, rest, section in zip(tetrahedra, weights, rests, sections, strict=True):
                section = int(section)
                last = self._eyelet_rest.get(section)
                if last is None or np.linalg.norm(rest - last) >= self.spacing:
                    self._make(section, int(tetrahedron), weight, rest)
                self._last_row[section], self._section = (int(tetrahedron), weight, rest), section
            self.rows = count
        if self._section is not None and any(event.get("kind") == "exit" for event in events):
            tetrahedron, weight, rest = self._last_row[self._section]
            if np.linalg.norm(rest - self._eyelet_rest[self._section]) >= 0.5 * self.spacing:
                self._make(self._section, tetrahedron, weight, rest)
