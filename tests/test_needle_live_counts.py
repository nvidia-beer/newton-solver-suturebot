# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""The live history counts are device data: kernels read their bounds from ``interaction._live``.

Before, a kernel took the number of track samples as a host integer or read a count from an
array shape. That text is rebuilt here from each kernel's source by undoing the edit, and both
texts are launched on the inputs of an embedded needle. While the storage of the history has
exactly the live size, both must write the same bits; with a count lowered by one, the kernel
must write what the earlier text writes on storage that is one row shorter. The track samples,
the openings and the visited ids are stored in a reserve that is larger than the live count:
there the kernel runs on the reserve, and its earlier text on views of exactly the live size or
over exactly the live rows.
"""

import inspect
import linecache
import os
import re
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import test_needle_trial_commit as commit_fixture
import warp as wp
from test_cooperative_shaft_runtime import original_packet
from test_needle_insertion_example import _example
from test_needle_shaft_preconditioner import setup

from newton_solver_suturebot._src.solvers.coupled.acceleration.surface_contact_tangent import SurfaceTangentAnchors
from newton_solver_suturebot._src.solvers.coupled.interface import (
    insertion,
    needle_trial_forces,
    shaft_projected_kernels,
    shaft_projection,
    surface_contact_forces,
    surface_facet_scratch,
)
from newton_solver_suturebot._src.solvers.coupled.interface.insertion import LIVE_OPENINGS, LIVE_TRACK, LIVE_VISITED

DEVICE = os.environ.get("NEWTON_TEST_DEVICE", "cpu")

# How each kernel read its bounds before: (pattern, earlier text, number of places).
_PARAMETER = r"    live: wp\.array\[int\],\n"
_HOST_COUNT = (_PARAMETER, "    track_count: int,\n", 1)
_NO_PARAMETER = (_PARAMETER, "", 1)
_TRACK_GUARD = (r"    if i >= live\[LIVE_TRACK\]:\n        return\n", "", 1)


def _passed_on(places):
    return (r"(?m)^(\s+)live,$", r"\1track_count,", places)


_EARLIER_TEXT = {
    "_surface_permitted": (
        insertion,
        [
            _HOST_COUNT,
            (r"range\(live\[LIVE_TRACK\]\)", "range(track_count)", 1),
            (r"range\(live\[LIVE_OPENINGS\]\)", "range(openings.shape[0])", 1),
        ],
    ),
    "_surface_forces": (insertion, [_HOST_COUNT, _passed_on(4)]),
    "surface_forces_with_tangent": (surface_contact_forces, [_HOST_COUNT, _passed_on(4)]),
    "collect_facets": (surface_facet_scratch, [_HOST_COUNT, _passed_on(1)]),
    "forces_from_facets": (surface_facet_scratch, [_HOST_COUNT, _passed_on(3)]),
    "track_segments": (needle_trial_forces, [_NO_PARAMETER, _TRACK_GUARD]),
    "track_geometry": (needle_trial_forces, [_NO_PARAMETER, _TRACK_GUARD]),
    "shaft_mask": (
        needle_trial_forces,
        [
            _NO_PARAMETER,
            (r"live\[LIVE_VISITED\] == 0", "visited.shape[0] == 0", 1),
            (
                r"    #[^\n]*\n    for k in range\(live\[LIVE_VISITED\]\):\n",
                "    for k in range(visited.shape[0]):\n"
                "        if k > 0 and visited[k] == visited[k - 1]:\n"
                "            break\n",
                1,
            ),
            (r"range\(live\[LIVE_TRACK\]\)", "range(track.shape[0])", 2),
        ],
    ),
    "opening_reuse_certificate": (
        needle_trial_forces,
        [_NO_PARAMETER, (r"    if i >= live\[LIVE_OPENINGS\]:\n        return\n", "", 1)],
    ),
    "_shaft_forces": (insertion, [_NO_PARAMETER, _TRACK_GUARD]),
    "shaft_forces_projected": (shaft_projected_kernels, [_NO_PARAMETER, _TRACK_GUARD]),
    "prepare_queries": (
        shaft_projection,
        [_NO_PARAMETER, (r"    if i >= live\[LIVE_TRACK\]:\n        dirty\[i\] = 0\n        return\n", "", 1)],
    ),
}


def earlier_text(name):
    """Source of a kernel with its bounds read from a host count or an array shape again."""
    module, rules = _EARLIER_TEXT[name]
    text = inspect.getsource(getattr(module, name).func)
    for pattern, replacement, places in rules:
        text, found = re.subn(pattern, replacement, text)
        if found != places:
            raise AssertionError(f"{name}: {pattern!r} matches {found} places, expected {places}")
    if re.search(r"\blive\b|LIVE_", text):
        raise AssertionError(f"{name}: the earlier text still reads the live counts")
    return text


def earlier_kernels():
    """Compile the earlier text of every changed kernel next to the globals of its own module."""
    compiled, namespaces = {}, {}
    for name, (module, _) in _EARLIER_TEXT.items():
        namespace = namespaces.get(module)
        if namespace is None:
            short = module.__name__.rsplit(".", 1)[-1]
            namespace = namespaces[module] = dict(vars(module), __name__=f"{__name__}.{short}")
            if "_surface_permitted" in compiled:
                # The surface kernels of every module call the earlier permission function.
                namespace["_surface_permitted"] = compiled["_surface_permitted"]
        text = earlier_text(name)
        filename = f"<{namespace['__name__']}.{name}>"
        # Warp reads a kernel's source through inspect, which serves this text from the line cache.
        linecache.cache[filename] = (len(text), None, text.splitlines(True), filename)
        exec(compile(text, filename, "exec"), namespace)
        compiled[name] = namespace[name]
    return compiled


def lowered(live, entries):
    """A copy of the live counts with the given entries lowered by one."""
    values = live.numpy().copy()
    values[list(entries)] -= 1
    return wp.array(values, dtype=int, device=live.device)


class LiveCountStorageTests(unittest.TestCase):
    def test_one_array_follows_every_change_of_the_accepted_history(self):
        fixture = commit_fixture.NeedleTrialCommitTests()
        fixture.setUp()
        x, state, forces = fixture.build()
        live, pointer = x._live, x._live.ptr
        np.testing.assert_array_equal(live.numpy(), [0, 0, 0])
        seen = {tuple(live.numpy())}
        # Entry, track growth, exit through the far face and re-entry: the path of test_needle_trial_commit.
        path = np.r_[
            -0.0001, np.linspace(0, 0.0012, 9), np.linspace(0.00105, -0.0012, 16), np.linspace(-0.00105, 0.0012, 16)
        ]
        for z in path:
            start = SimpleNamespace(particle_q=wp.clone(state.particle_q), body_q=wp.clone(state.body_q))
            state.body_q.assign([[0.002, 0.002, z, 0, 0, 0, 1]])
            state.body_f.zero_()
            forces.zero_()
            x.add_forces(state, 0, np.zeros(3), forces, 0.0005, implicit=True, start_state=start)
            x.accept(state, 0)
            self.assertNotEqual(x.status, "failed", x.reason)
            counts = (len(x._visited_tets), x.track_count, len(x._openings))
            np.testing.assert_array_equal(live.numpy(), counts)
            seen.add(counts)
        self.assertGreater(min(counts), 0)
        # The opening count and the track count each changed on a step of their own. The one
        # tetrahedron is visited with its first track sample; the next test has a step that changes
        # the visited count alone.
        self.assertGreater(len(seen), 3)
        self.assertEqual((live.numpy()[LIVE_VISITED], live.numpy()[LIVE_TRACK], live.numpy()[LIVE_OPENINGS]), counts)
        # A visited set edited without an accepted step is written by the helper's preparation.
        # The ids have one entry per tetrahedron, so the edit takes a real id out and puts it back.
        helper = needle_trial_forces._NeedleTrialForces(x)
        tet = min(x._visited_tets)
        x._visited_tets.discard(tet)
        np.testing.assert_array_equal(live.numpy(), counts)
        helper.prepare(x)
        np.testing.assert_array_equal(live.numpy(), (counts[0] - 1, *counts[1:]))
        x._visited_tets.add(tet)
        helper.prepare(x)
        np.testing.assert_array_equal(live.numpy(), counts)
        np.testing.assert_array_equal(helper.visited.numpy()[: counts[0]], sorted(x._visited_tets))
        # An opening registered directly, as tests do.
        x._register_opening(x.surface_triangles[0], np.full(3, 1 / 3), 0)
        np.testing.assert_array_equal(live.numpy(), (*counts[:2], counts[2] + 1))
        self.assertIs(x._live, live)
        self.assertEqual(live.ptr, pointer)
        self.assertEqual((live.shape, live.dtype), ((3,), wp.int32))

    def test_each_count_is_written_by_an_accepted_step_that_changes_it_alone(self):
        # The one tetrahedron of the test above is visited with its first track sample. On the insertion
        # fixture the tip enters a second tetrahedron within the track spacing of a sample: that step
        # lays no sample, so the only write of the counts is the one that follows the visited ids.
        accept, seen = insertion.NeedleTissueInteraction.accept, [(0, 0, 0)]

        def recorded(x, *arguments, **options):
            result = accept(x, *arguments, **options)
            counts = (len(x._visited_tets), x.track_count, len(x._openings))
            np.testing.assert_array_equal(x._live.numpy(), counts)
            seen.append(counts)
            return result

        with (
            _example() as ex,
            patch.object(insertion.NeedleTissueInteraction, "accept", autospec=True, side_effect=recorded),
        ):
            setup(ex, 0.0005, 1500)
        alone = set()
        for before, after in zip(seen, seen[1:], strict=False):
            changed = [index for index in range(3) if before[index] != after[index]]
            if len(changed) == 1:
                alone.add(changed[0])
        self.assertEqual(alone, {LIVE_VISITED, LIVE_TRACK, LIVE_OPENINGS})


@unittest.skipIf(wp.get_device(DEVICE).is_cuda, "bitwise on the CPU: sums over samples have no fixed order on CUDA")
class LiveCountKernelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.earlier = earlier_kernels()

    def compare(
        self,
        name,
        kernel,
        dim,
        inputs,
        *,
        host_count=None,
        exact=None,
        lower=(),
        shorter=None,
        fill=None,
        block_dim=None,
    ):
        """Launch a kernel and its earlier text from the same start values: every array must hold the same bits.

        ``host_count`` is the host integer that the earlier text takes in place of the live counts.
        ``exact(inputs, dim)`` gives the inputs and the launch size of the earlier text on storage of exactly
        the live size, where the kernel runs on a reserve. ``lower`` names the counts lowered by one for the
        kernel; ``shorter(inputs, dim)`` then gives the inputs and the launch size of the earlier text on
        storage that is one row shorter. ``fill`` sets start values of outputs. The arrays are left as they
        were. Returns the parameters whose arrays the kernel changed.
        """
        parameters = list(inspect.signature(kernel.func).parameters)
        self.assertEqual(len(parameters), len(inputs), name)
        inputs = list(inputs)
        old, old_dim = dict(zip(parameters, inputs, strict=True)), dim
        if exact is not None:
            old, old_dim = exact(old, old_dim)
        if lower:
            inputs[parameters.index("live")] = lowered(old["live"], lower)
            old, old_dim = shorter(old, old_dim)
        if host_count is None:
            del old["live"]
        else:
            old["live"] = host_count - int(LIVE_TRACK in lower)
        arrays = {
            parameter: value
            for parameter, value in zip(parameters, inputs, strict=True)
            if isinstance(value, wp.array) and value.size and parameter != "live"
        }
        original = {parameter: value.numpy().copy() for parameter, value in arrays.items()}
        for parameter, value in (fill or {}).items():
            arrays[parameter].fill_(value)
        start = {parameter: value.numpy().copy() for parameter, value in arrays.items()}
        options = {} if block_dim is None else {"block_dim": block_dim}
        wp.launch(kernel, dim, inputs, device=DEVICE, **options)
        written = {parameter: value.numpy().copy() for parameter, value in arrays.items()}
        for parameter, value in arrays.items():
            value.assign(start[parameter])
        wp.launch(self.earlier[name], old_dim, list(old.values()), device=DEVICE, **options)
        for parameter, value in arrays.items():
            self.assertEqual(value.numpy().tobytes(), written[parameter].tobytes(), f"{name}: {parameter}")
            value.assign(original[parameter])
        return {parameter for parameter in arrays if written[parameter].tobytes() != start[parameter].tobytes()}

    @staticmethod
    def record(call):
        """Argument lists of the changed kernels, as the launches of the package issue them."""
        launches = {}
        real = wp.launch

        def launch(kernel, dim, inputs, *args, **kwargs):
            if kernel.key in _EARLIER_TEXT:
                launches.setdefault(kernel.key, []).append((kernel, dim, list(inputs)))
            return real(kernel, dim, inputs, *args, **kwargs)

        with patch.object(wp, "launch", side_effect=launch):
            call()
        return launches

    def test_kernels_write_the_bits_of_their_earlier_text(self):
        with _example() as ex:
            setup(ex, 0.0005, 1500)
            x, state, body, forces = ex.interaction, ex.state_0, ex.needle_body, ex._external_forces
            helper = needle_trial_forces._NeedleTrialForces(x)
            counts = (len(x._visited_tets), x.track_count, len(x._openings))
            np.testing.assert_array_equal(x._live.numpy(), counts)
            self.assertGreater(min(counts), 0, "the fixture needs every kind of history")

            def evaluate():
                # The host path with both friction laws, then the helper with its device mask.
                for implicit in (True, False):
                    state.body_f.zero_()
                    forces.zero_()
                    x.add_forces(state, body, ex.drive.com, forces, 0.0005, trial=True, implicit=implicit)
                state.body_f.zero_()
                forces.zero_()
                helper.add_forces(state, body, ex.drive.com, forces, 0.0005)

            launches = self.record(evaluate)
            self.assertEqual(
                {name: len(calls) for name, calls in launches.items()},
                {
                    "_surface_forces": 3,
                    "_shaft_forces": 3,
                    "track_segments": 1,
                    "track_geometry": 1,
                    "shaft_mask": 1,
                    "opening_reuse_certificate": 1,
                },
            )
            changed = {name: set() for name in _EARLIER_TEXT if name != "_surface_permitted"}

            def both(name, kernel, dim, inputs, *, lower, shorter, **options):
                # Storage of exactly the live size first, then every bound that the kernel reads lowered by one.
                changed[name] |= self.compare(name, kernel, dim, inputs, **options)
                self.compare(name, kernel, dim, inputs, lower=lower, shorter=shorter, **options)

            def fewer_rows(values, dim):
                return values, dim - 1

            def fewer_openings(values, dim):
                return dict(values, openings=values["openings"][: counts[2] - 1]), dim

            def live_rows(values, dim):
                # Launched over the track reserve; the earlier text read every row of its launch.
                return values, counts[1]

            self.assertGreater(x.track_capacity, counts[1])

            def live_openings(values, dim):
                # The opening points have a row for every reserved opening: the earlier text reads the shape.
                self.assertGreater(values["openings"].shape[0], counts[2])
                return dict(values, openings=values["openings"][: counts[2]]), dim

            # Surface forces: the three recorded launches, each also with facet recording and through the scratch.
            surface = {"host_count": counts[1], "block_dim": 1, "exact": live_openings}
            fewer = {"lower": (LIVE_TRACK, LIVE_OPENINGS), "shorter": fewer_openings}
            anchors = SurfaceTangentAnchors(256, device=DEVICE)
            scratch = surface_facet_scratch.SurfaceFacetScratch(len(x.samples), DEVICE)
            for kernel, dim, inputs in launches["_surface_forces"]:
                both("_surface_forces", kernel, dim, inputs, **surface, **fewer)
                anchors.clear()
                recorded = [*inputs, *anchors.arguments(0.01)]
                both(
                    "surface_forces_with_tangent",
                    surface_contact_forces.surface_forces_with_tangent,
                    dim,
                    recorded,
                    **surface,
                    **fewer,
                )
                cached = [*inputs, *scratch.disabled_recording.arguments(1.0), *scratch.storage]
                scratch.valid.fill_(1)
                scratch.overflow.zero_()
                collect = surface_facet_scratch.collect_facets
                both("collect_facets", collect, dim, cached, fill={"counts": -3}, **surface, **fewer)
                # The second pass reads what the first one collected with the same counts.
                for lower in ((), fewer["lower"]):
                    first = [lowered(value, lower) if value is x._live else value for value in cached]
                    wp.launch(collect, dim, first, device=DEVICE, block_dim=1)
                    self.assertEqual(int(scratch.overflow.numpy()[0]), 0)
                    changed["forces_from_facets"] |= self.compare(
                        "forces_from_facets",
                        surface_facet_scratch.forces_from_facets,
                        dim,
                        cached,
                        lower=lower,
                        shorter=fewer_openings,
                        **surface,
                    )

            # Track geometry and the shaft mask of the helper.
            ((kernel, dim, inputs),) = launches["track_segments"]
            self.assertEqual(tuple(dim), (x.track_capacity, len(x.samples) - 1))
            both(
                "track_segments",
                kernel,
                dim,
                inputs,
                fill={"points": -3.0, "segments": -3.0},
                exact=lambda values, dim: (values, (counts[1], dim[1])),
                lower=(LIVE_TRACK,),
                shorter=lambda values, dim: (values, (dim[0] - 1, dim[1])),
            )
            ((kernel, dim, inputs),) = launches["track_geometry"]
            self.assertEqual(dim, x.track_capacity)
            both(
                "track_geometry",
                kernel,
                dim,
                inputs,
                fill={"projected_arc": -3.0, "associated": -3},
                exact=live_rows,
                lower=(LIVE_TRACK,),
                shorter=fewer_rows,
            )
            ((kernel, dim, inputs),) = launches["shaft_mask"]
            both(
                "shaft_mask",
                kernel,
                dim,
                inputs,
                fill={"mask": -3},
                # The earlier text read the visited ids and the track anchors up to their array shapes.
                exact=lambda values, dim: (
                    dict(values, visited=values["visited"][: counts[0]], track=values["track"][: counts[1]]),
                    dim,
                ),
                lower=(LIVE_VISITED, LIVE_TRACK),
                shorter=lambda values, dim: (
                    dict(values, visited=values["visited"][: counts[0] - 1], track=values["track"][: counts[1] - 1]),
                    dim,
                ),
            )
            # The entries of the visited ids from the live count on are never read: any value may stand there.
            self.assertGreater(helper.visited.shape[0], counts[0])
            mask = inputs[list(inspect.signature(kernel.func).parameters).index("mask")]
            before = mask.numpy().copy()
            padded = helper.visited.numpy()
            padded[counts[0] :] = -1
            helper.visited.assign(padded)
            mask.fill_(-3)
            wp.launch(kernel, dim, inputs, device=DEVICE)
            np.testing.assert_array_equal(mask.numpy(), before)

            # Opening reuse: the certificate returns at once inside the material. Outside, a capture radius
            # that reaches the opening makes it request the host path.
            ((kernel, dim, inputs),) = launches["opening_reuse_certificate"]
            # Launched over the opening reserve; its earlier text read every row of the launch.
            self.assertEqual(dim, x.opening_capacity)
            self.assertGreater(dim, counts[2])
            reaching = list(inputs)
            reaching[list(inspect.signature(kernel.func).parameters).index("capture")] = 1.0
            flags = helper.step_flags.numpy().copy()
            for outside in (False, True):
                if outside:
                    helper.step_flags.assign([0, 0, flags[2]])
                for values in (inputs, reaching):
                    both(
                        "opening_reuse_certificate",
                        kernel,
                        dim,
                        values,
                        fill={"fallback": 0},
                        exact=lambda values, dim: (values, counts[2]),
                        lower=(LIVE_OPENINGS,),
                        shorter=fewer_rows,
                    )
            helper.step_flags.assign(flags)

            # Shaft forces: the three recorded launches, and the projected variant of each. The interaction
            # launches over the live rows, the helper over the track reserve.
            self.assertEqual([dim for _, dim, _ in launches["_shaft_forces"]], [counts[1], counts[1], x.track_capacity])
            for kernel, dim, inputs in launches["_shaft_forces"]:
                both("_shaft_forces", kernel, dim, inputs, exact=live_rows, lower=(LIVE_TRACK,), shorter=fewer_rows)
                values = dict(zip(inspect.signature(kernel.func).parameters, inputs, strict=True))
                packet = wp.empty((dim, 8), dtype=float, device=DEVICE)
                wp.launch(
                    original_packet,
                    dim,
                    [
                        values["nodes"],
                        values["bary"],
                        values["q"],
                        values["body_q"],
                        values["body"],
                        0.5 * values["engagement_radius"],
                        values["samples"],
                        packet,
                    ],
                    device=DEVICE,
                )
                both(
                    "shaft_forces_projected",
                    shaft_projected_kernels.shaft_forces_projected,
                    dim,
                    [*inputs, packet],
                    exact=live_rows,
                    lower=(LIVE_TRACK,),
                    shorter=fewer_rows,
                )

            # Freshness test of the cooperative projection. Its binding is made on CUDA only: build one here.
            binding = shaft_projection.ProjectionBinding(x, state, body, float(x.radii.max()))
            self.assertIs(binding.live, x._live)
            self.assertTrue(any(owner is x._live for owner in binding.owners))
            queries = [
                binding.nodes,
                binding.bary,
                binding.live,
                binding.q,
                binding.body_q,
                binding.body,
                binding.radius,
                binding.tables.refresh(),
                binding.queries,
                binding.poses,
                binding.radii,
                binding.initialized,
                binding.generations,
                binding.dirty,
                binding.counts,
            ]
            both(
                "prepare_queries",
                shaft_projection.prepare_queries,
                binding.capacity,
                queries,
                fill={"queries": -3.0, "poses": -3.0, "radii": -3.0},
                exact=live_rows,
                lower=(LIVE_TRACK,),
                shorter=fewer_rows,
            )
            self.assertEqual(binding.capacity, x.track_capacity)
            # A row beyond the live count is marked clean, whatever it held: the projection skips it.
            binding.dirty.fill_(1)
            below = [lowered(value, (LIVE_TRACK,)) if value is x._live else value for value in queries]
            wp.launch(shaft_projection.prepare_queries, binding.capacity, below, device=DEVICE)
            live = counts[1] - 1
            np.testing.assert_array_equal(binding.dirty.numpy(), [1] * live + [0] * (binding.capacity - live))

            # Every kernel did work on these inputs: equal bits are not the bits of an early return.
            expected = {
                "_surface_forces": {"particle_f", "body_f", "diagnostics"},
                "surface_forces_with_tangent": {"particle_f", "body_f", "diagnostics", "anchor_count"},
                "collect_facets": {"counts", "face_ids"},
                "forces_from_facets": {"particle_f", "body_f", "diagnostics", "permissions"},
                "track_segments": {"points", "segments"},
                "track_geometry": {"projected_arc", "associated"},
                "shaft_mask": {"mask"},
                "opening_reuse_certificate": {"fallback"},
                "_shaft_forces": {"particle_f", "body_f", "diagnostics"},
                "shaft_forces_projected": {"particle_f", "body_f", "diagnostics"},
                "prepare_queries": {"queries", "poses", "radii", "dirty"},
            }
            self.assertEqual(
                {
                    name: parameters - changed[name]
                    for name, parameters in expected.items()
                    if parameters - changed[name]
                },
                {},
            )


if __name__ == "__main__":
    unittest.main()
