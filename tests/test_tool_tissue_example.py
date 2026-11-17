# SPDX-License-Identifier: Apache-2.0
"""The tool/tissue example: its scene, its scripted path, its contact switch and its start-up.

The scene and path tests run on the CPU with the native rigid backend, or on the device of
``NEWTON_TEST_DEVICE`` with MuJoCo Warp. The start-up tests need CUDA with conditional graphs.
"""

import contextlib
import io
import json
import math
import os
import unittest
from unittest.mock import patch

import numpy as np
import warp as wp
from newton.viewer import ViewerNull
from warp._src import context as warp_context

from newton_solver_suturebot._src.solvers.coupled.interface.swept_tool_tissue_contact import (
    STATION_RADII,
    capsule_pieces,
)
from newton_solver_suturebot._src.utils import capture
from newton_solver_suturebot.examples.grasp import example_fem_tool_tissue as example_module
from newton_solver_suturebot.examples.grasp._grasp_scene import GraspScene
from newton_solver_suturebot.examples.grasp._tool_tissue_contact import CONTACTS, make_contact
from newton_solver_suturebot.examples.grasp._tool_tissue_measure import HOLD_REMAINDER, Measurements
from newton_solver_suturebot.examples.grasp._tool_tissue_path import (
    CLEARANCE_JAW_RADII,
    FAST_APPROACH,
    FAST_RADII_PER_INTERVAL,
    PHASES,
    PINCH,
    RESTART_PHASES,
    plan_path,
)
from newton_solver_suturebot.examples.grasp._tool_tissue_scene import (
    TissueStepRejected,
    ToolTissueScene,
    _rigid_check,
)

DEVICE = os.environ.get("NEWTON_TEST_DEVICE", "cpu")


@wp.kernel(enable_backward=False)
def _signed_distances(mesh: wp.uint64, points: wp.array[wp.vec3], reach: float, distance: wp.array[float]):
    """Distance of every point from the closed surface [m], negative inside it (ray parity)."""
    index = wp.tid()
    query = wp.mesh_query_point(mesh, points[index], reach)
    distance[index] = wp.inf
    if query.result:
        nearest = wp.mesh_eval_position(mesh, query.face, query.u, query.v)
        distance[index] = query.sign * wp.length(nearest - points[index])


@wp.kernel(enable_backward=False)
def _heights_below(mesh: wp.uint64, points: wp.array[wp.vec3], height: wp.array[float]):
    """Height of the first surface point straight below every point [m]; NaN where there is none."""
    index = wp.tid()
    query = wp.mesh_query_ray(mesh, points[index], wp.vec3(0.0, 0.0, -1.0), 1.0)
    height[index] = wp.nan
    if query.result:
        height[index] = points[index][2] - query.t


# The contact that the example runs with when none is asked for.
DEFAULT_CONTACT = "off"
# First frame of the press: the jaws are at the tissue surface at its start.
PRESS_FRAME = 26


def _backend():
    return "mujoco-warp" if wp.get_device(DEVICE).is_cuda else "native"


def _scene(contact="off"):
    wp.set_device(DEVICE)
    args = ToolTissueScene.create_parser().parse_args([])
    return ToolTissueScene(ViewerNull(), args, rigid_backend=_backend(), contact=contact)


def _plan(scene, **changed):
    defaults = example_module.Example.create_parser().parse_args([])
    options = {
        "speed": defaults.speed,
        "hold_seconds": defaults.hold_seconds,
        "press_depth": defaults.press_depth,
        "deep_depth": defaults.deep_depth,
        "slide_length": defaults.slide_length,
        "fast_depth": defaults.fast_depth,
        "pinch_gap_fraction": example_module.PINCH_GAP_FRACTION,
    }
    return plan_path(scene, **{**options, **changed})


def _rotated(quaternion, points):
    """Points turned by a unit quaternion (xyzw)."""
    axis, w = np.asarray(quaternion[:3], dtype=float), float(quaternion[3])
    return points + 2.0 * np.cross(axis, np.cross(axis, points) + w * points)


def _example(*arguments, frames=3):
    wp.set_device(DEVICE)
    args = example_module.Example.create_parser().parse_args(
        ["--device", DEVICE, "--viewer", "null", "--num-frames", str(frames), *arguments]
    )
    with contextlib.redirect_stdout(io.StringIO()):
        return example_module.Example(ViewerNull(num_frames=frames), args)


@contextlib.contextmanager
def _host_reads_fail():
    """Inside it every read of array data on the host raises an AssertionError.

    ``array.numpy()`` and ``array.list()`` on any device, and on a CUDA device every
    device-to-host copy of Warp, which is what a read ends in whatever asks for it.
    """
    error = AssertionError("a device-to-host read")
    with contextlib.ExitStack() as stack:
        stack.enter_context(patch.object(wp.array, "numpy", side_effect=error))
        stack.enter_context(patch.object(wp.array, "list", side_effect=error))
        if wp.get_device(DEVICE).is_cuda:
            stack.enter_context(patch.object(warp_context.runtime.core, "wp_memcpy_d2h", side_effect=error))
        yield


def _axis_distance(scene, name, spacing_radii):
    """Of the capsules of the part ``name``, the one that reaches furthest into the tissue: its radius and the
    smallest signed distance of its axis from the tissue surface."""
    _, _, starts, ends, radii, _, firsts, counts = capsule_pieces(
        scene.capsule_bodies, scene.capsule_endpoints, scene.capsule_radii, scene.capsule_contact_radii
    )
    deepest = None
    for capsule, body in enumerate(scene.capsule_bodies):
        if scene.part_names[int(body)] != name:
            continue
        first, count = firsts[capsule], counts[capsule]
        radius = float(radii[first])
        start, end = starts[first], ends[first + count - 1]
        stations = int(math.ceil(np.linalg.norm(end - start) / (spacing_radii * radius))) + 1
        local = start + np.linspace(0.0, 1.0, stations)[:, None] * (end - start)
        pose = scene.state_0.body_q.numpy()[body].astype(float)
        axis, w = pose[3:6], pose[6]
        # Rotation by the unit quaternion (xyzw) of the body, then its translation.
        points = pose[:3] + local + 2.0 * np.cross(axis, np.cross(axis, local) + w * local)
        distance = wp.zeros(stations, dtype=float, device=scene.device)
        wp.launch(
            _signed_distances,
            stations,
            inputs=[scene.surface.mesh.id, wp.array(points, dtype=wp.vec3, device=scene.device), scene.surface_reach],
            outputs=[distance],
            device=scene.device,
        )
        nearest = float(distance.numpy().min())
        if deepest is None or radius - nearest > deepest[0] - deepest[1]:
            deepest = (radius, nearest)
    return deepest


class ToolTissueSceneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.init()
        cls.scene = _scene()
        cls.path, cls.arm = _plan(cls.scene)

    def test_scene_holds_the_tools_and_the_tissue_of_the_grasp_scene_and_nothing_else(self):
        scene = self.scene
        labels = list(scene.model.body_label)
        self.assertEqual(len(labels), 16)
        self.assertTrue(all(label.startswith(("PSM1/", "PSM2/")) for label in labels))
        self.assertEqual(scene.rigid_solver.rigid_substeps, 4)
        grasp = GraspScene.create_parser()
        for name in ("young_modulus", "poisson_ratio", "density", "material_damping", "linear_damping", "regions"):
            self.assertEqual(getattr(scene.args, name), grasp.get_default(name), name)
        self.assertEqual(scene.args.solver_iterations, grasp.get_default("solver_iterations"))
        self.assertEqual(scene.solver.linear_solver_tolerance, grasp.get_default("fem_tolerance"))
        self.assertEqual(scene.args.substeps, 4)
        self.assertAlmostEqual(scene.sim_dt, 1.0 / 240.0)
        # The block of the grasp scene: region 1 with its held border and underside.
        self.assertEqual(scene.model.particle_count, 7205)
        self.assertEqual(len(scene.fixed_vertex_indices), 1841)
        # The contact spheres of the tools' capsule set, as the first tool/tissue contact samples them.
        set_radii = np.concatenate([tool.capsule_radii for tool in scene.graspers])
        np.testing.assert_array_equal(scene.capsule_radii, set_radii)
        self.assertIs(scene.capsule_contact_radii, scene.capsule_radii)
        self.assertEqual(len(scene.sample_bodies), scene.describe()["contact_spheres"])
        # The jaw radius of the path is that of the capsule that carries the tip of the first jaw.
        self.assertEqual(scene.jaw_contact_radius, scene.tip_radius)
        self.assertIn(scene.tip_radius, set(scene.graspers[0].capsule_radii))
        self.assertEqual(self.path.phase_frames("press")[0], PRESS_FRAME)

    def test_contact_switch_has_three_settings(self):
        self.assertEqual(CONTACTS, ("off", "first", "swept"))
        self.assertEqual(self.scene.contact.name, "off")
        self.assertEqual(example_module.Example.create_parser().get_default("contact"), DEFAULT_CONTACT)
        self.assertIn(DEFAULT_CONTACT, CONTACTS)
        with self.assertRaises(ValueError):
            make_contact("other", self.scene)

    def test_first_contact_has_the_parameters_that_the_grasp_scene_derives(self):
        scene = self.scene
        contact = make_contact("first", scene)
        args = scene.args
        shear = args.young_modulus / (2.0 * (1.0 + args.poisson_ratio))
        depth = float(np.ptp(np.asarray(scene.selection.vertices)[:, 2]))
        parameters = contact.parameters()
        self.assertAlmostEqual(parameters["node_stiffness_N_m"], 0.25 * args.young_modulus * scene.mesh_spacing)
        self.assertAlmostEqual(
            parameters["shear_frequency_rad_s"], 0.5 * math.pi * math.sqrt(shear / args.density) / depth
        )
        # The numbers of the grasp scene (docs/method.md).
        self.assertAlmostEqual(parameters["node_stiffness_N_m"], 8.2018, places=3)
        self.assertAlmostEqual(parameters["shear_frequency_rad_s"], 401.26, places=1)
        self.assertEqual(parameters["samples"], len(scene.sample_bodies))
        self.assertTrue(contact.has_end_step and contact.keeps_surface)
        self.assertEqual(len(contact.state), 3)

    def test_path_has_the_seven_phases_with_lengths_derived_from_the_scene(self):
        scene, path = self.scene, self.path
        numbers = path.derivation
        self.assertEqual(tuple(dict.fromkeys(segment.phase for segment in path.segments)), PHASES)
        radius, interval = scene.jaw_contact_radius, scene.sim_dt
        constant = float(scene.actuator_time_constants.max())
        self.assertAlmostEqual(constant, 1.0 / 12.0, places=6)
        self.assertAlmostEqual(numbers["clearance_m"], CLEARANCE_JAW_RADII * radius)
        self.assertAlmostEqual(numbers["transit_speed_m_s"], CLEARANCE_JAW_RADII * radius / constant)
        self.assertAlmostEqual(numbers["fast_speed_m_s"], FAST_RADII_PER_INTERVAL * radius / interval)
        # Every motion is no faster than the speed it was planned with.
        for segment in path.segments:
            if segment.label == FAST_APPROACH:
                self.assertAlmostEqual(segment.speed, numbers["fast_speed_m_s"])
            elif segment.hold:
                self.assertEqual(segment.frames, round(numbers["hold_s"] / scene.frame_dt))
                self.assertEqual(segment.speed, 0.0)
            elif segment.in_tissue or segment.label in ("descend to the surface", "lift off"):
                self.assertLessEqual(segment.speed, numbers["speed_m_s"] * (1 + 1e-9))
            else:
                self.assertLessEqual(segment.speed, numbers["transit_speed_m_s"] * (1 + 1e-9))
        # The commanded depths at the end of the holds: the two presses, the border and the fast approach.
        ends = {(s.phase, s.label): path.depth[s.last_frame] for s in path.segments if s.hold}
        self.assertAlmostEqual(ends[("press", "hold")], numbers["press_depth_m"], delta=1e-7)
        self.assertAlmostEqual(ends[("press_deep", "hold")], numbers["deep_depth_m"], delta=1e-7)
        self.assertAlmostEqual(ends[("slide", "hold")], numbers["deep_depth_m"], delta=1e-7)
        self.assertAlmostEqual(ends[("border", "hold")], numbers["press_depth_m"], delta=1e-7)
        self.assertAlmostEqual(ends[("fast", "hold")], numbers["fast_depth_m"], delta=1e-7)
        self.assertAlmostEqual(ends[("lift", "hold")], -numbers["clearance_m"], delta=1e-7)
        slide = next(s for s in path.segments if s.label == "slide")
        depths = path.depth[slide.first_frame : slide.last_frame + 1]
        np.testing.assert_allclose(depths, numbers["deep_depth_m"], rtol=0, atol=1e-7)
        start = path.tip[slide.first_frame - 1]
        self.assertAlmostEqual(
            float(np.linalg.norm((path.tip[slide.last_frame] - start)[:2])), numbers["slide_length_m"], delta=1e-9
        )
        # The fast approach commands the tip more than its contact radius per interval, from the air.
        fast = path.fast_approach
        self.assertEqual(fast.label, FAST_APPROACH)
        travel = np.linalg.norm(path.tip[fast.first_frame] - path.tip[fast.first_frame - 1]) / scene.args.substeps
        self.assertGreater(travel, radius)
        self.assertAlmostEqual(travel, FAST_RADII_PER_INTERVAL * radius, delta=1e-9)
        self.assertLess(path.depth[fast.first_frame - 1], 0.0)
        # Its first frame ends at the fast depth, and its target goes on along the insertion axis at the same
        # speed: every frame of it is one more fast length from the start of the ramp.
        self.assertAlmostEqual(path.depth[fast.first_frame], numbers["fast_depth_m"], delta=1e-7)
        self.assertAlmostEqual(fast.depth, numbers["fast_depth_m"], delta=1e-7)
        insertion = np.asarray(numbers["insertion_axis"])
        start = path.poses[fast.first_frame - 1, :3]
        steps = np.diff(path.poses[fast.first_frame - 1 : fast.last_frame + 1, :3], axis=0)
        np.testing.assert_allclose(steps, np.tile(numbers["fast_length_m"] * insertion, (fast.frames, 1)), atol=1e-12)
        # Frames are planned until a tool that lags its target by the lag of the slowest actuator has touched the
        # surface: the commanded jaws touch it after fast_touch_travel_m.
        touch = start + numbers["fast_touch_travel_m"] * insertion
        self.assertAlmostEqual(scene.jaw_depth_at_rest([touch], path.start_pose[3:])[0], 0.0, delta=2e-7)
        self.assertAlmostEqual(numbers["fast_lag_m"], numbers["fast_speed_m_s"] * constant)
        planned = math.ceil((numbers["fast_touch_travel_m"] + numbers["fast_lag_m"]) / numbers["fast_length_m"] - 1e-9)
        self.assertEqual(numbers["fast_ramp_frames_for_lag"], planned)
        self.assertEqual(numbers["fast_ramp_frames"], fast.frames)
        self.assertEqual(fast.frames, planned)
        self.assertGreater(fast.frames, 1)
        # The hold after it is at the fast depth again.
        after = path.segment_of(fast.last_frame + 1)
        self.assertTrue(after.hold)
        np.testing.assert_allclose(path.poses[after.first_frame], path.poses[fast.first_frame], atol=1e-12)
        np.testing.assert_allclose(self.arm[after.first_frame + 1], self.arm[fast.first_frame + 1], atol=1e-5)
        # The rule of the ramp: it goes on while the tip of the tool is clear of the surface at rest, and the
        # first frame after which it is not is followed by the hold. No other frame reads the tool.
        for frame in range(fast.first_frame, fast.last_frame):
            self.assertTrue(path.reads_tool(frame))
            self.assertEqual(path.next_frame(frame, 1e-6), frame + 1)
            self.assertEqual(path.next_frame(frame, None), frame + 1)
            self.assertEqual(path.next_frame(frame, 0.0), after.first_frame)
            self.assertEqual(path.next_frame(frame, -1e-3), after.first_frame)
        for frame in (0, fast.first_frame - 1, fast.last_frame, after.first_frame, path.frames - 1):
            self.assertFalse(path.reads_tool(frame))
            self.assertEqual(path.next_frame(frame, -1e-3), frame + 1)
        # The border press is on a held node; the first press is away from every held node.
        rest = np.asarray(scene.selection.vertices)
        fixed = np.asarray(scene.fixed_vertex_indices)
        self.assertIn(numbers["border_node"], fixed)
        touched = np.r_[numbers["middle_site_m"], numbers["middle_surface_height_m"]]
        away = np.linalg.norm(rest[fixed] - touched, axis=1).min()
        self.assertGreater(away, 3 * scene.mesh_spacing)
        # One orientation, within the joint limits, and the phases that start in the air.
        np.testing.assert_array_equal(path.poses[:, 3:], np.tile(path.start_pose[3:], (path.frames, 1)))
        tool = scene.graspers[0]
        self.assertTrue(np.all(self.arm >= tool.arm_lower) and np.all(self.arm <= tool.arm_upper))
        self.assertEqual(self.arm.shape, (path.frames + 1, 6))
        self.assertEqual(list(path.restarts), ["approach", "border", "fast", "pinch"])
        for first in path.restarts.values():
            self.assertLess(path.depth[max(first - 1, 0)], 0.0)
        self.assertEqual(path.restart_after(50), path.restarts["border"])
        self.assertEqual(path.restart_after(path.restarts["fast"]), path.restarts["pinch"])
        self.assertIsNone(path.restart_after(path.restarts["pinch"]))

    def test_pinch_phase_follows_the_seven_phases_and_commands_no_opening_in_them(self):
        path = self.path
        self.assertEqual(len(PHASES), 8)
        self.assertEqual(PHASES[-1], PINCH)
        first, last = path.phase_frames(PINCH)
        self.assertEqual(first, path.phase_frames("fast")[1] + 1)
        self.assertEqual(last, path.frames - 1)
        self.assertEqual(path.derivation["pinch_first_frame"], first)
        self.assertEqual(path.derivation["pinch_frames"], last + 1 - first)
        # One opening per frame: closed jaws in the seven phases, and in the pinch until the tool is above the ridge.
        self.assertEqual(path.opening.shape, (path.frames,))
        np.testing.assert_array_equal(path.opening[:first], 0.0)
        segments = [segment for segment in path.segments if segment.phase == PINCH]
        self.assertEqual(
            [segment.label for segment in segments],
            [
                "lift off",
                "rise to the transit height",
                "transit to the ridge",
                "descend above the ridge",
                "open",
                "descend beside the ridge",
                "settle",
                "close",
                "squeeze hold",
                "lift",
                "lift hold",
                "lower",
                "lower hold",
                "release",
                "release hold",
                "retreat",
            ],
        )
        by_label = {segment.label: segment for segment in segments}
        opened = by_label["open"]
        np.testing.assert_array_equal(path.opening[first : opened.first_frame], 0.0)
        numbers = path.derivation
        wide, narrow = numbers["pinch_open_half_opening_rad"], numbers["pinch_closed_half_opening_rad"]
        self.assertGreater(wide, narrow)
        self.assertGreater(narrow, 0.0)
        self.assertLessEqual(wide, self.scene.maximum_half_opening)
        # The jaws turn in the three segments that are named for it, by equal parts per frame, and stand between them.
        for label, begin, end in (("open", 0.0, wide), ("close", wide, narrow), ("release", narrow, wide)):
            segment = by_label[label]
            frames = np.arange(1, segment.frames + 1) / segment.frames
            np.testing.assert_allclose(
                path.opening[segment.first_frame : segment.last_frame + 1], begin + (end - begin) * frames, atol=1e-15
            )
            self.assertEqual(path.opening_before(segment.first_frame), begin)
        for label, opening in (
            ("descend beside the ridge", wide),
            ("settle", wide),
            ("squeeze hold", narrow),
            ("lift", narrow),
            ("lift hold", narrow),
            ("lower", narrow),
            ("lower hold", narrow),
            ("release hold", wide),
            ("retreat", wide),
        ):
            segment = by_label[label]
            np.testing.assert_array_equal(path.opening[segment.first_frame : segment.last_frame + 1], opening)
        self.assertEqual(path.opening_before(0), 0.0)
        # A run can be placed in the phase: after its lift-off from the hold of the fast approach, where the tool
        # is clear of the tissue and its jaws are closed.
        self.assertEqual(RESTART_PHASES, ("approach", "border", "fast", PINCH))
        restart = path.restarts[PINCH]
        self.assertEqual(restart, by_label["lift off"].last_frame + 1)
        self.assertEqual(restart, numbers["pinch_restart_frame"])
        self.assertGreater(path.depth[first - 1], 0.0)
        self.assertLess(path.depth[restart - 1], 0.0)
        self.assertEqual(path.opening_before(restart), 0.0)
        self.assertEqual(example_module.Example.create_parser().get_default("start_phase"), None)

    def test_the_jaw_command_reads_nothing_from_the_device(self):
        # The start-up rule of the jaw command: a frame in which the opening changes writes the two actuator
        # targets of the jaws on the device and reads nothing back.
        scene = self.scene
        targets = scene.control.joint_target_q
        jaws = list(scene.graspers[0].jaw_q_indices)
        opening = 0.25 * scene.maximum_half_opening
        scene.place(self.arm[0])
        with _host_reads_fail():
            # The guard is one: a read of the targets fails inside it, by an array call and, on a CUDA device,
            # by a copy to the host.
            with self.assertRaisesRegex(AssertionError, "device-to-host read"):
                targets.numpy()
            if targets.device.is_cuda:
                with self.assertRaisesRegex(AssertionError, "device-to-host read"):
                    wp.copy(wp.empty(targets.shape, dtype=targets.dtype, device="cpu"), targets)
            scene.set_jaw_opening(opening)
        np.testing.assert_allclose(targets.numpy()[jaws], [opening, -opening], rtol=1e-6)
        with _host_reads_fail():
            scene.set_jaw_opening(0.0)
        np.testing.assert_array_equal(targets.numpy()[jaws], 0.0)

    def _tips_from_the_bodies(self, half_opening, tip=None):
        """The tip sphere of each jaw in the frame of the palm, from the bodies of the tool placed with that opening.

        Not from the hinge arithmetic of the planner: the tool is placed (``place``) and the poses
        of its palm and jaws are read. ``tip`` names the sphere of each jaw; without it, the one
        whose lowest point is lowest for the palm rotation of the path.
        """
        scene, path = self.scene, self.path
        tool = scene.graspers[0]
        scene.place(self.arm[0], half_opening)
        poses = scene.state_0.body_q.numpy().astype(float)
        palm = poses[tool.palm_body]
        inverse = np.r_[-palm[3:6], palm[6]]
        centres, radii, chosen = [], [], []
        for index, jaw in enumerate(tool.jaw_bodies):
            mine = np.flatnonzero(scene.sample_bodies == jaw)
            world = poses[jaw, :3] + _rotated(poses[jaw, 3:], scene.sample_points[mine])
            local = _rotated(inverse, world - palm[:3])
            if tip is None:
                low = _rotated(path.start_pose[3:], local)[:, 2] - scene.sample_radii[mine]
                chosen.append(int(np.argmin(low)))
            else:
                chosen.append(tip[index])
            centres.append(local[chosen[-1]])
            radii.append(float(scene.sample_radii[mine][chosen[-1]]))
        scene.place(self.arm[0])
        return np.asarray(centres), np.asarray(radii), chosen

    def test_pinch_plan_follows_from_the_surface_at_rest(self):
        scene, path = self.scene, self.path
        numbers = path.derivation
        radius, speed = scene.jaw_contact_radius, numbers["speed_m_s"]
        depth = example_module.DEEP_DEPTH
        rotation = path.start_pose[3:]
        closing = _rotated(rotation, np.array([1.0, 0.0, 0.0]))
        # The rotation of the path is a quaternion of single precision.
        np.testing.assert_allclose(closing, numbers["closing_axis"], rtol=0, atol=1e-6)
        along = closing[:2] / np.linalg.norm(closing[:2])
        middle = np.asarray(numbers["middle_site_m"])

        # The jaws from the placed tool: their tips, and the gap between them along the closing direction.
        closed, radii, tip = self._tips_from_the_bodies(0.0)

        def tips(half_opening):
            return self._tips_from_the_bodies(half_opening, tip)[0]

        def gap(centres):
            world = _rotated(rotation, centres)
            return abs(float((world[1] - world[0]) @ closing)) - float(radii.sum())

        np.testing.assert_allclose(radii, numbers["pinch_tip_radii_m"], rtol=1e-6)
        widest = gap(tips(scene.maximum_half_opening))
        self.assertAlmostEqual(widest, numbers["pinch_widest_tip_gap_m"], delta=2e-6)

        # The surface at rest along the closing direction through the middle site, by rays from above the pad.
        above = float(np.asarray(scene.reference.vertices)[:, 2].max()) + 4.0 * radius

        def heights(offsets):
            offsets = np.atleast_1d(np.asarray(offsets, dtype=float))
            points = np.column_stack((middle + offsets[:, None] * along, np.full(len(offsets), above)))
            result = wp.zeros(len(offsets), dtype=float, device=scene.device)
            wp.launch(
                _heights_below,
                len(offsets),
                inputs=[scene.rest_surface.id, wp.array(points, dtype=wp.vec3, device=scene.device)],
                outputs=[result],
                device=scene.device,
            )
            return result.numpy().astype(float)

        spacing = STATION_RADII * radius
        self.assertAlmostEqual(numbers["pinch_profile_spacing_m"], spacing)
        count = math.ceil(widest / spacing)
        offsets = spacing * np.arange(-count, count + 1)
        profile = heights(offsets)
        self.assertTrue(np.isfinite(profile).all())
        # Uphill from the site to the top of the ridge.
        top = count
        while True:
            higher = [index for index in (top - 1, top + 1) if profile[index] > profile[top]]
            if not higher:
                break
            top = max(higher, key=lambda index: profile[index])
        self.assertAlmostEqual(profile[top], numbers["pinch_ridge_top_m"], delta=1e-6)
        self.assertAlmostEqual(offsets[top], numbers["pinch_ridge_top_offset_m"], delta=1e-9)
        # The ridge: the surface above the height of the tip centres at the commanded depth, around the top.
        level = profile[top] - depth + radius
        first = last = top
        while profile[first - 1] > level:
            first -= 1
        while profile[last + 1] > level:
            last += 1

        def edge(outside, inside):
            while abs(inside - outside) > 1e-9:
                half = 0.5 * (outside + inside)
                outside, inside = (outside, half) if heights(half)[0] > level else (half, inside)
            return 0.5 * (outside + inside)

        left, right = edge(offsets[first - 1], offsets[first]), edge(offsets[last + 1], offsets[last])
        width, ridge_middle = right - left, 0.5 * (left + right)
        self.assertAlmostEqual(width, numbers["pinch_ridge_width_m"], delta=2e-6)
        self.assertAlmostEqual(ridge_middle, numbers["pinch_ridge_middle_offset_m"], delta=2e-6)
        # A ridge of some millimetres that the fully opened jaws can straddle.
        self.assertGreater(width, 2.0 * scene.mesh_spacing)
        self.assertLess(width, widest)

        # The commands: open to the width and a clearance on each side, closed to the chosen part of the width.
        constant = float(scene.actuator_time_constants.max())
        self.assertAlmostEqual(numbers["pinch_clearance_m"], speed * constant)
        wide, narrow = tips(numbers["pinch_open_half_opening_rad"]), tips(numbers["pinch_closed_half_opening_rad"])
        self.assertAlmostEqual(gap(wide), width + 2.0 * speed * constant, delta=4e-6)
        self.assertEqual(numbers["pinch_gap_fraction"], example_module.PINCH_GAP_FRACTION)
        self.assertAlmostEqual(gap(narrow), example_module.PINCH_GAP_FRACTION * width, delta=4e-6)

        # Where the tool stands: the middle between the tips over the middle of the ridge; above it with closed
        # jaws at the clearance of the other phases; beside it with the lowest point of the open tips DEEP_DEPTH
        # below the top (to a micrometre); lifted by the same.
        by_label = {segment.label: segment for segment in path.segments if segment.phase == PINCH}

        def placed(label, centres):
            world = path.poses[by_label[label].last_frame, :3] + _rotated(rotation, centres)
            return world.mean(axis=0)[:2], float((world[:, 2] - radii).min())

        site = middle + ridge_middle * along
        over, lowest = placed("descend above the ridge", closed)
        np.testing.assert_allclose(over, site, atol=2e-6)
        self.assertAlmostEqual(lowest, profile[top] + numbers["clearance_m"], delta=1e-6)
        over, lowest = placed("descend beside the ridge", wide)
        np.testing.assert_allclose(over, site, atol=2e-6)
        self.assertAlmostEqual(lowest, profile[top] - depth, delta=1e-6)
        self.assertEqual(numbers["pinch_depth_m"], depth)
        self.assertEqual(numbers["pinch_lift_m"], depth)
        low = path.poses[by_label["descend beside the ridge"].last_frame, :3]
        # The palm stays where the open jaws put the tips while the jaws close and open.
        for label in ("settle", "close", "squeeze hold", "lower", "lower hold", "release", "release hold"):
            segment = by_label[label]
            # The lowering ends there; the other segments are there from their first frame.
            first = segment.last_frame if label == "lower" else segment.first_frame
            block = path.poses[first : segment.last_frame + 1, :3]
            np.testing.assert_allclose(block, np.tile(low, (len(block), 1)), rtol=0, atol=1e-12)
        for label in ("lift", "lift hold"):
            np.testing.assert_allclose(
                path.poses[by_label[label].last_frame, :3] - low, (0.0, 0.0, depth), rtol=0, atol=1e-12
            )
        np.testing.assert_allclose(
            path.poses[by_label["retreat"].last_frame, :3],
            path.poses[by_label["descend above the ridge"].last_frame, :3],
            rtol=0,
            atol=1e-12,
        )
        # Motions in or beside the tissue at the speed of the example, also the jaw tips: the frames of the
        # closing are those that the arc of a tip about its hinge takes at that speed.
        for label in ("lift off", "descend beside the ridge", "lift", "lower", "retreat"):
            self.assertLessEqual(by_label[label].speed, speed * (1 + 1e-9))
            self.assertGreater(by_label[label].speed, 0.9 * speed)
        turn = numbers["pinch_open_half_opening_rad"] - numbers["pinch_closed_half_opening_rad"]
        chord = float(np.linalg.norm(wide - narrow, axis=1).max())
        arc = chord * (0.5 * turn) / math.sin(0.5 * turn)
        frames = by_label["close"].frames
        self.assertEqual(frames, by_label["release"].frames)
        self.assertGreaterEqual(frames * speed * scene.frame_dt, arc - 1e-7)
        self.assertLess((frames - 1) * speed * scene.frame_dt, arc + 1e-7)
        for value in numbers["pinch_jaw_tip_speeds_m_s"].values():
            self.assertLessEqual(value, speed * (1 + 1e-9))
        # Every hold of the phase lasts as long as the holds of the other phases.
        for segment in by_label.values():
            if segment.hold:
                self.assertEqual(segment.frames, round(numbers["hold_s"] / scene.frame_dt))

    def test_pinch_plan_is_refused_without_a_ridge_that_the_jaws_can_straddle(self):
        scene = self.scene
        for fraction in (-0.1, 1.0, float("nan")):
            with self.assertRaisesRegex(ValueError, "closed gap of the pinch"):
                _plan(scene, pinch_gap_fraction=fraction)
        # Jaws that open a tenth as far: the ridge at the middle site is wider than their tips come apart.
        widest = scene.maximum_half_opening
        scene.maximum_half_opening = 0.1 * widest
        try:
            with self.assertRaisesRegex(ValueError, "No ridge of the tissue at the middle site"):
                _plan(scene)
        finally:
            scene.maximum_half_opening = widest
        # Tips that are commanded no deeper below the top of the ridge than their own radius: no tissue between them.
        self.assertLess(0.0006, scene.jaw_contact_radius)
        with self.assertRaisesRegex(ValueError, "not deeper than their radius"):
            _plan(scene, press_depth=0.0005, deep_depth=0.0006)
        for lift in (0.0, -0.001, float("nan")):
            with self.assertRaisesRegex(ValueError, "lift of the pinch"):
                _plan(scene, pinch_lift=lift)

    def test_pinch_lift_is_its_own_quantity_of_the_plan(self):
        scene = self.scene
        path, _ = _plan(scene)
        numbers = path.derivation
        # Without a lift of its own the pinch lifts by the depth of its tips, as planned so far.
        self.assertEqual(numbers["pinch_lift_m"], numbers["pinch_depth_m"])
        higher, arm = _plan(scene, pinch_lift=0.005)
        changed = higher.derivation
        self.assertEqual((changed["pinch_lift_m"], changed["pinch_depth_m"]), (0.005, numbers["pinch_depth_m"]))
        self.assertAlmostEqual(
            changed["pinch_lifted_tip_height_m"],
            changed["pinch_ridge_top_m"] - changed["pinch_depth_m"] + 0.005,
            delta=1e-12,
        )
        segments = {segment.label: segment for segment in higher.segments if segment.phase == "pinch"}
        before = {segment.label: segment for segment in path.segments if segment.phase == "pinch"}
        # The same path up to the lift; the lift and the lowering are longer, at the same speed.
        lift, squeeze = segments["lift"], segments["squeeze hold"]
        self.assertEqual(lift.first_frame, before["lift"].first_frame)
        np.testing.assert_array_equal(higher.poses[: lift.first_frame], path.poses[: lift.first_frame])
        np.testing.assert_array_equal(higher.opening[: lift.first_frame], path.opening[: lift.first_frame])
        frames = math.ceil(0.005 / (numbers["speed_m_s"] * scene.frame_dt) - 1e-9)
        self.assertEqual((lift.frames, segments["lower"].frames), (frames, frames))
        self.assertEqual(higher.frames - path.frames, 2 * (frames - before["lift"].frames))
        raised = higher.poses[segments["lift hold"].last_frame, :3] - higher.poses[squeeze.last_frame, :3]
        np.testing.assert_allclose(raised, (0.0, 0.0, 0.005), atol=1e-12)
        np.testing.assert_allclose(
            higher.poses[segments["lower hold"].last_frame, :3], higher.poses[squeeze.last_frame, :3], atol=1e-12
        )
        self.assertEqual(len(arm), higher.frames + 1)
        # No capsule of the open jaws is inside the tissue at rest while they go down beside the ridge (with the
        # straight wrist of the default path; the plan records the depth, it does not refuse a tilted jaw).
        descent = before["descend beside the ridge"]
        deepest = float(path.depth[descent.first_frame : descent.last_frame + 1].max())
        self.assertAlmostEqual(numbers["pinch_open_descent_depth_m"], deepest, delta=1e-9)
        self.assertLessEqual(numbers["pinch_open_descent_depth_m"], 0.0)

    def test_fast_approach_is_planned_no_further_than_the_tool_reaches(self):
        scene = self.scene
        tilt = math.radians(30.0)
        whole, whole_arm = _plan(scene, jaw_tilt=tilt)
        lag_frames, ramp = whole.derivation["fast_ramp_frames_for_lag"], whole.fast_approach
        # With the jaw radius of the capsule set the ramp that covers the lag (6 frames of 5.17 mm) lies within
        # the insertion joint at every tilt the wrist allows (0 to 57 degrees, measured on the CPU; with the
        # 0.976 mm of the four typed capsules it was 7.8 mm a frame and left the joint at 30 degrees).
        self.assertEqual(ramp.frames, lag_frames)
        self.assertGreater(lag_frames, 4)
        # The hard case, stated: a tool whose insertion joint ends between the third and the fourth frame of
        # that ramp. Every other frame of the path is within it.
        kept = 3
        insertion = whole_arm[ramp.first_frame + 1 : ramp.last_frame + 2, 2]
        self.assertTrue(np.all(np.diff(insertion) > 0.0))
        reach = 0.5 * (insertion[kept - 1] + insertion[kept])
        self.assertLess(
            np.delete(whole_arm[:, 2], np.arange(ramp.first_frame + kept + 1, ramp.last_frame + 2)).max(), reach
        )
        solve = scene.solve_arm_targets

        def within_reach(poses, **options):
            arm = solve(poses, **options)
            if np.any(arm[:, 2] > reach):
                raise ValueError("The pose is beyond the stated reach of the insertion joint")
            return arm

        with patch.object(scene, "solve_arm_targets", within_reach):
            path, arm = _plan(scene, jaw_tilt=tilt)
            numbers, fast = path.derivation, path.fast_approach
            # The line of the ramp leaves the reach of the insertion joint before the lag is covered.
            self.assertLess(fast.frames, numbers["fast_ramp_frames_for_lag"])
            self.assertEqual((fast.frames, numbers["fast_ramp_frames_for_lag"]), (kept, lag_frames))
            self.assertGreater(fast.frames, 1)
            self.assertEqual(numbers["fast_ramp_frames"], fast.frames)
            self.assertEqual(arm.shape, (path.frames + 1, 6))
            tool = scene.graspers[0]
            self.assertTrue(np.all(arm >= tool.arm_lower) and np.all(arm <= tool.arm_upper))
            self.assertLessEqual(arm[:, 2].max(), reach)
            # The frames that were kept are those of the whole ramp, and one more frame is beyond the reach.
            np.testing.assert_array_equal(
                path.poses[fast.first_frame : fast.last_frame + 1],
                whole.poses[ramp.first_frame : ramp.first_frame + kept],
            )
            beyond = path.poses[fast.last_frame].copy()
            beyond[:3] += numbers["fast_length_m"] * np.asarray(numbers["insertion_axis"])
            np.testing.assert_allclose(beyond, whole.poses[ramp.first_frame + kept], rtol=0, atol=1e-12)
            with self.assertRaises(ValueError):
                within_reach(beyond[None])
        self.assertAlmostEqual(
            path.depth[path.segment_of(fast.last_frame + 1).last_frame], numbers["fast_depth_m"], delta=1e-7
        )
        # The joint limits of the tool end the ramp in the same way: at the largest tilt of the jaws that the
        # wrist allows for this path (57 degrees, measured) the whole ramp is within them and one more frame of
        # it is not.
        steep, steep_arm = _plan(scene, jaw_tilt=math.radians(57.0))
        last = steep.fast_approach.last_frame
        beyond = steep.poses[last].copy()
        beyond[:3] += steep.derivation["fast_length_m"] * np.asarray(steep.derivation["insertion_axis"])
        with self.assertRaises(ValueError):
            scene.pose_ik[0].solve(beyond, scene.rigid_solver.mj_data.qpos, initial=steep_arm[last + 1])

    def test_interval_check_records_the_interval_in_which_the_tip_crosses_the_surface_at_rest(self):
        scene, numbers = self.scene, self.path.derivation
        device = scene.device
        radius = scene.tip_radius
        # The tip sphere over a node of the upper surface near the middle site, moved along the node's normal:
        # the node is then no further from the centre of the sphere than the height that is set.
        rest = np.asarray(scene.selection.vertices, dtype=float)
        level = np.flatnonzero(scene.surface_normals[:, 2] > 0.9)
        nearest = level[
            np.argmin(np.linalg.norm(rest[scene.surface_nodes[level], :2] - numbers["middle_site_m"], axis=1))
        ]
        site, normal = rest[scene.surface_nodes[nearest]], scene.surface_normals[nearest]

        def run(before, after):
            poses = np.tile(
                np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32), (scene.model.body_count, 1)
            )
            start, end = poses.copy(), poses.copy()
            start[scene.tip_body, :3] = site + before * normal - scene.tip_local
            end[scene.tip_body, :3] = site + after * normal - scene.tip_local
            track = wp.zeros(5, dtype=float, device=device)
            check = wp.zeros(3, dtype=wp.float64, device=device)
            wp.launch(
                _rigid_check,
                scene.model.body_count,
                [
                    wp.array(start, dtype=wp.transform, device=device),
                    wp.array(end, dtype=wp.transform, device=device),
                    wp.zeros(scene.model.body_count, dtype=wp.spatial_vector, device=device),
                    scene.tip_body,
                    wp.vec3(*scene.tip_local),
                    radius,
                    scene.rest_surface.id,
                    scene.surface_reach,
                    check,
                    track,
                ],
                device=device,
            )
            self.assertEqual(check.numpy()[1], 0.0)
            return track.numpy()

        # From clear of the surface to inside it, in one interval of 1.5 radii: the travel of that interval and
        # the gap on both sides of it.
        track = run(2.0 * radius, 0.5 * radius)
        self.assertAlmostEqual(track[0], 1.5 * radius, delta=1e-7)
        self.assertAlmostEqual(track[2], 1.5 * radius, delta=1e-7)
        self.assertGreater(track[3], 0.0)
        self.assertLessEqual(track[4], 0.0)
        # The gap is a distance to the nearest surface point: it cannot change by more than the travel, and the
        # node is no further from the centre than the height that was set.
        self.assertLessEqual(track[3] - track[4], 1.5 * radius + 1e-7)
        self.assertLessEqual(track[3], radius + 1e-7)
        self.assertLessEqual(track[4], -0.5 * radius + 1e-7)
        # No crossing: an interval that stays clear, one that stays inside and one that leaves the tissue.
        for before, after in ((4.0 * radius, 2.5 * radius), (0.5 * radius, -radius), (0.5 * radius, 2.0 * radius)):
            track = run(before, after)
            self.assertAlmostEqual(track[0], abs(after - before), delta=1e-7)
            np.testing.assert_array_equal(track[2:], 0.0)
        # Beyond the reach of the search the tip is clear: coming from there is a crossing, staying there is not.
        far = 3.0 * scene.surface_reach
        np.testing.assert_array_equal(run(far + radius, far)[2:], 0.0)
        track = run(far, 0.5 * radius)
        self.assertAlmostEqual(track[2], far - 0.5 * radius, delta=1e-6)
        self.assertTrue(np.isinf(track[3]) and track[3] > 0)

    def test_frames_follow_the_path_and_the_jaws_pass_through_the_tissue_without_a_contact(self):
        scene, path, arm = self.scene, self.path, self.arm
        scene.reset_tissue()
        scene.place(arm[0])
        start = scene.observe()
        self.assertAlmostEqual(
            max(start["part_depth_m"]["PSM1/jaw_1"], start["part_depth_m"]["PSM1/jaw_2"]),
            -path.derivation["clearance_m"],
            delta=5e-5,
        )
        frames = 12
        constant = float(scene.actuator_time_constants.max())
        intervals = scene.accepted_intervals
        elapsed = scene.elapsed
        for frame in range(frames):
            scene.set_frame_targets(arm[frame], arm[frame + 1])
            scene.step()
            self.assertIsNone(scene.failure_reason)
        observation = scene.observe()
        self.assertEqual(scene.accepted_intervals - intervals, frames * scene.args.substeps)
        self.assertEqual(scene.rejected_intervals, 0)
        self.assertAlmostEqual(scene.elapsed - elapsed, frames * scene.frame_dt, places=12)
        palm = observation["body_q"][scene.graspers[0].palm_body]
        lag = float(np.linalg.norm(palm[:3] - path.poses[frames - 1, :3]))
        self.assertLess(lag, path.derivation["speed_m_s"] * constant + 4e-5)
        jaws = max(observation["part_depth_m"]["PSM1/jaw_1"], observation["part_depth_m"]["PSM1/jaw_2"])
        self.assertLess(abs(jaws - path.depth[frames - 1]), path.derivation["speed_m_s"] * constant + 4e-5)
        # Without a contact nothing reaches the tissue.
        self.assertLess(observation["surface_displacement_m"], 1e-7)
        self.assertEqual(observation["contact"]["tissue_force_N"], [0.0, 0.0, 0.0])
        integrity = scene.tissue_integrity()
        self.assertTrue(integrity["finite"] and integrity["held_nodes_at_rest"])
        self.assertGreater(integrity["minimum_volume_ratio"], 0.999)
        self.assertGreater(observation["tip_travel_m"], 0.0)
        # The tip sphere is one of the jaws' spheres: its gap to the surface at rest is no smaller than theirs.
        # In these frames it is still clear of the surface, and no interval crossed it.
        self.assertGreater(observation["tip_gap_m"], 0.0)
        self.assertGreaterEqual(observation["tip_gap_m"], -jaws - 1e-6)
        self.assertLess(observation["tip_gap_m"], -jaws + scene.jaw_contact_radius)
        self.assertIsNone(observation["tip_crossing"])
        again = scene.observe()
        self.assertEqual(again["tip_travel_m"], 0.0)
        self.assertEqual(again["tip_gap_m"], observation["tip_gap_m"])

    def _axis_distance(self, name, spacing_radii):
        return _axis_distance(self.scene, name, spacing_radii)

    def test_capsule_depth_is_the_radius_less_the_distance_of_the_axis_and_reads_surface_points_below_it(self):
        scene, path, arm = self.scene, self.path, self.arm
        scene.reset_tissue()
        # The jaws 0.3 mm inside the tissue at rest: their axes are outside it. The capsule is as deep as its
        # radius less the distance of its axis from the surface; against stations sixteen times as dense as
        # the measurement's (R / 512) it reads within its stated 0.24 micrometres, and not as low as points of
        # the capsule surface alone would (10 micrometres in the check of this example).
        frame = PRESS_FRAME + 5
        self.assertAlmostEqual(path.depth[frame], 3.0e-4, delta=5e-5)
        scene.place(arm[frame + 1])
        observation = scene.observe()
        for name in ("PSM1/jaw_1", "PSM1/jaw_2"):
            measured = observation["capsule_depth_m"][name]
            radius, distance = self._axis_distance(name, 1.0 / 512.0)
            self.assertGreater(distance, 0.5 * radius)
            self.assertLess(distance, radius)
            self.assertAlmostEqual(max(measured["yielding"], measured["held"]), radius - distance, delta=2.5e-7)
            self.assertEqual(measured["centre"], 0.0)
            # The chain of spheres of the other measurements lies inside the capsule.
            self.assertLessEqual(observation["part_depth_m"][name], radius - distance + 1e-9)
        for name in ("PSM1/shaft", "PSM1/wrist", "PSM2/jaw_1"):
            measured = observation["capsule_depth_m"][name]
            self.assertEqual((measured["yielding"], measured["held"], measured["centre"]), (0.0, 0.0, 0.0))
        # The jaws 3 mm inside (the hold of the deep press): their axes are under the surface. The centre line
        # is as deep as its deepest station, and the capsule is read on the points of its own surface: deeper
        # than the centre line by a good part of a radius (0.44 radii for the first jaw, whose axis is 2.35 mm
        # under the surface, and 1.00 for the second, 1.39 mm under it, measured on the CPU: the tissue wraps a
        # thin capsule from the sides, and depth is the distance to the nearest surface point) and by no more
        # than one (R + d bounds every point).
        deep = next(segment for segment in path.segments if segment.phase == "press_deep" and segment.hold).last_frame
        scene.place(arm[deep + 1])
        observation = scene.observe()
        for name in ("PSM1/jaw_1", "PSM1/jaw_2"):
            measured = observation["capsule_depth_m"][name]
            radius, distance = self._axis_distance(name, 1.0 / 512.0)
            self.assertLess(distance, -radius)
            self.assertAlmostEqual(measured["centre"], -distance, delta=2.5e-7)
            capsule = max(measured["yielding"], measured["held"])
            self.assertGreater(capsule, -distance + 0.4 * radius)
            self.assertLessEqual(capsule, radius - distance + 1e-7)
        scene.place(arm[0])

    def test_rejected_interval_is_repeated_from_its_start_with_half_the_interval(self):
        scene, arm = self.scene, self.arm
        scene.reset_tissue()
        scene.place(arm[0])
        scene.set_frame_targets(arm[0], arm[1])
        attempt = scene._attempt
        calls = []

        def reject_first(dt):
            calls.append(dt)
            if len(calls) == 1:
                # What the guard does with an interval it refuses, after the interval ran.
                attempt(dt)
                raise TissueStepRejected("Unresolved FEM tetrahedron motion: test")
            attempt(dt)

        before = (scene.accepted_intervals, scene.rejected_intervals, scene.elapsed)
        scene._attempt = reject_first
        try:
            scene.step()
        finally:
            scene._attempt = attempt
        self.assertIsNone(scene.failure_reason)
        self.assertEqual(scene.rejected_intervals - before[1], 1)
        # The first interval twice as two halves, and the halved interval to the end of the frame.
        self.assertEqual(calls, [scene.sim_dt] + [scene.sim_dt / 2] * (2 * scene.args.substeps))
        self.assertEqual(scene.accepted_intervals - before[0], 2 * scene.args.substeps)
        self.assertAlmostEqual(scene.elapsed - before[2], scene.frame_dt, places=14)
        self.assertEqual(scene.minimum_step_dt, scene.sim_dt / 2)
        # An interval that is refused down to the retry range fails the scene and leaves the accepted state.
        positions = scene.state_0.particle_q.numpy().copy()
        poses = scene.state_0.body_q.numpy().copy()

        def reject(dt):
            attempt(dt)
            raise TissueStepRejected("Unresolved FEM tetrahedron motion: test")

        scene._attempt = reject
        try:
            scene.step()
        finally:
            scene._attempt = attempt
        self.assertEqual(scene.failure_kind, "tissue")
        self.assertIn("Unresolved FEM tetrahedron motion", scene.failure_reason)
        self.assertEqual(scene.failed_intervals, 1)
        np.testing.assert_array_equal(scene.state_0.particle_q.numpy(), positions)
        np.testing.assert_array_equal(scene.state_0.body_q.numpy(), poses)
        elapsed = scene.elapsed
        scene.step()
        self.assertEqual(scene.elapsed, elapsed)
        scene.reset_tissue()
        self.assertIsNone(scene.failure_reason)
        scene.minimum_step_dt = scene.sim_dt


class SweptContactSceneTests(unittest.TestCase):
    """The third setting of the contact switch in the scene: the jaws press the tissue and the tissue gives way."""

    @classmethod
    def setUpClass(cls):
        wp.init()
        cls.scene = _scene("swept")
        cls.path, cls.arm = _plan(cls.scene)

    def test_swept_contact_takes_its_constants_from_the_scene(self):
        scene = self.scene
        contact = scene.contact
        self.assertEqual(
            (contact.name, contact.has_end_step, contact.keeps_surface, contact.moves_tissue),
            ("swept", True, True, True),
        )
        parameters = contact.parameters()
        self.assertEqual(parameters["sweep_horizon_s"], scene.sim_dt)
        self.assertEqual(parameters["min_stretch"], scene.solver.min_stretch)
        self.assertEqual(parameters["reach_m"], scene.surface_reach)
        self.assertEqual(parameters["axis_pieces"], len(scene.sample_bodies))
        # K_c = m_eff / h^2 per tool, with the lightest direction of its lightest capsule end: an end of a jaw
        # capsule across the jaw, between 12 and 24 g.
        mass = contact.effective_mass
        self.assertEqual(mass.shape, (2,))
        self.assertTrue(np.all(mass > 0.012) and np.all(mass < 0.024))
        self.assertEqual(parameters["effective_tool_mass_kg"], mass.tolist())
        self.assertAlmostEqual(
            parameters["held_stiffness_lightest_direction_N_m"], mass.min() / scene.sim_dt**2, delta=1e-6
        )
        self.assertAlmostEqual(
            parameters["held_stiffness_heaviest_direction_N_m"], mass.max() / scene.sim_dt**2, delta=1e-6
        )
        self.assertEqual(parameters["integral_ratio"], 0.0625)
        # The seven arrays of the law and those of the pinch, which the same exchange builds with the swept law.
        self.assertIsNotNone(contact.exchange.pinch)
        self.assertEqual(len(contact.state), 7 + len(contact.exchange.pinch.state))
        self.assertIs(scene.solver._node_holds, contact.law.holds)

    def test_effective_mass_is_that_of_the_placed_joint_coordinates(self):
        import mujoco

        scene, arm = self.scene, self.arm
        contact, law = scene.contact, scene.contact.law
        scene.reset_tissue()
        scene.place(arm[0])
        placed = contact.effective_mass.copy()
        # At the placed start: the driven tool with closed jaws 22.1 g, the parked one with its jaws open 13.0 g.
        self.assertAlmostEqual(placed[0], 0.0221, delta=2.0e-4)
        self.assertAlmostEqual(placed[1], 0.0130, delta=2.0e-4)
        # Every axis piece carries the mass of its own tool, in the array that the force kernel reads.
        tools = law.sample_tools_host
        self.assertEqual(sorted(set(tools.tolist())), [0, 1])
        np.testing.assert_allclose(law._inverse_mass.numpy(), np.eye(3)[None] / placed[tools][:, None, None], rtol=1e-6)
        self.assertAlmostEqual(
            law.parameters()["held_stiffness_lightest_direction_N_m"], placed.min() / scene.sim_dt**2, delta=1e-6
        )
        # The same number from the native model at the joint coordinates of the placed state, for the tip of
        # the first jaw of the driven tool (every joint of the tools has one coordinate: qpos is joint_q).
        rigid = scene.rigid_solver
        model = rigid.mj_model
        data = mujoco.MjData(model)
        data.qpos[:] = scene.state_0.joint_q.numpy()
        mujoco.mj_forward(model, data)
        body = int(np.flatnonzero(rigid.mjc_body_to_newton.numpy()[0] == scene.tip_body)[0])
        pieces = np.flatnonzero(law.sample_bodies_host == scene.tip_body)
        probes = law._probe_local.numpy()[pieces]
        lightest = np.inf
        jacobian, solved = np.zeros((3, model.nv)), np.zeros((3, model.nv))
        for local in (probes[0], probes[-1]):
            point = data.xpos[body] + data.xmat[body].reshape(3, 3) @ local
            mujoco.mj_jac(model, data, jacobian, None, point, body)
            mujoco.mj_solveM(model, data, solved, jacobian)
            lightest = min(lightest, 1.0 / float(np.linalg.eigvalsh(jacobian @ solved.T).max()))
        self.assertAlmostEqual(placed[0], lightest, delta=2.0e-5)
        # Another placement: the mass follows it (by under one percent along the path), the parked tool keeps its own.
        scene.place(arm[self.path.frames - 1])
        moved = contact.effective_mass
        self.assertNotEqual(moved[0], placed[0])
        self.assertLess(abs(moved[0] - placed[0]), 0.01 * placed[0])
        self.assertAlmostEqual(moved[1], placed[1], delta=1e-9)
        np.testing.assert_allclose(law._inverse_mass.numpy(), np.eye(3)[None] / moved[tools][:, None, None], rtol=1e-6)
        scene.place(arm[0])

    def test_pressed_jaws_make_a_dimple_and_stay_out_of_the_tissue(self):
        scene, arm = self.scene, self.arm
        scene.reset_tissue()
        scene.place(arm[PRESS_FRAME])
        frames = 8
        for frame in range(PRESS_FRAME, PRESS_FRAME + frames):
            scene.set_frame_targets(arm[frame], arm[frame + 1])
            scene.step()
            self.assertIsNone(scene.failure_reason)
        observation = scene.observe()
        commanded = self.path.depth[PRESS_FRAME + frames - 1]
        self.assertGreater(commanded, 3.0e-4)
        # The tissue follows the jaws: the dimple is as deep as the jaws are commanded (within the lag of the
        # arm behind its ramp), and the drawn capsules are inside the deformed surface by micrometres only.
        self.assertGreater(observation["dimple_depth_m"], 0.5 * commanded)
        jaws = [observation["capsule_depth_m"][f"PSM1/jaw_{index}"] for index in (1, 2)]
        self.assertLess(max(max(item["yielding"], item["held"], item["drawn"]) for item in jaws), 2.0e-5)
        self.assertEqual(max(item["centre"] for item in jaws), 0.0)
        contact = observation["contact"]
        self.assertGreater(contact["held_nodes"], 0)
        self.assertEqual(observation["rejected_intervals"], 0)
        # The tissue pushes the tool up and out, and the holds carry the opposite force.
        self.assertGreater(contact["tool_force_N"][2], 1.0e-3)
        self.assertLess(contact["tissue_force_N"][2], -1.0e-3)
        self.assertLess(max(observation["actuator_demand_over_limit"]), 1.0)
        self.assertGreater(scene.tissue_integrity()["minimum_volume_ratio"], 0.2)
        # Reset: the tissue is at rest again, nothing is held and no force is stored.
        scene.reset_tissue()
        cleared = scene.contact.observe()
        self.assertEqual((cleared["held_nodes"], cleared["stored_force_N"]), (0, 0.0))
        np.testing.assert_array_equal(np.asarray(cleared["tool_force_N"]), 0.0)


class ToolTissueExampleTests(unittest.TestCase):
    def setUp(self):
        wp.init()

    def test_example_runs_frames_and_its_self_check_passes_without_a_contact(self):
        example = _example(frames=4)
        self.assertEqual(example.frame_limit, 4)
        self.assertEqual(example.first_frame, 0)
        # Frames with closed jaws command no jaw opening: the seven phases issue what they issued before the pinch.
        commands = []
        command = example.scene.set_jaw_opening
        example.scene.set_jaw_opening = lambda opening: commands.append(opening) or command(opening)
        for _ in range(5):
            example.step()
        self.assertTrue(example.finished)
        self.assertEqual((example.frame, example.completed), (4, 4))
        self.assertEqual(commands, [])
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            example.test_final()
        result = json.loads(output.getvalue().strip().splitlines()[-1])
        self.assertTrue(result["physical_checks_passed"])
        self.assertEqual(result["contact"], "off")
        self.assertFalse(result["scenario_completed"])
        self.assertEqual(result["rejected_intervals"], 0)
        self.assertEqual(len(example.measurements.records), 4)
        record = example.measurements.records[-1]
        self.assertEqual(record["phase"], "approach")
        self.assertEqual(record["accepted_intervals"], 4)
        report = example.measurements.report(timing=example.timing())
        json.dumps(report)
        self.assertEqual(list(report["phases"]), list(PHASES))
        self.assertEqual(report["phases"]["approach"]["status"], "partly run")
        example.close()

    def test_fast_approach_goes_on_until_the_tool_has_crossed_the_surface_and_the_self_check_measures_it(self):
        # The kernel of this module's own distance check, loaded before the example starts: the example's
        # self-check counts every kernel module that is loaded after its start-up.
        wp.load_module(module=_signed_distances.module, device=DEVICE)
        example = _example(frames=0)
        scene, path, records = example.scene, example.path, example.measurements.records
        radius = scene.jaw_contact_radius
        fast = path.fast_approach
        hold = path.segment_of(fast.last_frame + 1)
        # The tool at rest where its wait before the fast approach leaves it.
        scene.place(example.arm_targets[fast.first_frame])
        example.frame = example._row = fast.first_frame
        with contextlib.redirect_stdout(io.StringIO()):
            while example.frame <= fast.last_frame:
                example.step()
        # The ramp went on while the tip was clear of the surface at rest and ended with the first frame after
        # which it was not: before the frames that were planned for the lag of the slowest actuator ran out.
        run = len(records)
        self.assertGreater(run, 1)
        self.assertLess(run, fast.frames)
        self.assertEqual([record["frame"] for record in records], list(range(fast.first_frame, fast.first_frame + run)))
        self.assertTrue(all(record["tip_gap_m"] > 0.0 for record in records[:-1]))
        self.assertLessEqual(records[-1]["tip_gap_m"], 0.0)
        self.assertEqual(example.frame, hold.first_frame)
        self.assertEqual(example.measurements.unused, [(fast.first_frame + run, hold.first_frame)])
        for record in records:
            self.assertAlmostEqual(record["commanded_tip_travel_per_interval_m"], FAST_RADII_PER_INTERVAL * radius)
        # The tool itself, not its command, travelled more than the contact radius of the tip in the interval in
        # which the tip sphere crossed the surface: from clear of it to inside it.
        travel = example.fast_travel()
        crossing = records[-1]
        self.assertEqual(travel["crossing_frame"], crossing["frame"])
        self.assertAlmostEqual(travel["in_crossing_interval"], crossing["tip_crossing_travel_m"] / radius)
        self.assertGreater(crossing["tip_crossing_travel_m"], scene.tip_radius)
        self.assertGreater(travel["in_crossing_interval"], 1.0)
        self.assertLess(travel["in_crossing_interval"], FAST_RADII_PER_INTERVAL)
        self.assertLessEqual(crossing["tip_crossing_travel_m"], crossing["tip_travel_per_interval_m"])
        self.assertGreater(crossing["tip_crossing_gap_before_m"], 0.0)
        self.assertLessEqual(crossing["tip_crossing_gap_after_m"], 0.0)
        self.assertLessEqual(
            crossing["tip_crossing_gap_before_m"] - crossing["tip_crossing_gap_after_m"],
            crossing["tip_crossing_travel_m"] + 1e-7,
        )
        self.assertTrue(all(record["tip_crossing_travel_m"] is None for record in records[:-1]))
        self.assertEqual((travel["approach_frames"], travel["approach_frames_planned"]), (run, fast.frames))
        # Two capsules end each jaw side by side and the tip sphere ends one of them, so the tip sphere is not
        # the part that leads: where it has crossed, the jaws are at least as deep in the tissue as the tip is
        # below the surface at rest, and as deep as the capsule that leads (0.218 mm deeper than the tip sphere
        # on the CPU, measured). The depth of the jaws, which the observation finds on the moving surface, is the
        # radius of that capsule less the distance of its axis from the surface, read here on stations of the
        # axis sixteen times as dense as the measurement's (within its stated 0.24 micrometres).
        self.assertGreaterEqual(crossing["jaw_depth_m"], -crossing["tip_gap_m"] - 1e-6)
        leading = max(
            radius - distance
            for radius, distance in (_axis_distance(scene, name, 1.0 / 512.0) for name in ("PSM1/jaw_1", "PSM1/jaw_2"))
        )
        self.assertAlmostEqual(crossing["jaw_depth_m"], leading, delta=1e-6)
        # The hold brings the target back from where the ramp ended to the commanded depth.
        np.testing.assert_array_equal(
            scene._ramp[1].numpy(), example.arm_targets[fast.first_frame + run].astype(np.float32)
        )
        with contextlib.redirect_stdout(io.StringIO()):
            example.step()
        np.testing.assert_array_equal(
            scene._ramp[0].numpy(), example.arm_targets[fast.first_frame + run].astype(np.float32)
        )
        np.testing.assert_array_equal(
            scene._ramp[1].numpy(), example.arm_targets[hold.first_frame + 1].astype(np.float32)
        )
        back = records[-1]
        self.assertEqual((back["frame"], back["segment"]), (hold.first_frame, hold.label))
        self.assertAlmostEqual(
            back["commanded_tip_travel_per_interval_m"],
            (run - 1) * path.derivation["fast_length_m"] / scene.args.substeps,
            delta=1e-9,
        )
        # Slow steps and periods are listed with the frame of the path, not with their count in the run.
        frames = [*range(fast.first_frame, fast.first_frame + run), hold.first_frame]
        self.assertEqual(example.step_frames, frames)
        limit = example_module.FRAME_LIMIT_MS
        example_module.FRAME_LIMIT_MS = 0.0
        try:
            timing = example.timing()
        finally:
            example_module.FRAME_LIMIT_MS = limit
        self.assertEqual([frame for frame, _ in timing["steps_over_limit"]], frames)
        self.assertEqual([frame for frame, _ in timing["periods_over_limit"]], frames[:-1])
        # To the end of the phase: the tool settles at the commanded depth.
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            while example.frame <= hold.last_frame:
                example.step()
        settled = records[-1]
        self.assertEqual(settled["frame"], hold.last_frame)
        self.assertAlmostEqual(settled["jaw_depth_m"], path.derivation["fast_depth_m"], delta=example._tolerance())
        # The pinch that follows has its own test: here the tool is placed where the last frame of the path
        # starts (as it was placed at the start of the fast approach), and that frame ends the scenario. The
        # self-check passes on what the tool did in the interval of the crossing.
        with contextlib.redirect_stdout(output):
            example._place(path.frames - 1)
            while not example.finished:
                example.step()
            example.test_final()
        self.assertEqual(example.frame, path.frames)
        result = json.loads(output.getvalue().strip().splitlines()[-1])
        self.assertTrue(result["physical_checks_passed"])
        self.assertTrue(result["scenario_completed"])
        self.assertEqual(result["fast_tip_travel_per_interval_jaw_radii"], example.fast_travel())
        # The command of the approach, without the frame of the hold that brings the target back.
        self.assertAlmostEqual(result["fast_tip_travel_per_interval_jaw_radii"]["commanded"], FAST_RADII_PER_INTERVAL)
        summary = example.measurements.phases()["fast"]
        self.assertEqual(summary["unused_frames"], fast.frames - run)
        self.assertEqual(summary["frames"] + summary["unused_frames"], hold.last_frame + 1 - fast.first_frame)
        # A crossing at no more than the contact radius, or none, fails the self-check.
        measured = crossing["tip_crossing_travel_m"]
        try:
            crossing["tip_crossing_travel_m"] = radius
            with (
                self.assertRaisesRegex(AssertionError, "of its contact radius"),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                example.test_final()
            crossing["tip_crossing_travel_m"] = None
            with self.assertRaisesRegex(AssertionError, "did not cross"), contextlib.redirect_stdout(io.StringIO()):
                example.test_final()
        finally:
            crossing["tip_crossing_travel_m"] = measured
        example.close()

    def test_pinch_phase_tracks_palm_and_jaws_without_a_contact(self):
        example = _example("--start-phase", PINCH, frames=0)
        scene, path, records = example.scene, example.path, example.measurements.records
        by_label = {segment.label: segment for segment in path.segments if segment.phase == PINCH}
        restart, last = path.restarts[PINCH], by_label["squeeze hold"].last_frame
        # The run begins where the phase is clear of the tissue, at rest, with closed jaws, and goes to the end of the path.
        self.assertEqual((example.first_frame, example.frame, example._row), (restart, restart, restart))
        self.assertEqual(example.frame_limit, path.frames)
        self.assertEqual(example._opening, 0.0)
        np.testing.assert_allclose(
            scene.state_0.joint_q.numpy()[list(scene.graspers[0].arm_q_indices)],
            example.arm_targets[restart],
            atol=1e-6,
        )
        commands = []
        command = scene.set_jaw_opening
        scene.set_jaw_opening = lambda opening: commands.append(opening) or command(opening)
        with contextlib.redirect_stdout(io.StringIO()):
            while example.frame <= last:
                example.step()
        self.assertIsNone(example.halted)
        self.assertEqual([record["frame"] for record in records], list(range(restart, last + 1)))
        # The jaws are commanded in the frames in which their opening changes, and in no other: while they open
        # and while they close.
        changed = by_label["open"].frames + by_label["close"].frames
        self.assertEqual(len(commands), changed)
        self.assertEqual(commands[-1], path.derivation["pinch_closed_half_opening_rad"])
        self.assertEqual(max(commands), path.derivation["pinch_open_half_opening_rad"])
        # At the end of both holds the palm is on its target and both jaw hinges are on their command, within
        # twice the tolerances of the inverse kinematics (about 40 micrometres and 2 mrad).
        tolerance = example._tolerance()
        hinge_tolerance = 2.0 * scene.pose_ik[0].orientation_tolerance_rad
        self.assertAlmostEqual(hinge_tolerance, 2.0e-3, delta=1e-12)
        ends = {
            record["segment"]: record for record in records if record["frame"] in (by_label["settle"].last_frame, last)
        }
        self.assertEqual(list(ends), ["settle", "squeeze hold"])
        for label, opening in (
            ("settle", path.derivation["pinch_open_half_opening_rad"]),
            ("squeeze hold", path.derivation["pinch_closed_half_opening_rad"]),
        ):
            record = ends[label]
            self.assertEqual(record["commanded_half_opening_rad"], opening)
            for angle in record["jaw_half_opening_rad"]:
                self.assertAlmostEqual(angle, opening, delta=hinge_tolerance)
            self.assertLess(record["palm_position_error_m"], tolerance)
        # While it moves the palm lags its target by no more than the lag of the slowest actuator behind the
        # fastest motion of the phase; the jaws pass through the tissue, which does not move.
        constant = path.derivation["actuator_time_constant_s"]
        fastest = max(segment.speed for segment in by_label.values())
        self.assertLess(max(record["palm_position_error_m"] for record in records), fastest * constant + tolerance)
        self.assertGreater(ends["squeeze hold"]["commanded_depth_m"], 1.0e-3)
        self.assertLess(max(record["surface_displacement_m"] for record in records), 1e-7)
        self.assertEqual({record["tissue_force_N"] for record in records}, {0.0})
        self.assertEqual(scene.rejected_intervals, 0)
        # The self-check of the example applies the same two bounds: it passes, and it fails for a jaw that
        # stands 3 mrad off its command at the end of a hold.
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            example.test_final()
        result = json.loads(output.getvalue().strip().splitlines()[-1])
        self.assertTrue(result["physical_checks_passed"])
        self.assertEqual((result["first_frame"], result["frames"]), (restart, last + 1 - restart))
        self.assertFalse(result["scenario_completed"])
        measured = list(ends["squeeze hold"]["jaw_half_opening_rad"])
        try:
            ends["squeeze hold"]["jaw_half_opening_rad"] = [measured[0] + 3.0e-3, measured[1]]
            with (
                self.assertRaisesRegex(AssertionError, "the jaws are open by"),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                example.test_final()
        finally:
            ends["squeeze hold"]["jaw_half_opening_rad"] = measured
        example.close()

    def test_default_frame_limit_is_the_whole_path_and_intervals_are_measured_only_when_asked(self):
        wp.set_device(DEVICE)
        parser = example_module.Example.create_parser()
        self.assertEqual(parser.get_default("num_frames"), 0)
        self.assertTrue(parser.get_default("measure"))
        self.assertFalse(parser.get_default("measure_intervals"))
        example = _example("--contact", "off", "--measure-intervals", frames=2)
        self.assertIsNotNone(example.scene.interval_observer)
        with contextlib.redirect_stdout(io.StringIO()):
            example.step()
        intervals = example.measurements.intervals
        self.assertEqual(len(intervals), example.scene.args.substeps)
        self.assertEqual([record["frame"] for record in intervals], [0] * len(intervals))
        np.testing.assert_allclose(np.diff([record["time_s"] for record in intervals]), example.scene.sim_dt, rtol=1e-9)
        for name in ("capsule_yielding_depth_m", "capsule_held_depth_m", "capsule_centre_depth_m", "tool_force_N"):
            self.assertEqual({record[name] for record in intervals}, {0.0})
        self.assertLess(max(record["actuator_demand_over_limit"] for record in intervals), 1.0)
        example.close()
        self.assertIsNone(_example("--contact", "off", frames=1).scene.interval_observer)

    def test_failed_phase_with_a_contact_restarts_at_the_next_phase_that_begins_in_the_air(self):
        example = _example("--contact", "first", frames=0)
        scene, path = example.scene, example.path
        self.assertEqual(example.frame_limit, path.frames)
        self.assertEqual(scene.contact.name, "first")
        example.step()
        self.assertEqual(example.frame, 1)
        # A tissue step that cannot be resolved while the jaws press.
        example.frame = path.phase_frames("press")[0] + 5
        scene.failure_reason, scene.failure_kind = "Unresolved FEM tetrahedron motion: test", "tissue"
        scene.contact.nodal_force.fill_(wp.vec3(1.0, 0.0, 0.0))
        with contextlib.redirect_stdout(io.StringIO()):
            example._fail(scene.failure_reason)
        restart = path.restarts["border"]
        self.assertEqual(example.frame, restart)
        self.assertIsNone(scene.failure_reason)
        self.assertIsNone(example.halted)
        self.assertEqual(float(np.abs(scene.contact.nodal_force.numpy()).max()), 0.0)
        np.testing.assert_allclose(
            scene.state_0.joint_q.numpy()[list(scene.graspers[0].arm_q_indices)],
            example.arm_targets[restart],
            atol=1e-6,
        )
        self.assertEqual(example.measurements.failures[0]["phase"], "press")
        self.assertEqual(example.measurements.failures[0]["restart_frame"], restart)
        phases = example.measurements.phases()
        self.assertEqual(phases["slide"]["status"], "skipped after a failure")
        # The run goes on at the restart frame, from the placement; a slow step is listed with that frame.
        with contextlib.redirect_stdout(io.StringIO()):
            example.step()
        np.testing.assert_array_equal(scene._ramp[0].numpy(), example.arm_targets[restart].astype(np.float32))
        np.testing.assert_array_equal(scene._ramp[1].numpy(), example.arm_targets[restart + 1].astype(np.float32))
        self.assertEqual(example.step_frames, [0, restart])
        self.assertEqual(example.measurements.records[-1]["frame"], restart)
        limit = example_module.FRAME_LIMIT_MS
        example_module.FRAME_LIMIT_MS = 0.0
        try:
            timing = example.timing()
        finally:
            example_module.FRAME_LIMIT_MS = limit
        self.assertEqual([frame for frame, _ in timing["steps_over_limit"]], [0, restart])
        self.assertEqual([frame for frame, _ in timing["periods_over_limit"]], [0])
        # A failure of the rigid solver halts the run, with or without a contact.
        scene.failure_reason, scene.failure_kind = "nonfinite rigid state", "rigid"
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(RuntimeError):
            example._fail(scene.failure_reason)
        example.close()


class PinchRowTests(unittest.TestCase):
    """Rows 3a, 3b, 8 and 9 of the pinch's self-check, on marks and frame records made by hand."""

    # The tracking tolerance of the tool in the example [m].
    TOLERANCE = 39.5e-6
    # Travel of a place on the jaws in one physics interval at the speed of the example [m].
    TRAVEL = 12.5e-6

    @staticmethod
    def _mark(carried, places, away=(), nodes=27, slipped=0.0, residual=2.5e-6, travel=12.5e-6, frame=0):
        """A mark of the measurements: mean rises [m], and nodes that are ``away`` [m] from their places (signed)."""
        return {
            "frame": frame,
            "pinch_carried_rise_m": carried,
            "pinch_jaw_point_rise_m": places,
            "pinch_node_lag_m": [*away, *([0.0] * (nodes - len(away)))],
            "pinch_slip_since_reference_m": slipped,
            "pinch_carry_residual_m": residual,
            "pinch_place_travel_m": travel,
        }

    def test_at_rest_every_node_is_at_its_place_at_both_holds(self):
        tolerance = self.TOLERANCE
        lifted = self._mark(4.339e-3, 4.349e-3)
        good = self._mark(4.445e-3, 4.438e-3)

        def verdict(lift_hold, lower_hold):
            return Measurements.rise_rows(lifted, lift_hold, lower_hold, good, tolerance)[0][2]

        rest = Measurements.rise_rows(lifted, good, good, good, tolerance)[0]
        self.assertEqual((rest[0], rest[2]), ("3a", True))
        self.assertIn("end of the lift hold", rest[3])
        self.assertIn("end of the lower hold", rest[3])
        self.assertIn("0 of 27 nodes more than 39.5 um from their place", rest[3])
        # One node 60 um from its place, behind or ahead, at either hold: not met, although the mean is inside
        # (60 um / 27 nodes = 2.2 um).
        for away in (-60.0e-6, 60.0e-6):
            one = self._mark(4.445e-3, 4.438e-3, away=(away,))
            self.assertIs(verdict(one, good), False)
            self.assertIs(verdict(good, one), False)
            self.assertIn("1 of 27 nodes", Measurements.rise_rows(lifted, one, good, good, tolerance)[0][3])
        # Inside the tolerance it passes; the measured landed carry (one node 472.5 um behind at the lift hold) does not.
        self.assertIs(verdict(self._mark(4.445e-3, 4.438e-3, away=(-39.0e-6, 39.0e-6)), good), True)
        self.assertIs(verdict(self._mark(2.5790e-3, 2.5948e-3, away=(-472.5e-6,)), good), False)
        # A pair that has slipped since the squeeze hold fails the row at the hold where it had.
        self.assertIs(verdict(self._mark(4.445e-3, 4.438e-3, slipped=1.0e-6), good), False)
        self.assertIs(verdict(good, self._mark(0.0, 0.0, slipped=1.0e-6)), False)
        # A hold that was not run leaves the row without a verdict, unless the other hold fails.
        self.assertIsNone(verdict(good, None))
        self.assertIs(verdict(self._mark(4.445e-3, 4.438e-3, away=(-60.0e-6,)), None), False)

    def test_moving_jaws_keep_the_tissue_within_one_interval_of_travel(self):
        tolerance, travel = self.TOLERANCE, self.TRAVEL
        hold = self._mark(4.445e-3, 4.438e-3)

        def verdict(lifted, last):
            return Measurements.rise_rows(lifted, hold, hold, last, tolerance)[1][2]

        lifted = self._mark(4.339e-3, 4.349e-3)
        moving = Measurements.rise_rows(lifted, hold, hold, self._mark(0.0, 0.0, frame=1557), tolerance)[1]
        self.assertEqual((moving[0], moving[2]), ("3b", True))
        self.assertIn("at most 2.50 um from its place", moving[3])
        self.assertIn(f"allowed {1e6 * (1.0 + HOLD_REMAINDER) * travel:.2f} um", moving[3])
        self.assertIn("-10.0 um in the mean", moving[3])
        self.assertIn("since the squeeze hold (up to frame 1557)", moving[3])
        # The hold of a node leaves at most the share of a move that its factor allows for the largest eigenvalue
        # of a block: three times the mean diagonal entry.
        self.assertEqual(HOLD_REMAINDER, 3.0 / 4096.0)
        # A residual up to one interval's travel (and the hold's share of it) passes, more does not: the measured
        # landed carry leaves its slowest node 1.6 mm from its place.
        self.assertIs(verdict(lifted, self._mark(0.0, 0.0, residual=(1.0 + HOLD_REMAINDER) * travel)), True)
        self.assertIs(verdict(lifted, self._mark(0.0, 0.0, residual=1.01 * travel)), False)
        self.assertIs(verdict(lifted, self._mark(0.0, 0.0, residual=1.6e-3)), False)
        # A carry that is slow by one part in 400 of the travel is outside when 400 intervals of a 5 mm lift have
        # added up: 400 x 12.5 um / 400.
        self.assertIs(verdict(lifted, self._mark(0.0, 0.0, residual=400 * travel / 400 + 1.0e-7)), False)
        # No interval in which the pair stuck and its places moved: nothing was carried.
        self.assertIs(verdict(lifted, self._mark(0.0, 0.0, residual=0.0, travel=0.0)), False)
        # Tissue that has not risen with its places at the end of the lift (here: not at all, and the landed
        # carry's 154.4 um in the mean), tissue ahead of them, and a pair that slipped.
        last = self._mark(0.0, 0.0)
        self.assertIs(verdict(self._mark(0.0, 4.349e-3), last), False)
        self.assertIs(verdict(self._mark(2.3670e-3, 2.5214e-3), last), False)
        self.assertIs(verdict(self._mark(4.349e-3 + tolerance + 1.1 * travel, 4.349e-3), last), False)
        self.assertIs(verdict(self._mark(4.349e-3 - tolerance - 0.9 * travel, 4.349e-3), last), True)
        self.assertIs(verdict(self._mark(4.339e-3, 4.349e-3, slipped=1.0e-6), last), False)

    def _cycle(self, reaction=1.0, follow=1.0, kept=0, halves=True, nodes=27, stiffness=115.0, lift=4.4e-3, frames=88):
        """Frame records of a lift, a hold, a lowering and a hold, by the measurements' own work of a frame.

        ``nodes`` carried nodes rise with two jaws (bodies 0 and 1) by ``lift`` [m] in
        ``frames`` frames and come down again; each pulls down with its share of ``stiffness``
        [N/m] times its rise. ``reaction`` scales what the tool gets of a node's force,
        ``follow`` is the share of the jaws' travel that the nodes make while the jaws move
        (they make up the rest in the hold that follows), and ``kept`` nodes are held by the
        pinch alone while both jaws drift half of their rise along the closing direction (x):
        their force along it goes half to each jaw (``halves``), or to no body.
        Returns the records and the marks of the squeeze hold, the lift hold and the lower hold.
        """
        measurements = Measurements.__new__(Measurements)
        measurements._pinch_previous = None
        ids = np.arange(100, 100 + nodes)
        body = (np.arange(nodes) % 2).astype(np.int64)
        start = np.stack([0.001 * (2 * body - 1), 0.0002 * np.arange(nodes), np.zeros(nodes)], axis=1)
        hold = 30
        jaw_rise = np.concatenate(
            (
                [0.0],
                np.linspace(0.0, lift, frames + 1)[1:],
                np.full(hold, lift),
                np.linspace(lift, 0.0, frames + 1)[1:],
                np.zeros(hold),
            )
        )
        labels = (
            ["squeeze hold"] + ["lift"] * frames + ["lift hold"] * hold + ["lower"] * frames + ["lower hold"] * hold
        )
        records, node_rise = [], 0.0
        for frame, (rise, label) in enumerate(zip(jaw_rise, labels, strict=True)):
            moving = frame > 0 and rise != jaw_rise[frame - 1]
            # While the jaws move the nodes make ``follow`` of the travel; in a hold they close a tenth of the gap.
            if moving:
                node_rise += follow * (rise - jaw_rise[frame - 1])
            else:
                node_rise += 0.1 * (rise - node_rise) if follow != 1.0 else rise - node_rise
            x = start + (0.5 * rise if kept else 0.0, 0.0, node_rise)
            force = np.zeros((nodes, 3))
            force[:, 2] = -stiffness * node_rise / nodes
            force[:, 0] = 0.1 * force[:, 2]
            along = np.zeros((nodes, 3))
            along[:kept, 0] = force[:kept, 0]
            body_q = np.zeros((2, 7))
            body_q[:, 0], body_q[:, 2], body_q[:, 6] = 0.5 * rise if kept else 0.0, rise, 1.0
            numbers = measurements._pinch_work(
                {
                    "frame": frame,
                    "ids": ids,
                    "body": body,
                    "other": 1 - body,
                    "x": x,
                    "force": force,
                    "tool": reaction * (force - along) + (0.5 * along if halves else 0.0),
                    "tool_other": 0.5 * along if halves else 0.0 * along,
                    "body_q": body_q,
                }
            )
            records.append(
                {
                    "frame": frame,
                    "segment": label,
                    "pinch_pull_N": stiffness * node_rise,
                    "pinch_law_force_vector_N": [0.0, 0.0, 0.39],
                    "pinch_slip_since_reference_m": 0.0,
                    "pinch_carry_residual_m": 2.5e-6,
                    "pinch_place_travel_m": 12.5e-6,
                    **numbers,
                }
            )
        ends = {label: next(r for r in reversed(records) if r["segment"] == label) for label in set(labels)}
        return records, ends["squeeze hold"], ends["lift hold"], ends["lower hold"]

    def test_the_work_of_a_cycle_balances_within_the_lag_of_the_reaction(self):
        records, squeeze, lift_hold, lowered = self._cycle()
        self.assertNotIn("pinch_work_tool_J", records[0])
        row = Measurements.work_row(records, squeeze, lift_hold, lowered)
        self.assertEqual((row[0], row[2]), (8, True))
        # The stroke: half of 115 N/m x (4.4 mm)^2 = 1.113 mJ up and the same back.
        self.assertIn("lift and lift hold: tool +1.1132 mJ, tissue +1.1132 mJ", row[3])
        self.assertIn("lower and lower hold: tool -1.1132 mJ, tissue -1.1132 mJ", row[3])
        # The bound: the largest pull (115 N/m x 4.4 mm = 506 mN) times the largest travel of a place in one
        # interval, and twice that pull times the largest distance of a node from its place.
        self.assertIn(f"allowed {1e3 * 0.506 * (12.5e-6 + 2 * 2.5e-6):.4f} mJ", row[3])
        self.assertIn("+390.0 mN at the squeeze hold and +390.0 mN at the lift hold", row[3])
        # A reaction that is a tenth short: the tool pays a tenth less on the way up and gets a tenth less back.
        # Over the cycle the two cancel; each stroke is held to its own half of the bound.
        short = Measurements.work_row(*self._cycle(reaction=0.9))
        self.assertIs(short[2], False)
        self.assertIn("lift and lift hold: tool +1.0019 mJ, tissue +1.1132 mJ", short[3])
        # Each stroke has half of the bound (4.4 uJ of 8.9 uJ): a reaction that is 0.6 % short (6.7 uJ in a stroke)
        # is outside it, one that is 0.3 % short (3.3 uJ) is inside.
        self.assertIs(Measurements.work_row(*self._cycle(reaction=0.994))[2], False)
        self.assertIs(Measurements.work_row(*self._cycle(reaction=0.997))[2], True)
        # Tissue that is behind the jaws while they move and is taken to its places while they stand: the tissue
        # receives that work in the hold and the tool pays none of it.
        self.assertIs(Measurements.work_row(*self._cycle(follow=0.9))[2], False)
        # Nodes that only the pinch holds, with jaws that drift along the closing direction together: their force
        # along it is paid by the two jaws, half each, and the work balances (9 of 27 nodes, a tenth of the pull
        # along the closing direction, half of the rise as drift: 1.1132 mJ / 60 on top of the stroke).
        kept = Measurements.work_row(*self._cycle(kept=9))
        self.assertIs(kept[2], True)
        self.assertIn("lift and lift hold: tool +1.1689 mJ, tissue +1.1689 mJ", kept[3])
        # Without the halves that force reaches no body and the tissue receives work that no jaw paid.
        unpaid = Measurements.work_row(*self._cycle(kept=9, halves=False))
        self.assertIs(unpaid[2], False)
        self.assertIn("lift and lift hold: tool +1.1503 mJ, tissue +1.1689 mJ", unpaid[3])
        # A residual that a carry which keeps up cannot leave does not widen the bound.
        records, squeeze, lift_hold, lowered = self._cycle(follow=0.9)
        wide = Measurements.work_row(records, squeeze, lift_hold, {**lowered, "pinch_carry_residual_m": 1.6e-3})
        self.assertIs(wide[2], False)
        # A pair that slipped is not judged, and frames that were not run leave the row without a verdict.
        slipped = Measurements.work_row(records, squeeze, lift_hold, {**lowered, "pinch_slip_since_reference_m": 1e-4})
        self.assertIsNone(slipped[2])
        self.assertIn("not judged, the pair slipped 0.100 mm", slipped[3])
        self.assertIsNone(Measurements.work_row(records, squeeze, lift_hold, None)[2])

    def test_the_grip_holds_the_lift_while_friction_and_the_tent_allow_it(self):
        def marks(limit=0.465, pull=0.370, stretch=0.38):
            squeeze = {"frame": 10, "pinch_squeeze_N": 0.280, "pinch_limit_N": 0.465, "pinch_pull_N": 0.02}
            lifted = {
                "frame": 110,
                "pinch_squeeze_N": limit / 1.66,
                "pinch_limit_N": limit,
                "pinch_pull_N": pull,
                "pinch_carried_rise_m": 3.2e-3,
            }
            lift_hold = {**lifted, "frame": 140, "pinch_tent_min_stretch": stretch}
            return squeeze, lifted, lift_hold

        def records(short=()):
            frames = []
            for frame in range(10, 300):
                segment = "squeeze hold" if frame <= 10 else ("lift" if frame <= 110 else "lift hold")
                pull = 0.370 * min(1.0, (frame - 10) / 100.0)
                frames.append(
                    {
                        "frame": frame,
                        "segment": segment,
                        "pinch_pull_N": pull,
                        "pinch_limit_N": 0.9 * pull if frame in short else 0.465,
                    }
                )
            return frames

        none = {"closed_intervals": 0}
        row = Measurements.grip_row(records(), *marks(), none, 0.2)
        self.assertEqual((row[0], row[2]), (9, True))
        # The formula's limit: 2 mu S_0 / k_T = 465 mN / (370 mN / 3.2 mm) = 4.02 mm of rise.
        self.assertIn("S_0 = 280.0 mN", row[3])
        self.assertIn("k_T = 115.6 N/m", row[3])
        self.assertIn("2 mu S_0 / k_T = 4.02 mm of rise", row[3])
        self.assertIn("limit over pull 1.26 at the end of the lift and 1.26 at the end of its hold", row[3])
        self.assertIn("smallest stretch of the tent 0.380", row[3])
        # A slip while the jaws were closed, and a friction limit below the pull at the end of a frame.
        self.assertIs(Measurements.grip_row(records(), *marks(), {"closed_intervals": 3}, 0.2)[2], False)
        short = Measurements.grip_row(records(short=(90,)), *marks(), none, 0.2)
        self.assertIs(short[2], False)
        self.assertIn("at the end of 1 of 290 frames (first: frame 90, lift)", short[3])
        # The tent at the stretch limit of the material.
        self.assertIs(Measurements.grip_row(records(), *marks(stretch=0.2), none, 0.2)[2], False)
        # The margin is stated, not judged: a grip at 1.05 times the pull passes.
        self.assertIs(Measurements.grip_row(records(), *marks(limit=0.3885), none, 0.2)[2], True)
        # Without the lift there is no verdict.
        self.assertIsNone(Measurements.grip_row(records(), marks()[0], None, None, none, 0.2)[2])


class ToolTissueStartupTests(unittest.TestCase):
    def setUp(self):
        wp.init()
        device = wp.get_device(DEVICE)
        if not device.is_cuda or not wp.is_conditional_graph_supported():
            self.skipTest("The start-up call captures CUDA graphs")

    def _run(self, contact, frames=6, first_frame=0, openings=None):
        wp.set_device(DEVICE)
        args = example_module.Example.create_parser().parse_args(
            ["--device", DEVICE, "--viewer", "null", "--num-frames", str(frames), "--contact", contact]
        )
        states = {}
        start = ToolTissueScene.start

        def checked_start(scene):
            # The start-up call runs real intervals: every array must come back bit for bit.
            names = ("particle_q", "particle_qd", "body_q", "body_qd", "joint_q", "joint_qd")
            before = [getattr(scene.state_0, name).numpy().copy() for name in names]
            before.append(scene.control.joint_target_q.numpy().copy())
            before.extend(array.numpy().copy() for array in scene.contact.state)
            start(scene)
            after = [getattr(scene.state_0, name).numpy() for name in names]
            after.append(scene.control.joint_target_q.numpy())
            after.extend(array.numpy() for array in scene.contact.state)
            states["equal"] = all(np.array_equal(a, b) for a, b in zip(before, after, strict=True))
            states["status"] = scene._startup_status

        ToolTissueScene.start = checked_start
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                example = example_module.Example(ViewerNull(num_frames=frames), args)
        finally:
            ToolTissueScene.start = start
        self.assertTrue(states["equal"])
        scene = example.scene
        self.assertTrue(scene._startup_ready, states["status"])
        self.assertEqual(scene.elapsed, 0.0)
        self.assertEqual(scene.accepted_intervals, 0)
        self.assertEqual(len(scene._graphs), 2)
        self.assertEqual(len(scene._end_graphs), 0 if contact == "off" else 2)
        if first_frame:
            # A reset-time placement, as after a failed phase: the jaws at the tissue surface.
            scene.place(example.arm_targets[first_frame])
            example.frame = first_frame
        captures = capture.census.captures
        reads = []
        numpy = wp.array.numpy

        def counted(array, *args, **kwargs):
            reads.append(array.shape)
            return numpy(array, *args, **kwargs)

        with contextlib.redirect_stdout(io.StringIO()):
            for index in range(frames):
                # The step alone: one read per interval, the result of its check. A jaw command of the frame
                # (``openings``: per frame, the half-opening in parts of the widest) is issued as the example
                # issues it and adds none.
                wp.array.numpy = counted
                try:
                    example.scene.set_frame_targets(
                        example.arm_targets[example.frame], example.arm_targets[example.frame + 1]
                    )
                    if openings is not None:
                        example.scene.set_jaw_opening(openings[index] * scene.maximum_half_opening)
                    example.scene.step()
                finally:
                    wp.array.numpy = numpy
                self.assertIsNone(scene.failure_reason)
                example.frame += 1
        self.assertEqual(reads, [(3,)] * (frames * scene.args.substeps))
        self.assertEqual(capture.census.captures, captures)
        census = scene.census()
        self.assertEqual(census["capture_census_status"], "checked")
        for name in ("captures_after_startup", "solver_captures_after_startup", "module_loads_after_startup"):
            self.assertEqual(census[name], 0, (name, census["late_sites"], census["late_modules"]))
        self.assertEqual(census["startup_unlaunched_graphs"], 0)
        observation = scene.observe()
        # What the jaws of the driven tool are commanded to at the end [rad], in parts of the widest half-opening.
        jaws = list(scene.graspers[0].jaw_q_indices)
        observation["jaw_targets"] = scene.control.joint_target_q.numpy()[jaws] / scene.maximum_half_opening
        observation["jaw_angles"] = np.asarray(observation["joint_q"])[jaws] / scene.maximum_half_opening
        example.close()
        return observation

    def test_start_up_leaves_the_state_and_frames_capture_and_load_nothing_without_a_contact(self):
        self._run("off")

    def test_start_up_leaves_the_state_and_frames_capture_and_load_nothing_with_the_swept_contact(self):
        self._run("swept")
        # The same while the jaws press the tissue: nodes are held, nothing is captured, loaded or read.
        observation = self._run("swept", frames=8, first_frame=PRESS_FRAME)
        self.assertGreater(observation["contact"]["held_nodes"], 0)
        self.assertGreater(observation["dimple_depth_m"], 2.0e-4)

    def test_start_up_leaves_the_state_and_frames_capture_and_load_nothing_with_the_first_contact(self):
        self._run("first")

    def test_jaw_commands_in_frames_read_capture_and_load_nothing(self):
        # The start-up rule of the jaw command, on the device: frames in which the jaws are commanded to another
        # opening issue no graph capture, no kernel module load and no read beside the one of the interval check.
        frames = 6
        openings = [0.5 * (index + 1) / frames for index in range(frames)]
        for contact in ("off", "swept"):
            with self.subTest(contact=contact):
                observation = self._run(contact, frames=frames, openings=openings)
                # The last command is on the device, and both hinges have begun to follow it.
                np.testing.assert_allclose(observation["jaw_targets"], [openings[-1], -openings[-1]], rtol=1e-6)
                self.assertGreater(float(observation["jaw_angles"][0]), 0.0)
                self.assertLess(float(observation["jaw_angles"][1]), 0.0)


if __name__ == "__main__":
    unittest.main()
