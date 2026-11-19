# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Lagged frictionless contact of tool capsules with the tissue surface: law, balance, state and switch."""

import os
import unittest
from types import SimpleNamespace

import newton
import numpy as np
import warp as wp
from test_worker_thread import scene_fixture

from newton_solver_suturebot._src.solvers.coupled.interface.tool_tissue_contact import (
    ToolTissueContact,
    capsule_samples,
)
from newton_solver_suturebot._src.solvers.thread.worker_thread import WorkerThread

DEVICE = os.environ.get("NEWTON_TEST_DEVICE", "cpu")
# A square of 10 mm in the plane z = 0 whose two triangles face +z: the tissue is below it.
POINTS = np.array([[0, 0, 0], [0.01, 0, 0], [0.01, 0.01, 0], [0, 0.01, 0]], dtype=np.float32)
TRIANGLES = np.array([0, 1, 2, 0, 2, 3], dtype=np.int32)
STIFFNESS, FREQUENCY, RADIUS = 8.0, 400.0, 0.001
# One capsule of 4 mm on body 1, along its x axis: five spheres of 1 mm radius, 1 mm apart.
ENDPOINTS = np.array([[[-0.002, 0, 0], [0.002, 0, 0]]])
TILT = 0.2
COM = np.array([0.0005, -0.0003, 0.0002])


def _pose(height, tilt=TILT, x=0.0055, y=0.003):
    rotation = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), tilt)
    return np.array([x, y, height, *rotation], dtype=np.float32)


def _centres(pose, local):
    transform = wp.transform(wp.vec3(*pose[:3]), wp.quat(*pose[3:]))
    return np.array([wp.transform_point(transform, wp.vec3(*point)) for point in local], dtype=float)


class _Rig:
    """A flat tissue patch of four particles and two rigid bodies; body 1 carries the capsule."""

    def __init__(self, *, point_particle=None, dt=1 / 240, reaction=True):
        builder = newton.ModelBuilder()
        for point in POINTS:
            builder.add_particle(wp.vec3(*point), wp.vec3(), mass=1e-4)
        for _ in range(2):
            builder.add_body(mass=0.01, inertia=wp.mat33(1e-8, 0, 0, 0, 1e-8, 0, 0, 0, 1e-8))
        self.model = builder.finalize(device=DEVICE)
        com = self.model.body_com.numpy()
        com[1] = COM
        self.model.body_com.assign(com)
        self.points = wp.array(POINTS, dtype=wp.vec3, device=DEVICE)
        self.triangles = wp.array(TRIANGLES, dtype=int, device=DEVICE)
        self.mesh = wp.Mesh(self.points, self.triangles)
        self.map = np.arange(4, dtype=np.int32) if point_particle is None else np.asarray(point_particle, np.int32)
        self.step_seconds = wp.array([dt], dtype=wp.float64, device=DEVICE)
        self.bodies, self.local, self.radii = capsule_samples([1], ENDPOINTS, [RADIUS])
        self.contact = ToolTissueContact(
            self.model,
            self.mesh,
            self.triangles,
            wp.array(self.map, dtype=int, device=DEVICE),
            self.bodies,
            self.local,
            self.radii,
            node_stiffness=STIFFNESS,
            shear_frequency=FREQUENCY,
            step_seconds=self.step_seconds,
            reach=0.02,
            reaction=reaction,
        )
        self.body_q = wp.array(
            np.array([[0, 0, 1, 0, 0, 0, 1], _pose(1.0)], dtype=np.float32), dtype=wp.transform, device=DEVICE
        )
        self.particle_f = wp.zeros(4, dtype=wp.vec3, device=DEVICE)

    def place(self, pose):
        poses = self.body_q.numpy()
        poses[1] = pose
        self.body_q.assign(poses)

    def update(self):
        self.particle_f.zero_()
        self.contact.update(self.body_q, self.particle_f)
        return self.contact.force.numpy().copy()


def _weights(point):
    """Barycentric weights of a point of the patch in the triangle (0, 1, 2), where y < x."""
    x, y = point[0] / 0.01, point[1] / 0.01
    return np.array([1.0 - x, x - y, y])


class CapsuleSampleTests(unittest.TestCase):
    def test_chain_is_at_most_one_radius_apart_and_spans_the_capsule_from_cap_to_cap(self):
        bodies = [3, 4, 6]
        endpoints = np.array(
            [
                [[0, 0, -0.243], [0, 0, -0.003]],
                [[0, 0, 0], [0.008, 0, 0]],
                [[-0.00075, 0.0015, 0], [-0.00075, 0.009, 0]],
            ]
        )
        radii = np.array([0.0043, 0.0022, 0.00065])
        sample_bodies, points, sample_radii = capsule_samples(bodies, endpoints, radii)
        # The capsules' own radii: ceil(L / r) + 1 spheres from end point to end point.
        self.assertEqual(np.bincount(sample_bodies).tolist()[3:], [57, 5, 0, 13])
        for body, (start, end), radius in zip(bodies, endpoints, radii, strict=True):
            chain = points[sample_bodies == body]
            np.testing.assert_allclose(chain[0], start, atol=1e-15)
            np.testing.assert_allclose(chain[-1], end, atol=1e-15)
            self.assertLessEqual(np.linalg.norm(np.diff(chain, axis=0), axis=1).max(), radius * (1 + 1e-12))
            np.testing.assert_array_equal(sample_radii[sample_bodies == body], radius)
        # Another contact radius: the caps of the chain stay where the caps of the capsule are.
        contact = np.array([0.0042, 0.0030, 0.00095])
        sample_bodies, points, sample_radii = capsule_samples(bodies, endpoints, radii, contact)
        for body, (start, end), radius, wide in zip(bodies, endpoints, radii, contact, strict=True):
            chain = points[sample_bodies == body]
            axis = (end - start) / np.linalg.norm(end - start)
            np.testing.assert_allclose((chain[0] - start) @ axis - wide, -radius, atol=1e-12)
            np.testing.assert_allclose((chain[-1] - end) @ axis + wide, radius, atol=1e-12)
            np.testing.assert_allclose(np.cross(chain - start, axis), 0, atol=1e-12)
            self.assertLessEqual(np.linalg.norm(np.diff(chain, axis=0), axis=1).max(), wide * (1 + 1e-12))
            np.testing.assert_array_equal(sample_radii[sample_bodies == body], wide)
        self.assertEqual(np.bincount(sample_bodies).tolist()[3:], [59, 4, 0, 9])
        # A capsule that is shorter, cap to cap, than one sphere: one sphere at its middle.
        sample_bodies, points, sample_radii = capsule_samples([0], [[[0, 0, 0], [0.001, 0, 0]]], [0.0005], [0.002])
        np.testing.assert_allclose(points, [[0.0005, 0, 0]])
        self.assertEqual((sample_bodies.tolist(), sample_radii.tolist()), ([0], [0.002]))
        for arguments in (
            ([0], [[[0, 0, 0], [0, 0, 0]]], [0.001]),
            ([0], [[[0, 0, 0], [0.001, 0, 0]]], [0.0]),
            ([0], [[[0, 0, 0], [0.001, 0, 0]]], [0.001], [np.nan]),
            ([], np.empty((0, 2, 3)), []),
        ):
            with self.assertRaises(ValueError):
                capsule_samples(*arguments)


class ToolTissueContactTests(unittest.TestCase):
    def expected(self, rig, pose):
        centres = _centres(pose, rig.local)
        return centres, centres[:, 2] - RADIUS

    def test_force_law_matches_a_hand_computation_and_the_loads_balance(self):
        rig = _Rig()
        pose = _pose(0.00085)
        rig.place(pose)
        centres, gaps = self.expected(rig, pose)
        # Heights of the five centres: 0.85 mm -+ 1 and 2 mm times sin(0.2). Three are inside.
        np.testing.assert_allclose(
            gaps, 0.00085 - RADIUS + np.array([2, 1, 0, -1, -2]) * 0.001 * np.sin(TILT), atol=1e-9
        )
        inside = gaps < 0
        self.assertEqual(int(inside.sum()), 3)
        # (omega h)^2 / 2 = (400 / 240)^2 / 2 > 1: the whole step K / n.
        first = np.where(inside, -(STIFFNESS / 3) * gaps, 0.0)
        np.testing.assert_allclose(rig.update(), first, rtol=2e-4, atol=1e-9)
        self.assertEqual(rig.contact._count.numpy()[0], 3)
        np.testing.assert_allclose(rig.contact._gap.numpy(), gaps, atol=2e-8)
        np.testing.assert_allclose(rig.contact._normal.numpy(), np.tile([0, 0, 1.0], (5, 1)), atol=1e-5)
        # The tissue does not move: the same gaps raise the forces by the same step again.
        forces = rig.update()
        np.testing.assert_allclose(forces, 2 * first, rtol=2e-4, atol=1e-9)
        # On the tissue: minus the force at the surface point, by the barycentric weights.
        nodal = np.zeros((4, 3))
        for centre, force in zip(centres, forces, strict=True):
            nodal[:3, 2] -= force * _weights(centre)
        np.testing.assert_allclose(rig.particle_f.numpy(), nodal, rtol=2e-4, atol=1e-9)
        # On the tool: the force at the surface point, torque about the centre of mass.
        wrench = rig.contact.wrench.numpy()
        np.testing.assert_array_equal(wrench[0], 0)
        push = np.zeros((5, 3))
        push[:, 2] = forces
        surface = centres * [1, 1, 0]
        centre_of_mass = _centres(pose, [COM])[0]
        np.testing.assert_allclose(wrench[1, :3], push.sum(axis=0), rtol=2e-4, atol=1e-9)
        np.testing.assert_allclose(
            wrench[1, 3:], np.cross(surface - centre_of_mass, push).sum(axis=0), rtol=2e-4, atol=1e-10
        )
        # Equal and opposite: forces sum to zero, and so do the moments about the origin.
        nodal_f = rig.particle_f.numpy().astype(float)
        np.testing.assert_allclose(nodal_f.sum(axis=0), -wrench[1, :3], rtol=1e-5, atol=1e-10)
        tool_moment = wrench[1, 3:] + np.cross(centre_of_mass, wrench[1, :3])
        tissue_moment = np.cross(POINTS.astype(float), nodal_f).sum(axis=0)
        np.testing.assert_allclose(tissue_moment, -tool_moment, rtol=1e-4, atol=1e-10)
        # The stored reaction is what add_forces adds, and adding it changes no contact state.
        body_f = wp.zeros(2, dtype=wp.spatial_vector, device=DEVICE)
        saved = [array.numpy().copy() for array in rig.contact.state]
        for _ in range(2):
            rig.contact.add_forces(body_f)
        np.testing.assert_allclose(body_f.numpy(), 2 * wrench, rtol=1e-6)
        for array, value in zip(rig.contact.state, saved, strict=True):
            np.testing.assert_array_equal(array.numpy(), value)
        report = rig.contact.diagnostics()
        self.assertEqual((report["active_samples"], report["loaded_samples"]), (3, 3))
        self.assertAlmostEqual(report["deepest_gap_m"], gaps.min(), delta=2e-8)
        self.assertAlmostEqual(report["total_force_N"], forces.sum(), delta=1e-8)

    def test_update_and_reaction_read_nothing_back_from_the_device(self):
        from unittest.mock import patch

        rig = _Rig()
        rig.place(_pose(0.00085))
        body_f = wp.zeros(2, dtype=wp.spatial_vector, device=DEVICE)
        reads = []
        with (
            patch.object(wp.array, "numpy", side_effect=lambda *a, **k: reads.append("numpy")),
            patch.object(wp.array, "list", side_effect=lambda *a, **k: reads.append("list")),
            patch.object(wp.array, "__getitem__", side_effect=lambda *a, **k: reads.append("item")),
        ):
            for _ in range(3):
                rig.contact.update(rig.body_q, rig.particle_f)
                rig.contact.add_forces(body_f)
        self.assertEqual(reads, [])
        # The count of active samples was taken on the device all the same.
        self.assertEqual(rig.contact._count.numpy()[0], 3)
        self.assertGreater(rig.contact.force.numpy().max(), 0)

    def test_step_fraction_follows_the_time_step_that_the_device_holds(self):
        # (omega h)^2 / 2 at 1/480 s: 0.347 of the step; at 1/960 s: 0.087.
        for dt in (1 / 480, 1 / 960):
            with self.subTest(dt=dt):
                rig = _Rig(dt=dt)
                pose = _pose(0.00085)
                rig.place(pose)
                _, gaps = self.expected(rig, pose)
                fraction = 0.5 * (FREQUENCY * dt) ** 2
                self.assertLess(fraction, 1)
                expected = np.where(gaps < 0, -fraction * (STIFFNESS / 3) * gaps, 0.0)
                np.testing.assert_allclose(rig.update(), expected, rtol=2e-4, atol=1e-10)
        # The step is device data: writing it changes the next update, with no other call.
        rig.step_seconds.fill_(1 / 240)
        np.testing.assert_allclose(
            rig.update(), expected + np.where(gaps < 0, -(STIFFNESS / 3) * gaps, 0.0), rtol=2e-4, atol=1e-10
        )

    def test_no_force_at_a_gap(self):
        rig = _Rig()
        # The lowest sphere is 0.05 mm clear of the surface.
        rig.place(_pose(0.00105 + 0.002 * np.sin(TILT)))
        for _ in range(3):
            np.testing.assert_array_equal(rig.update(), 0)
            self.assertEqual(rig.contact._count.numpy()[0], 0)
            np.testing.assert_array_equal(rig.particle_f.numpy(), 0)
            np.testing.assert_array_equal(rig.contact.wrench.numpy(), 0)
        self.assertGreater(rig.contact._gap.numpy().min(), 0)
        self.assertTrue((rig.contact._face.numpy() >= 0).all())

    def test_force_is_released_when_the_capsule_leaves(self):
        rig = _Rig()
        rig.place(_pose(0.00085))
        for _ in range(4):
            loaded = rig.update()
        self.assertEqual(int((loaded > 0).sum()), 3)
        # Lifted by 0.7 mm every sphere has a gap: each loaded one loses (K / n) gap per step, n
        # the number that still carry force, until its force is zero. No force goes below zero.
        pose = _pose(0.00155)
        rig.place(pose)
        _, gaps = self.expected(rig, pose)
        self.assertGreater(gaps.min(), 0)
        expected, counts = loaded.astype(float), []
        for _ in range(12):
            active = expected > 0
            counts.append(int(active.sum()))
            if active.any():
                expected = np.where(active, np.maximum(0.0, expected - (STIFFNESS / active.sum()) * gaps), 0.0)
            forces = rig.update()
            self.assertEqual(rig.contact._count.numpy()[0], counts[-1])
            np.testing.assert_allclose(forces, expected, rtol=1e-3, atol=1e-9)
            self.assertGreaterEqual(forces.min(), 0)
        self.assertEqual(counts[0], 3)
        self.assertEqual(counts[-1], 0)
        np.testing.assert_array_equal(forces, 0)
        np.testing.assert_array_equal(rig.particle_f.numpy(), 0)
        np.testing.assert_array_equal(rig.contact.wrench.numpy(), 0)
        # Beyond the reach of the surface search the force is dropped at once.
        rig.place(_pose(0.00085))
        self.assertGreater(rig.update().max(), 0)
        rig.place(_pose(0.5))
        np.testing.assert_array_equal(rig.update(), 0)
        np.testing.assert_array_equal(rig.contact._face.numpy(), -1)
        np.testing.assert_array_equal(rig.contact.wrench.numpy(), 0)

    def test_surface_vertices_that_are_no_particles_take_their_share_to_the_ground(self):
        # Vertex 1 of the patch is not a tissue particle: particles 0..2 are vertices 0, 2, 3.
        rig = _Rig(point_particle=[0, -1, 1, 2])
        pose = _pose(0.00085)
        rig.place(pose)
        centres, _ = self.expected(rig, pose)
        forces = rig.update()
        nodal = np.zeros((4, 3))
        for centre, force in zip(centres, forces, strict=True):
            weights = _weights(centre)
            nodal[0, 2] -= force * weights[0]
            nodal[1, 2] -= force * weights[2]
        np.testing.assert_allclose(rig.particle_f.numpy(), nodal, rtol=2e-4, atol=1e-9)
        # The tool still gets the whole reaction.
        self.assertAlmostEqual(rig.contact.wrench.numpy()[1, 2], forces.sum(), delta=1e-8)

    def test_a_sample_inside_the_tissue_is_pushed_out_along_the_surface_normal(self):
        # Centres under the surface: the gap is minus (depth + radius), the direction still +z.
        rig = _Rig()
        pose = _pose(-0.0006, tilt=0.0)
        rig.place(pose)
        forces = rig.update()
        np.testing.assert_allclose(rig.contact._gap.numpy(), -0.0016, atol=2e-8)
        np.testing.assert_allclose(rig.contact._normal.numpy(), np.tile([0, 0, 1.0], (5, 1)), atol=1e-5)
        np.testing.assert_allclose(forces, (STIFFNESS / 5) * 0.0016, rtol=2e-4)
        # A centre exactly on the surface takes the triangle's normal.
        rig.place(_pose(0.0, tilt=0.0))
        rig.update()
        np.testing.assert_allclose(rig.contact._normal.numpy(), np.tile([0, 0, 1.0], (5, 1)), atol=1e-5)
        np.testing.assert_allclose(rig.contact._gap.numpy(), -RADIUS, atol=2e-8)

    def test_over_an_edge_the_force_acts_between_the_surface_point_and_the_centre(self):
        # The capsule lies along the border x = 0 of the patch, 0.5 mm outside it and 0.5 mm above
        # its plane: every nearest point is on that edge, and the direction is that of the offset.
        rig = _Rig()
        rotation = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 0.5 * np.pi)
        pose = np.array([-0.0005, 0.005, 0.0005, *rotation], dtype=np.float32)
        rig.place(pose)
        centres = _centres(pose, rig.local)
        np.testing.assert_allclose(centres[:, 1], 0.005 + np.array([-2, -1, 0, 1, 2]) * 0.001, atol=1e-9)
        forces = rig.update()
        direction = np.array([-1.0, 0.0, 1.0]) / np.sqrt(2.0)
        gap = 0.0005 * np.sqrt(2.0) - RADIUS
        np.testing.assert_allclose(rig.contact._normal.numpy(), np.tile(direction, (5, 1)), atol=1e-5)
        np.testing.assert_allclose(rig.contact._gap.numpy(), gap, atol=2e-8)
        np.testing.assert_allclose(forces, -(STIFFNESS / 5) * gap, rtol=2e-4)
        # The two vertices of the edge share each force by the position along the edge.
        nodal = np.zeros((4, 3))
        for centre, force in zip(centres, forces, strict=True):
            nodal[0] -= force * (1 - centre[1] / 0.01) * direction
            nodal[3] -= force * (centre[1] / 0.01) * direction
        np.testing.assert_allclose(rig.particle_f.numpy(), nodal, rtol=2e-4, atol=1e-8)
        np.testing.assert_allclose(rig.contact.wrench.numpy()[1, :3], forces.sum() * direction, rtol=1e-4, atol=1e-9)

    def test_two_contacts_share_no_state_and_reaction_can_be_left_out(self):
        first, second = _Rig(), _Rig(reaction=False)
        for rig in (first, second):
            rig.place(_pose(0.00085))
        forces = first.update()
        self.assertGreater(forces.max(), 0)
        np.testing.assert_array_equal(second.contact.force.numpy(), 0)
        np.testing.assert_array_equal(second.contact.wrench.numpy(), 0)
        for own, other in zip(first.contact.state, second.contact.state, strict=True):
            self.assertNotEqual(own.ptr, other.ptr)
        second.update()
        np.testing.assert_array_equal(second.contact.force.numpy(), forces)
        np.testing.assert_array_equal(first.contact.force.numpy(), forces)
        body_f = wp.zeros(2, dtype=wp.spatial_vector, device=DEVICE)
        second.contact.add_forces(body_f)
        np.testing.assert_array_equal(body_f.numpy(), 0)
        self.assertGreater(np.abs(second.contact.wrench.numpy()).max(), 0)

    def test_invalid_arguments_are_refused(self):
        rig = _Rig()
        common = dict(node_stiffness=STIFFNESS, shear_frequency=FREQUENCY, step_seconds=rig.step_seconds, reach=0.02)
        mapping = wp.array(rig.map, dtype=int, device=DEVICE)
        for bodies, local, radii, changed in (
            ([2], rig.local[:1], rig.radii[:1], {}),
            ([1], rig.local[:2], rig.radii[:1], {}),
            ([1], rig.local[:1], [0.0], {}),
            ([1], rig.local[:1], rig.radii[:1], {"node_stiffness": 0.0}),
            ([1], rig.local[:1], rig.radii[:1], {"shear_frequency": np.nan}),
            ([1], rig.local[:1], rig.radii[:1], {"step_seconds": wp.zeros(1, dtype=float, device=DEVICE)}),
        ):
            with self.assertRaises(ValueError):
                ToolTissueContact(
                    rig.model, rig.mesh, rig.triangles, mapping, bodies, local, radii, **{**common, **changed}
                )


def _worker_scene():
    """The worker fixture with the material values the contact derives its parameters from."""
    scene, config = scene_fixture()
    scene.args = SimpleNamespace(young_modulus=300.0, shaft_friction=0.2, poisson_ratio=0.35, density=1000.0)
    return scene, config


def _press(scene, body=1, depth=0.0004):
    """Lay the capsule of a tool body (1 mm radius, along y) on the top of the slab, ``depth`` inside it."""
    poses = scene.state_0.body_q.numpy()
    poses[body] = [0.01, 0.0, 0.001 - depth, 0, 0, 0, 1]
    scene.state_0.body_q.assign(poses)


class WorkerThreadToolTissueTests(unittest.TestCase):
    def test_contact_is_off_by_default_and_the_worker_is_then_as_before(self):
        scene, config = _worker_scene()
        worker = WorkerThread(scene, config)
        self.assertIsNone(worker.tool_tissue)
        self.assertIsNone(worker.tool_tissue_radii)
        expected = (*worker.solver.state, *worker.tissue.state, worker.attachment.wrench, worker._nodal_force)
        self.assertEqual([array.ptr for array in worker.state], [array.ptr for array in expected])
        # The fixture of the other worker tests has no Poisson ratio or density: none is read.
        plain_scene, plain_config = scene_fixture()
        self.assertFalse(hasattr(plain_scene.args, "density"))
        self.assertIsNone(WorkerThread(plain_scene, plain_config).tool_tissue)
        with self.assertRaises(AttributeError):
            WorkerThread(plain_scene, plain_config, tool_tissue=True)

    def test_parameters_follow_from_the_scene(self):
        scene, config = _worker_scene()
        worker = WorkerThread(scene, config, tool_tissue=True)
        contact = worker.tool_tissue
        # One tissue node: a quarter of E h, the stiffness that relaxes the thread's tissue forces.
        self.assertAlmostEqual(contact.node_stiffness, 0.25 * 300.0 * worker.tissue_edge_m, delta=1e-12)
        self.assertEqual(contact.node_stiffness, 0.25 * worker.contact_stiffness)
        # Lowest shear mode of the 5 mm slab on its base: (pi / 2) sqrt(G / rho) / H.
        shear = 300.0 / (2 * 1.35)
        self.assertAlmostEqual(contact.shear_frequency, 0.5 * np.pi * np.sqrt(shear / 1000.0) / 0.005, delta=1e-9)
        # The search reaches as far as the pad is thick.
        self.assertAlmostEqual(contact.reach, 0.005, delta=1e-12)
        # The two tool capsules (20 mm, radius 1 mm) and not the needle: 21 spheres each.
        self.assertEqual(contact.sample_count, 42)
        self.assertEqual(sorted(set(contact._sample_body.numpy().tolist())), [1, 2])
        self.assertEqual(worker.tool_tissue_radii, [0.001, 0.001])
        np.testing.assert_allclose(contact._sample_radius.numpy(), 0.001)
        self.assertIs(contact._step_seconds, worker.solver.step_time.seconds_f64)
        mesh, triangles, point_particle = worker.tissue.surface
        self.assertIs(contact.mesh, mesh)
        self.assertIs(contact._triangles, triangles)
        self.assertIs(contact._point_particle, point_particle)
        # Radii of the drawn tool replace the capsule radii.
        wide = WorkerThread(scene, config, tool_tissue=True, tool_tissue_radii=[0.002, 0.004])
        self.assertEqual(wide.tool_tissue_radii, [0.002, 0.004])
        radii = wide.tool_tissue._sample_radius.numpy()
        bodies = wide.tool_tissue._sample_body.numpy()
        np.testing.assert_allclose(radii[bodies == 1], 0.002)
        np.testing.assert_allclose(radii[bodies == 2], 0.004)
        # 18 mm and 14 mm from cap to cap of the centres: 10 and 5 spheres.
        self.assertEqual(np.bincount(bodies).tolist(), [0, 10, 5])
        with self.assertRaises(ValueError):
            WorkerThread(scene, config, tool_tissue=True, tool_tissue_radii=[0.002])

    def test_accepted_step_updates_the_contact_and_its_loads_join_the_threads(self):
        scene, config = _worker_scene()
        _press(scene)
        worker = WorkerThread(scene, config, tool_tissue=True)
        contact = worker.tool_tissue
        self.assertEqual(len(worker.state), len(worker.solver.state) + len(worker.tissue.state) + 4)
        self.assertEqual([array.ptr for array in worker.state[-2:]], [contact.force.ptr, contact.wrench.ptr])
        np.testing.assert_array_equal(contact.force.numpy(), 0)
        dt = 1 / 240
        worker.advance(dt)
        forces = contact.force.numpy()
        # The 21 spheres of body 1 are 0.4 mm inside the top of the slab; body 2 is far above.
        bodies = contact._sample_body.numpy()
        fraction = min(1.0, 0.5 * (contact.shear_frequency * dt) ** 2)
        np.testing.assert_allclose(forces[bodies == 1], fraction * (contact.node_stiffness / 21) * 0.0004, rtol=1e-3)
        np.testing.assert_array_equal(forces[bodies == 2], 0)
        self.assertEqual(contact._count.numpy()[0], 21)
        # The stored nodal forces are the thread's own plus the contact's; the thread's reaction
        # that the diagnostics report is still the thread's own.
        own = worker.tissue.applied_force.numpy()
        nodal = worker._nodal_force.numpy()
        np.testing.assert_allclose((nodal - own).sum(axis=0), [0, 0, -forces.sum()], rtol=1e-5, atol=1e-9)
        self.assertAlmostEqual(
            worker.maximum_tissue_reaction(), float(np.linalg.norm(own, axis=-1).max(initial=0)), delta=1e-12
        )
        # Tool reaction: up, on body 1 only, beside the thread's capsule reaction.
        wrench = contact.wrench.numpy()
        self.assertAlmostEqual(wrench[1, 2], forces.sum(), delta=1e-9)
        np.testing.assert_array_equal(wrench[[0, 2]], 0)
        scene.state_0.body_f.zero_()
        external = wp.zeros(scene.model.particle_count, dtype=wp.vec3, device=DEVICE)
        worker.add_forces(scene.state_0.body_f, external)
        expected = worker.rigid_contact.wrench.numpy() + wrench
        expected[scene.needle_body] += worker.attachment.wrench.numpy()[0]
        np.testing.assert_allclose(scene.state_0.body_f.numpy(), expected, rtol=1e-6, atol=1e-12)
        np.testing.assert_array_equal(external.numpy(), nodal)
        # A rejected attempt adds the loads again and advances nothing.
        saved = [array.numpy().copy() for array in worker.state]
        for _ in range(2):
            scene.state_0.body_f.zero_()
            worker.add_forces(scene.state_0.body_f, external)
        for value, array in zip(saved, worker.state, strict=True):
            np.testing.assert_array_equal(value, array.numpy())
        # The tool leaves: after enough accepted steps nothing is left on either side.
        poses = scene.state_0.body_q.numpy()
        poses[1, 2] = 0.003
        scene.state_0.body_q.assign(poses)
        for _ in range(3):
            worker.advance(dt)
        np.testing.assert_array_equal(contact.force.numpy(), 0)
        np.testing.assert_array_equal(contact.wrench.numpy(), 0)
        np.testing.assert_array_equal(worker._nodal_force.numpy(), worker.tissue.applied_force.numpy())

    def test_two_workers_share_no_contact_state(self):
        scene, config = _worker_scene()
        _press(scene)
        first, second = WorkerThread(scene, config, tool_tissue=True), WorkerThread(scene, config, tool_tissue=True)
        first.advance(1 / 240)
        self.assertGreater(first.tool_tissue.force.numpy().max(), 0)
        np.testing.assert_array_equal(second.tool_tissue.force.numpy(), 0)
        np.testing.assert_array_equal(second.tool_tissue.wrench.numpy(), 0)
        np.testing.assert_array_equal(second._nodal_force.numpy(), 0)
        for own, other in zip(first.tool_tissue.state, second.tool_tissue.state, strict=True):
            self.assertNotEqual(own.ptr, other.ptr)
        # A new worker, as a reset of the scene makes one, starts without contact forces.
        fresh = WorkerThread(scene, config, tool_tissue=True)
        np.testing.assert_array_equal(fresh.tool_tissue.force.numpy(), 0)
        np.testing.assert_array_equal(fresh.tool_tissue.wrench.numpy(), 0)

    def test_contact_off_gives_the_bits_of_a_worker_without_the_option(self):
        scene, config = _worker_scene()
        _press(scene)
        plain, off = WorkerThread(scene, config), WorkerThread(scene, config, tool_tissue=False)
        for _ in range(3):
            plain.advance(1 / 240)
            off.advance(1 / 240)

        def same(actual, expected):
            # Two workers on a CUDA device differ in the last bits (the order of atomic sums).
            if wp.get_device(DEVICE).is_cuda:
                np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-10)
            else:
                np.testing.assert_array_equal(actual, expected)

        for expected, actual in zip(plain.state, off.state, strict=True):
            same(actual.numpy(), expected.numpy())
        forces = []
        for worker in (plain, off):
            scene.state_0.body_f.zero_()
            external = wp.zeros(scene.model.particle_count, dtype=wp.vec3, device=DEVICE)
            worker.add_forces(scene.state_0.body_f, external)
            forces.append((scene.state_0.body_f.numpy().copy(), external.numpy().copy()))
        for expected, actual in zip(*forces, strict=True):
            same(actual, expected)
        same(off.maximum_tissue_reaction(), plain.maximum_tissue_reaction())

    def test_start_up_capture_leaves_the_contact_state_as_it_found_it(self):
        if not wp.get_device(DEVICE).is_cuda or not wp.is_conditional_graph_supported():
            self.skipTest("The thread's start-up capture needs CUDA with conditional graphs")
        scene, config = _worker_scene()
        _press(scene)
        states = (scene.state_0, scene.model.state())
        states[1].body_q.assign(states[0].body_q.numpy())
        states[1].particle_q.assign(states[0].particle_q.numpy())
        reference = WorkerThread(scene, config, capture=False, tool_tissue=True)
        worker = WorkerThread(scene, config, tool_tissue=True)
        dt = 1 / 240
        for _ in range(2):
            reference.advance(dt)
            worker.advance(dt)
        worker.invalidate_step_graphs()
        self.assertGreater(worker.tool_tissue.force.numpy().max(), 0)
        before = [(str(array.dtype), array.shape, array.numpy().tobytes()) for array in worker.state]
        steps, launches = worker.accepted_steps, worker.graph_launches
        self.assertTrue(worker.capture_graphs(states, dt))
        after = [(str(array.dtype), array.shape, array.numpy().tobytes()) for array in worker.state]
        # Every state array, the contact's forces and stored wrench among them, has its bits back:
        # the discarded steps of the capture pressed on with the tool inside the tissue.
        self.assertEqual(after, before)
        self.assertEqual((worker.accepted_steps, worker.graph_launches, len(worker._step_graphs)), (steps, launches, 2))
        # The graphs hold the contact's update: replayed steps on either buffer equal the eager worker.
        for step in range(4):
            scene.state_0 = states[step % 2]
            reference.advance(dt)
            worker.advance(dt)
            np.testing.assert_allclose(
                worker.tool_tissue.force.numpy(), reference.tool_tissue.force.numpy(), rtol=1e-6, atol=1e-12
            )
            np.testing.assert_allclose(
                worker.tool_tissue.wrench.numpy(), reference.tool_tissue.wrench.numpy(), rtol=1e-5, atol=1e-10
            )
            np.testing.assert_allclose(
                worker._nodal_force.numpy(), reference._nodal_force.numpy(), rtol=1e-5, atol=1e-10
            )
        self.assertEqual(worker.graph_captures - len(worker._step_graphs), 1)
        self.assertEqual(worker.graph_launches, launches + 4)
        self.assertGreater(worker.tool_tissue.force.numpy().max(), 0)


class GraspSceneToolTissueTests(unittest.TestCase):
    """The switch of the grasp scene, on the small real mesh of the scene tests."""

    ARGUMENTS = ("--tool-tissue-contact",)

    def test_scene_has_no_contact_unless_its_option_is_given(self):
        from test_grasp_scene import _geometry, _scene

        with _scene() as (scene, config):
            self.assertFalse(scene.args.tool_tissue_contact)
            self.assertIsNone(scene.worker_thread.tool_tissue)
            self.assertNotIn("tool_tissue_contact", scene.report())
            self.assertTrue(_geometry(scene, config)["scope"].endswith("lagged rigid/tissue reactions"))
            self.assertFalse(hasattr(scene, "_capsule_contact_radii"))

    def test_scene_option_builds_the_contact_of_the_drawn_tool_and_a_reset_clears_its_forces(self):
        from test_grasp_scene import _geometry, _reset_scene, _scene

        with _scene(example_arguments=self.ARGUMENTS) as (scene, config):
            self.assertTrue(scene.args.tool_tissue_contact)
            worker = scene.worker_thread
            contact = worker.tool_tissue
            self.assertIsNotNone(contact)
            # The capsule set of the tools, as their models publish it: the contact has no radii of its own.
            capsules = np.concatenate([tool.capsule_radii for tool in scene.graspers])
            np.testing.assert_array_equal(worker.tool_tissue_radii, capsules)
            sample_bodies, _, sample_radii = capsule_samples(
                np.concatenate([np.asarray(tool.capsule_bodies) for tool in scene.graspers]),
                np.concatenate([np.asarray(tool.capsule_endpoints, dtype=float) for tool in scene.graspers]),
                capsules,
            )
            np.testing.assert_array_equal(contact._sample_body.numpy(), sample_bodies)
            np.testing.assert_allclose(contact._sample_radius.numpy(), sample_radii, rtol=0, atol=1e-10)
            self.assertEqual(contact.sample_count, len(sample_bodies))
            # The parameters of the scene's material and mesh.
            args = scene.args
            self.assertEqual(contact.node_stiffness, 0.25 * worker.contact_stiffness)
            depth = float(np.ptp(scene.selection.vertices[:, 2]))
            shear = args.young_modulus / (2 * (1 + args.poisson_ratio))
            self.assertAlmostEqual(
                contact.shear_frequency, 0.5 * np.pi * np.sqrt(shear / args.density) / depth, delta=1e-9
            )
            self.assertAlmostEqual(contact.reach, float(np.ptp(scene.reference.vertices[:, 2])), delta=1e-12)
            report = scene.report()["tool_tissue_contact"]
            self.assertEqual(report["samples"], len(sample_bodies))
            self.assertEqual(report["capsule_contact_radii_m"], worker.tool_tissue_radii)
            self.assertEqual(report["node_stiffness_N_m"], contact.node_stiffness)
            self.assertIn("tool capsules with the tissue surface", _geometry(scene, config)["scope"])
            self.assertEqual(len(worker.state), len(worker.solver.state) + len(worker.tissue.state) + 4)
            # Frames run with it and the worker advances once per accepted interval.
            for _ in range(2):
                scene.step()
                self.assertTrue(scene.running, scene.failure_reason)
            self.assertEqual(worker.accepted_steps, scene.accepted_physics_steps)
            self.assertTrue(np.isfinite(contact.force.numpy()).all())
            scene._check_tissue_integrity()
            # A reset builds a new worker: forces that the old one held are gone.
            contact.force.fill_(0.01)
            contact.wrench.fill_(wp.spatial_vector(0.0, 0.0, 0.01, 0.0, 0.0, 0.0))
            _reset_scene(scene, config, rebuild=True)
            fresh = scene.worker_thread.tool_tissue
            self.assertIsNot(fresh, contact)
            np.testing.assert_array_equal(fresh.force.numpy(), 0)
            np.testing.assert_array_equal(fresh.wrench.numpy(), 0)
            np.testing.assert_array_equal(scene.worker_thread._nodal_force.numpy(), 0)
            self.assertEqual(scene.worker_thread.tool_tissue_radii, worker.tool_tissue_radii)
            # The scene keeps no radii beside those of the tools.
            self.assertFalse(hasattr(scene, "_capsule_contact_radii"))

    def test_start_up_call_leaves_the_contact_as_it_found_it_and_frames_capture_and_load_nothing(self):
        device = wp.get_device(DEVICE)
        if not device.is_cuda or not wp.is_conditional_graph_supported():
            self.skipTest("The start-up call captures CUDA graphs")
        from test_grasp_scene import (
            _NOTHING_LATE,
            _counted_captures,
            _late,
            _mark_startup,
            _observe,
            _recorded_startup,
            _scene,
        )

        with (
            _recorded_startup() as records,
            _scene(rigid_backend="mujoco-warp", example_arguments=self.ARGUMENTS) as (scene, _),
        ):
            self.assertEqual(len(records), 1)
            (arrays, host, _), (arrays_after, host_after, _), returned = records[0]
            self.assertTrue(returned)
            worker = scene.worker_thread
            contact = worker.tool_tissue
            # The record holds every array of the worker's state, the contact's two among them.
            self.assertEqual(sum(name.startswith("thread.") for name in arrays), len(worker.state) + 1, sorted(arrays))
            self.assertEqual([array.ptr for array in worker.state[-2:]], [contact.force.ptr, contact.wrench.ptr])
            self.assertEqual(sorted(arrays_after), sorted(arrays))
            for name, bits in arrays.items():
                self.assertEqual(arrays_after[name], bits, name)
            self.assertEqual(host_after, host)
            np.testing.assert_array_equal(contact.force.numpy(), 0)
            np.testing.assert_array_equal(contact.wrench.numpy(), 0)
            # The thread's two graphs hold the contact's update; frames replay them and capture
            # nothing, and the contact's kernel module was loaded by start-up.
            self.assertEqual((worker.graph_captures, worker.graph_launches, len(worker._step_graphs)), (2, 0, 2))
            # As a worker: the first observation, then the mark that ends start-up.
            _observe(scene, 0)
            _mark_startup(scene)
            with _counted_captures() as begun:
                for _ in range(3):
                    scene.step()
                    self.assertTrue(scene.running, scene.failure_reason)
                self.assertEqual(begun, [])
            self.assertEqual((worker.graph_captures, worker.graph_launches), (2, worker.accepted_steps))
            self.assertEqual(worker.accepted_steps, scene.accepted_physics_steps)
            scene.refresh_capture_census()
            report = scene.report()
            census = report["capture_census"]
            self.assertEqual(census["status"], "checked")
            self.assertEqual(_late(report), _NOTHING_LATE, census)
            self.assertEqual((census["late_sites"], census["late_modules"]), ({}, {}))
            self.assertEqual(census["startup_unlaunched_graphs"], 0)


class ExampleOptionTests(unittest.TestCase):
    def test_example_option_reaches_the_scene_and_the_scene_keeps_the_contact_off_by_default(self):
        from newton_solver_suturebot.examples.grasp import example_fem_needle_grasp as grasp_example
        from newton_solver_suturebot.examples.grasp._grasp_scene import GraspScene
        from newton_solver_suturebot.examples.grasp.parallel import SutureSceneConfig
        from newton_solver_suturebot.examples.needle import example_fem_needle_insertion as insertion

        parser = grasp_example.Example.create_parser()
        self.assertIs(parser.parse_args([]).tool_tissue_contact, grasp_example.TOOL_TISSUE_CONTACT)
        off = grasp_example.Example._coupling(parser.parse_args(["--no-tool-tissue-contact"]))
        self.assertNotIn("--tool-tissue-contact", off)
        # The first-version law is a law of the scene: its argument without a value, after the
        # arguments of the example. The example itself offers the swept law or none.
        on = (*off, "--tool-tissue-contact")
        # The example's default is the swept law: an argument namespace without the setting takes
        # it, as the parser does.
        self.assertEqual(grasp_example.TOOL_TISSUE_CONTACT, "swept")
        default = grasp_example.Example._coupling(SimpleNamespace())
        self.assertEqual(default, (*off, "--tool-tissue-contact", "swept"))
        self.assertEqual(default, grasp_example.Example._coupling(parser.parse_args([])))
        # The grasp scene parses the argument; without it the scene has no contact.
        scene_parser = GraspScene.create_parser()
        self.assertTrue(scene_parser.parse_args(list(on)).tool_tissue_contact)
        self.assertFalse(scene_parser.parse_args(list(off)).tool_tissue_contact)
        self.assertFalse(scene_parser.parse_args([]).tool_tissue_contact)
        # The scene configuration of Python callers passes no argument, and the free-needle
        # example, which has no tools, does not know the option.
        self.assertEqual(SutureSceneConfig(device="cpu").example_arguments, ())
        self.assertFalse(hasattr(insertion.Example.create_parser().parse_args([]), "tool_tissue_contact"))


if __name__ == "__main__":
    unittest.main()
