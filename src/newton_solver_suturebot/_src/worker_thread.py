# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Physical worker-owned thread with lagged needle, tool and tissue reactions.

The caller advances this object only after accepting a rigid/FEM interval and
adds its stored loads before the next rigid/FEM trial. Rejected trials never
advance the thread. No renderer owns this state and no scheduled puncture track
or eyelet is introduced. The far end remains free.
"""

import math
from collections import OrderedDict

import numpy as np
import warp as wp

from .solvers.fem.differentiation import active_tape
from .solvers.thread import SolverThread, ThreadAttachment, ThreadModel, ThreadTissueCoupling
from .solvers.thread.kernels import _centerline_point, _shape
from .thread_initialization import loose_thread_coordinates
from .thread_rigid_contact import ThreadRigidCapsuleContact


@wp.kernel(enable_backward=False)
def _add_nodal(source: wp.array[wp.vec3], target: wp.array[wp.vec3]):
    node = wp.tid()
    target[node] += source[node]


@wp.kernel(enable_backward=False)
def _sample(q: wp.array[wp.float64], lengths: wp.array[wp.float64], samples: int, points: wp.array[wp.vec3]):
    index = wp.tid()
    element = wp.min(index // samples, lengths.shape[0] - 1)
    xi = wp.float64(index - element * samples) / wp.float64(samples)
    points[index] = wp.vec3(_centerline_point(q, element, _shape(xi, lengths[element])))


class WorkerThread:
    """A single scene's physical ANCF thread; all lengths are metres.

    ``scene`` supplies model, state_0 (accepted state), geometry, needle_body,
    reference, selection, args and grasper assemblies. ``config`` supplies
    thread_elements, thread_length_m, thread_radius_m and thread_modulus_pa.
    Optional thread_iterations defaults to2, matching the source stitch demo.

    Attachment uses the real free needle's mass and inertia with zero assumed
    external drive support. Its lagged compliant head can differ from the
    needle tail during force convergence; diagnostics report that discrepancy.
    This is not a welded gripper/needle constraint or a prescribed needle pose.
    """

    samples_per_element = 2
    density = 1140.0
    gravity = 9.81
    mass_damping = 2.0

    def __init__(self, scene, config, *, capture=True):
        self.scene, self.config = scene, config
        model, geometry = scene.model, scene.geometry
        self.device = model.device
        self.radius = float(config.thread_radius_m)
        self.length = float(config.thread_length_m)
        self.elements = config.thread_elements
        self.young_modulus = float(config.thread_modulus_pa)
        self.iteration_limit = int(getattr(config, "thread_iterations", 2))
        if isinstance(self.elements, bool) or not isinstance(self.elements, (int, np.integer)) or self.elements < 2:
            raise ValueError("thread_elements must be an integer at least two")
        if (
            not np.isfinite([self.radius, self.length, self.young_modulus]).all()
            or min(self.radius, self.length, self.young_modulus) <= 0
        ):
            raise ValueError("Thread dimensions and modulus must be finite and positive")
        if self.iteration_limit < 1:
            raise ValueError("thread_iterations must be positive")
        self.tail = np.asarray(geometry.tail, dtype=float)
        tangent = np.asarray(geometry.centerline[-1] - geometry.centerline[-2], dtype=float)
        if self.tail.shape != (3,) or not np.isfinite(self.tail).all() or np.linalg.norm(tangent) <= 1e-12:
            raise ValueError("Needle tail and tangent must be finite and nondegenerate")
        self.tangent = tangent / np.linalg.norm(tangent)
        full = np.asarray(scene.reference.vertices, dtype=float)
        global_ids = np.asarray(scene.reference.global_vertex_indices)
        active_global = np.asarray(scene.selection.global_vertex_indices)
        ids = np.searchsorted(global_ids, active_global)
        if np.any(ids >= len(global_ids)) or not np.array_equal(global_ids[ids], active_global):
            raise ValueError("Every active tissue vertex must have an exact full-pad global identity")
        self.active_reference_indices = ids
        current = scene.state_0.particle_q.numpy()
        if current.shape != (len(ids), 3):
            raise ValueError("Worker thread needs one accepted particle per active FEM vertex")
        pad = full.copy()
        pad[ids] = current
        pose = scene.state_0.body_q.numpy()[scene.needle_body]
        transform = wp.transform(wp.vec3(pose[:3]), wp.quat(pose[3:]))
        head_point = np.asarray(wp.transform_point(transform, wp.vec3(self.tail)), dtype=float)
        head_tangent = np.asarray(wp.quat_rotate(wp.quat(pose[3:]), wp.vec3(self.tangent)), dtype=float)
        head_tangent /= np.linalg.norm(head_tangent)
        initial = loose_thread_coordinates(
            np.r_[head_point, head_tangent], pad, length=self.length, elements=self.elements, radius=self.radius
        )
        straight = np.zeros((self.elements + 1, 3))
        straight[:, 0] = np.linspace(0, self.length, self.elements + 1)
        self.thread_model = ThreadModel.create(
            straight, young_modulus=self.young_modulus, radius=self.radius, density=self.density, device=self.device
        )
        tetrahedra = np.asarray(scene.selection.indices, dtype=np.int32)
        edges = (
            np.asarray(scene.selection.vertices)[tetrahedra[:, 1:]]
            - np.asarray(scene.selection.vertices)[tetrahedra[:, :1]]
        )
        self.tissue_edge_m = float(np.linalg.norm(edges, axis=-1).mean())
        self.contact_stiffness = float(scene.args.young_modulus) * self.tissue_edge_m
        self.friction = float(scene.args.shaft_friction)
        if not np.isfinite(self.contact_stiffness) or self.contact_stiffness <= 0:
            raise ValueError("Thread contact needs a finite positive tissue modulus/edge scale")
        element_length = self.length / self.elements
        node_weight = self.density * math.pi * self.radius**2 * element_length * self.gravity
        self.solver = SolverThread(
            model,
            self.thread_model,
            gravity=(0, 0, -self.gravity),
            mass_damping=self.mass_damping,
            max_iterations=self.iteration_limit,
            tolerance=0.1 * node_weight,
            contact_distance=self.radius,
            contact_stiffness=self.contact_stiffness,
            contact_smoothing=0.1 * self.radius,
            friction=self.friction,
            self_contact=True,
            max_eyelets=0,
        )
        self.solver.q.assign(initial.reshape(-1))
        self.solver.boundary_q.assign(self.solver.q)
        self.attachment = ThreadAttachment(
            self.solver, model, [scene.needle_body], self.tail, self.tangent, lagged=True
        )
        # Deliberately no support from the old needle servo: the gripper acts
        # on this independent free body through actual MuJoCo contact alone.
        self.tissue = ThreadTissueCoupling(
            self.solver,
            full,
            scene.reference.exterior_surface_triangles,
            tetrahedra,
            ids,
            tissue_stiffness=0.25 * self.contact_stiffness,
            contact_margin=0.005,
            max_tracks=1,
        )
        centerline = np.asarray(geometry.centerline, dtype=float)
        bodies = [scene.needle_body] * (len(centerline) - 1)
        endpoints = [np.stack((a, b)) for a, b in zip(centerline[:-1], centerline[1:], strict=True)]
        radii = [float(geometry.thickness)] * len(bodies)
        for grasper in scene.graspers:
            bodies.extend(grasper.capsule_bodies)
            endpoints.extend(grasper.capsule_endpoints)
            radii.extend(grasper.capsule_radii)
        self.rigid_contact = ThreadRigidCapsuleContact(
            self.solver,
            model,
            bodies,
            endpoints,
            radii,
            thread_radius=self.radius,
            friction=self.friction,
            skip_by_body={scene.needle_body: 2},
        )
        self._nodal_force = wp.zeros(model.particle_count, dtype=wp.vec3, device=self.device)
        self._positions = wp.empty(self.elements * self.samples_per_element + 1, dtype=wp.vec3, device=self.device)
        self.accepted_steps, self.physical_time_s = 0, 0.0
        self.capture = bool(capture)
        self._step_graphs = OrderedDict()
        self._graph_warmed = False
        self.graph_captures = self.graph_launches = 0

    @property
    def state(self):
        """Mutable device arrays for accepted-state snapshots/capture warm-up."""
        return (*self.solver.state, *self.tissue.state, self.attachment.wrench, self._nodal_force)

    def add_forces(self, body_f, external_nodal_forces):
        """Add stored reactions; does not advance thread or friction history."""
        self.attachment.add_forces(body_f)
        self.rigid_contact.add_forces(body_f)
        wp.launch(
            _add_nodal,
            self.scene.model.particle_count,
            inputs=[self._nodal_force, external_nodal_forces],
            device=self.device,
        )

    def invalidate_step_graphs(self):
        """Call after replacing worker storage or changing contact/material settings.

        Accepted scene buffers, timestep and Newton stopping settings are
        tracked automatically. In-place state updates need no invalidation.
        The next real accepted step warms resources without a discarded solve.
        """
        self._step_graphs.clear()
        self._graph_warmed = False

    def _advance_device(self, accepted, dt):
        """Device-only accepted-step operations, shared by eager and captured paths."""
        self.attachment.attach(accepted.body_q, dt)
        self.rigid_contact.update(accepted.body_q, accepted.body_qd, dt)
        self.tissue.update(accepted.particle_q)
        self.solver.step(None, None, None, None, dt)
        self.attachment.update()
        self.rigid_contact.react(accepted.body_q)
        self._nodal_force.zero_()
        self.tissue.add_forces(self._nodal_force)

    def advance(self, dt):
        """Advance once after the caller accepts a rigid/FEM interval of dt.

        CUDA replay keeps the conditional thread Newton loop on the device;
        eager execution otherwise polls its active flag from Python. Capture
        records operations without an extra physical step or friction update.
        """
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("Accepted thread interval must be finite and positive")
        accepted = self.scene.state_0
        arrays = (accepted.body_q, accepted.body_qd, accepted.particle_q)
        if (
            not self.capture
            or not self.device.is_cuda
            or not wp.is_conditional_graph_supported()
            or active_tape() is not None
            or getattr(self.scene.model, "requires_grad", False)
            or any(array.requires_grad for array in (*arrays, *self.solver.state))
        ):
            self._advance_device(accepted, dt)
        elif not self._graph_warmed:
            self._advance_device(accepted, dt)
            self._graph_warmed = True
        else:
            key = (
                float(dt),
                self.solver.max_iterations,
                self.solver.tolerance,
                tuple((array.ptr, array.shape, array.strides, array.dtype) for array in arrays),
            )
            entry = self._step_graphs.get(key)
            if entry is None:
                with wp.ScopedCapture(device=self.device) as captured:
                    self._advance_device(accepted, dt)
                entry = (captured.graph, arrays)
                self._step_graphs[key] = entry
                if len(self._step_graphs) > 16:
                    self._step_graphs.popitem(last=False)
                self.graph_captures += 1
            else:
                self._step_graphs.move_to_end(key)
            # Keep the contact object's lightweight Python metadata consistent
            # when alternating previously captured accepted-state buffers.
            self.rigid_contact._body_q, self.rigid_contact._dt = accepted.body_q, float(dt)
            wp.capture_launch(entry[0])
            self.graph_launches += 1
        self.accepted_steps += 1
        self.physical_time_s += dt

    def positions(self):
        """Actual cubic centreline samples [2*elements+1,3], copied to host."""
        wp.launch(
            _sample,
            len(self._positions),
            inputs=[self.solver.q, self.thread_model.lengths, self.samples_per_element, self._positions],
            device=self.device,
        )
        result = self._positions.numpy().copy()
        if not np.isfinite(result).all():
            raise RuntimeError("Physical worker thread has nonfinite centreline coordinates")
        return result

    def diagnostics(self, *, needle_pose=None):
        q, qd = self.solver.q.numpy().reshape(-1, 6), self.solver.qd.numpy().reshape(-1, 6)
        pose = self.scene.state_0.body_q.numpy()[self.scene.needle_body] if needle_pose is None else needle_pose
        actual_tail = np.asarray(
            wp.transform_point(wp.transform(wp.vec3(pose[:3]), wp.quat(pose[3:])), wp.vec3(self.tail))
        )
        values = {
            "scope": "physical worker ANCF thread; lagged relaxed rigid/FEM forces; no scheduled eyelets",
            "physical_time_s": self.physical_time_s,
            "accepted_steps": self.accepted_steps,
            "graph_captures": self.graph_captures,
            "graph_launches": self.graph_launches,
            "radius_m": self.radius,
            "length_m": self.length,
            "elements": int(self.elements),
            "young_modulus_pa": self.young_modulus,
            "density_kg_m3": self.density,
            "newton_iteration_limit": self.iteration_limit,
            "last_newton_updates": self.solver.iterations.numpy().tolist(),
            "contact_stiffness_N_m": self.contact_stiffness,
            "contact_friction": self.friction,
            "head_to_needle_tail_error_m": float(np.linalg.norm(q[0, :3] - actual_tail)),
            "maximum_speed_m_s": float(np.linalg.norm(qd[:, :3], axis=-1).max()),
            "maximum_chord_stretch_ratio": float(
                np.max(np.linalg.norm(np.diff(q[:, :3], axis=0), axis=-1) / (self.length / self.elements))
            ),
            "needle_attachment_wrench_N_Nm": self.attachment.wrench.numpy()[0].tolist(),
            "maximum_tissue_reaction_N": float(np.linalg.norm(self._nodal_force.numpy(), axis=-1).max(initial=0)),
            "needle_drive_support_assumed": False,
            "eyelets": 0,
            "gripper_capsules": sum(len(g.capsule_bodies) for g in self.scene.graspers),
            "rigid_contact": self.rigid_contact.diagnostics(),
        }
        if not np.isfinite(q).all() or not np.isfinite(qd).all():
            raise RuntimeError("Nonfinite physical thread coordinates or velocities")
        return values
