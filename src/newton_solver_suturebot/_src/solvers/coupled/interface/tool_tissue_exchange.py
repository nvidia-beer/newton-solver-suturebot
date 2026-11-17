# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""What the tools of a scene and its tissue exchange: one object, built from one description of the scene.

A scene with tools describes itself once, as a :class:`ToolTissueSetup` (plain data), and builds
one :class:`ToolTissueExchange` from it. The exchange builds the law that the setup names (the
first-version law, :class:`ToolTissueContact`, or the contact on the sweep of the capsules,
:class:`SweptToolTissueContact`) and is the only code that constructs one. Its owner issues
three calls, in this order in every interval:

- :meth:`ToolTissueExchange.add_body_loads` before the rigid interval of every attempt;
- :meth:`ToolTissueExchange.add_nodal_loads` before the tissue step of every attempt;
- :meth:`ToolTissueExchange.update` at an accepted state.

All three only launch kernels, so the owner captures them in its CUDA graphs: the worker
thread of the grasp scene in the graph of its accepted step, the tool/tissue example in the
graphs of its own interval. Which graph issues them and who refits the surface mesh before
the update is the owner's; the launches are the same list in every scene.

With the swept law, and for every tool whose model names two jaws and a friction coefficient
between jaws and tissue, the exchange also builds the pinch (:class:`JawTissuePinch`): tissue
that the contact holds between the two jaws of a tool. It is part of :meth:`ToolTissueExchange.update`
in every scene, after the law's update; no scene asks for it and none can leave it out.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .jaw_tissue_pinch import JawTissuePinch
from .swept_tool_tissue_contact import SweptToolTissueContact, capsule_pieces
from .tool_tissue_contact import ToolTissueContact, capsule_samples

# Names of the laws that a setup can ask for.
LAWS = ("first", "swept")


def tool_tissue_law(value):
    """The law that a setting of the tool/tissue contact names: ``None`` (no contact), "first" or "swept".

    ``False``, ``None`` and "off" name no contact; ``True`` and "first" the first-version law
    (:class:`ToolTissueContact`, sample spheres at the accepted state); "swept" the contact on
    the sweep of the capsules (:class:`SweptToolTissueContact`).
    """
    if value is None or value is False or value == "off":
        return None
    if value is True or value == "first":
        return "first"
    if value == "swept":
        return "swept"
    raise ValueError(f"Unknown tool/tissue contact {value!r}; choose off, first or swept")


def tool_capsules(graspers):
    """The capsule set of the tools of a scene, tool after tool, as their models publish it.

    Args:
        graspers: The tool models of the scene, each with ``capsule_bodies``, ``capsule_endpoints``
            and ``capsule_radii``, and with ``jaw_bodies`` where it names its jaws.

    Returns:
        The body of each capsule, the end points of its axis in the body frame [m], shape
        (capsules, 2, 3), its radius [m], its tool (numbered from zero), and for each tool the
        bodies of its jaws (an empty tuple for a tool whose model names none).
    """
    bodies, endpoints, radii, tools, jaws = [], [], [], [], []
    for tool, grasper in enumerate(graspers):
        bodies.extend(grasper.capsule_bodies)
        endpoints.extend(grasper.capsule_endpoints)
        radii.extend(grasper.capsule_radii)
        tools.extend([tool] * len(grasper.capsule_bodies))
        jaws.append(tuple(int(body) for body in getattr(grasper, "jaw_bodies", ())))
    return (
        np.asarray(bodies),
        np.asarray(endpoints),
        np.asarray(radii, dtype=float),
        np.asarray(tools, dtype=np.int64),
        tuple(jaws),
    )


def tool_tissue_frictions(graspers):
    """For each tool of a scene, the friction coefficient between its jaws and the tissue that its model gives.

    ``config.tissue_friction`` of the tool model; ``None`` for a model that gives none (such a
    tool gets no pinch).
    """
    return tuple(getattr(getattr(grasper, "config", None), "tissue_friction", None) for grasper in graspers)


@dataclass(frozen=True)
class ToolTissueSetup:
    """What a scene with tools gives its tool/tissue exchange; host values only.

    Attributes:
        law: "first" or "swept" (:data:`LAWS`).
        capsule_bodies: Body of each tool capsule, shape (capsules,), tool after tool.
        capsule_endpoints: End points of each capsule axis in the body frame [m], shape (capsules, 2, 3).
        capsule_radii: Radii of the capsules [m].
        contact_radii: Radii with which the capsules touch the tissue [m]: the capsule radii,
            unless the owner gave others (the laws still take the two lists).
        capsule_tools: Tool of each capsule, numbered from zero.
        jaw_bodies: For each tool, the bodies of its two jaws, or an empty tuple for a tool
            whose model names none.
        tissue_friction: For each tool, the friction coefficient between its jaws and the tissue
            (the pinch, swept law), or ``None`` for a tool whose model gives none; ``None``
            for no tool.
        reach: Largest distance at which the tissue surface is looked for [m]: the thickness of the pad.
        node_stiffness: First law: lower bound of the stiffness of one tissue node [N/m].
        shear_frequency: First law: lowest shear frequency of the tissue layer [rad/s].
        effective_mass: Swept law: for each tool, its smallest effective mass at an end of one
            of its capsules [kg], from which the held-tissue stiffness follows.
        fixed_particles: Swept law: particles that the tissue solver holds fixed.
        full_step: Swept law: the full interval of the scene [s], the sweep horizon.
    """

    law: str
    capsule_bodies: np.ndarray
    capsule_endpoints: np.ndarray
    capsule_radii: np.ndarray
    contact_radii: np.ndarray
    capsule_tools: np.ndarray
    jaw_bodies: tuple
    reach: float
    node_stiffness: float | None = None
    shear_frequency: float | None = None
    effective_mass: np.ndarray | None = None
    fixed_particles: np.ndarray | None = None
    full_step: float | None = None
    tissue_friction: tuple | None = None

    @classmethod
    def of_scene(cls, scene, law, *, effective_mass=None, contact_radii=None):
        """The setup of ``scene`` for a law: everything but the effective masses is read from the scene here.

        Read of the scene, for either law: ``graspers`` (the capsule set that each tool model
        publishes, its jaw bodies, :func:`tool_capsules`, and its friction with the tissue,
        :func:`tool_tissue_frictions`) and ``reference.vertices`` (the pad). Rules:

        - Reach of the surface search: the thickness of the pad; a sample centre inside the
          pad is nearer to its surface than that.
        - First law, node stiffness: ``0.25 E h`` with the mean tetrahedron edge ``h`` of the
          simulated region, the lower bound of the stiffness of one tissue node that also
          relaxes the thread's forces on the tissue (``args.young_modulus``, ``selection``).
        - First law, shear frequency: ``(pi / 2) sqrt(G / rho) / H``, the lowest shear mode of a
          layer of depth ``H`` on a fixed base, with ``G = E / (2 (1 + nu))`` and ``H`` the
          vertical extent of the simulated region, which is held on its underside
          (``args.poisson_ratio``, ``args.density``).
        - Swept law, sweep horizon: the full interval of the scene (``sim_dt``), also when an
          attempt is shorter. Nodes that cannot be moved: the fixed particles of the scene
          (``fixed_vertex_indices``) and the surface vertices outside the simulated region.
        - Swept law, effective masses: the caller's, one per tool [kg] or one for all (the rule
          that needs the rigid solver is the scene's: ``tool_effective_masses``).

        Args:
            scene: A scene with tools and tissue.
            law: A setting of :func:`tool_tissue_law` that names a law.
            effective_mass: Swept law: the effective mass of each tool [kg], or one for all.
            contact_radii: Radii of the contact of each capsule [m]; ``None`` takes the capsule radii.
        """
        law = tool_tissue_law(law)
        if law is None:
            raise ValueError("A tool/tissue setup needs a law: first or swept")
        bodies, endpoints, radii, tools, jaws = tool_capsules(scene.graspers)
        if not len(bodies):
            raise ValueError("Tool/tissue contact needs the capsules of at least one tool")
        contact_radii = radii if contact_radii is None else np.asarray(contact_radii, dtype=float)
        if contact_radii.shape != radii.shape:
            raise ValueError("Capsule and contact radii must be finite positive metres, one per capsule")
        pad = np.asarray(scene.reference.vertices, dtype=float)
        values = {"reach": float(np.ptp(pad[:, 2]))}
        if law == "first":
            args = scene.args
            modulus, poisson, density = float(args.young_modulus), float(args.poisson_ratio), float(args.density)
            vertices = np.asarray(scene.selection.vertices)
            tetrahedra = np.asarray(scene.selection.indices, dtype=np.int32)
            edges = vertices[tetrahedra[:, 1:]] - vertices[tetrahedra[:, :1]]
            edge = float(np.linalg.norm(edges, axis=-1).mean())
            depth = float(np.ptp(np.asarray(scene.selection.vertices, dtype=float)[:, 2]))
            if (
                not np.isfinite([poisson, density, depth, values["reach"]]).all()
                or min(1.0 + poisson, density, depth) <= 0
            ):
                raise ValueError("Tool/tissue contact needs a finite tissue density, Poisson ratio and depth")
            shear_modulus = modulus / (2.0 * (1.0 + poisson))
            values.update(
                node_stiffness=0.25 * (modulus * edge),
                shear_frequency=0.5 * math.pi * math.sqrt(shear_modulus / density) / depth,
            )
        else:
            if effective_mass is None:
                raise ValueError("The swept tool/tissue contact needs the effective mass of the tools")
            count = int(max(tools)) + 1
            masses = np.asarray(effective_mass, dtype=float)
            if masses.ndim == 0:
                masses = np.full(count, float(masses))
            if masses.shape != (count,) or not np.isfinite(masses).all() or masses.min() <= 0:
                raise ValueError("The effective masses must be finite and positive, one per tool")
            values.update(
                effective_mass=masses,
                fixed_particles=np.asarray(scene.fixed_vertex_indices),
                full_step=float(scene.sim_dt),
            )
        return cls(
            law=law,
            capsule_bodies=bodies,
            capsule_endpoints=endpoints,
            capsule_radii=radii,
            contact_radii=contact_radii,
            capsule_tools=tools,
            jaw_bodies=jaws,
            tissue_friction=tool_tissue_frictions(scene.graspers),
            **values,
        )


class ToolTissueExchange:
    """The tool/tissue law of a scene and its calls; the one place that constructs a law.

    Args:
        model: Model of the bodies and of the tissue particles and tetrahedra.
        tissue_solver: The tissue solver (``SolverFEM``); the swept law writes its holds into
            the matrix of that solver. The first law takes none.
        surface: The closed triangle surface of the tissue that the owner refits at every
            accepted state, before :meth:`update`: the ``wp.Mesh``, its triangles as flat
            vertex indices and the tissue particle of each vertex (-1 for a vertex that is not one).
        setup: The :class:`ToolTissueSetup` of the scene.
        step_seconds: One ``float64`` on the device that holds the interval [s] of the accepted
            step when :meth:`update` runs.
        particle_q: Swept law: the accepted tissue positions. The holds are written at fixed
            slots of the tissue matrix, so its sparse structure must be final: a tissue solver
            that has not built its tangent yet builds it here, at these positions and the full
            interval, as the first frame does.

    Attributes:
        law: The law object (:class:`ToolTissueContact` or :class:`SweptToolTissueContact`).
        law_name: "first" or "swept".
        pinch: Swept law: the :class:`JawTissuePinch` of the tools whose models name two jaws
            and a friction with the tissue, or ``None`` if no tool does.
        contact_radii: The radius with which each capsule touches the tissue [m].
        effective_mass: Swept law: the effective mass of each tool [kg] that the law holds now.
    """

    def __init__(self, model, tissue_solver, surface, setup, step_seconds, *, particle_q=None):
        if setup.law not in LAWS:
            raise ValueError(f"Unknown tool/tissue law {setup.law!r}; choose one of {', '.join(LAWS)}")
        if not len(setup.capsule_bodies):
            raise ValueError("Tool/tissue contact needs the capsules of at least one tool")
        self.model, self.setup, self.law_name = model, setup, setup.law
        self.contact_radii = [float(radius) for radius in setup.contact_radii]
        self.effective_mass = None
        self.pinch = None
        mesh, triangles, point_particle = surface
        bodies, endpoints = setup.capsule_bodies, np.asarray(setup.capsule_endpoints)
        if setup.law == "first":
            # Samples: at most one contact radius apart on every capsule axis (capsule_samples).
            sample_bodies, points, sample_radii = capsule_samples(
                bodies, endpoints, setup.capsule_radii, setup.contact_radii
            )
            self.law = ToolTissueContact(
                model,
                mesh,
                triangles,
                point_particle,
                sample_bodies,
                points,
                sample_radii,
                node_stiffness=setup.node_stiffness,
                shear_frequency=setup.shear_frequency,
                step_seconds=step_seconds,
                reach=setup.reach,
            )
            return
        # Tool of every axis piece, in the order of the law's pieces: the inverse mass of that tool, in every
        # direction. K_c = effective mass / full step^2 is the held-tissue stiffness of a tool.
        self._piece_tool = setup.capsule_tools[
            capsule_pieces(bodies, endpoints, setup.capsule_radii, setup.contact_radii)[5]
        ]
        self.effective_mass = np.asarray(setup.effective_mass, dtype=float).copy()
        if tissue_solver._tangent_triplet_slots is None:
            tissue_solver.refresh_tangent(particle_q, setup.full_step)
        self.law = SweptToolTissueContact(
            model,
            tissue_solver,
            mesh,
            triangles,
            point_particle,
            np.asarray(setup.fixed_particles),
            bodies,
            endpoints,
            setup.capsule_radii,
            setup.contact_radii,
            setup.capsule_tools,
            step_seconds=step_seconds,
            full_step=setup.full_step,
            inverse_mass=self._inverse_mass(self.effective_mass),
            reach=setup.reach,
        )
        # The pinch: for every tool with two jaw bodies and a friction coefficient with the tissue.
        frictions = setup.tissue_friction or (None,) * len(setup.jaw_bodies)
        pairs = tuple(
            tuple(pair) if len(pair) and friction is not None else ()
            for pair, friction in zip(setup.jaw_bodies, frictions, strict=True)
        )
        if any(pairs):
            self.pinch = JawTissuePinch(
                model,
                self.law,
                bodies,
                endpoints,
                setup.contact_radii,
                pairs,
                [0.0 if friction is None else float(friction) for friction in frictions],
                self._held_stiffness(),
            )

    def _held_stiffness(self):
        """Per tool: the held-tissue stiffness ``K_c = effective mass / full step^2`` [N/m]."""
        return self.effective_mass / self.setup.full_step**2

    def _inverse_mass(self, masses):
        """Per axis piece: the inverse mass matrix of its tool, the same in every direction [1/kg]."""
        return np.eye(3)[None] / masses[self._piece_tool][:, None, None]

    # ------------------------------------------------------------------ the three calls of an interval

    def add_body_loads(self, body_f):
        """Before the rigid interval of every attempt: add the stored reaction on the tools [N, N m].

        The law's, and beside it the reaction of the tissue that the pinch carries.
        """
        self.law.add_forces(body_f)
        if self.pinch is not None:
            self.pinch.add_forces(body_f)

    def add_nodal_loads(self, particle_f):
        """Before the tissue step of every attempt: add what the law keeps for the tissue solve [N].

        Swept law: for every held node the force that balances its elastic force of the accepted
        state, so that its share of the right-hand side is only its change. The forces of the
        first law on the tissue are in the owner's array (:meth:`update`), which the owner adds.
        """
        if self.law_name == "swept":
            self.law.add_balance(particle_f)

    def update(self, body_q, body_qd, particle_q, particle_qd, nodal_force=None):
        """At an accepted state, with the surface refit to it: update the law for the next interval.

        Swept law: the law's update (reactions of the interval that ended, sweep, rows, forces,
        placement of the held nodes of ``particle_q`` and ``particle_qd``, holds), then the
        pinch on what that update published (it gives the nodes that a clamping pair of jaws
        carries the velocity that takes them with the jaws, and keeps those of them that the
        law has let go), then the holds once more.
        First law: that law's update, which adds its forces on the tissue to the owner's
        ``nodal_force``.
        """
        if self.law_name == "swept":
            self.law.update(body_q, body_qd, particle_q, particle_qd)
            if self.pinch is not None:
                self.pinch.update(body_q, particle_q, particle_qd)
                # The pinch keeps nodes that the law has let go: the tissue matrix follows the flags once more.
                self.law.holds.apply()
        else:
            self.law.update(body_q, nodal_force)

    # ------------------------------------------------------------------ outside the step

    @property
    def state(self):
        """The arrays that carry over from one accepted interval to the next (for a capture that puts a state back)."""
        if self.pinch is not None:
            return (*self.law.state, *self.pinch.state)
        return self.law.state

    @property
    def moves_tissue(self):
        """Whether :meth:`update` writes tissue nodes of the accepted state (their positions and velocities)."""
        return self.law_name == "swept"

    def restore_holds(self):
        """After the flags of :attr:`state` were put back: the tissue matrix follows them."""
        if self.law_name == "swept":
            self.law.holds.apply()

    def set_effective_mass(self, masses):
        """Swept law: replace the effective mass of each tool [kg], after a placement of the tools (a host write)."""
        masses = np.asarray(masses, dtype=float)
        if masses.shape != self.effective_mass.shape or not np.isfinite(masses).all() or masses.min() <= 0:
            raise ValueError("The effective masses must be finite and positive, one per tool")
        self.effective_mass = masses.copy()
        self.law.set_inverse_mass(self._inverse_mass(self.effective_mass))
        if self.pinch is not None:
            # The smallest squeeze that clamps follows the stiffness of the tools.
            self.pinch.set_stiffness(self._held_stiffness())

    def reset(self):
        """Clear what the law carries: no stored force, no reaction, and with the swept law no held node."""
        if self.law_name == "swept":
            if self.pinch is not None:
                self.pinch.reset()
            self.law.reset()
        else:
            for array in self.law.state:
                array.zero_()

    def modules(self):
        """Kernel modules that the calls launch, with their block sizes, for a start-up that loads by name."""
        if self.law_name == "swept":
            return self.law.modules() if self.pinch is None else (*self.law.modules(), *self.pinch.modules())
        from . import tool_tissue_contact

        # The law's surface query is launched with one query per block for a small batch.
        return ((tool_tissue_contact, tuple(sorted({256, self.law._query_block}))),)

    def parameters(self):
        """Host values of the law, and of the pinch under ``pinch``, without a device read."""
        if self.pinch is not None:
            return {**self.law.parameters(), "pinch": self.pinch.parameters()}
        return self.law.parameters()

    def diagnostics(self):
        """Explicit host readback of the law, and of the pinch under ``pinch``; never called by a step."""
        if self.pinch is not None:
            return {**self.law.diagnostics(), "pinch": self.pinch.diagnostics()}
        return self.law.diagnostics()
