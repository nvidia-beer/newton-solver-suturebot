# SPDX-License-Identifier: Apache-2.0
"""The tool/tissue contact of the tool/tissue example: the scene's hooks onto the one tool/tissue exchange.

``ToolTissueScene`` calls its contact object

- ``add_reaction(body_f)`` before the rigid interval of every attempt: the reaction on the
  tools [N, N m], force first, torque about the centre of mass, world frame;
- ``begin_step(state_in, state_out, particle_f, dt)`` after the rigid interval and before the
  tissue step: ``state_in`` holds the accepted tissue and the tool poses at the start of the
  interval, ``state_out.body_q`` the tool poses at its end, so the pair is the motion of the
  tools over the interval; the object adds its forces on the tissue particles [N];
- ``correct_step(state_in, state_out, dt)`` after the tissue step and before the check of the
  interval: ``state_out`` holds the end state of tools and tissue. Nothing is corrected inside
  an attempt today; a contact that did would write ``state_out.particle_q`` and
  ``state_out.particle_qd`` here, and the guard of the interval would check the result;
- ``end_step(accepted, dt)`` once the interval is accepted, with the accepted state: the
  object updates what it carries to the next interval.

All of them only launch kernels, so the scene captures them in its CUDA graphs. An attempt can
be rejected and repeated with a shorter interval: ``add_reaction``, ``begin_step`` and
``correct_step`` must not change what ``state`` lists, only ``end_step`` may. ``dt`` is the interval [s] as a host
number; a launch that needs it on the device reads ``scene.solver.step_time``, which holds
it. Outside the step the scene calls ``placed()`` after every placement of the tools (start-up
and a reset) and ``reset()`` when the tissue is put back at rest: host work is allowed there.

There is one contact object (:class:`SceneContact`) for the three settings of the switch. It
holds no law of its own: what tools and tissue exchange is the ``ToolTissueExchange`` of the
shared package, built from the scene by ``tool_tissue_setup``, the same function and the same
object as in ``fem_needle_grasp``. There the worker thread issues the three calls of the
exchange (in the graph of its accepted step); here the hooks above issue them, in the scene's
own graphs. A new law is added to the exchange, not here; the scripted path and the
measurements do not change.
"""

import numpy as np
import warp as wp

from ..._src.solvers.coupled.interface.tool_tissue_exchange import ToolTissueExchange
from ._grasp_scene import placed_tool_effective_masses, tool_tissue_setup

# Names of the contact switch: off (the tools pass through the tissue), the first-version law
# of the grasp scene, and the contact built on the sweep of the capsules.
CONTACTS = ("off", "first", "swept")


class SceneContact:
    """The tool/tissue exchange of the scene under the names of its hooks; with "off" there is none.

    Where the calls of the exchange sit in an interval:

    - ``add_reaction`` (before the rigid interval of every attempt): the stored reaction on the
      tools (``add_body_loads``). First law: that of the forces of the last accepted state.
      Swept law: that of the nodes held through the last accepted interval and of the
      held-tissue forces.
    - ``begin_step`` (before the tissue step of every attempt): what the law keeps for the tissue
      solve (``add_nodal_loads``). First law: its forces on the tissue particles, which this
      object keeps in an array of its own, as the worker thread keeps them with the thread's.
      Swept law: the forces that balance the elastic force on the held nodes, so that the
      right-hand side of the tissue solve holds only their change.
    - ``end_step`` (after an interval is accepted, at the accepted state): the surface is refit
      and the law is updated (``update``). First law: forces at the nearest surface point of
      every sample sphere, applied through the whole next interval. Swept law: reactions of the
      interval that ended, sweep of every capsule to the end of the next interval, rows,
      held-tissue forces, placement of the held nodes of the accepted state and their holds in
      the tissue matrix.

    Constants: all from the scene, by the one setup function (``tool_tissue_setup``). The swept
    law takes ``K_c`` of a tool from its smallest effective mass at an end of one of its capsules
    (``placed_tool_effective_masses``), at the joint coordinates that ``scene.state_0`` holds:
    when the object is built, and again in :meth:`placed` after every placement of the tools,
    so a run uses the masses of its placed start (in this scene the driven tool changes by under
    one percent along the scripted path).

    Attributes:
        name: The setting of the switch (:data:`CONTACTS`).
        exchange: The ``ToolTissueExchange`` of the scene, or ``None`` with "off".
        nodal_force: First law: its forces on the tissue particles [N] of the last accepted interval.
    """

    def __init__(self, scene, name):
        if name not in CONTACTS:
            raise ValueError(f"Unknown tool/tissue contact {name!r}; choose one of {', '.join(CONTACTS)}")
        self.scene, self.name = scene, name
        self.exchange, self.nodal_force = None, None
        setup = tool_tissue_setup(scene, name)
        if setup is None:
            return
        surface = scene.surface
        self.exchange = ToolTissueExchange(
            scene.model,
            scene.solver,
            (surface.mesh, surface.triangles, surface.point_particle),
            setup,
            # The laws read the interval from the tissue solver's step time, which holds that of the accepted step.
            scene.solver.step_time.seconds_f64,
            particle_q=scene.state_0.particle_q,
        )
        if setup.law == "first":
            self.nodal_force = wp.zeros(scene.model.particle_count, dtype=wp.vec3, device=scene.device)
        # The bodies of each tool that carry capsules, each once (a body carries several capsules of the set).
        self._tool_bodies = [np.unique(np.asarray(tool.capsule_bodies)) for tool in scene.graspers]

    # ------------------------------------------------------------------ what the scene asks before it captures

    @property
    def law(self):
        """The law object of the exchange, or ``None`` without a contact."""
        return None if self.exchange is None else self.exchange.law

    @property
    def effective_mass(self):
        """Swept law: the effective mass of each tool [kg] that the law holds now."""
        return None if self.exchange is None else self.exchange.effective_mass

    @property
    def state(self):
        """Arrays that carry over from one accepted interval to the next (restored by the start-up capture)."""
        if self.exchange is None:
            return ()
        return self.exchange.state if self.nodal_force is None else (*self.exchange.state, self.nodal_force)

    @property
    def has_end_step(self):
        """Whether ``end_step`` launches anything (the scene then captures it)."""
        return self.exchange is not None

    @property
    def keeps_surface(self):
        """Whether ``end_step`` leaves ``scene.surface`` at the accepted state (an observation then uses it as it is)."""
        return self.exchange is not None

    @property
    def moves_tissue(self):
        """Whether ``end_step`` moves tissue nodes of the accepted state (an observation then also measures the
        state as it is drawn)."""
        return self.exchange is not None and self.exchange.moves_tissue

    # ------------------------------------------------------------------ the hooks of an interval

    def add_reaction(self, body_f):
        if self.exchange is not None:
            self.exchange.add_body_loads(body_f)

    def begin_step(self, state_in, state_out, particle_f, dt):
        if self.exchange is None:
            return
        if self.nodal_force is not None:
            from ..._src.solvers.thread.worker_thread import _add_nodal

            wp.launch(
                _add_nodal,
                self.scene.model.particle_count,
                inputs=[self.nodal_force, particle_f],
                device=self.scene.device,
            )
        self.exchange.add_nodal_loads(particle_f)

    def correct_step(self, state_in, state_out, dt):
        pass

    def end_step(self, accepted, dt):
        if self.exchange is None:
            return
        self.scene.surface.update(accepted.particle_q)
        if self.nodal_force is not None:
            self.nodal_force.zero_()
        self.exchange.update(
            accepted.body_q, accepted.body_qd, accepted.particle_q, accepted.particle_qd, nodal_force=self.nodal_force
        )

    # ------------------------------------------------------------------ outside the step

    def reset(self):
        if self.exchange is None:
            return
        self.exchange.reset()
        if self.nodal_force is not None:
            self.nodal_force.zero_()

    def placed(self):
        """After a placement of the tools (``ToolTissueScene.place``): ``K_c`` of every tool from the placed joints."""
        if self.exchange is not None and self.exchange.effective_mass is not None:
            self.exchange.set_effective_mass(placed_tool_effective_masses(self.scene))

    def modules(self):
        """Kernel modules that this object launches, with their block sizes, for the start-up of the scene."""
        if self.exchange is None:
            return ()
        if self.nodal_force is None:
            return self.exchange.modules()
        from ..._src.solvers.thread import worker_thread

        # The forces of the first law on the tissue are added by the kernel of the worker thread.
        return (*self.exchange.modules(), (worker_thread, (256,)))

    def parameters(self):
        if self.exchange is None:
            return {"law": "none: the tools pass through the tissue"}
        parameters = self.exchange.parameters()
        if self.exchange.effective_mass is not None:
            parameters = {
                **parameters,
                "effective_tool_mass_kg": self.exchange.effective_mass.tolist(),
                "effective_tool_mass_rule": "per tool, at the joint coordinates of its last placement: smallest "
                "1 / lambda_max(J M^-1 J^T) over the ends of its capsules",
            }
        return parameters

    def observe(self):
        """Explicit read of the contact loads for an observation; never called by a step."""
        if self.exchange is None:
            return {"tissue_force_N": [0.0, 0.0, 0.0], "tool_force_N": [0.0, 0.0, 0.0], "loaded_samples": 0}
        report = self.exchange.diagnostics()
        wrench = np.asarray(report.pop("applied_body_wrench_N_Nm"))
        report.pop("law", None)
        if self.nodal_force is not None:
            return {
                "tissue_force_N": self.nodal_force.numpy().sum(axis=0).tolist(),
                "tool_force_N": wrench[:, :3].sum(axis=0).tolist(),
                **report,
            }
        balance = self.exchange.law.balance.numpy()
        return {
            # What the holds carry: minus the elastic force of the tissue on the held nodes.
            "tissue_force_N": balance.sum(axis=0).tolist(),
            "tool_force_N": wrench[:, :3].sum(axis=0).tolist(),
            "tool_force_by_tool_N": [wrench[bodies, :3].sum(axis=0).tolist() for bodies in self._tool_bodies],
            # The law's wrench on every body that carries capsules (force, torque about its centre of mass).
            "body_wrench_N_Nm": {int(body): wrench[body].tolist() for bodies in self._tool_bodies for body in bodies},
            "loaded_samples": report["loaded_pieces"],
            **report,
        }


def make_contact(name, scene):
    """The contact object of ``scene`` for a name of :data:`CONTACTS`."""
    return SceneContact(scene, name)
