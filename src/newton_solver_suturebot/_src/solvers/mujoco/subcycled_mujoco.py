# SPDX-License-Identifier: Apache-2.0
"""Resolve native rigid contacts inside one partitioned FEM coupling interval."""

import math

import newton
import numpy as np
import warp as wp
from newton.solvers import SolverMuJoCo

from .mujoco_free_body import synchronizes_every_step
from .mujoco_native_transfer import NativeMuJoCoTransfer
from .native_step_guard import (
    NativeMicrostepGuard,
    NativeRigidStepRejected,
    restore_native_warnings,
    snapshot_native_warnings,
)


class SubcycledMuJoCo(SolverMuJoCo):
    """Native CPU MuJoCo steps with held external wrenches and actuator targets.

    The outer coupling iteration supplies one world-frame wrench per body.
    These loads and the control targets are held for the complete interval;
    native contacts and finite joint actuators are solved at every microstep.
    Only the final rigid/joint state is exported. Particle/FEM arrays in the
    caller's output remain untouched. The enclosing coupling transaction owns
    rollback of native integration state; scratch states carry no history.
    """

    def __init__(self, model, *, rigid_substeps=4, batch_native=True, packed_transfer=True, **kwargs):
        if type(rigid_substeps) is not int or rigid_substeps < 1:
            raise ValueError("rigid_substeps must be a positive integer")
        if not kwargs.get("use_mujoco_cpu", False):
            raise ValueError("Rigid subcycling requires native CPU MuJoCo")
        super().__init__(model, **kwargs)
        self.rigid_substeps = rigid_substeps
        self.batch_native = bool(batch_native)
        self.packed_transfer = bool(packed_transfer)
        self._native_transfer = None
        self._rigid_scratch = None
        # Scalar joint targets and world COM wrenches are independent of the
        # evolving coordinates, so their native conversion stays valid for
        # every microstep. BALL-joint torque/velocity targets require rotating
        # their control each microstep and retain the general adapter.
        allowed = [newton.JointType.FIXED, newton.JointType.PRISMATIC, newton.JointType.REVOLUTE, newton.JointType.FREE]
        self._constant_native_control = bool(np.isin(model.joint_type.numpy(), allowed).all())
        self.native_batches = 0
        self.general_intervals = 0
        self._native_step_guard = NativeMicrostepGuard(self._mujoco, self.mj_data)
        self.native_guard_rejections = 0
        self.native_guard_last_failure = None
        self.native_guard_microsteps = 0

    def notify_model_changed(self, flags):
        # Graphs reference the old conversion/model arrays. Dropping the bridge
        # also refreshes the persistent native mapping after a notification.
        transfer = getattr(self, "_native_transfer", None)
        if transfer is not None:
            wp.synchronize_stream(transfer.device)
        self._native_transfer = None
        super().notify_model_changed(flags)

    def _can_batch(self):
        return (
            self.batch_native
            and self._constant_native_control
            and self.model.world_count == 1
            and synchronizes_every_step(self)
            and self._kinematic_weld_eq is None
        )

    def begin_repeated_input(self, state_in, state_out, control, body):
        """Borrow an input transaction for the supported native transfer path.

        The caller must call ``end_repeated_input`` on the returned transfer in
        ``finally``. Unsupported configurations return ``None`` unchanged.
        """
        if not (
            self._can_batch()
            and self.packed_transfer
            and NativeMuJoCoTransfer.supports(self, state_in, state_out, control, None)
        ):
            return None
        if self._native_transfer is None:
            self._native_transfer = NativeMuJoCoTransfer(self)
        return self._native_transfer.begin_repeated_input(state_in, control, body)

    def _step_native_batch(self, state_in, state_out, control, contacts, dt, observables, before_native_step):
        """Import once, integrate at native precision, then export the endpoint.

        The previous implementation exported and reimported float32 Newton
        coordinates between native microsteps. Those three synchronization
        roundtrips had no physical purpose: FEM never observes intermediate
        rigid states and all external loads stay held over the interval.
        Native contacts, finite actuators and warm starts still advance at
        every microstep with the same integrator and numerical budgets.
        """
        self.validate_observables(observables, contacts)
        self._require_unreduced_contacts(contacts)
        transfer = None
        if self.packed_transfer and NativeMuJoCoTransfer.supports(self, state_in, state_out, control, observables):
            if self._native_transfer is None:
                self._native_transfer = NativeMuJoCoTransfer(self)
            transfer = self._native_transfer
            transfer.push(state_in, control)
        else:
            self._apply_mjc_control(self.model, state_in, control, self.mj_data)
            self._update_mjc_data(self.mj_data, self.model, state_in)
        if before_native_step is not None:
            before_native_step()
        self.mj_model.opt.timestep = dt / self.rigid_substeps
        kind = self.ObservableKind
        needs_body = (
            state_out.body_qdd is not None
            or state_out.body_parent_f is not None
            or (
                observables is not None
                and (observables.is_requested(kind.BODY_QDD) or observables.is_requested(kind.BODY_PARENT_F))
            )
        )
        needs_contact = observables is not None and observables.is_requested(kind.CONTACT_F)
        for index in range(self.rigid_substeps):
            final = index == self.rigid_substeps - 1
            self._checked_microstep(
                state_out,
                observables if final else None,
                needs_body=final and needs_body,
                needs_contact=final and needs_contact,
            )
            self._step += 1
        if transfer is not None:
            transfer.pull(state_in, state_out)
        else:
            self._update_newton_state(self.model, state_out, self.mj_data, state_prev=state_in, observables=observables)
        self.native_batches += 1

    def _checked_microstep(self, state_out, observables, *, needs_body, needs_contact):
        """Run original native arithmetic, checking before every endpoint export."""
        mujoco, model, data = self._mujoco, self.mj_model, self.mj_data
        guard = self._native_step_guard
        if guard.data is not data:
            # A native model rebuild may replace MjData; warning views belong
            # to one allocation and must never inspect the retired data.
            guard = self._native_step_guard = NativeMicrostepGuard(mujoco, data)
        token = guard.begin(model.opt.timestep)
        try:
            if needs_body or (needs_contact and model.opt.integrator == mujoco.mjtIntegrator.mjINT_RK4):
                # Same native pipeline as SolverMuJoCo's observable path;
                # validate forward dynamics before exporting observables.
                integrator = model.opt.integrator
                if integrator not in (
                    mujoco.mjtIntegrator.mjINT_EULER,
                    mujoco.mjtIntegrator.mjINT_RK4,
                    mujoco.mjtIntegrator.mjINT_IMPLICIT,
                    mujoco.mjtIntegrator.mjINT_IMPLICITFAST,
                ):
                    raise ValueError("Unsupported native observable integrator")
                mujoco.mj_checkPos(model, data)
                mujoco.mj_checkVel(model, data)
                mujoco.mj_forward(model, data)
                mujoco.mj_checkAcc(model, data)
                guard.check(token, integrated=False)
                if model.opt.enableflags & mujoco.mjtEnableBit.mjENBL_FWDINV:
                    mujoco.mj_compareFwdInv(model, data)
                self._update_body_observables(self.model, state_out, data, observables)
                if needs_contact:
                    self._populate_contact_observables(
                        observables.contacts, observables.contact_f, warning_stacklevel=6
                    )
                if integrator == mujoco.mjtIntegrator.mjINT_EULER:
                    mujoco.mj_Euler(model, data)
                elif integrator == mujoco.mjtIntegrator.mjINT_RK4:
                    mujoco.mj_RungeKutta(model, data, 4)
                else:
                    mujoco.mj_implicit(model, data)
                guard.check(token)
            else:
                mujoco.mj_step(model, data)
                guard.check(token)
                if needs_contact:
                    self._populate_contact_observables(
                        observables.contacts, observables.contact_f, warning_stacklevel=5
                    )
            self.native_guard_microsteps += 1
        except NativeRigidStepRejected as error:
            self.native_guard_rejections += 1
            self.native_guard_last_failure = str(error)
            raise
        finally:
            guard.finish(token)

    def _general_native_step(self, state_in, state_out, control, contacts, dt, observables):
        """Original CPU adapter push/step/pull, with checked native integration."""
        self.validate_observables(observables, contacts)
        self._require_unreduced_contacts(contacts)
        self._apply_mjc_control(self.model, state_in, control, self.mj_data)
        if self._update_data_interval > 0 and self._step % self._update_data_interval == 0:
            self._update_mjc_data(self.mj_data, self.model, state_in)
        self._update_kinematic_weld_references()
        self.mj_model.opt.timestep = dt
        kind = self.ObservableKind
        needs_body = (
            state_out.body_qdd is not None
            or state_out.body_parent_f is not None
            or observables is not None
            and (observables.is_requested(kind.BODY_QDD) or observables.is_requested(kind.BODY_PARENT_F))
        )
        needs_contact = observables is not None and observables.is_requested(kind.CONTACT_F)
        self._checked_microstep(state_out, observables, needs_body=needs_body, needs_contact=needs_contact)
        self._update_newton_state(self.model, state_out, self.mj_data, state_prev=state_in, observables=observables)
        self._step += 1

    def step(self, state_in, state_out, control, contacts, dt, *, observables=None, before_native_step=None):
        """Reject invalid native trials before the caller can accept their state."""
        warnings = snapshot_native_warnings(self.mj_data)
        try:
            return self._step_interval(
                state_in,
                state_out,
                control,
                contacts,
                dt,
                observables=observables,
                before_native_step=before_native_step,
            )
        except BaseException:
            restore_native_warnings(self.mj_data, warnings)
            raise

    def _step_interval(self, state_in, state_out, control, contacts, dt, *, observables=None, before_native_step=None):
        """Advance rigid dynamics, optionally overlapping independent device work.

        ``before_native_step`` runs once, after the batched native import and
        before CPU integration. It may enqueue independent particle/FEM work
        on the current stream, allowing that work to overlap native dynamics.
        It must not mutate rigid inputs, controls, native state or the stream.
        The final rigid export is ordered after its device work. The general
        adapter fallback invokes it once before the first ordinary microstep.
        Callback failures propagate to the enclosing coupling transaction,
        which owns rollback just as for a failed native integration.
        """
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("Rigid coupling interval must be finite and positive")
        if state_in is state_out:
            raise ValueError("Rigid subcycling requires distinct input and output states")
        if self._can_batch():
            self._step_native_batch(state_in, state_out, control, contacts, dt, observables, before_native_step)
            return
        if before_native_step is not None:
            before_native_step()
        self.general_intervals += 1
        if self._rigid_scratch is None:
            self._rigid_scratch = (self.model.state(), self.model.state())
        current = state_in
        for index in range(self.rigid_substeps):
            final = index == self.rigid_substeps - 1
            following = state_out if final else self._rigid_scratch[index % 2]
            if current is not state_in:
                # SolverMuJoCo writes coordinates, not external forces. Refill
                # each scratch input explicitly, including zero-load intervals.
                wp.copy(current.body_f, state_in.body_f)
            self._general_native_step(
                current, following, control, contacts, dt / self.rigid_substeps, observables if final else None
            )
            current = following
