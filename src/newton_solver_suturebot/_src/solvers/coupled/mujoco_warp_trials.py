"""Captured articulated rigid/FEM trials; exact events retain host fallback.

This adapter retains the current physical acceptance tests and linear budgets.
Its explicit device-Aitken or FP64 small-Gram Anderson policy replaces host
SVD acceleration;
iteration trajectories need not match that host numerical policy. Rigid motion
uses checkpointed MuJoCoWarp contacts.
A frozen contact search metric is condensed inside the graph from the Hessian
that the GPU constraint solve ended with; native integration and host decisions
never run in trials.
The contact-preconditioner CG starts from the guarded response-basis guess,
whose Schur coefficients are solved by device kernels in every correction.
The guess is one object from the first response-basis core on; whether a
launch may use it is a device flag, so its key in the graph signature does not
follow the track.
"""

import math

import warp as wp

from ..mujoco.subcycled_mujoco_warp import SubcycledMuJoCoWarp
from .coupling_graph import _CapturedNeedleTrials


@wp.kernel(enable_backward=False)
def _include_overflow(overflow: wp.array[int], finite: wp.array[int]):
    world = wp.tid()
    if overflow[world] != 0:
        wp.atomic_min(finite, 0, 0)


@wp.kernel(enable_backward=False)
def _surface_map_epoch(
    valid: wp.array[int],
    previous: wp.array[int],
    state: wp.array[int],
    values: wp.array[wp.float64],
    relaxation: wp.float64,
):
    if valid[0] != previous[0]:
        previous[0] = valid[0]
        state[1] = state[0]
        values[0] = relaxation


class ArticulatedRigidTrials:
    """The narrow rigid protocol consumed by the existing captured loop."""

    def __init__(self, coupler):
        self.coupler = coupler
        self.solver = coupler.rigid_solver
        self.checkpoint = coupler._gpu_checkpoint
        self.surface = coupler._surface_preconditioner
        self.surface_preconditioned = self.surface is not None
        self.surface_selected = wp.zeros(1, dtype=int, device=coupler.model.device)
        self.metric_selected = wp.zeros(1, dtype=int, device=coupler.model.device)
        self.dt = None
        self.projection = None
        self.metric = None
        # True while a capture holds begin_trials: its trials skip what that issued once.
        self._first_state_prepared = False
        # False keeps the zero CG start in the graph, for search-path comparisons.
        self.device_guess = bool(getattr(coupler, "_gpu_contact_guess", True))
        self.guess = None
        self.guess_key = None
        self.anderson = None
        if getattr(coupler, "_gpu_trial_policy", "current") == "device-anderson":
            from .acceleration.surface_anderson_graph import CapturedSurfaceAnderson

            self.anderson = CapturedSurfaceAnderson(coupler._trial_load)
        if type(self.solver) is not SubcycledMuJoCoWarp or self.checkpoint is None:
            raise ValueError("Articulated trials require the checkpointed MuJoCoWarp adapter")
        if coupler.model.world_count != 1:
            raise ValueError("The initial articulated coupling graph supports one independent world")

    @property
    def capture_owners(self):
        # A backend model update replaces the rigid step while an old graph
        # still exists. Keep the step whose pool, solver context and tree
        # tables that graph addresses alive until its own cache entry is
        # invalidated.
        pc = self.coupler._shaft_preconditioner
        return (
            self.solver.rigid_step,
            self.solver._first_step,
            self.solver._prepared_overflow,
            self.solver._trial_warm_slots,
            self.solver._trial_warm_valid,
            self.checkpoint,
            self.surface,
            self.surface_selected,
            self.metric_selected,
            self.projection,
            self.metric,
            self.anderson,
            () if self.anderson is None else self.anderson.owners,
            self.guess,
            () if self.guess is None else self.guess.owners + self.guess.device_solve.owners,
            # The anchors of the shaft correction: replaced only when the track storage grows.
            pc.matrices,
            pc.levers,
            pc.shears,
        )

    def prepare(self, dt):
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("A coupled interval must be finite and positive")
        self.checkpoint._validate()
        self.dt = float(dt)
        pc, interaction = self.coupler._shaft_preconditioner, self.coupler.interaction
        track = interaction.track_count > 0
        if (
            track
            and getattr(interaction, "tunnel_alm", None) is None
            and interaction.shaft_stiffness_per_m == 0
            and interaction.shaft_damping_per_m == 0
        ):
            raise ValueError("Unsupported captured contact operator")
        # Time step and damping are written where they changed, and the anchors keep their
        # storage at the track reserve. Empty shaft support is a valid surface-only numerical
        # system; there an active tape is no refusal.
        if not pc.stage(dt, tape=track):
            raise ValueError("Unsupported captured contact operator" if track else "Invalid rigid damping")
        from .interface.shaft_projection import projection_binding

        self.projection = projection_binding(
            self.coupler.interaction,
            self.coupler._candidate,
            self.coupler.needle_body,
            float(self.coupler.interaction.radii.max()),
        )
        # Bind the response basis of the contact-PC guess on the host, once per
        # launch. Its key joins the graph signature; coefficients stay on device.
        # The guess exists from the first response-basis core on and its key is
        # constant from its first binding: whether this launch may use the basis,
        # and the size of its solve, are device data that the binding writes.
        manager = getattr(self.coupler.fem_solver, "_affine_predictor", None)
        guess = pc.start_guess(manager) if self.device_guess else None
        if (
            guess is not None
            and pc._woodbury_guess is None
            and manager is not None
            and manager.active
            and manager.core is not None
        ):
            # A host-controlled correction starts from the same object.
            pc._woodbury_guess = guess
        self.guess_key = None if guess is None else guess.prepare_device(manager)
        self.guess = None if self.guess_key is None else guess

    def reset_numerical_map(self):
        self.surface_selected.zero_()
        self.metric_selected.zero_()
        if self.anderson is not None:
            self.anderson.reset_device()
        if self.guess is not None:
            # Tracks and response basis are fixed within one launch. Issued behind the
            # device flag of the guess: a launch that may not use the basis skips it.
            self.guess.device_solve.enqueue_topology()

    def validate_surface(self):
        if self.surface is not None:
            self.surface.validate()

    def update_numerical_map(self, state, values, relaxation):
        self.metric.prepare(self.dt)
        wp.launch(
            _surface_map_epoch,
            1,
            [self.metric.valid, self.metric_selected, state, values, relaxation],
            device=self.coupler.model.device,
        )
        if self.surface is not None:
            wp.launch(
                _surface_map_epoch,
                1,
                [self.surface.valid, self.surface_selected, state, values, relaxation],
                device=self.coupler.model.device,
            )

    def apply_preconditioner(self, pc, state, dt, residual, output):
        # Current geometry, surface anchors and ALM values remain device inputs.
        # The Woodbury guess replaces its host packet by device coefficients.
        if self.metric is None:
            raise RuntimeError("Create the rigid contact metric (prepare the trials) before a captured correction")

        def solve():
            previous, guess = pc._woodbury_guess, self.guess
            try:
                pc._woodbury_guess = guess
                if guess is not None:
                    # The start of the solve takes its coefficients from the device kernels,
                    # behind the device flag of the guess, after the contact inputs.
                    guess.device_coefficients = True
                pc._enqueue_inputs(state, dt, residual)
                pc._solve(residual, output)
            finally:
                pc._woodbury_guess = previous
                if guess is not None:
                    guess.device_coefficients = False

        def fallback():
            wp.copy(output, residual)

        # Invalid approximate Hessians only disable the numerical correction;
        # they cannot bypass any physical force or linear residual certificate.
        wp.capture_if(self.metric.valid, solve, fallback)

    # The rigid interval reads the applied wrench, joint state and controls and
    # writes rigid state only, so the captured loop may run it beside the tissue solve.
    concurrent_with_tissue = True

    def begin_trials(self, state_in, control, dt):
        """Capture, before the trial loop, the part of the rigid interval that no trial changes.

        Every trial starts from the checkpoint and applies another wrench, so the stages of
        its first microstep that read only the state are issued here, once per launch.
        Pair with ``end_trials`` when the capture of the loop is complete.
        """
        self.checkpoint.restore()
        self._first_state_prepared = self.solver.prepare_trials(state_in, control, dt)

    def end_trials(self):
        self._first_state_prepared = False

    def step(self, state_in, state_out, control, dt, *, after_conditional=None):
        # Every nonlinear trial starts from the same accepted integration
        # history. Applying a new interface guess must not advance physical time.
        # ``after_conditional`` is called after the solver loop of each microstep is
        # captured, for a caller that captures this interval as a concurrent graph branch.
        self.checkpoint.restore()
        self.solver._step_uncaptured(
            state_in,
            state_out,
            control,
            None,
            dt,
            after_solver_loop=after_conditional,
            first_state_prepared=self._first_state_prepared,
        )

    def include_validity(self, finite):
        wp.launch(
            _include_overflow,
            self.solver.mjw_data.nworld,
            [self.solver.mjw_data.overflow, finite],
            device=self.coupler.model.device,
        )


class CapturedArticulatedTrials(_CapturedNeedleTrials):
    """One conditional outer solve with unchanged FEM/ALM acceptance tests."""

    def __init__(self, coupler):
        super().__init__(coupler, rigid=ArticulatedRigidTrials(coupler))
        self._articulated_signature = None

    def prepare(self, state_in, control, dt):
        c = self.c
        # Host preflight is outside ScopedCapture and outside every trial.
        # The backend validates current numerical options and all data owners.
        signature = c.rigid_solver.validate_capture_bindings(state_in, c._candidate, control, dt)
        if not c.rigid_solver._rigid_warmed:
            # Prepare generated MJWarp kernels/context outside outer capture.
            # The temporary output is discarded, and all integration history
            # returns to the accepted checkpoint even if preparation fails.
            temporary = c.model.state()
            try:
                self.rigid.checkpoint.restore()
                c.rigid_solver.step(state_in, temporary, control, None, dt)
            finally:
                self.rigid.checkpoint.restore()
            signature = c.rigid_solver.validate_capture_bindings(state_in, c._candidate, control, dt)
        # Input/output buffers alternate normally. Their complete bindings are
        # already owned by the base graph cache; invalidate all captures only
        # when backend/options/checkpoint storage changes, not on every swap.
        invariant = signature[:2] + (
            tuple((array.ptr, array.shape, array.strides, array.dtype) for array in self.rigid.checkpoint.saved),
        )
        if invariant != self._articulated_signature:
            self.invalidate()
            self._articulated_signature = invariant
        from ..mujoco.rigid_contact_metric import CapturedRigidContactMetric

        if self.rigid.metric is None or self.rigid.metric.rigid_step is not c.rigid_solver.rigid_step:
            self.rigid.metric = CapturedRigidContactMetric(
                c.rigid_solver.rigid_step,
                c.moment_length,
                c._shaft_preconditioner.rigid_blocks,
                step_time=c._shaft_preconditioner.step_time,
            )
        warm = not self._fem_prepared
        super().prepare(state_in, control, dt)
        if warm:
            # Compile numerical operator/Krylov kernels outside conditional
            # capture. This changes only scratch, never state or integration.
            c._residual.zero_()
            self.rigid.metric.prepare(dt)
            self.rigid.apply_preconditioner(c._shaft_preconditioner, c._candidate, dt, c._residual, c._direction)
