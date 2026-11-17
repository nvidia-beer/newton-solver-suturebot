"""Step parity and launch census of the rigid step against MuJoCo Warp's step.

``run`` rolls every case out with one of two steps, ``mujoco_warp.step`` (``--step
stock``) or the solver's ``RigidStep.step`` (``--step rigid``), and writes one trajectory
archive; ``compare`` then reports bitwise equality or the largest relative difference
of qpos, qvel, qacc and the constraint forces of every step, plus the smooth-dynamics
intermediates of the last step.

    python tests/_mjw_step_parity.py dump-grasp --out DIR            # CPU, native scene
    python tests/_mjw_step_parity.py run --step stock --device cpu --grasp DIR --out a.npz
    python tests/_mjw_step_parity.py run --step rigid --device cpu --grasp DIR --out b.npz
    python tests/_mjw_step_parity.py run --step rigid --device cuda:0 --graph --grasp DIR --reference a.npz --out c.npz
    python tests/_mjw_step_parity.py compare a.npz b.npz
    python tests/_mjw_step_parity.py census --grasp DIR              # launches per step, CUDA

A CPU trajectory is reproducible bit for bit. A CUDA trajectory is not, with either
step: contacts and constraint rows take their slots, and sums their order, from races
between threads, and contact dynamics amplify the rounding differences within a few
steps. ``--reference`` therefore restarts every step from the state that a CPU archive
recorded for it, so two CUDA archives differ by the effect of one step only, and the
difference between the two steps can be set against the difference between two runs
of the same step.

``census`` counts the kernel launches and memory operations that one step of each kind
issues on the CUDA device, read from Warp's activity timing. With the solver loop as a
loop in Python of 0 and of 3 iterations, the counts split into a fixed part and a part
per solver iteration. ``test_mjw_fused_launches.py`` asserts these counts for its test
scenes; ``census`` reports them for the grasp model and MuJoCo Warp's test models.

``dump-grasp`` saves the MuJoCo model that the grasp scene builds (two dVRK tools and
the free needle) and its integration state after a few frames.
"""

import argparse
import collections
import os
import sys
from pathlib import Path

import mujoco
import numpy as np

STATE = ("qpos", "qvel", "act", "ctrl", "qacc_warmstart", "qfrc_applied", "xfrc_applied", "mocap_pos", "mocap_quat")
STEP_FIELDS = ("qpos", "qvel", "qacc", "act", "time", "efc_force", "nefc", "solver_niter")
LAST_FIELDS = ("subtree_com", "cinert", "crb", "M", "cacc", "cfrc_int", "qfrc_bias", "qfrc_passive", "qfrc_constraint")

# The rigid options of the grasp scene, which the rigid step requires: implicitfast and elliptic cones.
SCENE = {"integrator": mujoco.mjtIntegrator.mjINT_IMPLICITFAST, "cone": mujoco.mjtCone.mjCONE_ELLIPTIC}

# name, xml below MuJoCo Warp's test_data, MjOption edits, keyframe, worlds, steps, constraint rows per world
CASES = (
    ("humanoid_scene_options", "humanoid/humanoid.xml", {**SCENE, "impratio": 10.0}, "squat", 1, 300, None),
    ("humanoid_3_worlds", "humanoid/humanoid.xml", SCENE, "stand_on_left_leg", 3, 300, None),
    (
        "position_actuators",
        "actuation/position.xml",
        {**SCENE, "iterations": 100, "ls_iterations": 50},
        0,
        2,
        300,
        None,
    ),
    ("muscle_activation", "actuation/muscle.xml", SCENE, 0, 1, 300, None),
    ("collision_free_bodies", "collision.xml", SCENE, 0, 2, 300, None),
)


def _test_data():
    import mujoco_warp

    return Path(mujoco_warp.__file__).resolve().parent / "test_data"


def _native(name, grasp):
    """MuJoCo model and initial data of one case."""
    if name == "grasp":
        mjm = mujoco.MjModel.from_binary_path(str(Path(grasp) / "grasp.mjb"))
        state = np.load(Path(grasp) / "grasp_state.npz")
        mjd = mujoco.MjData(mjm)
        for field in STATE:
            getattr(mjd, field)[:] = state[field]
        mjd.time = float(state["time"])
        return mjm, mjd, 1, int(state["steps"]), None
    _, path, options, keyframe, worlds, steps, njmax = next(case for case in CASES if case[0] == name)
    mjm = mujoco.MjModel.from_xml_path(str(_test_data() / path))
    for key, value in options.items():
        setattr(mjm.opt, key, value)
    mjd = mujoco.MjData(mjm)
    if isinstance(keyframe, str):
        keyframe = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_KEY, keyframe)
    if mjm.nkey:
        mujoco.mj_resetDataKeyframe(mjm, mjd, keyframe)
    random = np.random.default_rng(7)
    mjd.qvel[:] = random.uniform(-0.2, 0.2, mjm.nv)
    mjd.ctrl[:] = random.uniform(-0.1, 0.1, mjm.nu)
    mujoco.mj_step(mjm, mjd, 5)
    return mjm, mjd, worlds, steps, njmax


def _device_model(name, grasp):
    import mujoco_warp as mjw
    import warp as wp

    mjm, mjd, worlds, steps, njmax = _native(name, grasp)
    mujoco.mj_forward(mjm, mjd)
    mjd.qacc_warmstart[:] = mjd.qacc
    m = mjw.put_model(mjm)
    d = mjw.put_data(mjm, mjd, nworld=worlds, njmax=njmax)
    if worlds > 1:
        # Distinct worlds, so that a per-world indexing error cannot cancel.
        random = np.random.default_rng(11)
        qvel = d.qvel.numpy()
        qvel += random.uniform(-0.05, 0.05, qvel.shape).astype(qvel.dtype)
        d.qvel.assign(qvel)
    wp.synchronize()
    return mjm, m, d, steps


def _stepper(kind, m, d):
    """The step of one kind for a Model and Data: MuJoCo Warp's, or the solver's rigid step."""
    import mujoco_warp as mjw

    if kind == "stock":
        return lambda: mjw.step(m, d)
    from newton_solver_suturebot._src.solvers.mujoco.warp_step import RigidStep

    return RigidStep(m, d).step


def _needle_wrench(mjm, d, interval):
    """Held external wrench on the free needle, changed every four microsteps like the tissue reaction.

    Without it the grasp rests in equilibrium and every solve converges in one iteration.
    """
    body = int(mjm.jnt_bodyid[mjm.jnt_type == mujoco.mjtJoint.mjJNT_FREE][0])
    wrench = d.xfrc_applied.numpy()
    wrench[:, body, :3] = 0.3 * np.array([np.sin(0.9 * interval), np.cos(1.3 * interval), np.sin(0.5 * interval + 1.0)])
    wrench[:, body, 3:] = 1.0e-3 * np.array([np.cos(0.7 * interval), np.sin(1.1 * interval), np.cos(0.4 * interval)])
    d.xfrc_applied.assign(wrench)


def _record(d, rows):
    nefc = d.nefc.numpy().copy()
    force = d.efc.force.numpy().copy()
    for world, count in enumerate(nefc):
        force[world, count:] = 0.0
    rows["qpos"].append(d.qpos.numpy().copy())
    rows["qvel"].append(d.qvel.numpy().copy())
    rows["qacc"].append(d.qacc.numpy().copy())
    rows["act"].append(d.act.numpy().copy())
    rows["time"].append(d.time.numpy().copy())
    rows["efc_force"].append(force)
    rows["nefc"].append(nefc)
    rows["solver_niter"].append(d.solver_niter.numpy().copy())


def _names(options):
    names = [case[0] for case in CASES]
    if options.grasp:
        names.insert(0, "grasp")
    if options.cases:
        names = [name for name in names if name in options.cases]
    return names


def _restart(d, reference, name, step):
    """Put the state that follows step ``step - 1`` of the reference trajectory into ``d``."""
    if reference is None or step == 0:
        return
    for field, source in (
        ("qpos", "qpos"),
        ("qvel", "qvel"),
        ("act", "act"),
        ("time", "time"),
        ("qacc_warmstart", "qacc"),
    ):
        if getattr(d, field).size:
            getattr(d, field).assign(reference[f"{name}/{source}"][step - 1])


def run(options):
    import warp as wp

    wp.config.log_level = wp.LOG_WARNING
    wp.set_device(options.device)
    reference = np.load(options.reference) if options.reference else None
    result = {"step": np.array(options.step)}
    for name in _names(options):
        mjm, m, d, steps = _device_model(name, options.grasp)
        step_once = _stepper(options.step, m, d)
        steps = options.steps or steps
        rows = {field: [] for field in STEP_FIELDS}
        captured = name == "grasp" and options.graph and wp.get_device().is_cuda
        if captured:
            # The coupled solver replays its microsteps from a CUDA graph, captured after one eager step.
            _needle_wrench(mjm, d, 0)
            step_once()
            _record(d, rows)
            replayed = 1 if reference is not None else 4
            with wp.ScopedCapture() as capture:
                for _ in range(replayed):
                    step_once()
            for step in range(1, steps - replayed + 1, replayed):
                _restart(d, reference, name, step)
                _needle_wrench(mjm, d, step // 4)
                wp.capture_launch(capture.graph)
                _record(d, rows)
        else:
            for step in range(steps):
                _restart(d, reference, name, step)
                if name == "grasp" and (step % 4 == 0 or reference is not None):
                    _needle_wrench(mjm, d, step // 4)
                step_once()
                _record(d, rows)
        for field in STEP_FIELDS:
            result[f"{name}/{field}"] = np.stack(rows[field])
        for field in LAST_FIELDS:
            result[f"{name}/last_{field}"] = getattr(d, field).numpy().copy()
        result[f"{name}/overflow"] = d.overflow.numpy().copy()
        niter = result[f"{name}/solver_niter"]
        print(
            f"{name:36s} steps {len(niter):4d}  nefc max {int(result[f'{name}/nefc'].max()):4d}  "
            f"solver iterations mean {float(niter.mean()):.2f} max {int(niter.max())}  "
            f"finite {bool(np.isfinite(result[f'{name}/qpos']).all())}"
        )
    np.savez(options.out, **result)


def compare(options):
    first, second = np.load(options.first), np.load(options.second)
    print(f"A {first['step']} step\nB {second['step']} step")
    worst, identical = 0.0, 0
    for key in first.files:
        if key == "step":
            continue
        a, b = first[key], second[key]
        if key.endswith("/efc_force"):
            # constraint rows take their slots from a race on CUDA; compare each step's forces as a set
            a, b = np.sort(a, axis=-1, kind="stable"), np.sort(b, axis=-1, kind="stable")
        if a.shape != b.shape:
            print(f"{key:52s} SHAPE {a.shape} != {b.shape}")
            worst = float("inf")
            continue
        if a.tobytes() == b.tobytes():
            identical += 1
            if options.all:
                print(f"{key:52s} bitwise identical")
            continue
        a64, b64 = a.astype(np.float64), b.astype(np.float64)
        difference = np.abs(a64 - b64)
        # Relative to the largest entry of the same array and step: entries
        # that cancel to zero must not be compared against themselves.
        axes = tuple(range(1, a.ndim)) if a.ndim > 1 else None
        scale = np.maximum(np.abs(a64), np.abs(b64)).max(axis=axes, keepdims=a.ndim > 1)
        relative = np.divide(difference, scale, out=np.zeros_like(difference), where=scale > 0)
        changed = np.argwhere(difference > 0)
        if not len(changed):
            # sorted rows may order -0.0 and +0.0 differently
            identical += 1
            continue
        worst = max(worst, float(relative.max()))
        # the median over steps separates the usual rounding difference from the rare large one
        steps = relative.reshape(len(relative), -1).max(axis=1) if a.ndim > 1 else relative
        print(
            f"{key:52s} max abs {difference.max():.3e}  max rel {relative.max():.3e}  "
            f"median step rel {np.median(steps):.3e}  steps differing {int((steps > 0).sum())}/{len(steps)}  "
            f"first differing index {tuple(int(i) for i in changed[0])}"
        )
    print(f"{identical} arrays bitwise identical; largest relative difference {worst:.3e}")


def census(options):
    """Count the kernel launches and memory operations of one step of each kind, by kernel."""
    import warp as wp
    from _warp_step_support import activities

    wp.config.log_level = wp.LOG_WARNING
    wp.set_device(options.device)
    if not wp.get_device().is_cuda:
        raise SystemExit("census reads Warp's activity timing, which reports CUDA devices only")
    iterations = 3
    for name in _names(options):
        counts = {}
        for kind in ("stock", "rigid"):
            for count in (0, iterations):
                _, m, d, _ = _device_model(name, options.grasp)
                # a loop in Python issues exactly opt.iterations solver iterations
                m.opt.graph_conditional = False
                m.opt.iterations = count
                m.opt.warn_overflow = 0
                step_once = _stepper(kind, m, d)
                for _ in range(3):
                    step_once()
                kernels, memory = activities(step_once)
                counts[kind, count] = collections.Counter({f"kernel {key}": n for key, n in kernels.items()}) + memory
        for kind in ("stock", "rigid"):
            fixed, loop = counts[kind, 0], counts[kind, iterations] - counts[kind, 0]
            kernels = sum(value for key, value in fixed.items() if key.startswith("kernel"))
            print(
                f"{name} {kind}: fixed {kernels} kernels and {sum(fixed.values()) - kernels} memory operations, "
                f"per solver iteration {sum(loop.values()) / iterations:g}"
            )
            if options.verbose:
                for label, part, scale in (("fixed", fixed, 1), ("per solver iteration", loop, iterations)):
                    print(f"  {label}:")
                    for key, value in sorted(part.items()):
                        print(f"    {value / scale:6g}  {key}")
        if options.verbose:
            print("  whole step with 3 iterations, where the two kinds differ (stock -> rigid):")
            stock, rigid = counts["stock", iterations], counts["rigid", iterations]
            for key in sorted(set(stock) | set(rigid)):
                if stock[key] != rigid[key]:
                    print(f"    {stock[key]:3d} -> {rigid[key]:3d}  {key}")


def dump_grasp(options):
    """Save the grasp scene's MuJoCo model and a grasping state (native backend, CPU)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    os.environ["NEWTON_TEST_DEVICE"] = "cpu"
    from test_grasp_scene import _scene

    out = Path(options.out)
    out.mkdir(parents=True, exist_ok=True)
    with _scene() as (scene, _):
        for _ in range(options.frames):
            scene.step()
        solver = scene.rigid_solver
        mjm, mjd = solver.mj_model, solver.mj_data
        # The GPU backend integrates 4 microsteps per coupled substep of a 60 Hz frame with 4 substeps.
        mjm.opt.timestep = 1.0 / 960.0
        mujoco.mj_saveModel(mjm, str(out / "grasp.mjb"), None)
        state = {field: np.array(getattr(mjd, field)) for field in STATE}
        np.savez(out / "grasp_state.npz", time=mjd.time, steps=400, **state)
        depth = np.zeros(mjm.nbody, dtype=int)
        for body in range(1, mjm.nbody):
            depth[body] = depth[mjm.body_parentid[body]] + 1
        print(
            f"grasp model: nbody {mjm.nbody} nv {mjm.nv} nu {mjm.nu} na {mjm.na} ntendon {mjm.ntendon} neq {mjm.neq} "
            f"ngeom {mjm.ngeom} levels {depth.max() + 1} widest level {np.bincount(depth).max()} "
            f"ncon {mjd.ncon} nefc {mjd.nefc} integrator {mjm.opt.integrator} cone {mjm.opt.cone} "
            f"solver {mjm.opt.solver} impratio {mjm.opt.impratio}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "census"):
        command = commands.add_parser(name)
        command.add_argument("--device", default="cuda:0" if name == "census" else "cpu")
        command.add_argument("--grasp", default=None, help="directory written by dump-grasp")
        command.add_argument("--cases", nargs="*", default=None)
        if name == "run":
            command.add_argument("--step", choices=("stock", "rigid"), required=True, help="the step to roll out")
            command.add_argument("--out", required=True)
            command.add_argument("--steps", type=int, default=0)
            command.add_argument("--graph", action="store_true", help="replay the grasp case from a CUDA graph")
            command.add_argument("--reference", default=None, help="CPU archive whose states restart every step")
        else:
            command.add_argument("--verbose", action="store_true", help="list the counts by kernel")
    command = commands.add_parser("compare")
    command.add_argument("first")
    command.add_argument("second")
    command.add_argument("--all", action="store_true", help="also list the arrays that are bitwise identical")
    command = commands.add_parser("dump-grasp")
    command.add_argument("--out", required=True)
    command.add_argument("--frames", type=int, default=12)
    options = parser.parse_args()
    if options.command == "run" and options.graph and options.step != "rigid":
        parser.error("--graph replays the rigid step: use --step rigid")
    handlers = {"run": run, "compare": compare, "census": census, "dump-grasp": dump_grasp}
    handlers[options.command](options)


if __name__ == "__main__":
    main()
