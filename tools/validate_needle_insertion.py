# SPDX-License-Identifier: Apache-2.0
"""Repeat insertion refinement or regional needle-handling robustness scenarios."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import warp as wp
from newton.viewer import ViewerNull

import newton_solver_suturebot
from newton_solver_suturebot.examples.needle.example_fem_needle_insertion import Example

REFINEMENT_CASES = ("timestep", "spacing")
ROBUSTNESS_SCENARIOS = ("continuous", "gap", "lift", "lower", "abrupt-up", "abrupt-down", "region2", "region3")
# Prepared undeformed surface picks; each region is simulated separately.
REGION_PAIRS = {
    1: ((0.030, -0.040), (0.038, -0.040)),
    2: ((-0.029622442569402822, -0.032509430292103556), (-0.02396558831991044, -0.03816628454159593)),
    3: ((0.03248692747859243, -0.015903875169191804), (0.040486927478592435, -0.015903875169191804)),
}


def case_arguments(name, arguments=(), *, region=1, substeps=None):
    entry_xy, exit_xy = REGION_PAIRS[region]
    if name == "gap":
        exit_xy = (0.030, -0.048)
    args = Example.create_parser().parse_args(
        [
            "--regions",
            str(region),
            "--entry-xy",
            *map(str, entry_xy),
            "--exit-xy",
            *map(str, exit_xy),
            "--start",
            "--substeps",
            "32",
            *arguments,
        ]
    )
    if substeps is not None:
        args.substeps = substeps
    # Refine the effective user-specified baseline, not defaults that later
    # forwarded arguments could silently override in both cases.
    if name == "timestep":
        args.substeps *= 2
    elif name == "spacing":
        args.track_spacing *= 0.5
        args.shaft_spacing *= 0.5
    return args


def _snapshot(example, frame, phase):
    task = example.report()
    return {
        "frame": frame,
        "phase": phase,
        "time_s": example.elapsed,
        "state_evaluation": "accepted_state",
        "force_evaluation": (
            "rejected_coupling_trial" if "coupling did not converge" in task.get("outcome", "") else "accepted_endpoint"
        ),
        "status": example.interaction.status,
        "task_status": task.get("task_status"),
        "success": task.get("success"),
        **{
            key: task.get(key)
            for key in ("entry_count", "exit_count", "puncture_count", "track_section_count", "track_length_m")
        },
        "track_count": example.interaction.track_count,
        "tip_m": example.actual_tip().tolist(),
        "force_N": example.interaction.interface_force.tolist(),
        "entry_m": example.entry.position(example.state_0.particle_q).tolist(),
        "exit_m": example.exit.position(example.state_0.particle_q).tolist(),
        "deformation_m": task.get("current_deformation_m"),
        "engaged_track_length_m": task.get("engaged_track_length_m"),
    }


def run_case(name, *, device, arguments=()):
    args = case_arguments(name, arguments)
    trace = []
    with wp.ScopedDevice(device):
        viewer = ViewerNull()
        example, result, latest = None, {}, None
        try:
            example = Example(viewer, args)
            previous = None
            for frame in range(420):
                example.step()
                status = example.interaction.status
                latest = _snapshot(example, frame, "automatic")
                if status != previous or frame % 30 == 0 or not example.running:
                    trace.append(latest)
                    print(name, json.dumps(latest), flush=True)
                example.test_final()
                previous = status
                if not example.running:
                    break
            result = example.report()
        except Exception as error:
            # Numerical failures are evidence: retain the partial run and let
            # the CLI write every case before returning a failing exit status.
            result = {} if example is None else example.report()
            result.update(success=False, validation_error=f"{type(error).__name__}: {error}")
            if latest is not None and (not trace or trace[-1] is not latest):
                trace.append(latest)
        finally:
            viewer.close()
    return result | {
        "trace": trace,
        "selection": {"regions": args.regions, "entry_xy_m": args.entry_xy, "exit_xy_m": args.exit_xy},
    }


def handling_phases(scenario, insertion):
    """60 Hz slider inputs; interpolation is a test gesture, not a drive limiter."""
    if scenario.startswith("abrupt"):
        height = 0.002 if scenario == "abrupt-up" else -0.002
        return [("click_hold", 0.2, height, None, True), ("return_hold", 0.2, 0.0, None, True)]
    height = 0.001 if scenario == "lift" else -0.001
    travel_time = max(1.5, insertion / 0.003)
    return [
        ("drag", 1.0, height, None, False),
        ("loaded_hold", 0.2, None, None, False),
        ("return", 1.0, 0.0, None, False),
        ("recovery", 0.2, None, None, False),
        ("withdraw", travel_time, None, 0.0, False),
        ("outside_hold", 0.2, None, None, False),
        ("reenter", travel_time, None, insertion, False),
        ("reentry_hold", 0.2, None, None, False),
    ]


def run_robustness_case(scenario, substeps, *, device, arguments=()):
    region = int(scenario[-1]) if scenario.startswith("region") else 1
    automatic = scenario in ("continuous", "gap", "region2", "region3")
    args = case_arguments("gap" if scenario == "gap" else "base", arguments, region=region, substeps=substeps)
    trace, phases, attempts, checks = [], {}, [], {}
    example, result, phase = None, {}, "automatic"
    completed, error = False, None
    with wp.ScopedDevice(device):
        viewer = ViewerNull()
        try:
            example = Example(viewer, args)
            checks["selected_region_only"] = np.unique(example.selection.tetrahedron_region_ids).tolist() == [region]
            original_step = example.coupler.step

            def observe(state_in, state_out, control, dt, **kwargs):
                before = example.coupler.evaluations
                record = {"phase": phase, "start_time_s": example.elapsed, "dt_s": dt, "accepted": False}
                try:
                    original_step(state_in, state_out, control, dt, **kwargs)
                    record["accepted"] = True
                except Exception as failure:
                    record["error"] = f"{type(failure).__name__}: {failure}"
                    raise
                finally:
                    record["force_map_trials"] = example.coupler.evaluations - before
                    record["interface_residual_N"] = example.coupler.residual
                    alm = example.interaction.tunnel_alm
                    if alm is not None:
                        record["tunnel_normal_residual_m"] = alm.normal_error
                        record["tunnel_dual_force_residual_N"] = alm.dual_residual
                        record["dual_updates"] = alm.dual_updates
                    attempts.append(record)

            example.coupler.step = observe

            def advance(name, duration, lift=None, insertion=None, jump=False):
                nonlocal phase
                phase = name
                first = len(trace)
                initial = example.manual_target.copy()
                frames = int(round(duration / example.frame_dt))
                for index in range(frames):
                    fraction = 1.0 if jump else (index + 1) / frames
                    target = {}
                    for component, goal, key in ((0, insertion, "insertion"), (1, lift, "lift")):
                        if goal is not None:
                            target[key] = initial[component] + fraction * (goal - initial[component])
                    if target:
                        example.set_manual_target(**target)
                    example.step()
                    trace.append(_snapshot(example, len(trace), name))
                    example.test_final()
                    if not example.running:
                        raise RuntimeError(example.outcome)
                phases[name] = {
                    "start_time_s": trace[first]["time_s"] - example.frame_dt,
                    "end_time_s": example.elapsed,
                    "peak_deformation_m": max(item["deformation_m"] for item in trace[first:]),
                    "peak_force_N": max(float(np.linalg.norm(item["force_N"])) for item in trace[first:]),
                    "end": trace[-1],
                }

            if automatic:
                for frame in range(420):
                    example.step()
                    trace.append(_snapshot(example, frame, phase))
                    example.test_final()
                    if not example.running:
                        break
                completed = bool(example.report()["success"])
                checks["reached_selected_exit"] = completed
                if scenario == "gap":
                    for metric in ("entry_count", "exit_count", "puncture_count", "track_section_count"):
                        checks[f"gap_{metric}_at_least_two"] = example.report()[metric] >= 2
            else:
                advance("automatic_entry", 2.2)
                checks["embedded_before_handling"] = (
                    example.interaction.tip_inside and example.interaction.engaged_track_length > 0
                )
                if not checks["embedded_before_handling"]:
                    raise RuntimeError("The actual insertion did not establish an embedded material track")
                example.use_manual_control()
                advance("settle", 0.2)
                initial_deformation = example.current_deformation
                initial_entries = example.interaction.entry_count
                for item in handling_phases(scenario, float(example.manual_target[0])):
                    advance(*item)
                    if item[0] == "outside_hold":
                        checks["withdrawn_track_disengaged"] = example.interaction.engaged_track_length == 0
                        checks["withdrawn_force_released"] = (
                            np.linalg.norm(example.interaction.interface_force) <= example.coupler.absolute_tolerance
                        )
                loaded = phases["click_hold" if scenario.startswith("abrupt") else "loaded_hold"]
                # Detect a physical response, not a calibrated deformation amount.
                checks["loaded_tissue_response"] = loaded["peak_deformation_m"] > initial_deformation + 1e-7
                if not scenario.startswith("abrupt"):
                    checks["reentered_material"] = (
                        example.interaction.entry_count > initial_entries
                        and example.interaction.engaged_track_length > 0
                    )
                completed = True
            checks["finite_fixed_uninverted"] = True
            result = example.report()
        except Exception as failure:
            error = f"{type(failure).__name__}: {failure}"
            result = {} if example is None else example.report()
            completed = False
        finally:
            viewer.close()
    checks["completed"] = completed
    return {
        "scenario": scenario,
        "nominal_substeps": substeps,
        "selection": {"regions": args.regions, "entry_xy_m": args.entry_xy, "exit_xy_m": args.exit_xy},
        "completed": completed,
        "validation_error": error,
        "checks": {key: bool(value) for key, value in checks.items()},
        "report": result,
        "trace": trace,
        "phases": phases,
        "attempts": attempts,
        "force_map_trials": sum(item["force_map_trials"] for item in attempts),
        "scope": "Live physics with scripted public slider inputs; nominal substeps permit adaptive refinement. Integrity checks cover completed frames and the final partial frame, not every accepted substep. Frame traces are observations, not force-history convergence or interactive FPS evidence. Loaded response means deformation increases by more than0.1 micrometre, a detection threshold rather than calibration.",
    }


def build_robustness_report(results):
    return {
        "suite": "robustness",
        "cases": results,
        "checks": {
            f"{name}:{check}": bool(value)
            for name, result in results.items()
            for check, value in result["checks"].items()
        },
        "force_law_assessment": {"convergence_established": False},
    }


def compare(results):
    """Declared engineering checks, not physiological acceptance criteria."""
    checks = {"all_reached_exit": all(item["success"] for item in results.values())}
    if "gap" in results:
        # This fresh two-material scenario must actually exit and re-enter. Physics has no planned count input.
        for metric in ("entry_count", "exit_count", "puncture_count", "track_section_count"):
            count = results["gap"].get(metric)
            checks[f"gap_{metric}_at_least_two"] = count is not None and count >= 2
    if "base" not in results:
        return checks
    base = results["base"]
    for name, result in results.items():
        if name not in REFINEMENT_CASES:
            continue
        checks[f"{name}_exit_change_under_0.2mm"] = (
            result.get("exit_error_m") is not None
            and base.get("exit_error_m") is not None
            and abs(result["exit_error_m"] - base["exit_error_m"]) <= 0.0002
        )
        for metric in ("maximum_interface_force_N", "maximum_deformation_m"):
            checks[f"{name}_{metric}_change_under_20percent"] = (
                result.get(metric) is not None
                and base.get(metric) is not None
                and abs(result[metric] - base[metric]) <= 0.2 * max(base[metric], 1e-12)
            )
    return checks


def _exit_vector(result):
    """Recover the measured tip-to-deformed-target vector from an existing run."""
    if not _successful_exit(result) or not result.get("trace"):
        return None
    final = result["trace"][-1]
    if "tip_m" not in final or "exit_m" not in final:
        return None
    vector = np.asarray(final["tip_m"], dtype=float) - np.asarray(final["exit_m"], dtype=float)
    return vector if vector.shape == (3,) and np.isfinite(vector).all() else None


def _successful_exit(result):
    # Old saved baseline reports used a terminal physical state; new code keeps task semantics in the controller.
    return bool(result.get("success", result.get("status") == "emerged"))


def _post_entry_trace(result):
    trace = result.get("trace", [])
    first_entry = next((index for index, item in enumerate(trace) if item.get("status") == "inserted"), None)
    if first_entry is None:
        return None
    samples = [
        (item["time_s"], item["force_N"])
        for item in trace[first_entry:]
        if "time_s" in item and "force_N" in item and item.get("force_evaluation") != "rejected_coupling_trial"
    ]
    if len(samples) < 2:
        return None
    times = np.asarray([sample[0] for sample in samples], dtype=float)
    forces = np.asarray([sample[1] for sample in samples], dtype=float)
    if (
        forces.shape != (len(times), 3)
        or not np.isfinite(times).all()
        or not np.isfinite(forces).all()
        or np.any(np.diff(times) <= 0)
    ):
        return None
    return times, forces


def _compare_post_entry_force_trace(result, base):
    """Compare sparse saved force vectors without aligning away event-time changes."""
    refined_trace, base_trace = _post_entry_trace(result), _post_entry_trace(base)
    if refined_trace is None or base_trace is None:
        return None
    times, forces = refined_trace
    base_times, base_forces = base_trace
    start, end = max(times[0], base_times[0]), min(times[-1], base_times[-1])
    if end <= start:
        return None
    knots = np.unique(np.concatenate(([start, end], times, base_times)))
    knots = knots[(knots >= start) & (knots <= end)]
    observed = np.column_stack([np.interp(knots, times, forces[:, axis]) for axis in range(3)])
    reference = np.column_stack([np.interp(knots, base_times, base_forces[:, axis]) for axis in range(3)])
    difference = observed - reference

    def rms(values):
        # Exact integral for the piecewise-linear interpolant, not the unsampled physical trajectory.
        integrals = np.diff(knots) / 3 * np.sum(values[:-1] ** 2 + values[:-1] * values[1:] + values[1:] ** 2, axis=1)
        return float(np.sqrt(np.sum(integrals) / (end - start)))

    difference_rms, reference_rms = rms(difference), rms(reference)
    return {
        "shared_time_interval_s": [float(start), float(end)],
        "base_saved_sample_count": len(base_times),
        "refined_saved_sample_count": len(times),
        "interpolation_knot_count": len(knots),
        "vector_difference_rms_N": difference_rms,
        "base_vector_rms_N": reference_rms,
        "vector_difference_relative_rms": None if reference_rms == 0 else difference_rms / reference_rms,
        "maximum_interpolated_vector_difference_N": float(np.max(np.linalg.norm(difference, axis=1))),
        "scope": (
            "Piecewise-linear interpolation of sparse snapshots after first observed entry over shared physical "
            "time; no time warping or acceptance threshold. Unresolved substep transients and the "
            "entry transition before both first inserted samples are excluded. Rejected coupling-trial "
            "force diagnostics are excluded."
        ),
    }


def build_report(results):
    """Assess saved cases without rerunning physics or conflating task and force accuracy.

    Missing evidence remains null instead of being treated as a passing check.
    """
    task_checks = compare(results)
    case_observations, refinements, balance_checks = {}, {}, {}
    base = results.get("base")
    base_vector = None if base is None else _exit_vector(base)
    for name, result in results.items():
        vector = _exit_vector(result)
        case_observations[name] = {
            "final_exit_error_vector_m": None if vector is None else vector.tolist(),
        }
        for metric, bound, units in (
            ("maximum_force_balance_error_N", 1.0e-8, "N"),
            ("maximum_torque_balance_error_Nm", 1.0e-9, "Nm"),
        ):
            value = result.get(metric)
            balance_checks[f"{name}_{metric}_under_{bound:g}{units}"] = None if value is None else value <= bound
        if base is None or name not in REFINEMENT_CASES:
            continue
        vector_delta = None if vector is None or base_vector is None else float(np.linalg.norm(vector - base_vector))
        refinements[name] = {
            "exit_error_vector_change_m": vector_delta,
            "emergence_time_change_s": (
                result["physical_time_s"] - base["physical_time_s"]
                if _successful_exit(result) and _successful_exit(base)
                else None
            ),
            "post_entry_force_trace_comparison": _compare_post_entry_force_trace(result, base),
        }
        task_checks[f"{name}_exit_vector_change_under_0.2mm"] = vector_delta is not None and vector_delta <= 0.0002
    return {
        "cases": results,
        "checks": task_checks,  # Preserve the original report key for downstream readers.
        "task_checks": task_checks,
        "coupling_balance_checks": balance_checks,
        "numerical_observations": {"cases": case_observations, "refinements": refinements},
        "force_law_assessment": {
            "convergence_established": False,
            "scope": (
                "Task checks assess the selected insertion scenarios. Force/moment residuals assess endpoint "
                "force-law scatter, not every finite-tolerance applied iterate. Applied-load balance is "
                "covered independently by coupler regression tests. Sparse post-entry force-vector "
                "comparisons are observations without an "
                "acceptance threshold. Passing task checks does not establish convergence of full "
                "force histories or coupled energy."
            ),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--suite", choices=("refinement", "robustness"), default="refinement")
    parser.add_argument("--case", choices=("base", "timestep", "spacing", "gap", "all"), default="all")
    parser.add_argument("--scenario", nargs="+", choices=ROBUSTNESS_SCENARIOS, default=list(ROBUSTNESS_SCENARIOS))
    parser.add_argument("--substep-counts", nargs="+", type=int, choices=(1, 4, 8), default=[1, 4, 8])
    parser.add_argument("--output", type=Path, required=True)
    args, extra = parser.parse_known_args()
    if extra[:1] == ["--"]:
        extra = extra[1:]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.suite == "robustness":
        forbidden = {"--substeps", "--regions", "--entry-xy", "--exit-xy", "--report"}
        if any(item.split("=", 1)[0] in forbidden for item in extra):
            parser.error(
                "Robustness geometry/timesteps use --scenario and --substep-counts; do not forward region, pick, substep or report overrides"
            )
        package = Path(newton_solver_suturebot.__file__).resolve().parent
        sources = {
            str(path.relative_to(package)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in package.rglob("*.py")
        }
        requested = [
            (scenario, count)
            for scenario in dict.fromkeys(args.scenario)
            for count in dict.fromkeys(args.substep_counts)
        ]
        results = {}
        for scenario, count in requested:
            name = f"{scenario}-h{count}"
            results[name] = run_robustness_case(scenario, count, device=args.device, arguments=extra)
            report = build_robustness_report(results)
            report.update(
                requested_cases=[f"{scenario}-h{count}" for scenario, count in requested],
                source_sha256=sources,
                validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            )
            report["checks"]["all_cases_recorded"] = len(results) == len(requested)
            report["checks"]["runtime_source_unchanged"] = all(
                hashlib.sha256((package / name).read_bytes()).hexdigest() == digest for name, digest in sources.items()
            )
            temporary = args.output.with_suffix(args.output.suffix + ".tmp")
            temporary.write_text(json.dumps(report, indent=2) + "\n")
            temporary.replace(args.output)
            print(name, json.dumps(results[name]["checks"]), flush=True)
        if not all(report["checks"].values()):
            raise SystemExit("Needle handling robustness failed; inspect the saved partial traces and checks")
        return
    cases = ("base", "timestep", "spacing", "gap") if args.case == "all" else (args.case,)
    results = {case: run_case(case, device=args.device, arguments=extra) for case in cases}
    report = build_report(results)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["task_checks"], indent=2))
    print(json.dumps(report["force_law_assessment"], indent=2))
    if not all(report["task_checks"].values()) or not all(report["coupling_balance_checks"].values()):
        raise SystemExit("Needle insertion validation failed or lacks required evidence; inspect the saved report")


if __name__ == "__main__":
    main()
