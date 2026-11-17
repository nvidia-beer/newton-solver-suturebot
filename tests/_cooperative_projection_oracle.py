"""Independent float64 evaluation of the authored-polyline mathematical map.

No quaternion renormalization: match Warp's unit-quaternion convention on the
actual stored FP32 components. Rounded prefix sums are separately reported.
The conservative error scale is a conditioning diagnostic, not a physical
convergence tolerance or a formal compiler interval proof.
"""

import numpy as np


def rotate(q, x):
    v, w = q[:3], q[3]
    return x * (2 * w * w - 1) + v * (2 * np.dot(v, x)) + np.cross(v, x) * (2 * w)


def evaluate(samples, point, pose, radius, *, segment_override=None):
    samples = np.asarray(samples, dtype=np.float64)
    point = np.asarray(point, dtype=np.float64)
    pose = np.asarray(pose, dtype=np.float64)
    qi = pose[3:].copy()
    qi[:3] *= -1
    local = rotate(qi, point) - rotate(qi, pose[:3])
    direction = np.diff(samples, axis=0)
    lengths = np.linalg.norm(direction, axis=1)
    prefix = np.r_[0, np.cumsum(lengths)]
    fractions = np.einsum("ij,ij->i", local - samples[:-1], direction) / np.maximum(
        np.einsum("ij,ij->i", direction, direction), 1e-20
    )
    candidates = samples[:-1] + np.clip(fractions, 0, 1)[:, None] * direction
    distances = np.einsum("ij,ij->i", candidates - local, candidates - local)
    exact_winner = int(np.argmin(distances))
    winner = exact_winner if segment_override is None else int(segment_override)
    fraction = fractions[winner]
    arc = prefix[winner] + np.clip(fraction, 0, 1) * lengths[winner]
    before = max(0, arc - radius)
    after = min(prefix[-1], arc + radius)

    def arc_point(value):
        selected = np.flatnonzero((prefix[:-1] <= value) & (value <= prefix[:-1] + lengths))
        if not len(selected):
            return np.zeros(3), -1
        i = int(selected[-1])
        return samples[i] + ((value - prefix[i]) / lengths[i]) * direction[i], i

    lo, loi = arc_point(before)
    hi, hii = arc_point(after)
    secant = hi - lo
    use_secant = np.dot(secant, secant) > 1e-20
    tangent = secant / np.linalg.norm(secant) if use_secant else direction[winner] / lengths[winner]
    world = rotate(pose[3:], candidates[winner]) + pose[:3]
    tangent = rotate(pose[3:], tangent)
    covered = not ((winner == 0 and fraction < 0) or (winner == len(lengths) - 1 and fraction > 1))
    displacement = world - point
    displacement -= np.dot(displacement, tangent) * tangent
    magnitude = np.linalg.norm(displacement)
    # Constants are the exact FP32 values sent to the original test kernel.
    clearance = float(np.float32(0.00002))
    compliance = float(np.float32(0.001))
    reaction = (1 - clearance / magnitude) * displacement / compliance if magnitude > clearance else np.zeros(3)
    output = np.r_[world, tangent, distances[winner], float(covered), reaction]
    branch = np.array(
        [
            winner,
            int(fraction < 0) - int(fraction > 1),
            loi,
            hii,
            int(use_secant),
            int(covered),
        ]
    )
    # Error estimates separate absolute scale from near-zero relative error.
    # A quaternion rotation has fewer than64 rounded scalar operations.
    # Prefix accumulation adds one rounded sum per authored segment. Endpoint
    # error is amplified by normalization's 2/||secant|| condition bound.
    u = 2**-24

    def gamma(n):
        return n * u / (1 - n * u)

    scale = max(np.max(np.abs(point)), np.max(np.abs(pose[:3])), np.max(np.abs(samples)), 1e-30)
    local_error = gamma(96) * scale
    closest_error = 4 * local_error + gamma(64) * scale
    arc_error = 2 * local_error + gamma(len(lengths) + 12) * prefix[-1]
    endpoint_error = 4 * (local_error + arc_error) + gamma(16) * scale
    tangent_length = np.linalg.norm(secant) if use_secant else lengths[winner]
    tangent_error = (4 * endpoint_error / max(tangent_length, 1e-30)) + gamma(64)
    displacement_error = closest_error + (2 * np.linalg.norm(tangent) + tangent_error) * tangent_error * np.linalg.norm(
        world - point
    )
    reaction_error = displacement_error / compliance + gamma(32) * np.linalg.norm(reaction)
    estimate = np.r_[
        np.full(3, closest_error),
        np.full(3, tangent_error),
        2 * np.linalg.norm(candidates[winner] - local) * (2 * local_error) + 4 * local_error**2,
        0,
        np.full(3, reaction_error),
    ]
    return (
        output,
        branch,
        estimate,
        {
            "secant_length": float(tangent_length),
            "prefix_length": float(prefix[-1]),
            "exact_nearest_segment": exact_winner,
            "evaluated_segment": winner,
            "nearest_distance_gap": float(distances[winner] - distances[exact_winner]),
            "nearest_distance_fp32_ulp": float(np.spacing(np.float32(distances[exact_winner]))),
        },
    )
