"""Offline validation for the Yurameki C++/OpenMP core.

Run with the exact Python ABI used to build the module, for example:

    <Blender Python 3.13> tools/validate_native.py
"""

from __future__ import annotations

import os
import sys

import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NATIVE_DIR = os.environ.get("YURAMEKI_NATIVE_DIR", ROOT)
sys.path.insert(0, NATIVE_DIR)

import _yurameki_native_0_3_0 as native  # noqa: E402


EMPTY_VERTICES = np.empty((0, 3), dtype=np.float32)
EMPTY_TRIANGLES = np.empty((0, 3), dtype=np.int32)


def strand_lengths(points: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.diff(points, axis=1), axis=2)


def max_joint_angle(points: np.ndarray) -> float:
    edges = np.diff(points, axis=1)
    directions = edges / np.maximum(np.linalg.norm(edges, axis=2, keepdims=True), 1.0e-12)
    cosine = np.sum(directions[:, :-1] * directions[:, 1:], axis=2)
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))).max(initial=0.0))


def make_straight(strands: int = 64, points_per_strand: int = 12) -> np.ndarray:
    result = np.empty((strands, points_per_strand, 3), dtype=np.float32)
    for strand in range(strands):
        result[strand, :, 0] = (strand % 8) * 0.01
        result[strand, :, 1] = (strand // 8) * 0.01
        result[strand, :, 2] = -np.arange(points_per_strand, dtype=np.float32) * 0.02
    return result


def run_frame(simulator, target, *, threads: int = 0, gravity=(0.0, 0.0, 0.0)):
    return simulator.frame(
        target, EMPTY_VERTICES, EMPTY_TRIANGLES, EMPTY_VERTICES, EMPTY_TRIANGLES,
        1.0 / 24.0,
        {"gravity": np.asarray(gravity, dtype=np.float32), "openmp_threads": threads},
    )


def main() -> None:
    print(f"native={native.__version__} openmp={native.openmp_enabled()} max_threads={native.max_threads()}")

    print("[1] exact rest-state stability")
    rest = make_straight()
    simulator = native.Simulator(rest, rest, rest.shape[1], 3, 0.003, 1)
    result = run_frame(simulator, rest, threads=4)
    error = float(np.max(np.abs(result["positions"] - rest)))
    assert result["stats"]["repaired_strands"] == 0, result["stats"]
    assert error < 2.0e-6, error
    print(f"    max position error={error:.3e} m")

    print("[2] frame-1 rod lengths remain the constitutive lengths")
    initial = rest.copy()
    initial[:, 3:, 2] -= 0.01
    simulator = native.Simulator(initial, rest, rest.shape[1], 3, 0.003, 1)
    audit = simulator.audit(initial, EMPTY_VERTICES, EMPTY_TRIANGLES, EMPTY_VERTICES, EMPTY_TRIANGLES)
    assert audit["length_bad_before"] == rest.shape[0], audit
    assert audit["length_bad_rods_before"] == rest.shape[0], audit
    result = run_frame(simulator, initial, threads=4)
    length_error = float(np.max(np.abs(strand_lengths(result["positions"]) - strand_lengths(rest))))
    assert length_error < 2.0e-6, length_error
    print(f"    max rod-length error={length_error * 1000.0:.6f} mm")

    print("[3] folded input is groomed with a bounded convergence loop")
    one_rest = make_straight(1)
    folded = one_rest.copy()
    folded[0, 6:, 2] = folded[0, 5, 2] + np.arange(1, 7, dtype=np.float32) * 0.02
    simulator = native.Simulator(one_rest, one_rest, one_rest.shape[1], 3, 0.003, 1)
    result = simulator.settle_external(folded, {"openmp_threads": 2})
    angle = max_joint_angle(result["positions"])
    assert result["stats"]["repaired_strands"] == 1, result["stats"]
    assert result["stats"]["max_settle_iterations"] <= 12, result["stats"]
    assert np.isfinite(result["positions"]).all()
    assert angle <= 90.01, angle
    print(f"    max angle={angle:.3f} deg, stats={result['stats']}")

    print("[4] deep Body penetration is found beyond the search radius")
    plane_vertices = np.array(((-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0)), dtype=np.float32)
    plane_triangles = np.array(((0, 1, 2), (0, 2, 3)), dtype=np.int32)
    above = np.zeros((1, 12, 3), dtype=np.float32)
    above[0, :, 0] = np.linspace(-0.2, 0.2, 12, dtype=np.float32)
    above[0, :, 2] = 0.01
    simulator = native.Simulator(above, above, 12, 1, 0.003, 1)
    simulator.frame(
        above, plane_vertices, plane_triangles, EMPTY_VERTICES, EMPTY_TRIANGLES,
        1.0 / 24.0, {"gravity": np.zeros(3, dtype=np.float32)},
    )
    penetrating = above.copy()
    penetrating[0, :, 2] = np.linspace(0.01, -0.05, 12, dtype=np.float32)
    audit = simulator.audit(
        penetrating, plane_vertices, plane_triangles, EMPTY_VERTICES, EMPTY_TRIANGLES,
        {"keep_length": False, "collision_margin": 0.001, "collision_search": 0.002},
    )
    assert audit["collision_bad_before"] == 1 and audit["repaired_strands"] == 1, audit
    result = simulator.settle_external(
        penetrating,
        {
            "keep_length": False, "collision_margin": 0.001,
            "collision_search": 0.002, "collision_max_correction": 0.1,
            "settle_relaxation": 1.0, "settle_iterations": 24,
            "angle_change_limit_deg": 180.0, "fold_limit_deg": 180.0,
            "collision_smooth_passes": 0,
        },
    )
    assert result["stats"]["collision_bad_before"] == 1, result["stats"]
    assert result["stats"]["collision_bad_after"] == 0, result["stats"]
    assert float(result["positions"][0, 1:, 2].min()) >= 0.00099
    print(f"    minimum repaired height={result['positions'][0, 1:, 2].min() * 1000.0:.3f} mm")

    print("[5] gravity cantilever stays planar, smooth, and inextensible")
    cantilever = np.zeros((1, 12, 3), dtype=np.float32)
    cantilever[0, :, 0] = np.arange(12, dtype=np.float32) * 0.02
    simulator = native.Simulator(cantilever, cantilever, 12, 3, 0.003, 1)
    for _frame in range(24):
        result = run_frame(simulator, cantilever, threads=4, gravity=(0.0, 0.0, -9.81))
    tip = result["positions"][0, -1]
    length_error = float(np.max(np.abs(strand_lengths(result["positions"]) - 0.02)))
    angle = max_joint_angle(result["positions"])
    assert abs(float(tip[1])) < 1.0e-7, tip
    assert float(tip[0]) > 0.1 and float(tip[2]) < -0.02, tip
    assert angle <= 30.01, angle
    assert length_error < 2.0e-6, length_error
    assert result["stats"]["failed_strands"] == 0, result["stats"]
    print(f"    tip={tip.tolist()}, max angle={angle:.3f} deg")

    print("[6] OpenMP strand parallelism is deterministic")
    target = rest.copy()
    target[:, 3:, 0] += np.linspace(0.0, 0.03, target.shape[1] - 3, dtype=np.float32)
    simulator_one = native.Simulator(rest, rest, rest.shape[1], 3, 0.003, 1)
    simulator_many = native.Simulator(rest, rest, rest.shape[1], 3, 0.003, 1)
    one = run_frame(simulator_one, target, threads=1, gravity=(0.0, 0.0, -9.81))["positions"]
    many = run_frame(simulator_many, target, threads=4, gravity=(0.0, 0.0, -9.81))["positions"]
    difference = float(np.max(np.abs(one - many)))
    assert difference < 1.0e-7, difference
    print(f"    max 1-thread/4-thread difference={difference:.3e} m")

    print("[7] malformed native inputs fail before corrupting state")
    try:
        simulator_one.frame(
            rest, plane_vertices, np.array(((0, 1, 99),), dtype=np.int32),
            EMPTY_VERTICES, EMPTY_TRIANGLES, 1.0 / 24.0,
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("out-of-range triangle index was accepted")
    try:
        run_frame(simulator_one, rest, gravity=(0.0, 0.0, float("nan")))
    except ValueError:
        pass
    else:
        raise AssertionError("NaN parameter was accepted")
    try:
        simulator_one.frame(
            rest, EMPTY_VERTICES, EMPTY_TRIANGLES, EMPTY_VERTICES, EMPTY_TRIANGLES,
            0.0,
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("zero dt_frame was accepted")
    print("    invalid index, NaN, and zero dt rejected")

    print("PASS")


if __name__ == "__main__":
    main()
