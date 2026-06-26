"""CUDA XPBD solver whose state is shared directly with Warp collision."""
from __future__ import annotations

import numpy as np
import warp as wp


@wp.kernel
def _advect_by_root_motion(
    pos: wp.array(dtype=wp.vec3),
    roots: wp.array(dtype=wp.vec3),
    points_per_strand: int,
    tip_weight: float,
):
    strand = wp.tid()
    base = strand * points_per_strand
    root_delta = roots[strand] - pos[base]
    denom = float(points_per_strand - 3)
    point = int(2)
    while point < points_per_strand:
        if denom > 0.0:
            t = float(points_per_strand - 1 - point) / denom
        else:
            t = 0.0
        weight = tip_weight + (1.0 - tip_weight) * t
        i = base + point
        pos[i] = pos[i] + root_delta * weight
        point += 1


@wp.kernel
def _predict(
    pos: wp.array(dtype=wp.vec3),
    vel: wp.array(dtype=wp.vec3),
    predicted: wp.array(dtype=wp.vec3),
    roots: wp.array(dtype=wp.vec3),
    point1s: wp.array(dtype=wp.vec3),
    points_per_strand: int,
    dt: float,
    gravity_x: float,
    gravity_y: float,
    gravity_z: float,
):
    i = wp.tid()
    strand = i // points_per_strand
    point = i % points_per_strand
    if point == 0:
        pos[i] = roots[strand]
        predicted[i] = roots[strand]
    elif point == 1:
        pos[i] = point1s[strand]
        predicted[i] = point1s[strand]
    else:
        velocity = vel[i]
        velocity = velocity + wp.vec3(
            gravity_x * dt,
            gravity_y * dt,
            gravity_z * dt,
        )
        vel[i] = velocity
        predicted[i] = pos[i] + velocity * dt


@wp.kernel
def _solve_springs(
    predicted: wp.array(dtype=wp.vec3),
    inverse_mass: wp.array(dtype=float),
    segment_rest: wp.array(dtype=float),
    bending_rest: wp.array(dtype=float),
    points_per_strand: int,
    dt: float,
    segment_stiffness: float,
    root_bending_stiffness: float,
    bending_stiffness: float,
    bending_enabled: int,
):
    strand = wp.tid()
    base = strand * points_per_strand
    segment_base = strand * (points_per_strand - 1)
    bending_base = strand * (points_per_strand - 2)

    for k in range(points_per_strand - 1):
        i = base + k
        j = i + 1
        wi = inverse_mass[i]
        wj = inverse_mass[j]
        if wi + wj > 1.0e-10:
            delta = predicted[i] - predicted[j]
            distance = wp.length(delta)
            if distance > 1.0e-8:
                constraint = distance - segment_rest[segment_base + k]
                alpha = 1.0 / (segment_stiffness * dt * dt)
                delta_lambda = -constraint / (wi + wj + alpha)
                gradient = delta / distance
                predicted[i] = predicted[i] + wi * delta_lambda * gradient
                predicted[j] = predicted[j] - wj * delta_lambda * gradient

    if bending_enabled == 1:
        for k in range(points_per_strand - 2):
            i = base + k
            j = i + 2
            wi = inverse_mass[i]
            wj = inverse_mass[j]
            if wi + wj > 1.0e-10:
                delta = predicted[i] - predicted[j]
                distance = wp.length(delta)
                if distance > 1.0e-8:
                    stiffness = bending_stiffness
                    if k < 2:
                        stiffness = root_bending_stiffness
                    constraint = distance - bending_rest[bending_base + k]
                    alpha = 1.0 / (stiffness * dt * dt)
                    delta_lambda = -constraint / (wi + wj + alpha)
                    gradient = delta / distance
                    predicted[i] = predicted[i] + wi * delta_lambda * gradient
                    predicted[j] = predicted[j] - wj * delta_lambda * gradient


@wp.kernel
def _solve_angle_limits(
    predicted: wp.array(dtype=wp.vec3),
    inverse_mass: wp.array(dtype=float),
    angle_chord_min: wp.array(dtype=float),
    angle_chord_max: wp.array(dtype=float),
    points_per_strand: int,
    dt: float,
    stiffness: float,
    enabled: int,
):
    if enabled != 1:
        return

    strand = wp.tid()
    base = strand * points_per_strand
    angle_base = strand * (points_per_strand - 2)
    alpha = 1.0 / (stiffness * dt * dt)

    for k in range(points_per_strand - 2):
        i = base + k
        j = i + 2
        wi = inverse_mass[i]
        wj = inverse_mass[j]
        if wi + wj > 1.0e-10:
            delta = predicted[i] - predicted[j]
            distance = wp.length(delta)
            if distance > 1.0e-8:
                lower = angle_chord_min[angle_base + k]
                upper = angle_chord_max[angle_base + k]
                target = distance
                if distance < lower:
                    target = lower
                elif distance > upper:
                    target = upper

                constraint = distance - target
                if wp.abs(constraint) > 1.0e-8:
                    delta_lambda = -constraint / (wi + wj + alpha)
                    gradient = delta / distance
                    predicted[i] = predicted[i] + wi * delta_lambda * gradient
                    predicted[j] = predicted[j] - wj * delta_lambda * gradient


@wp.kernel
def _derive_velocity(
    pos: wp.array(dtype=wp.vec3),
    predicted: wp.array(dtype=wp.vec3),
    vel: wp.array(dtype=wp.vec3),
    inverse_mass: wp.array(dtype=float),
    dt: float,
    damping: float,
):
    i = wp.tid()
    if inverse_mass[i] > 0.0:
        vel[i] = (predicted[i] - pos[i]) / dt * (1.0 - damping)
    else:
        vel[i] = wp.vec3(0.0, 0.0, 0.0)


@wp.kernel
def _commit_positions(
    pos: wp.array(dtype=wp.vec3),
    predicted: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    pos[i] = predicted[i]


class WarpXPBDSolver:
    """XPBD solver that keeps simulation and collision state on one GPU."""

    uses_warp_device_collision = True
    keeps_state_on_device = True

    def __init__(
        self,
        n_total: int,
        n_strands: int,
        pps: int,
        init_pos: np.ndarray,
        particle_mass: float,
        bending_enabled: bool,
    ):
        if not wp.is_cuda_available():
            raise RuntimeError("NVIDIA Warp CUDA device is unavailable")
        self.device = "cuda:0"
        self.n_total = int(n_total)
        self.n_strands = int(n_strands)
        self.pps = int(pps)

        positions = np.ascontiguousarray(init_pos, dtype=np.float32)
        self.rest_positions = positions.copy()
        roots = np.arange(n_strands, dtype=np.int32) * pps
        inverse_mass = np.full(n_total, 1.0 / particle_mass, dtype=np.float32)
        inverse_mass[roots] = 0.0
        inverse_mass[roots + 1] = 0.0

        segment_rest = np.empty((n_strands, pps - 1), dtype=np.float32)
        for strand in range(n_strands):
            base = strand * pps
            delta = positions[base : base + pps - 1] - positions[
                base + 1 : base + pps
            ]
            segment_rest[strand] = np.maximum(
                np.linalg.norm(delta, axis=1), 1.0e-6
            )

        bending_rest = np.ones(
            (n_strands, max(pps - 2, 1)), dtype=np.float32
        )
        angle_chord_min = np.zeros(
            (n_strands, max(pps - 2, 1)), dtype=np.float32
        )
        angle_chord_max = np.zeros(
            (n_strands, max(pps - 2, 1)), dtype=np.float32
        )
        if bending_enabled and pps >= 3:
            for strand in range(n_strands):
                base = strand * pps
                delta = positions[base : base + pps - 2] - positions[
                    base + 2 : base + pps
                ]
                bending_rest[strand, : pps - 2] = np.maximum(
                    np.linalg.norm(delta, axis=1), 1.0e-6
                )
        if pps >= 3:
            for strand in range(n_strands):
                base = strand * pps
                for k in range(pps - 2):
                    p0 = positions[base + k]
                    p1 = positions[base + k + 1]
                    p2 = positions[base + k + 2]
                    a = max(float(np.linalg.norm(p0 - p1)), 1.0e-6)
                    b = max(float(np.linalg.norm(p2 - p1)), 1.0e-6)
                    c = float(np.linalg.norm(p0 - p2))
                    cos_theta = (a * a + b * b - c * c) / (2.0 * a * b)
                    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
                    rest_angle = float(np.arccos(cos_theta))
                    min_angle = max(0.0, rest_angle - 1.0)
                    max_angle = min(float(np.pi), rest_angle + 1.0)
                    chord_min = np.sqrt(
                        max(a * a + b * b - 2.0 * a * b * np.cos(min_angle), 0.0)
                    )
                    chord_max = np.sqrt(
                        max(a * a + b * b - 2.0 * a * b * np.cos(max_angle), 0.0)
                    )
                    angle_chord_min[strand, k] = max(float(chord_min), 1.0e-6)
                    angle_chord_max[strand, k] = max(float(chord_max), 1.0e-6)

        self.pos = wp.array(positions, dtype=wp.vec3, device=self.device)
        self.vel = wp.zeros(n_total, dtype=wp.vec3, device=self.device)
        self.predicted = wp.array(
            positions, dtype=wp.vec3, device=self.device
        )
        self.inverse_mass = wp.array(
            inverse_mass, dtype=float, device=self.device
        )
        self.segment_rest = wp.array(
            segment_rest.ravel(), dtype=float, device=self.device
        )
        self.bending_rest = wp.array(
            bending_rest.ravel(), dtype=float, device=self.device
        )
        self.angle_chord_min = wp.array(
            angle_chord_min.ravel(), dtype=float, device=self.device
        )
        self.angle_chord_max = wp.array(
            angle_chord_max.ravel(), dtype=float, device=self.device
        )
        self._angle_limit_rad = 1.0
        self.roots = wp.empty(
            n_strands, dtype=wp.vec3, device=self.device
        )
        self.point1s = wp.empty(
            n_strands, dtype=wp.vec3, device=self.device
        )
        self.seg1_offset = (
            positions[roots + 1] - positions[roots]
        ).astype(np.float32, copy=False)

    def set_positions_velocities(self, pos_np, vel_np):
        self.pos.assign(np.ascontiguousarray(pos_np, dtype=np.float32))
        self.vel.assign(np.ascontiguousarray(vel_np, dtype=np.float32))

    def get_positions_numpy(self):
        return self.pos.numpy()

    def get_velocities_numpy(self):
        return self.vel.numpy()

    def _host_collision(
        self,
        collision,
        allow_sweep: bool,
        final_cleanup: bool = False,
    ):
        pos_np = self.pos.numpy()
        pred_np = self.predicted.numpy()
        vel_np = self.vel.numpy()
        collision(
            pos_np,
            pred_np,
            vel_np,
            allow_sweep=allow_sweep,
            final_cleanup=final_cleanup,
        )
        self.predicted.assign(pred_np)
        self.vel.assign(vel_np)

    def _collide(
        self,
        collision,
        allow_sweep: bool,
        final_cleanup: bool = False,
    ):
        apply_device = getattr(collision, "apply_device", None)
        if apply_device is not None:
            apply_device(
                self.pos,
                self.predicted,
                self.vel,
                allow_sweep=allow_sweep,
                final_cleanup=final_cleanup,
            )
        else:
            self._host_collision(collision, allow_sweep, final_cleanup)

    def run_frame(
        self,
        dt,
        n_substeps,
        n_iter,
        gravity,
        new_root_world,
        seg_ke,
        root_bend_ke,
        bend_ke,
        damping,
        bending_enabled,
        new_point1_world=None,
        body_collision_fn=None,
        post_collision_iterations=4,
        root_advection_tip_weight=1.0,
        angle_limit_enabled=False,
        angle_limit_rad=1.0,
        angle_limit_ke=1.0e6,
    ):
        self._set_angle_limit(float(angle_limit_rad))
        dt_sub = float(dt) / float(n_substeps)
        gravity_np = np.asarray(gravity, dtype=np.float32).reshape(3)
        roots_np = np.ascontiguousarray(new_root_world, dtype=np.float32)
        if new_point1_world is None:
            point1_np = roots_np + self.seg1_offset
        else:
            point1_np = np.ascontiguousarray(
                new_point1_world, dtype=np.float32
            )
        self.roots.assign(roots_np)
        self.point1s.assign(point1_np)

        for _ in range(n_substeps):
            wp.launch(
                _advect_by_root_motion,
                dim=self.n_strands,
                inputs=[
                    self.pos,
                    self.roots,
                    self.pps,
                    float(root_advection_tip_weight),
                ],
                device=self.device,
            )
            wp.launch(
                _predict,
                dim=self.n_total,
                inputs=[
                    self.pos,
                    self.vel,
                    self.predicted,
                    self.roots,
                    self.point1s,
                    self.pps,
                    dt_sub,
                    float(gravity_np[0]),
                    float(gravity_np[1]),
                    float(gravity_np[2]),
                ],
                device=self.device,
            )
            for _ in range(n_iter):
                self._solve(
                    dt_sub,
                    seg_ke,
                    root_bend_ke,
                    bend_ke,
                    bending_enabled,
                    angle_limit_enabled,
                    angle_limit_ke,
                )

            if body_collision_fn is None:
                wp.launch(
                    _derive_velocity,
                    dim=self.n_total,
                    inputs=[
                        self.pos,
                        self.predicted,
                        self.vel,
                        self.inverse_mass,
                        dt_sub,
                        float(damping),
                    ],
                    device=self.device,
                )
            else:
                wp.launch(
                    _derive_velocity,
                    dim=self.n_total,
                    inputs=[
                        self.pos,
                        self.predicted,
                        self.vel,
                        self.inverse_mass,
                        dt_sub,
                        float(damping),
                    ],
                    device=self.device,
                )
                self._collide(body_collision_fn, allow_sweep=True)
                for _ in range(max(int(post_collision_iterations), 0)):
                    self._solve(
                        dt_sub,
                        seg_ke,
                        root_bend_ke,
                        bend_ke,
                        bending_enabled,
                        angle_limit_enabled,
                        angle_limit_ke,
                    )
                    self._collide(body_collision_fn, allow_sweep=False)
                self._collide(
                    body_collision_fn,
                    allow_sweep=False,
                    final_cleanup=True,
                )

            wp.launch(
                _commit_positions,
                dim=self.n_total,
                inputs=[self.pos, self.predicted],
                device=self.device,
            )

        wp.synchronize()
        return self.pos.numpy()

    def _solve(
        self,
        dt,
        seg_ke,
        root_bend_ke,
        bend_ke,
        bending_enabled,
        angle_limit_enabled=False,
        angle_limit_ke=1.0e6,
    ):
        wp.launch(
            _solve_springs,
            dim=self.n_strands,
            inputs=[
                self.predicted,
                self.inverse_mass,
                self.segment_rest,
                self.bending_rest,
                self.pps,
                float(dt),
                float(seg_ke),
                float(root_bend_ke),
                float(bend_ke),
                int(bool(bending_enabled)),
            ],
            device=self.device,
        )
        wp.launch(
            _solve_angle_limits,
            dim=self.n_strands,
            inputs=[
                self.predicted,
                self.inverse_mass,
                self.angle_chord_min,
                self.angle_chord_max,
                self.pps,
                float(dt),
                float(angle_limit_ke),
                int(bool(angle_limit_enabled)),
            ],
            device=self.device,
        )

    def _set_angle_limit(self, limit_rad: float):
        limit = max(float(limit_rad), 0.0)
        if getattr(self, "_angle_limit_rad", None) == limit:
            return
        positions = self.rest_positions
        pps = self.pps
        angle_chord_min = np.zeros(
            (self.n_strands, max(pps - 2, 1)), dtype=np.float32
        )
        angle_chord_max = np.zeros(
            (self.n_strands, max(pps - 2, 1)), dtype=np.float32
        )
        if pps >= 3:
            for strand in range(self.n_strands):
                base = strand * pps
                for k in range(pps - 2):
                    p0 = positions[base + k]
                    p1 = positions[base + k + 1]
                    p2 = positions[base + k + 2]
                    a = max(float(np.linalg.norm(p0 - p1)), 1.0e-6)
                    b = max(float(np.linalg.norm(p2 - p1)), 1.0e-6)
                    c = float(np.linalg.norm(p0 - p2))
                    cos_theta = (a * a + b * b - c * c) / (2.0 * a * b)
                    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
                    rest_angle = float(np.arccos(cos_theta))
                    min_angle = max(0.0, rest_angle - limit)
                    max_angle = min(float(np.pi), rest_angle + limit)
                    chord_min = np.sqrt(
                        max(a * a + b * b - 2.0 * a * b * np.cos(min_angle), 0.0)
                    )
                    chord_max = np.sqrt(
                        max(a * a + b * b - 2.0 * a * b * np.cos(max_angle), 0.0)
                    )
                    angle_chord_min[strand, k] = max(float(chord_min), 1.0e-6)
                    angle_chord_max[strand, k] = max(float(chord_max), 1.0e-6)
        self.angle_chord_min.assign(angle_chord_min.ravel())
        self.angle_chord_max.assign(angle_chord_max.ravel())
        self._angle_limit_rad = limit
