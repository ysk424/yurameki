"""CUDA XPBD solver whose state is shared directly with Warp collision."""
from __future__ import annotations

import numpy as np
import warp as wp


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
        if bending_enabled and pps >= 3:
            for strand in range(n_strands):
                base = strand * pps
                delta = positions[base : base + pps - 2] - positions[
                    base + 2 : base + pps
                ]
                bending_rest[strand, : pps - 2] = np.maximum(
                    np.linalg.norm(delta, axis=1), 1.0e-6
                )

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
    ):
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
