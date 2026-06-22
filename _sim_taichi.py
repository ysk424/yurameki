"""Taichi XPBD solver for hair simulation.

Design choice: all @ti.kernel methods use SCALAR arguments only
(int, float / ti.f32 etc.).  ndarray inputs go through ti.field.from_numpy()
before the kernel, outputs come back via ti.field.to_numpy() after.
This avoids a Python-3.13 + Taichi-1.7.4 incompatibility where
ndarray / ti.template() annotations inside @ti.data_oriented class
kernels raise TaichiSyntaxError.
"""
import sys
import site
import numpy as np

_ti          = None
_SolverClass = None
_backend     = None


def _ensure_taichi(backend: str = "CUDA"):
    global _ti, _SolverClass, _backend
    backend = backend.upper()
    if _ti is not None and _backend == backend:
        return _ti
    user_site = site.getusersitepackages()
    if user_site not in sys.path:
        site.addsitedir(user_site)
    import taichi as ti
    if _ti is not None:
        ti.reset()
        _SolverClass = None
    arch = {
        "CUDA": ti.cuda,
        "VULKAN": ti.vulkan,
        "CPU": ti.cpu,
    }.get(backend)
    if arch is None:
        raise ValueError(f"Unsupported Taichi backend: {backend!r}")
    kwargs = {"arch": arch}
    if backend == "CUDA":
        kwargs["device_memory_fraction"] = 0.5
    ti.init(**kwargs)
    print(f"[yurameki/taichi] {backend} initialized")
    _ti = ti
    _backend = backend
    return ti


def get_solver_class(backend: str = "CUDA"):
    """Return (creating if needed) the TaichiXPBDSolver class."""
    global _SolverClass
    ti = _ensure_taichi(backend)
    if _SolverClass is not None:
        return _SolverClass

    @ti.data_oriented
    class TaichiXPBDSolver:
        """XPBD strand solver. All coordinates in world space (metres).

        All kernels use only scalar arguments (int / ti.f32 etc.).
        ndarray I/O goes through .from_numpy() / .to_numpy() in
        Python-side helper methods.
        """

        def __init__(
            self,
            n_total:         int,
            n_strands:       int,
            pps:             int,           # points per strand
            init_pos:        np.ndarray,    # (n_total, 3) float32 world
            particle_mass:   float,
            bending_enabled: bool,
        ):
            self.n_total   = n_total
            self.n_strands = n_strands
            self.pps       = pps

            # Main simulation fields
            self.pos       = ti.Vector.field(3, dtype=ti.f32, shape=n_total)
            self.vel       = ti.Vector.field(3, dtype=ti.f32, shape=n_total)
            self.pos_pred  = ti.Vector.field(3, dtype=ti.f32, shape=n_total)
            self.inv_mass  = ti.field(dtype=ti.f32, shape=n_total)

            # Rest-length tables
            self.seg_rest  = ti.field(dtype=ti.f32, shape=(n_strands, pps - 1))
            bend_shape     = (n_strands, max(pps - 2, 1))
            self.bend_rest = ti.field(dtype=ti.f32, shape=bend_shape)

            # Kinematic follicle points written by Python before each substep.
            # Point 0 is the root and point 1 locks the growth direction.
            self.roots      = ti.Vector.field(3, dtype=ti.f32, shape=n_strands)
            self.point1s    = ti.Vector.field(3, dtype=ti.f32, shape=n_strands)

            # Rest offset from root (point 0) to point 1 in world coords,
            # captured at Start. Point 1 is kinematic: driven as
            # root_current + seg1_offset each substep, simulating a hair
            # follicle that locks the first segment to the scalp direction.
            self.seg1_offset = ti.Vector.field(3, dtype=ti.f32, shape=n_strands)

            self._setup(init_pos, particle_mass, bending_enabled)

        # ------------------------------------------------------------------ #
        # Initialisation
        # ------------------------------------------------------------------ #

        def _setup(self, init_pos: np.ndarray, particle_mass: float, bending_enabled: bool):
            n, ns, pps = self.n_total, self.n_strands, self.pps
            pos_f = np.ascontiguousarray(init_pos, dtype=np.float32)

            self.pos.from_numpy(pos_f)
            self.vel.from_numpy(np.zeros((n, 3), dtype=np.float32))
            self.pos_pred.from_numpy(pos_f)

            inv_m = np.full(n, 1.0 / particle_mass, dtype=np.float32)
            root_idx = np.arange(ns, dtype=np.int32) * pps
            inv_m[root_idx]     = 0.0   # point 0: kinematic (head-tracked)
            inv_m[root_idx + 1] = 0.0   # point 1: kinematic (follicle direction)
            self.inv_mass.from_numpy(inv_m)

            # seg1_offset: world-space offset from root to point 1 at rest.
            # As the root moves with the head, point 1 follows: p1 = root + offset.
            seg1_off = pos_f[root_idx + 1] - pos_f[root_idx]  # (ns, 3)
            self.seg1_offset.from_numpy(np.ascontiguousarray(seg1_off, dtype=np.float32))

            # Segment rest lengths
            sr = np.zeros((ns, pps - 1), dtype=np.float32)
            for s in range(ns):
                b = s * pps
                for k in range(pps - 1):
                    sr[s, k] = max(float(np.linalg.norm(pos_f[b+k] - pos_f[b+k+1])), 1e-6)
            self.seg_rest.from_numpy(sr)

            # Bending rest lengths
            if bending_enabled and pps >= 3:
                br = np.zeros((ns, pps - 2), dtype=np.float32)
                for s in range(ns):
                    b = s * pps
                    for k in range(pps - 2):
                        br[s, k] = max(float(np.linalg.norm(pos_f[b+k] - pos_f[b+k+2])), 1e-6)
                self.bend_rest.from_numpy(br)

            # Initialise roots field from init_pos
            root_np = np.ascontiguousarray(pos_f[np.arange(ns)*pps], dtype=np.float32)
            self.roots.from_numpy(root_np)
            point1_np = np.ascontiguousarray(
                pos_f[np.arange(ns)*pps + 1], dtype=np.float32
            )
            self.point1s.from_numpy(point1_np)

        # ------------------------------------------------------------------ #
        # Kernels — scalar arguments only
        # ------------------------------------------------------------------ #

        @ti.kernel
        def _predict(
            self, dt: ti.f32,
            gravity_x: ti.f32, gravity_y: ti.f32, gravity_z: ti.f32,
        ):
            """Apply gravity, advance free particles; set kinematic points.

            k == 0: root — driven by head animation (roots field).
            k == 1: follicle anchor — fixed at root + seg1_offset,
                    simulating the skin-embedded hair follicle that locks
                    the first segment to the scalp growth direction.
            k >= 2: free particles — gravity + XPBD constraints.
            """
            for i in range(self.n_total):
                s = i // self.pps
                k = i %  self.pps
                if k == 0:
                    self.pos[i]      = self.roots[s]
                    self.pos_pred[i] = self.roots[s]
                elif k == 1:
                    # Follicle anchor follows the evaluated surface direction.
                    p1 = self.point1s[s]
                    self.pos[i]      = p1
                    self.pos_pred[i] = p1
                else:
                    self.vel[i] += ti.Vector([
                        gravity_x, gravity_y, gravity_z
                    ]) * dt
                    self.pos_pred[i]  = self.pos[i] + self.vel[i] * dt

        @ti.kernel
        def _solve_springs(
            self,
            dt: ti.f32, seg_ke: ti.f32,
            root_bend_ke: ti.f32, bend_ke: ti.f32,
            do_bend: int,
        ):
            """XPBD distance constraints — parallel over strands,
            sequential within each strand (Gauss-Seidel).

            Bending stiffness gradient: first 2 bending springs from root
            use root_bend_ke (stiff → hair stands up from scalp);
            remaining springs use bend_ke (soft → hair drapes naturally).
            """
            for s in range(self.n_strands):   # parallel
                base = s * self.pps

                # Segment springs (i, i+1)
                for k in range(self.pps - 1):
                    i  = base + k
                    j  = base + k + 1
                    wi = self.inv_mass[i]
                    wj = self.inv_mass[j]
                    if wi + wj > 1e-10:
                        d    = self.pos_pred[i] - self.pos_pred[j]
                        dist = d.norm()
                        if dist > 1e-8:
                            C     = dist - self.seg_rest[s, k]
                            alpha = 1.0 / (seg_ke * dt * dt)
                            dlam  = -C / (wi + wj + alpha)
                            grad  = d / dist
                            self.pos_pred[i] += wi * dlam * grad
                            self.pos_pred[j] -= wj * dlam * grad

                # Bending springs (i, i+2) with stiffness gradient
                if do_bend == 1:
                    for k in range(self.pps - 2):
                        i  = base + k
                        j  = base + k + 2
                        wi = self.inv_mass[i]
                        wj = self.inv_mass[j]
                        if wi + wj > 1e-10:
                            d    = self.pos_pred[i] - self.pos_pred[j]
                            dist = d.norm()
                            if dist > 1e-8:
                                # Root area (k < 2): stiff → stands up from scalp
                                # Distal area (k >= 2): soft → drapes naturally
                                ke    = ti.select(k < 2, root_bend_ke, bend_ke)
                                C     = dist - self.bend_rest[s, k]
                                alpha = 1.0 / (ke * dt * dt)
                                dlam  = -C / (wi + wj + alpha)
                                grad  = d / dist
                                self.pos_pred[i] += wi * dlam * grad
                                self.pos_pred[j] -= wj * dlam * grad

        @ti.kernel
        def _update_vel_pos(self, dt: ti.f32, damping: ti.f32):
            """Derive velocity from position change, apply damping, advance pos."""
            for i in range(self.n_total):
                if self.inv_mass[i] > 0.0:
                    self.vel[i] = (self.pos_pred[i] - self.pos[i]) / dt * (1.0 - damping)
                    self.pos[i] = self.pos_pred[i]

        # ------------------------------------------------------------------ #
        # Python-side upload / download
        # ------------------------------------------------------------------ #

        def set_positions_velocities(self, pos_np: np.ndarray, vel_np: np.ndarray):
            self.pos.from_numpy(np.ascontiguousarray(pos_np, dtype=np.float32))
            self.vel.from_numpy(np.ascontiguousarray(vel_np, dtype=np.float32))

        def upload_corrected_positions(self, pos_np: np.ndarray):
            """Sync corrected positions (e.g. after body collision) into pos."""
            self.pos.from_numpy(np.ascontiguousarray(pos_np, dtype=np.float32))

        def upload_pred_positions(self, pred_np: np.ndarray):
            """Sync corrected positions into pos_pred (called within substep
            after body collision, before _update_vel_pos)."""
            self.pos_pred.from_numpy(np.ascontiguousarray(pred_np, dtype=np.float32))

        def get_positions_numpy(self) -> np.ndarray:
            return self.pos.to_numpy()

        def get_velocities_numpy(self) -> np.ndarray:
            return self.vel.to_numpy()

        # ------------------------------------------------------------------ #
        # High-level entry point
        # ------------------------------------------------------------------ #

        def run_frame(
            self,
            dt:              float,
            n_substeps:      int,
            n_iter:          int,
            gravity,
            new_root_world:  np.ndarray,    # (n_strands, 3)
            seg_ke:          float,
            root_bend_ke:    float,
            bend_ke:         float,
            damping:         float,
            bending_enabled: bool,
            new_point1_world = None,         # (n_strands, 3), optional
            body_collision_fn = None,       # callable(pred_np) → None, or None
            post_collision_iterations: int = 4,
        ) -> np.ndarray:
            """Run one Blender frame → return final (n_total, 3) positions.

            body_collision_fn: modifies predicted positions and velocities.
            Collision displacement is excluded from velocity. A short
            spring/collision reconciliation loop follows the first contact
            pass so contact correction does not leave stretched segments.
            """
            dt_sub   = float(dt) / float(n_substeps)
            gravity_np = np.asarray(gravity, dtype=np.float32).reshape(3)
            roots_np = np.ascontiguousarray(new_root_world, dtype=np.float32)
            if new_point1_world is None:
                point1_np = roots_np + self.seg1_offset.to_numpy()
            else:
                point1_np = np.ascontiguousarray(
                    new_point1_world, dtype=np.float32
                )
            do_bend  = int(bending_enabled)

            for _ in range(n_substeps):
                self.roots.from_numpy(roots_np)
                self.point1s.from_numpy(point1_np)
                self._predict(
                    dt_sub,
                    float(gravity_np[0]),
                    float(gravity_np[1]),
                    float(gravity_np[2]),
                )
                for _ in range(n_iter):
                    self._solve_springs(dt_sub, float(seg_ke),
                                        float(root_bend_ke), float(bend_ke),
                                        do_bend)
                # Body collision may modify positions and velocities. Collision
                # displacement is deliberately excluded from velocity so a
                # push-out does not become a bounce impulse.
                if body_collision_fn is not None:
                    pos_np  = self.pos.to_numpy()
                    pred_np = self.pos_pred.to_numpy()
                    vel_np  = (
                        (pred_np - pos_np) / dt_sub * (1.0 - float(damping))
                    ).astype(np.float32, copy=False)
                    body_collision_fn(
                        pos_np, pred_np, vel_np, allow_sweep=True
                    )
                    self.upload_pred_positions(pred_np)
                    for _ in range(max(int(post_collision_iterations), 0)):
                        self._solve_springs(
                            dt_sub, float(seg_ke),
                            float(root_bend_ke), float(bend_ke),
                            do_bend,
                        )
                        pred_np = self.pos_pred.to_numpy()
                        body_collision_fn(
                            pos_np, pred_np, vel_np, allow_sweep=False
                        )
                        self.upload_pred_positions(pred_np)
                    pred_np = self.pos_pred.to_numpy()
                    body_collision_fn(
                        pos_np, pred_np, vel_np,
                        allow_sweep=False, final_cleanup=True,
                    )
                    self.pos.from_numpy(
                        np.ascontiguousarray(pred_np, dtype=np.float32)
                    )
                    self.vel.from_numpy(
                        np.ascontiguousarray(vel_np, dtype=np.float32)
                    )
                else:
                    self._update_vel_pos(dt_sub, float(damping))

            return self.pos.to_numpy()

    _SolverClass = TaichiXPBDSolver
    return TaichiXPBDSolver


# ------------------------------------------------------------------ #
# Body collision (Python / Blender BVHTree — called after run_frame)
# ------------------------------------------------------------------ #

def build_body_bvh(body_name: str):
    """Build a world-space BVH from the evaluated (skinned) body mesh."""
    import bpy
    from mathutils.bvhtree import BVHTree
    body_obj = bpy.data.objects.get(body_name)
    if body_obj is None or body_obj.type != "MESH":
        return None
    dg = bpy.context.evaluated_depsgraph_get()
    eval_obj = body_obj.evaluated_get(dg)
    mesh = eval_obj.to_mesh()
    try:
        matrix = eval_obj.matrix_world
        vertices = [matrix @ vertex.co for vertex in mesh.vertices]
        polygons = [tuple(poly.vertices) for poly in mesh.polygons]
        return BVHTree.FromPolygons(vertices, polygons)
    finally:
        eval_obj.to_mesh_clear()
