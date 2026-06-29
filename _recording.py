"""Timeline recording and persistent playback cache for Yurameki."""
from __future__ import annotations

import os
from pathlib import Path

import bpy
import numpy as np
from mathutils.kdtree import KDTree

from . import _world_passthrough as _wp


POINTS_PER_STRAND = 9
CACHE_SUFFIX = ".yurameki-cache.npz"
AUTO_INTERPOLATION_MIN = 1
AUTO_INTERPOLATION_MAX = 64
AUTO_SPACING_FRACTION = 0.25
AUTO_SAFETY_FACTOR = 1.1


def _find_curves_obj():
    objects = [obj for obj in bpy.data.objects if obj.type == "CURVES"]
    return objects[0] if len(objects) == 1 else None


def _cache_path() -> Path | None:
    # During extension install/enable Blender can temporarily expose
    # `_RedirectData` instead of the normal BlendData object.
    filepath = getattr(bpy.data, "filepath", "")
    if not filepath:
        return None
    blend_path = Path(filepath)
    return blend_path.with_name(blend_path.name + CACHE_SUFFIX)


def _median_root_spacing(roots: np.ndarray) -> float:
    """Return the median nearest-neighbour root spacing in world metres."""
    count = len(roots)
    if count < 2:
        return 0.001
    tree = KDTree(count)
    for index, point in enumerate(roots):
        tree.insert(tuple(point), index)
    tree.balance()
    distances = []
    for point in roots:
        nearest = tree.find_n(tuple(point), 2)
        if len(nearest) > 1 and nearest[1][2] > 1.0e-8:
            distances.append(float(nearest[1][2]))
    if not distances:
        return 0.001
    return float(np.median(np.asarray(distances, dtype=np.float64)))


def _auto_interpolation_count(
    previous_roots: np.ndarray,
    target_roots: np.ndarray,
    median_spacing: float,
) -> int:
    """Choose a motion-based interpolation count."""
    if (
        previous_roots.shape != target_roots.shape
        or len(previous_roots) == 0
        or median_spacing <= 1.0e-8
    ):
        return AUTO_INTERPOLATION_MIN
    max_move = float(
        np.linalg.norm(target_roots - previous_roots, axis=1).max()
    )
    if max_move <= 1.0e-8:
        return AUTO_INTERPOLATION_MIN
    target_step = median_spacing * AUTO_SPACING_FRACTION
    count = int(np.ceil(max_move * AUTO_SAFETY_FACTOR / target_step))
    return max(
        AUTO_INTERPOLATION_MIN,
        min(AUTO_INTERPOLATION_MAX, count),
    )


class RecordingManager:
    def __init__(self):
        self.frames: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self.obj_name = ""
        self.n_total = 0
        self.last_frame: int | None = None
        self.positions: np.ndarray | None = None
        self.velocities: np.ndarray | None = None
        self.solver = None
        self.collision = None
        self.root_indices: np.ndarray | None = None
        self.previous_roots: np.ndarray | None = None
        self.median_root_spacing = 0.001
        self._inside_frame_eval = False
        self.dirty = False
        self.previous_sync_mode: str | None = None

    def _set_mode(self, mode: str) -> None:
        wm = bpy.context.window_manager
        if wm is not None and hasattr(wm, "yurameki_record_mode"):
            wm.yurameki_record_mode = mode

    def is_recording(self) -> bool:
        wm = bpy.context.window_manager
        return (
            wm is not None
            and getattr(wm, "yurameki_record_mode", "PLAYBACK") == "RECORDING"
        )

    def start(self, scene) -> tuple[bool, str]:
        obj = _find_curves_obj()
        if obj is None:
            return False, "Need exactly one Curves object"
        body_name = bpy.context.window_manager.yurameki_body_obj.strip()
        body = bpy.data.objects.get(body_name)
        if body is None or body.type != "MESH":
            return False, "Select a Body Mesh first"
        cloth_name = bpy.context.window_manager.yurameki_cloth_obj.strip()
        if cloth_name:
            cloth = bpy.data.objects.get(cloth_name)
            if cloth is None or cloth.type != "MESH":
                return False, "Cloth Collider must be a mesh"

        attr = obj.data.attributes.get("position")
        if attr is None or len(attr.data) == 0:
            return False, "Curves object has no points"
        n_total = len(attr.data)
        if n_total % POINTS_PER_STRAND:
            return False, (
                f"Point count {n_total} is not divisible by "
                f"{POINTS_PER_STRAND}"
            )

        frame = int(scene.frame_current)
        depsgraph = bpy.context.evaluated_depsgraph_get()
        obj_eval = obj.evaluated_get(depsgraph)
        eval_world = _wp._read_world(
            obj_eval.data, n_total, obj_eval.matrix_world
        )
        orig_world = _wp._read_world(
            obj.data, n_total, obj.matrix_world
        )
        if eval_world is None or orig_world is None:
            return False, "Could not read evaluated Curves positions"

        cached = self.frames.get(frame)
        if (
            cached is not None
            and cached[0].shape == (n_total, 3)
            and self.obj_name == obj.name
        ):
            positions = cached[0].copy()
            velocities = cached[1].copy()
        else:
            positions = eval_world.copy()
            velocities = np.zeros_like(positions)

        # Re-recording replaces this frame and everything after it.
        for old_frame in [key for key in self.frames if key >= frame]:
            del self.frames[old_frame]

        n_strands = n_total // POINTS_PER_STRAND
        roots = np.arange(n_strands, dtype=np.int32) * POINTS_PER_STRAND
        try:
            solver_kwargs = dict(
                n_total=n_total,
                n_strands=n_strands,
                pps=POINTS_PER_STRAND,
                init_pos=positions,
                particle_mass=_wp.PARTICLE_MASS,
                bending_enabled=_wp.BENDING_ENABLED,
            )
            from ._sim_warp import WarpXPBDSolver
            solver = WarpXPBDSolver(**solver_kwargs)
            solver.set_positions_velocities(positions, velocities)
        except Exception as exc:
            return False, f"Warp CUDA solver build failed: {exc!r}"

        try:
            from ._collision_warp import WarpBodyCollider
            collision = WarpBodyCollider(
                collider_names=_wp.collision_target_names(),
                n_total=n_total,
                points_per_strand=POINTS_PER_STRAND,
                margin=_wp.COLLISION_MARGIN,
                search_distance=_wp.COLLISION_SEARCH,
            )
            print("[yurameki/record] Warp CUDA collision cache enabled")
        except Exception as exc:
            return False, f"Warp CUDA collision unavailable: {exc!r}"

        self.obj_name = obj.name
        self.n_total = n_total
        self.last_frame = frame
        self.positions = positions
        self.velocities = velocities
        self.solver = solver
        self.collision = collision
        self.root_indices = roots
        self.previous_roots = eval_world[roots].copy()
        self.median_root_spacing = _median_root_spacing(
            self.previous_roots
        )
        wm = bpy.context.window_manager
        wm.yurameki_auto_interpolation_current = 1
        self.frames[frame] = (positions.copy(), velocities.copy())
        self.dirty = True
        self.previous_sync_mode = scene.sync_mode
        scene.sync_mode = "NONE"
        self._set_mode("RECORDING")
        print(
            f"[yurameki/record] recording from frame {frame}; "
            f"sync {self.previous_sync_mode} -> NONE; "
            f"median root spacing={self.median_root_spacing:.6g} m"
        )
        return True, f"Recording from frame {frame}"

    def stop(self, reason: str = "stopped") -> None:
        if self.is_recording():
            print(f"[yurameki/record] {reason}; entering playback")
        scene = getattr(bpy.context, "scene", None)
        if scene is not None and self.previous_sync_mode is not None:
            scene.sync_mode = self.previous_sync_mode
        self.previous_sync_mode = None
        self._set_mode("PLAYBACK")
        self.solver = None
        self.collision = None
        self.last_frame = None
        self.positions = None
        self.velocities = None
        self.previous_roots = None

    def toggle(self, scene) -> tuple[bool, str]:
        if self.is_recording():
            self.stop("REC toggled off")
            return True, "Playback"
        return self.start(scene)

    def bake_range(
        self,
        scene,
        start: int,
        end: int,
        progress_callback=None,
    ) -> tuple[bool, str]:
        """Simulate frames [start, end] in one batch and cache every frame.

        Reuses the same per-frame solver path as interactive recording, but
        drives the timeline itself instead of relying on the user pressing
        Play. The result is stored in `self.frames`, so the baked motion plays
        back over the timeline exactly like a recording.
        """
        if end < start:
            return False, f"End frame {end} is before start frame {start}"
        scene.frame_set(int(start))
        bpy.context.view_layer.update()
        ok, message = self.start(scene)
        if not ok:
            return False, message
        if progress_callback is not None:
            progress_callback(int(start), int(start), int(end), 1)
        try:
            for frame in range(int(start) + 1, int(end) + 1):
                if not self._simulate_next(scene, frame):
                    self.stop("bake aborted by simulation error")
                    return False, f"Simulation failed at frame {frame}"
                if progress_callback is not None:
                    progress_callback(
                        int(frame),
                        int(start),
                        int(end),
                        int(frame) - int(start) + 1,
                    )
        finally:
            baked = len(self.frames)
            self.stop("bake complete")
        scene.frame_set(int(start))
        self.restore(scene, int(start))
        return True, f"Baked {baked} frames ({start}-{end})"

    def _evaluate_subframe(self, scene, frame: int, subframe: float) -> None:
        self._inside_frame_eval = True
        try:
            scene.frame_set(frame, subframe=subframe)
            bpy.context.view_layer.update()
        finally:
            self._inside_frame_eval = False

    def _simulate_next(self, scene, target_frame: int) -> bool:
        obj = bpy.data.objects.get(self.obj_name)
        if obj is None or obj.type != "CURVES":
            return False
        wm = bpy.context.window_manager
        current_eval = obj.evaluated_get(
            bpy.context.evaluated_depsgraph_get()
        )
        current_world = _wp._read_world(
            current_eval.data, self.n_total, current_eval.matrix_world
        )
        if current_world is None:
            return False
        target_roots = current_world[self.root_indices].copy()
        if (
            wm.yurameki_auto_frame_interpolation
            and self.previous_roots is not None
        ):
            interpolation = _auto_interpolation_count(
                self.previous_roots,
                target_roots,
                self.median_root_spacing,
            )
        else:
            interpolation = max(1, int(wm.yurameki_frame_interpolation))
        interpolation *= max(1, int(wm.yurameki_interpolation_mag))
        wm.yurameki_auto_interpolation_current = interpolation
        fps = float(scene.render.fps) / float(scene.render.fps_base)
        if fps <= 0.0:
            return False
        dt_subframe = (1.0 / fps) / float(interpolation)
        previous_frame = target_frame - 1

        keeps_state_on_device = bool(
            getattr(self.solver, "keeps_state_on_device", False)
        )
        for index in range(1, interpolation + 1):
            if index == interpolation:
                self._evaluate_subframe(scene, target_frame, 0.0)
            else:
                self._evaluate_subframe(
                    scene, previous_frame, index / float(interpolation)
                )

            dg = bpy.context.evaluated_depsgraph_get()
            obj_eval = obj.evaluated_get(dg)
            eval_world = _wp._read_world(
                obj_eval.data, self.n_total, obj_eval.matrix_world
            )
            orig_world = _wp._read_world(
                obj.data, self.n_total, obj.matrix_world
            )
            if eval_world is None or orig_world is None:
                return False
            offset_world = eval_world - orig_world
            roots = eval_world[self.root_indices]
            point1s = eval_world[self.root_indices + 1]
            try:
                self.collision.update_from_collider_names()
                collision = self.collision
            except Exception as exc:
                print(
                    "[yurameki/record] Warp collision update failed: "
                    f"{exc!r}"
                )
                return False

            # Warp CUDA retains the authoritative position and velocity state
            # between interpolation steps. Re-upload only for solvers whose
            # state may have been modified on the host.
            if not keeps_state_on_device:
                self.solver.set_positions_velocities(
                    self.positions, self.velocities
                )
            self.positions = self.solver.run_frame(
                dt=dt_subframe,
                n_substeps=_wp.SUBSTEPS,
                n_iter=_wp.ITERATIONS,
                gravity=_wp.GRAVITY,
                new_root_world=roots,
                new_point1_world=point1s,
                seg_ke=_wp.SPRING_KE,
                root_bend_ke=_wp.ROOT_BENDING_KE,
                bend_ke=_wp.BENDING_KE,
                damping=_wp.DAMPING,
                bending_enabled=_wp.BENDING_ENABLED,
                body_collision_fn=collision,
                post_collision_iterations=_wp.POST_COLLISION_ITERATIONS,
                angle_limit_enabled=_wp.ANGLE_LIMIT_ENABLED,
                angle_limit_rad=_wp.ANGLE_LIMIT_RAD,
                angle_limit_ke=_wp.ANGLE_LIMIT_KE,
            )
            if not keeps_state_on_device:
                self.velocities = self.solver.get_velocities_numpy()
            _wp._write_world(obj, self.positions, offset=offset_world)

        if keeps_state_on_device:
            # Velocity is needed on the host only for the completed-frame
            # recording cache. Intermediate interpolation steps stay on GPU.
            self.velocities = self.solver.get_velocities_numpy()

        self.last_frame = target_frame
        self.previous_roots = target_roots
        self.frames[target_frame] = (
            self.positions.copy(),
            self.velocities.copy(),
        )
        self.dirty = True
        print(
            f"[yurameki/record] frame {target_frame} cached "
            f"({interpolation} interpolation steps)"
        )
        return True

    def restore(self, scene, frame: int) -> bool:
        cached = self.frames.get(int(frame))
        obj = bpy.data.objects.get(self.obj_name)
        if cached is None or obj is None or obj.type != "CURVES":
            return False
        if cached[0].shape != (self.n_total, 3):
            return False
        dg = bpy.context.evaluated_depsgraph_get()
        obj_eval = obj.evaluated_get(dg)
        eval_world = _wp._read_world(
            obj_eval.data, self.n_total, obj_eval.matrix_world
        )
        orig_world = _wp._read_world(
            obj.data, self.n_total, obj.matrix_world
        )
        if eval_world is None or orig_world is None:
            return False
        _wp._write_world(obj, cached[0], offset=eval_world - orig_world)
        return True

    def on_frame_change(self, scene) -> None:
        if self._inside_frame_eval:
            return
        frame = int(scene.frame_current)
        if self.is_recording():
            if self.last_frame is None or frame != self.last_frame + 1:
                self.stop("recording aborted by reverse playback or jump")
                self.restore(scene, frame)
                return
            if not self._simulate_next(scene, frame):
                self.stop("recording aborted by simulation error")
        else:
            self.restore(scene, frame)

    def save_cache(self) -> bool:
        path = _cache_path()
        if path is None:
            return False
        if not self.frames:
            if self.dirty and path.exists():
                try:
                    path.unlink()
                    print(f"[yurameki/record] removed stale cache {path}")
                except Exception as exc:
                    print(
                        f"[yurameki/record] stale cache removal failed: {exc!r}"
                    )
                    return False
            self.dirty = False
            return True
        frames = np.array(sorted(self.frames), dtype=np.int32)
        positions = np.stack([self.frames[int(f)][0] for f in frames])
        velocities = np.stack([self.frames[int(f)][1] for f in frames])
        temp_path = path.with_name(path.name + ".tmp")
        try:
            with open(temp_path, "wb") as handle:
                np.savez_compressed(
                    handle,
                    format_version=np.array([1], dtype=np.int32),
                    object_name=np.array([self.obj_name]),
                    points_per_strand=np.array(
                        [POINTS_PER_STRAND], dtype=np.int32
                    ),
                    frames=frames,
                    positions=positions.astype(np.float32, copy=False),
                    velocities=velocities.astype(np.float32, copy=False),
                )
            os.replace(temp_path, path)
            self.dirty = False
            print(
                f"[yurameki/record] saved {len(frames)} frames to {path}"
            )
            return True
        except Exception as exc:
            print(f"[yurameki/record] cache save failed: {exc!r}")
            try:
                temp_path.unlink(missing_ok=True)
            except Exception:
                pass
            return False

    def load_cache(self) -> bool:
        self.stop("blend load")
        self.frames.clear()
        self.obj_name = ""
        self.n_total = 0
        path = _cache_path()
        if path is None or not path.exists():
            return False
        try:
            with np.load(path, allow_pickle=False) as data:
                global POINTS_PER_STRAND
                pps = int(data["points_per_strand"][0])
                if pps < 3:
                    raise ValueError(f"unsupported points-per-strand: {pps}")
                POINTS_PER_STRAND = pps
                _wp.POINTS_PER_STRAND = pps
                frames = data["frames"].astype(np.int32, copy=False)
                positions = data["positions"].astype(np.float32, copy=False)
                velocities = data["velocities"].astype(np.float32, copy=False)
                obj_name = str(data["object_name"][0])
                if (
                    positions.ndim != 3
                    or velocities.shape != positions.shape
                    or positions.shape[0] != len(frames)
                ):
                    raise ValueError("invalid cache array shapes")
                self.frames = {
                    int(frame): (
                        positions[index].copy(),
                        velocities[index].copy(),
                    )
                    for index, frame in enumerate(frames)
                }
                self.obj_name = obj_name
                self.n_total = int(positions.shape[1])
            self.dirty = False
            print(
                f"[yurameki/record] loaded {len(self.frames)} frames from {path}"
            )
            return True
        except Exception as exc:
            self.frames.clear()
            print(f"[yurameki/record] cache load failed: {exc!r}")
            return False

    def clear(self) -> None:
        self.stop("cache cleared")
        self.frames.clear()
        self.obj_name = ""
        self.n_total = 0
        self.dirty = True


manager = RecordingManager()
