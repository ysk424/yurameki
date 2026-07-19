"""Yurameki -- C++/OpenMP elastic-rod long-hair simulator."""

from __future__ import annotations

import os
import sys
import math

import bpy
from bpy.props import (
    BoolProperty,
    EnumProperty,
    FloatProperty,
    FloatVectorProperty,
    IntProperty,
    StringProperty,
)
from bpy.types import Operator, WindowManager

from . import ui


def _load_defaults():
    import json

    path = os.path.join(os.path.dirname(__file__), "yurameki_defaults.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _default_particle_mass_g(defaults):
    if "PARTICLE_MASS_G" in defaults:
        return float(defaults.get("PARTICLE_MASS_G", 0.01))
    return float(defaults.get("PARTICLE_MASS_KG", 0.00001)) * 1000.0


def _default_log10(defaults, key: str, fallback: float) -> float:
    value = max(float(defaults.get(key, fallback)), 1.0e-12)
    return math.log10(value)


def _value_from_log10(value) -> float:
    return 10.0 ** float(value)


def _find_curves_obj(context=None):
    wm = context.window_manager if context is not None else bpy.context.window_manager
    name = getattr(wm, "yurameki_curves_obj", "").strip()
    if name:
        obj = bpy.data.objects.get(name)
        return obj if obj is not None and obj.type == "CURVES" else None
    objs = [obj for obj in bpy.data.objects if obj.type == "CURVES"]
    return objs[0] if len(objs) == 1 else None


def _curve_spans(curves_data):
    return [
        (int(curve.first_point_index), int(curve.points_length))
        for curve in curves_data.curves
    ]


def _points_per_strand(obj):
    spans = _curve_spans(obj.data)
    if not spans:
        raise ValueError("Curves object has no strands")
    lengths = sorted({length for _start, length in spans})
    if min(lengths) < 3:
        raise ValueError(f"{min(lengths)} points per strand is too small")
    if len(lengths) != 1:
        raise ValueError(f"Native path requires uniform points per strand: {lengths[:8]}")
    return int(lengths[0]), len(spans)


def _source_body_collider(context):
    name = getattr(context.window_manager, "yurameki_collider_obj", "").strip()
    obj = bpy.data.objects.get(name)
    return obj if obj is not None and obj.type == "MESH" else None


def _source_clothes_collider(context):
    name = getattr(context.window_manager, "yurameki_clothes_obj", "").strip()
    obj = bpy.data.objects.get(name)
    return obj if obj is not None and obj.type == "MESH" else None


def _compute_colliders(context):
    body = _source_body_collider(context)
    if body is None:
        raise ValueError("Select a Body mesh first")
    colliders = [body]
    clothes = _source_clothes_collider(context)
    if clothes is not None:
        colliders.append(clothes)
    return colliders


def _make_sim_generator(context):
    """Build the frame-by-frame simulation generator, or return (None, message).

    The generator (``_native_sim.simulate_iter``) yields once per computed frame so
    a modal operator can step it, draw each frame, and stop early.
    """
    obj = _find_curves_obj(context)
    if obj is None:
        return None, "Pick one Hair Curves object"
    wm = context.window_manager
    try:
        from . import _native_sim

        points_per_strand, _strand_count = _points_per_strand(obj)
        wm.yurameki_points_per_strand = points_per_strand
        colliders = _compute_colliders(context)
    except ImportError as exc:
        return None, f"Yurameki native module import failed: {exc}. Build the C++ module for this Blender Python."
    except Exception as exc:
        return None, f"Simulation setup failed: {exc!r}"
    gen = _native_sim.simulate_iter(
        obj,
        colliders,
        start_frame=int(wm.yurameki_sim_start_frame),
        end_frame=int(wm.yurameki_sim_end_frame),
        root_locked_points=int(wm.yurameki_root_locked_points),
        adaptive_root_lock=bool(wm.yurameki_adaptive_root_lock),
        gravity=tuple(float(v) for v in wm.yurameki_gravity),
        damping=float(wm.yurameki_damping),
        max_velocity_mps=float(wm.yurameki_max_velocity_mps),
        particle_mass=float(wm.yurameki_particle_mass_g) * 1.0e-3,
        iterations=int(wm.yurameki_iterations),
        bend_stiffness=_value_from_log10(wm.yurameki_bend_stiffness_log10),
        collision_margin_m=float(wm.yurameki_collision_margin_mm) * 1.0e-3,
        collision_search_m=float(wm.yurameki_collision_search_mm) * 1.0e-3,
        collision_max_correction_m=float(wm.yurameki_collision_max_correction_mm) * 1.0e-3,
        collision_response=float(wm.yurameki_collision_response),
        collision_velocity_damping=float(wm.yurameki_collision_velocity_damping),
        collision_passes=int(wm.yurameki_collision_passes),
        post_collision_iterations=int(wm.yurameki_post_collision_iterations),
        max_move_per_substep_m=float(wm.yurameki_auto_substep_mm) * 1.0e-3,
        max_substeps=int(wm.yurameki_max_substeps),
        bake_mode=wm.yurameki_sim_bake_mode,
        guide_decimation=int(wm.yurameki_guide_decimation),
        keep_length=bool(wm.yurameki_keep_length),
        internal_damping=float(wm.yurameki_internal_damping),
        collision_smoothing=float(wm.yurameki_collision_smoothing),
        settle_iterations=int(wm.yurameki_settle_iterations),
        settle_relaxation=float(wm.yurameki_settle_relaxation),
        groom_strength=float(wm.yurameki_groom_strength),
        groom_repair_strength=float(wm.yurameki_groom_repair_strength),
        length_tolerance_m=float(wm.yurameki_groom_length_tolerance_mm) * 1.0e-3,
        angle_change_limit_deg=float(wm.yurameki_groom_angle_change_deg),
        fold_limit_deg=float(wm.yurameki_groom_fold_deg),
        roughness_factor=float(wm.yurameki_groom_roughness_factor),
        settle_stagnation=int(wm.yurameki_settle_stagnation),
        collision_smooth_passes=int(wm.yurameki_groom_collision_smooth_passes),
        surface_feedback_iterations=int(wm.yurameki_surface_feedback_iterations),
        openmp_threads=int(wm.yurameki_openmp_threads),
    )
    return gen, None


def _format_sim_stats(stats):
    cache_text = f", cache={stats.cache_path}" if stats.cache_path else ""
    return (
        f"Simulate: frames={stats.start_frame}-{stats.end_frame}, "
        f"steps={stats.frame_steps}, substeps={stats.total_substeps} "
        f"(max {stats.max_substeps}), "
        f"strands={stats.n_strands}, sim={stats.simulated_strands}, "
        f"decim={stats.guide_decimation}, pps={stats.points_per_strand}, "
        f"locked={stats.root_locked_points}, "
        f"adaptLock={'on' if stats.adaptive_root_lock else 'off'}"
        f"(max{stats.adaptive_lock_max_points},sf{stats.adaptive_lock_strand_frames}), "
        f"keep_len={'on' if stats.keep_length else 'off'} "
        f"(F{stats.keep_length_source_frame}, finalErr={stats.max_keep_length_error_mm:.6f}mm, "
        f"preGroomErr={stats.max_pre_groom_length_error_mm:.6f}mm), "
        f"auto_move={stats.max_auto_move_mm:.3f}mm, "
        f"vmax={stats.max_velocity_mps:.2f}m/s, "
        f"corr<={stats.collision_max_correction_mm:.2f}mm, "
        f"resp={stats.collision_response:.2f}, "
        f"contact_damp={stats.collision_velocity_damping:.2f}, "
        f"groomed={stats.groomed_strand_frames}, "
        f"settleFail={stats.settle_failed_strand_frames}, "
        f"shapeBad={stats.shape_bad_strand_frames}, rough={stats.rough_strand_frames}, "
        f"lengthBad={stats.length_bad_strand_frames}strands/{stats.length_bad_rod_frames}rods, "
        f"folded={stats.folded_strand_frames}, "
        f"collisionBad={stats.collision_bad_strand_frames}, "
        f"unresolved={stats.unresolved_shape_strand_frames}/{stats.unresolved_collision_strand_frames}, "
        f"settleIter<={stats.max_settle_iterations}, feedback={stats.surface_feedback_repairs}, "
        f"hits={stats.total_hits}, tris={stats.n_triangles_last}, "
        f"{stats.device} ({stats.device_name}), "
        f"bake={stats.bake_mode.lower()}{cache_text}, time={stats.elapsed_sec:.2f}s"
    )


def _bake_cache(context):
    obj = _find_curves_obj(context)
    if obj is None:
        return False, "Pick one Hair Curves object"
    try:
        from . import _native_sim

        stats = _native_sim.bake_runtime_cache(obj)
    except ImportError as exc:
        return False, f"Yurameki native module import failed: {exc}"
    except Exception as exc:
        return False, f"Bake cache failed: {exc!r}"
    return (
        True,
        f"Bake Cache: frames={stats['start_frame']}-{stats['end_frame']}, "
        f"points={stats['n_points']}, fcurves={stats['fcurves']}, "
        f"keys={stats['keys']}",
    )


class YURAMEKI_OT_pick_curves(Operator):
    bl_idname = "yurameki.pick_curves"
    bl_label = "Pick Hair Curves"
    bl_description = "Use the active Curves object as the simulated hair"

    def execute(self, context):
        obj = context.active_object
        if obj is None or obj.type != "CURVES":
            self.report({"ERROR"}, "Active object must be Curves")
            return {"CANCELLED"}
        try:
            points_per_strand, _strand_count = _points_per_strand(obj)
        except ValueError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        context.window_manager.yurameki_curves_obj = obj.name
        context.window_manager.yurameki_points_per_strand = points_per_strand
        self.report({"INFO"}, f"Hair Curves: {obj.name}")
        return {"FINISHED"}


class YURAMEKI_OT_pick_collider(Operator):
    bl_idname = "yurameki.pick_collider"
    bl_label = "Pick Body"
    bl_description = "Use the active mesh as the Body collider"

    def execute(self, context):
        obj = context.active_object
        if obj is None or obj.type != "MESH":
            self.report({"ERROR"}, "Active object must be a mesh")
            return {"CANCELLED"}
        context.window_manager.yurameki_collider_obj = obj.name
        self.report({"INFO"}, f"Body: {obj.name}")
        return {"FINISHED"}


class YURAMEKI_OT_pick_clothes(Operator):
    bl_idname = "yurameki.pick_clothes"
    bl_label = "Pick Clothes"
    bl_description = "Use the active mesh as the optional Clothes collider"

    def execute(self, context):
        obj = context.active_object
        if obj is None or obj.type != "MESH":
            self.report({"ERROR"}, "Active object must be a mesh")
            return {"CANCELLED"}
        context.window_manager.yurameki_clothes_obj = obj.name
        self.report({"INFO"}, f"Clothes: {obj.name}")
        return {"FINISHED"}


class YURAMEKI_OT_simulate(Operator):
    bl_idname = "yurameki.simulate"
    bl_label = "Simulate"
    bl_description = (
        "Run the C++/OpenMP simulation. Each frame is drawn as it is computed so "
        "you can watch it; press Stop or Esc to end early and keep the frames so far"
    )

    _timer = None
    _gen = None
    _stats = None
    _error = None

    def invoke(self, context, event):
        wm = context.window_manager
        if getattr(wm, "yurameki_sim_running", False):
            self.report({"WARNING"}, "A simulation is already running")
            return {"CANCELLED"}
        gen, err = _make_sim_generator(context)
        if gen is None:
            self.report({"ERROR"}, err)
            return {"CANCELLED"}
        self._gen = gen
        self._stats = None
        self._error = None
        wm.yurameki_sim_cancel = False
        wm.yurameki_sim_running = True
        wm.yurameki_sim_status = "Starting..."
        self._timer = wm.event_timer_add(0.01, window=context.window)
        wm.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        # Non-interactive fallback (headless/scripts): run to completion.
        gen, err = _make_sim_generator(context)
        if gen is None:
            self.report({"ERROR"}, err)
            return {"CANCELLED"}
        try:
            while True:
                next(gen)
        except StopIteration as stop:
            self.report({"INFO"}, _format_sim_stats(stop.value))
            return {"FINISHED"}
        except Exception as exc:
            self.report({"ERROR"}, f"Simulation failed: {exc!r}")
            return {"CANCELLED"}

    def modal(self, context, event):
        wm = context.window_manager
        if event.type == "ESC" or getattr(wm, "yurameki_sim_cancel", False):
            return self._end(context, cancelled=True)
        if event.type == "TIMER":
            try:
                progress = next(self._gen)
            except StopIteration as stop:
                self._stats = stop.value
                return self._end(context, cancelled=False)
            except Exception as exc:
                self._error = f"{exc!r}"
                return self._end(context, cancelled=True)
            wm.yurameki_sim_status = (
                f"Frame {progress['frame']}/{progress['end_frame']} "
                f"({progress['completed']}/{progress['total']}) - Stop/Esc to cancel"
            )
            if context.area is not None:
                context.area.tag_redraw()
            return {"RUNNING_MODAL"}
        return {"PASS_THROUGH"}

    def _end(self, context, cancelled):
        wm = context.window_manager
        if self._timer is not None:
            wm.event_timer_remove(self._timer)
            self._timer = None
        if self._gen is not None:
            try:
                self._gen.close()  # early stop -> generator keeps its computed frames
            except Exception:
                pass
            self._gen = None
        wm.yurameki_sim_running = False
        wm.yurameki_sim_cancel = False
        if context.area is not None:
            context.area.tag_redraw()
        if cancelled:
            if self._error:
                wm.yurameki_sim_status = f"Error: {self._error}"
                self.report({"ERROR"}, self._error)
            else:
                wm.yurameki_sim_status = "Stopped (kept computed frames)"
                self.report({"INFO"}, "Simulation stopped; computed frames kept")
            return {"CANCELLED"}
        wm.yurameki_sim_status = "Done"
        self.report({"INFO"}, _format_sim_stats(self._stats) if self._stats else "Simulate done")
        return {"FINISHED"}


class YURAMEKI_OT_stop_simulate(Operator):
    bl_idname = "yurameki.stop_simulate"
    bl_label = "Stop"
    bl_description = "Stop the running simulation and keep the frames computed so far"

    def execute(self, context):
        context.window_manager.yurameki_sim_cancel = True
        return {"FINISHED"}


class YURAMEKI_OT_bake_cache(Operator):
    bl_idname = "yurameki.bake_cache"
    bl_label = "Bake Cache"
    bl_description = "Convert the current Yurameki simulation cache to Curves position keyframes"

    def execute(self, context):
        ok, message = _bake_cache(context)
        self.report({"INFO"} if ok else {"ERROR"}, message)
        return {"FINISHED"} if ok else {"CANCELLED"}


_classes = (
    YURAMEKI_OT_pick_curves,
    YURAMEKI_OT_pick_collider,
    YURAMEKI_OT_pick_clothes,
    YURAMEKI_OT_simulate,
    YURAMEKI_OT_stop_simulate,
    YURAMEKI_OT_bake_cache,
)


_PROP_NAMES = (
    "yurameki_points_per_strand",
    "yurameki_sim_running",
    "yurameki_sim_cancel",
    "yurameki_sim_status",
    "yurameki_curves_obj",
    "yurameki_collider_obj",
    "yurameki_clothes_obj",
    "yurameki_sim_start_frame",
    "yurameki_sim_end_frame",
    "yurameki_root_locked_points",
    "yurameki_adaptive_root_lock",
    "yurameki_gravity",
    "yurameki_damping",
    "yurameki_internal_damping",
    "yurameki_collision_smoothing",
    "yurameki_max_velocity_mps",
    "yurameki_particle_mass_g",
    "yurameki_iterations",
    "yurameki_bend_stiffness_log10",
    "yurameki_collision_margin_mm",
    "yurameki_collision_search_mm",
    "yurameki_collision_max_correction_mm",
    "yurameki_collision_response",
    "yurameki_collision_velocity_damping",
    "yurameki_collision_passes",
    "yurameki_post_collision_iterations",
    "yurameki_auto_substep_mm",
    "yurameki_max_substeps",
    "yurameki_guide_decimation",
    "yurameki_keep_length",
    "yurameki_sim_bake_mode",
    "yurameki_settle_iterations",
    "yurameki_settle_relaxation",
    "yurameki_groom_strength",
    "yurameki_groom_repair_strength",
    "yurameki_groom_length_tolerance_mm",
    "yurameki_groom_angle_change_deg",
    "yurameki_groom_fold_deg",
    "yurameki_groom_roughness_factor",
    "yurameki_settle_stagnation",
    "yurameki_groom_collision_smooth_passes",
    "yurameki_surface_feedback_iterations",
    "yurameki_openmp_threads",
)


def _clear_props():
    for name in _PROP_NAMES:
        try:
            delattr(WindowManager, name)
        except Exception:
            pass


def register():
    defaults = _load_defaults()
    registered_classes = []
    ui_registered = False
    try:
        for cls in _classes:
            bpy.utils.register_class(cls)
            registered_classes.append(cls)

        WindowManager.yurameki_points_per_strand = IntProperty(
            name="Points Per Strand",
            default=12,
            min=3,
            max=256,
            options={"SKIP_SAVE"},
        )
        WindowManager.yurameki_sim_running = BoolProperty(
            name="Simulation Running",
            default=False,
            options={"SKIP_SAVE"},
        )
        WindowManager.yurameki_sim_cancel = BoolProperty(
            name="Stop Requested",
            default=False,
            options={"SKIP_SAVE"},
        )
        WindowManager.yurameki_sim_status = StringProperty(
            name="Status",
            default="",
            options={"SKIP_SAVE"},
        )
        WindowManager.yurameki_curves_obj = StringProperty(
            name="Hair",
            default=str(defaults.get("CURVES_OBJECT", "")),
            options={"SKIP_SAVE"},
        )
        WindowManager.yurameki_collider_obj = StringProperty(
            name="Body",
            default=str(defaults.get("COLLIDER_OBJECT", "")),
            options={"SKIP_SAVE"},
        )
        WindowManager.yurameki_clothes_obj = StringProperty(
            name="Clothes",
            default=str(defaults.get("CLOTHES_OBJECT", "")),
            options={"SKIP_SAVE"},
        )
        WindowManager.yurameki_sim_start_frame = IntProperty(
            name="Start Frame",
            default=int(defaults.get("SIM_START_FRAME", 1)),
            min=-1048574,
            max=1048574,
            options={"SKIP_SAVE"},
        )
        WindowManager.yurameki_sim_end_frame = IntProperty(
            name="End Frame",
            default=int(defaults.get("SIM_END_FRAME", 24)),
            min=-1048574,
            max=1048574,
            options={"SKIP_SAVE"},
        )
        WindowManager.yurameki_root_locked_points = IntProperty(
            name="Root Locked Points",
            default=int(defaults.get("ROOT_LOCKED_POINTS", 3)),
            min=1,
            max=128,
            options={"SKIP_SAVE"},
        )
        WindowManager.yurameki_adaptive_root_lock = BoolProperty(
            name="Adaptive Root Lock",
            description=(
                "As the head advances into a strand, lock (make kinematic) the "
                "crown-side joints of that strand so it rides the skull instead "
                "of being headbutted and flung. The number of locked joints ramps "
                "smoothly with the angular relation to the head's motion, and "
                "back out when it stops, so it costs no collision work on the "
                "leading side. Baseline is Root Locked Points"
            ),
            default=bool(defaults.get("ADAPTIVE_ROOT_LOCK", True)),
            options={"SKIP_SAVE"},
        )
        WindowManager.yurameki_gravity = FloatVectorProperty(
            name="Gravity m/s2",
            default=tuple(defaults.get("GRAVITY", (0.0, 0.0, -9.81))),
            size=3,
            subtype="XYZ",
            min=-1000.0,
            max=1000.0,
            step=10,
            precision=3,
            options={"SKIP_SAVE"},
        )
        WindowManager.yurameki_damping = FloatProperty(
            name="Damping",
            default=float(defaults.get("DAMPING", 0.08)),
            min=0.0,
            max=0.99,
            precision=3,
            options={"SKIP_SAVE"},
        )
        WindowManager.yurameki_internal_damping = FloatProperty(
            name="Internal Damping",
            description=(
                "Strain-rate (viscoelastic) damping between a strand's joints. "
                "Removes internal ringing, jitter, and frizz while preserving the "
                "bulk motion that follows the head and gravity"
            ),
            default=float(defaults.get("INTERNAL_DAMPING", 0.05)),
            min=0.0,
            max=0.5,
            precision=3,
            options={"SKIP_SAVE"},
        )
        WindowManager.yurameki_collision_smoothing = FloatProperty(
            name="Collision Smoothing",
            description=(
                "Distributes each collision push-out along the strand before it is "
                "applied (anti-kink). Stops a scalp-adjacent strand from being "
                "folded into a sharp bend at one joint where the body pushes it. "
                "0 disables it; higher spreads the push over more joints"
            ),
            default=float(defaults.get("COLLISION_SMOOTHING", 0.5)),
            min=0.0,
            max=1.0,
            precision=2,
            options={"SKIP_SAVE"},
        )
        WindowManager.yurameki_max_velocity_mps = FloatProperty(
            name="Max Velocity m/s",
            default=float(defaults.get("MAX_VELOCITY_MPS", 1.0)),
            min=0.0,
            max=100.0,
            precision=3,
            options={"SKIP_SAVE"},
        )
        WindowManager.yurameki_particle_mass_g = FloatProperty(
            name="Particle Mass g",
            default=_default_particle_mass_g(defaults),
            min=0.001,
            max=1000.0,
            precision=3,
            options={"SKIP_SAVE"},
        )
        WindowManager.yurameki_iterations = IntProperty(
            name="Iterations",
            description="Cosserat rod solver outer iterations (position + orientation sweeps)",
            default=int(defaults.get("ITERATIONS", 20)),
            min=1,
            max=256,
            options={"SKIP_SAVE"},
        )
        WindowManager.yurameki_bend_stiffness_log10 = FloatProperty(
            name="Bend Stiffness log10",
            description=(
                "Rod bend/twist stiffness (log10). Higher is stiffer and "
                "straighter; lower is floppier and whips more. Default -3 = 1e-3"
            ),
            default=_default_log10(defaults, "BEND_STIFFNESS", 1.0e-3),
            min=-8.0,
            max=2.0,
            precision=2,
            options={"SKIP_SAVE"},
        )
        WindowManager.yurameki_collision_margin_mm = FloatProperty(
            name="Collision Margin mm",
            default=float(defaults.get("COLLISION_MARGIN_MM", 0.8)),
            min=0.01,
            max=100.0,
            precision=3,
            options={"SKIP_SAVE"},
        )
        WindowManager.yurameki_collision_search_mm = FloatProperty(
            name="Collision Search mm",
            default=float(defaults.get("COLLISION_SEARCH_MM", 20.0)),
            min=0.1,
            max=1000.0,
            precision=3,
            options={"SKIP_SAVE"},
        )
        WindowManager.yurameki_collision_max_correction_mm = FloatProperty(
            name="Collision Max Correction mm",
            default=float(defaults.get("COLLISION_MAX_CORRECTION_MM", 5.0)),
            min=0.01,
            max=500.0,
            precision=3,
            options={"SKIP_SAVE"},
        )
        WindowManager.yurameki_collision_response = FloatProperty(
            name="Collision Response",
            default=float(defaults.get("COLLISION_RESPONSE", 1.0)),
            min=0.0,
            max=1.0,
            precision=3,
            options={"SKIP_SAVE"},
        )
        WindowManager.yurameki_collision_velocity_damping = FloatProperty(
            name="Collision Velocity Damping",
            default=float(defaults.get("COLLISION_VELOCITY_DAMPING", 1.0)),
            min=0.0,
            max=1.0,
            precision=3,
            options={"SKIP_SAVE"},
        )
        WindowManager.yurameki_collision_passes = IntProperty(
            name="Collision Passes",
            default=int(defaults.get("COLLISION_PASSES", 1)),
            min=1,
            max=32,
            options={"SKIP_SAVE"},
        )
        WindowManager.yurameki_post_collision_iterations = IntProperty(
            name="Post Collision Iterations",
            default=int(defaults.get("POST_COLLISION_ITERATIONS", 2)),
            min=0,
            max=64,
            options={"SKIP_SAVE"},
        )
        WindowManager.yurameki_auto_substep_mm = FloatProperty(
            name="Auto Substep mm",
            default=float(defaults.get("AUTO_SUBSTEP_MM", 1.0)),
            min=0.05,
            max=100.0,
            precision=3,
            options={"SKIP_SAVE"},
        )
        WindowManager.yurameki_max_substeps = IntProperty(
            name="Max Substeps",
            default=int(defaults.get("MAX_SUBSTEPS", 16)),
            min=1,
            max=512,
            options={"SKIP_SAVE"},
        )
        WindowManager.yurameki_guide_decimation = IntProperty(
            name="Guide Decimation",
            description="Simulate one guide strand for every N strands, then interpolate the full cache",
            default=int(defaults.get("GUIDE_DECIMATION", 1)),
            min=1,
            max=1000,
            options={"SKIP_SAVE"},
        )
        WindowManager.yurameki_keep_length = BoolProperty(
            name="KEEP LENGTH",
            description="Rebuild each strand from the frame-1 rod lengths before previewing, caching, or baking",
            default=bool(defaults.get("KEEP_LENGTH", True)),
            options={"SKIP_SAVE"},
        )
        WindowManager.yurameki_sim_bake_mode = EnumProperty(
            name="Output",
            items=(
                ("CACHE", "Runtime Cache", "Store simulated frames in the Yurameki runtime cache for preview playback"),
                ("KEYFRAMES", "Position Keyframes", "Bake every simulated frame as Curves position keyframes"),
                ("FINAL", "Final Preview", "Write only the final simulated frame as a static preview"),
            ),
            default=str(defaults.get("SIM_BAKE_MODE", "CACHE")),
            options={"SKIP_SAVE"},
        )
        WindowManager.yurameki_settle_iterations = IntProperty(
            name="Max Settle Iterations",
            description="Hard cap for the per-strand groom/evaluate/collision convergence loop",
            default=int(defaults.get("SETTLE_ITERATIONS", 12)),
            min=1,
            max=128,
            options={"SKIP_SAVE"},
        )
        WindowManager.yurameki_settle_relaxation = FloatProperty(
            name="Settle Relaxation",
            description="Fraction of each grooming or collision repair accepted per convergence iteration",
            default=float(defaults.get("SETTLE_RELAXATION", 0.5)),
            min=0.01,
            max=1.0,
            precision=3,
            options={"SKIP_SAVE"},
        )
        WindowManager.yurameki_groom_strength = FloatProperty(
            name="Groom Strength",
            description="Initial tangent smoothing strength for a strand that fails evaluation",
            default=float(defaults.get("GROOM_STRENGTH", 0.15)),
            min=0.0,
            max=1.0,
            precision=3,
            options={"SKIP_SAVE"},
        )
        WindowManager.yurameki_groom_repair_strength = FloatProperty(
            name="Repair Strength",
            description="Tangent smoothing strength used after the first failed evaluation",
            default=float(defaults.get("GROOM_REPAIR_STRENGTH", 0.4)),
            min=0.0,
            max=1.0,
            precision=3,
            options={"SKIP_SAVE"},
        )
        WindowManager.yurameki_groom_length_tolerance_mm = FloatProperty(
            name="Length Tolerance mm",
            description="Maximum absolute frame-1 rod-length error accepted by the evaluator",
            default=float(defaults.get("GROOM_LENGTH_TOLERANCE_MM", 0.1)),
            min=0.0001,
            max=10.0,
            precision=4,
            options={"SKIP_SAVE"},
        )
        WindowManager.yurameki_groom_angle_change_deg = FloatProperty(
            name="Curvature Change deg",
            description="Maximum joint-angle change from frame 1 before a strand needs grooming",
            default=float(defaults.get("GROOM_ANGLE_CHANGE_DEG", 30.0)),
            min=0.1,
            max=180.0,
            precision=2,
            options={"SKIP_SAVE"},
        )
        WindowManager.yurameki_groom_fold_deg = FloatProperty(
            name="Fold Limit deg",
            description="Maximum angle between adjacent rods before the strand is considered folded",
            default=float(defaults.get("GROOM_FOLD_DEG", 90.0)),
            min=1.0,
            max=180.0,
            precision=2,
            options={"SKIP_SAVE"},
        )
        WindowManager.yurameki_groom_roughness_factor = FloatProperty(
            name="Roughness Factor",
            description="Multiplier on the frame-1 mean tangent-second-difference limit",
            default=float(defaults.get("GROOM_ROUGHNESS_FACTOR", 1.25)),
            min=0.1,
            max=10.0,
            precision=3,
            options={"SKIP_SAVE"},
        )
        WindowManager.yurameki_settle_stagnation = IntProperty(
            name="Stagnation Limit",
            description="Stop and keep the best candidate after this many non-improving iterations",
            default=int(defaults.get("SETTLE_STAGNATION", 3)),
            min=1,
            max=32,
            options={"SKIP_SAVE"},
        )
        WindowManager.yurameki_groom_collision_smooth_passes = IntProperty(
            name="Repair Smooth Passes",
            description="How far collision-repair displacement is diffused along a strand",
            default=int(defaults.get("GROOM_COLLISION_SMOOTH_PASSES", 4)),
            min=0,
            max=32,
            options={"SKIP_SAVE"},
        )
        WindowManager.yurameki_surface_feedback_iterations = IntProperty(
            name="Surface Feedback",
            description="Maximum write/evaluate/re-groom cycles for modifier feedback per frame",
            default=int(defaults.get("SURFACE_FEEDBACK_ITERATIONS", 4)),
            min=1,
            max=32,
            options={"SKIP_SAVE"},
        )
        WindowManager.yurameki_openmp_threads = IntProperty(
            name="OpenMP Threads",
            description="Worker threads; 0 uses all logical processors",
            default=int(defaults.get("OPENMP_THREADS", 0)),
            min=0,
            max=256,
            options={"SKIP_SAVE"},
        )
        ui.register()
        ui_registered = True
    except Exception:
        if ui_registered:
            try:
                ui.unregister()
            except Exception:
                pass
        _clear_props()
        for cls in reversed(registered_classes):
            try:
                bpy.utils.unregister_class(cls)
            except Exception:
                pass
        raise


def unregister():
    mod = sys.modules.get(__name__ + "._native_sim")
    if mod is not None:
        try:
            mod.unregister_cache_handler()
        except Exception:
            pass
    ui.unregister()
    _clear_props()
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)
