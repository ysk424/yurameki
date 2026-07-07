"""Yurameki 0.7.x -- NVIDIA Warp long straight-hair simulator."""

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
        return float(defaults.get("PARTICLE_MASS_G", 1.0))
    return float(defaults.get("PARTICLE_MASS_KG", 0.001)) * 1000.0


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
        raise ValueError(f"Warp path requires uniform points per strand: {lengths[:8]}")
    return int(lengths[0]), len(spans)


def _source_body_collider(context):
    name = getattr(context.window_manager, "yurameki_collider_obj", "").strip()
    obj = bpy.data.objects.get(name)
    return obj if obj is not None and obj.type == "MESH" else None


def _source_clothes_collider(context):
    name = getattr(context.window_manager, "yurameki_clothes_obj", "").strip()
    obj = bpy.data.objects.get(name)
    return obj if obj is not None and obj.type == "MESH" else None


def _ensure_body_proxy(context):
    from . import collider_proxy

    body = _source_body_collider(context)
    if body is None:
        raise ValueError("Select a Body mesh first")
    proxy = collider_proxy.get_valid_proxy(
        body,
        getattr(context.window_manager, "yurameki_collider_proxy_obj", ""),
    )
    if proxy is not None:
        return proxy, None
    stats = collider_proxy.build_filled_proxy(
        body,
        getattr(context.window_manager, "yurameki_collider_proxy_obj", ""),
    )
    context.window_manager.yurameki_collider_proxy_obj = stats["proxy_name"]
    proxy = bpy.data.objects.get(stats["proxy_name"])
    return proxy if proxy is not None else body, stats


def _compute_colliders(context):
    body_or_proxy, proxy_stats = _ensure_body_proxy(context)
    colliders = [body_or_proxy]
    clothes = _source_clothes_collider(context)
    if clothes is not None:
        colliders.append(clothes)
    return colliders, proxy_stats


def _check_hair(context):
    obj = _find_curves_obj(context)
    if obj is None:
        return False, "Pick one Hair Curves object"
    try:
        from . import _warp_sim

        pps, strands = _points_per_strand(obj)
        colliders, proxy_stats = _compute_colliders(context)
        stats = _warp_sim.check_warp_ready(
            obj,
            colliders,
            root_locked_points=int(context.window_manager.yurameki_root_locked_points),
            particle_mass=float(context.window_manager.yurameki_particle_mass_g) * 1.0e-3,
        )
    except ImportError as exc:
        return False, f"Warp import failed: {exc}. Install NVIDIA warp-lang for Blender Python."
    except Exception as exc:
        return False, f"Check failed: {exc}"
    context.window_manager.yurameki_points_per_strand = pps
    proxy_text = "reused"
    if proxy_stats is not None:
        proxy_text = (
            f"created {proxy_stats['proxy_name']} "
            f"filled={proxy_stats['faces_added']} boundary={proxy_stats['boundary_edges_after']}"
        )
    clothes = _source_clothes_collider(context)
    return (
        True,
        f"Check PASS: warp={stats.warp_version} {stats.device} "
        f"{stats.device_name} sm_{stats.device_arch}, "
        f"strands={strands}, points={stats.n_points}, pps={stats.points_per_strand}, "
        f"locked={stats.root_locked_points}, tris={stats.n_triangles}, "
        f"body_proxy={proxy_text}, clothes={clothes.name if clothes else 'none'}",
    )


def _simulate(context):
    obj = _find_curves_obj(context)
    if obj is None:
        return False, "Pick one Hair Curves object"
    wm = context.window_manager
    try:
        from . import _warp_sim

        _points_per_strand(obj)
        colliders, _proxy_stats = _compute_colliders(context)
        stats = _warp_sim.simulate(
            obj,
            colliders,
            start_frame=int(wm.yurameki_sim_start_frame),
            end_frame=int(wm.yurameki_sim_end_frame),
            root_locked_points=int(wm.yurameki_root_locked_points),
            gravity=tuple(float(v) for v in wm.yurameki_gravity),
            damping=float(wm.yurameki_damping),
            max_velocity_mps=float(wm.yurameki_max_velocity_mps),
            particle_mass=float(wm.yurameki_particle_mass_g) * 1.0e-3,
            iterations=int(wm.yurameki_iterations),
            stretch_compliance=_value_from_log10(wm.yurameki_stretch_compliance_log10),
            bend_compliance=_value_from_log10(wm.yurameki_bend_compliance_log10),
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
        )
    except ImportError as exc:
        return False, f"Warp import failed: {exc}. Install NVIDIA warp-lang for Blender Python."
    except Exception as exc:
        return False, f"Simulation failed: {exc!r}"
    cache_text = f", cache={stats.cache_path}" if stats.cache_path else ""
    return (
        True,
        f"Simulate: frames={stats.start_frame}-{stats.end_frame}, "
        f"steps={stats.frame_steps}, substeps={stats.total_substeps} "
        f"(max {stats.max_substeps}), "
        f"strands={stats.n_strands}, sim={stats.simulated_strands}, "
        f"decim={stats.guide_decimation}, pps={stats.points_per_strand}, "
        f"locked={stats.root_locked_points}, "
        f"keep_len={'on' if stats.keep_length else 'off'} "
        f"(F{stats.keep_length_source_frame}, err={stats.max_keep_length_error_mm:.6f}mm), "
        f"auto_move={stats.max_auto_move_mm:.3f}mm, "
        f"vmax={stats.max_velocity_mps:.2f}m/s, "
        f"corr<={stats.collision_max_correction_mm:.2f}mm, "
        f"resp={stats.collision_response:.2f}, "
        f"contact_damp={stats.collision_velocity_damping:.2f}, "
        f"hits={stats.total_hits}, tris={stats.n_triangles_last}, "
        f"{stats.device} sm_{stats.device_arch}, "
        f"bake={stats.bake_mode.lower()}{cache_text}, time={stats.elapsed_sec:.2f}s",
    )


def _bake_cache(context):
    obj = _find_curves_obj(context)
    if obj is None:
        return False, "Pick one Hair Curves object"
    try:
        from . import _warp_sim

        stats = _warp_sim.bake_runtime_cache(obj)
    except ImportError as exc:
        return False, f"Warp import failed: {exc}. Install NVIDIA warp-lang for Blender Python."
    except Exception as exc:
        return False, f"Bake cache failed: {exc!r}"
    return (
        True,
        f"Bake Cache: frames={stats['start_frame']}-{stats['end_frame']}, "
        f"points={stats['n_points']}, fcurves={stats['fcurves']}, "
        f"keys={stats['keys']}",
    )


class YURAMEKI_OT_check_hair(Operator):
    bl_idname = "yurameki.check_hair"
    bl_label = "Check"
    bl_description = "Validate inputs, build/reuse the Body proxy, and initialize NVIDIA Warp CUDA"

    def execute(self, context):
        ok, message = _check_hair(context)
        context.window_manager.yurameki_hair_check_status = message
        self.report({"INFO"} if ok else {"ERROR"}, message)
        return {"FINISHED"} if ok else {"CANCELLED"}


class YURAMEKI_OT_pick_curves(Operator):
    bl_idname = "yurameki.pick_curves"
    bl_label = "Pick Hair Curves"
    bl_description = "Use the active Curves object as the simulated hair"

    def execute(self, context):
        obj = context.active_object
        if obj is None or obj.type != "CURVES":
            self.report({"ERROR"}, "Active object must be Curves")
            return {"CANCELLED"}
        context.window_manager.yurameki_curves_obj = obj.name
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
        from . import collider_proxy

        collider_proxy.clear_proxy(getattr(context.window_manager, "yurameki_collider_proxy_obj", ""))
        context.window_manager.yurameki_collider_proxy_obj = ""
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
    bl_description = "Run the NVIDIA Warp joint-chain simulation"

    def execute(self, context):
        ok, message = _simulate(context)
        self.report({"INFO"} if ok else {"ERROR"}, message)
        return {"FINISHED"} if ok else {"CANCELLED"}


class YURAMEKI_OT_bake_cache(Operator):
    bl_idname = "yurameki.bake_cache"
    bl_label = "Bake Cache"
    bl_description = "Convert the current Yurameki simulation cache to Curves position keyframes"

    def execute(self, context):
        ok, message = _bake_cache(context)
        self.report({"INFO"} if ok else {"ERROR"}, message)
        return {"FINISHED"} if ok else {"CANCELLED"}


_classes = (
    YURAMEKI_OT_check_hair,
    YURAMEKI_OT_pick_curves,
    YURAMEKI_OT_pick_collider,
    YURAMEKI_OT_pick_clothes,
    YURAMEKI_OT_simulate,
    YURAMEKI_OT_bake_cache,
)


_PROP_NAMES = (
    "yurameki_points_per_strand",
    "yurameki_hair_check_status",
    "yurameki_curves_obj",
    "yurameki_collider_obj",
    "yurameki_collider_proxy_obj",
    "yurameki_clothes_obj",
    "yurameki_sim_start_frame",
    "yurameki_sim_end_frame",
    "yurameki_root_locked_points",
    "yurameki_gravity",
    "yurameki_damping",
    "yurameki_max_velocity_mps",
    "yurameki_particle_mass_g",
    "yurameki_iterations",
    "yurameki_stretch_compliance_log10",
    "yurameki_bend_compliance_log10",
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
        WindowManager.yurameki_hair_check_status = StringProperty(
            name="Check",
            default="Not checked",
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
        WindowManager.yurameki_collider_proxy_obj = StringProperty(
            name="Collider Proxy",
            default="",
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
            default=int(defaults.get("ITERATIONS", 8)),
            min=1,
            max=256,
            options={"SKIP_SAVE"},
        )
        WindowManager.yurameki_stretch_compliance_log10 = FloatProperty(
            name="Stretch Compliance log10",
            default=_default_log10(defaults, "STRETCH_COMPLIANCE", 1.0e-8),
            min=-12.0,
            max=0.0,
            precision=2,
            options={"SKIP_SAVE"},
        )
        WindowManager.yurameki_bend_compliance_log10 = FloatProperty(
            name="Bend Compliance log10",
            default=_default_log10(defaults, "BEND_COMPLIANCE", 1.0e-5),
            min=-12.0,
            max=0.0,
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
                ("CACHE", "Cache", "Store simulated frames in the Yurameki runtime cache for preview playback"),
                ("KEYFRAMES", "Keyframes", "Bake every simulated frame as Curves position keyframes"),
                ("FINAL", "Final Preview", "Write only the final simulated frame as a static preview"),
            ),
            default=str(defaults.get("SIM_BAKE_MODE", "CACHE")),
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
    mod = sys.modules.get(__name__ + "._warp_sim")
    if mod is not None:
        try:
            mod.unregister_cache_handler()
        except Exception:
            pass
    ui.unregister()
    _clear_props()
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)
