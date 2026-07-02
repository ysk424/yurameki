"""Yurameki 0.4.x -- CUDA straight long-hair solver prototype."""

from __future__ import annotations

import os

import bpy
from bpy.props import EnumProperty, FloatProperty, IntProperty, StringProperty
from bpy.types import Operator, WindowManager

from . import ui


def _load_defaults():
    import json

    path = os.path.join(os.path.dirname(__file__), "yurameki_defaults.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _find_curves_obj():
    objs = [o for o in bpy.data.objects if o.type == "CURVES"]
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
    if len(lengths) != 1:
        raise ValueError(f"strands have different point counts: {lengths[:8]}")
    pps = int(lengths[0])
    if pps < 2:
        raise ValueError(f"{pps} points per strand is too small")
    return pps, len(spans)


def _solver_kwargs(context, pps: int) -> dict:
    wm = context.window_manager
    return dict(
        points_per_strand=pps,
        sort_axis=wm.yurameki_solver_sort_axis,
        target_length_m=float(wm.yurameki_cylinder_length_cm) * 1.0e-2,
    )


def _check_hair(context):
    obj = _find_curves_obj()
    if obj is None:
        return False, "Need exactly one Curves object"
    try:
        pps, strands = _points_per_strand(obj)
    except Exception as exc:
        return False, f"Hair Check failed: {exc}"
    context.window_manager.yurameki_points_per_strand = pps
    return True, f"Hair Check PASS: {strands} strands, {pps} points per strand"


def _apply_solver_step(context):
    from . import cuda_collider
    from . import solver_interface as si

    obj = _find_curves_obj()
    if obj is None:
        return False, "Need exactly one Curves object"
    wm = context.window_manager
    collider = bpy.data.objects.get(wm.yurameki_collider_obj.strip())
    try:
        pps, _strands = _points_per_strand(obj)
        step_index = int(wm.yurameki_solver_step_index)
        gravity_stats = si.apply_directional_gravity_fk_step(
            obj,
            **_solver_kwargs(context, pps),
            gravity_step_m=float(wm.yurameki_gravity_step_mm) * 1.0e-3,
            step_index=step_index,
            gravity_blend_steps=int(wm.yurameki_gravity_blend_steps),
        )
        collider_text = "collider skipped"
        if collider is not None and collider.type == "MESH":
            collider_result = cuda_collider.apply_capsule_mesh_avoidance(
                obj,
                collider,
                points_per_strand=pps,
                radius_m=float(wm.yurameki_collider_radius_mm) * 1.0e-3,
                sort_axis=wm.yurameki_solver_sort_axis,
                cylinder_length_m=float(wm.yurameki_cylinder_length_cm) * 1.0e-2,
                n_substeps=int(wm.yurameki_collider_substeps),
                max_move_m=float(wm.yurameki_collider_max_move_mm) * 1.0e-3,
            )
            collider_text = (
                f"hits={collider_result.hit_count}/{collider_result.n_cylinders}, "
                f"tip_adjust={collider_result.max_tip_adjust_mm:.3f}mm"
            )
        wm.yurameki_solver_step_index = step_index + 1
    except Exception as exc:
        return False, f"Solver step failed: {exc!r}"
    gx, gy, gz = gravity_stats["gravity_dir"]
    return (
        True,
        f"Solver step {step_index}: "
        f"gravity=({gx:.2f},{gy:.2f},{gz:.2f}), "
        f"len_err={gravity_stats['max_len_err_mm']:.6f}mm, "
        f"fk_tip={gravity_stats['max_tip_displacement_mm']:.3f}mm, "
        f"{collider_text}",
    )


def _settle_hair_to_back(context):
    from . import initial_groom

    obj = _find_curves_obj()
    if obj is None:
        return False, "Need exactly one Curves object"
    wm = context.window_manager
    collider = bpy.data.objects.get(wm.yurameki_collider_obj.strip())
    if collider is None or collider.type != "MESH":
        return False, "Set a Mesh collider object first"
    try:
        stats = initial_groom.settle_hair_back(
            obj,
            collider,
            max_strands=int(wm.yurameki_groom_strands),
            collision_radius_m=float(wm.yurameki_groom_radius_mm) * 1.0e-3,
            follow_radius_m=float(wm.yurameki_groom_follow_mm) * 1.0e-3,
            release_probe_m=float(wm.yurameki_groom_release_mm) * 1.0e-3,
        )
    except Exception as exc:
        return False, f"Settle hair failed: {exc!r}"
    return (
        True,
        f"Settle Hair Back: strands={stats['processed_strands']}, "
        f"time={stats['elapsed_sec']:.2f}s, "
        f"len_err={stats['max_length_error_mm']:.6f}mm, "
        f"close={stats['remaining_close_points']}, "
        f"tip_down={stats['avg_tip_down_dot']:.3f}",
    )


def _detect_cuda_collider(context):
    from . import cuda_collider

    obj = _find_curves_obj()
    if obj is None:
        return False, "Need exactly one Curves object"
    wm = context.window_manager
    collider = bpy.data.objects.get(wm.yurameki_collider_obj.strip())
    if collider is None or collider.type != "MESH":
        return False, "Set a Mesh collider object first"
    try:
        pps, _strands = _points_per_strand(obj)
        result = cuda_collider.detect_capsule_mesh(
            obj,
            collider,
            points_per_strand=pps,
            radius_m=float(wm.yurameki_collider_radius_mm) * 1.0e-3,
            sort_axis=wm.yurameki_solver_sort_axis,
            cylinder_length_m=float(wm.yurameki_cylinder_length_cm) * 1.0e-2,
        )
    except Exception as exc:
        return False, f"CUDA collider failed: {exc!r}"
    return (
        True,
        f"CUDA collider hits: {result.hit_count} / {result.n_cylinders} "
        f"cylinders, triangles={result.n_triangles}",
    )


class YURAMEKI_OT_check_hair(Operator):
    bl_idname = "yurameki.check_hair"
    bl_label = "Check Hair"
    bl_description = "Validate Curves hair input for the prototype solver"

    def execute(self, context):
        ok, message = _check_hair(context)
        wm = context.window_manager
        wm.yurameki_hair_check_status = message
        self.report({"INFO"} if ok else {"ERROR"}, message)
        return {"FINISHED"} if ok else {"CANCELLED"}


class YURAMEKI_OT_apply_solver_step(Operator):
    bl_idname = "yurameki.apply_solver_step"
    bl_label = "Apply Solver Step"
    bl_description = "Apply one long-hair solver step: startup gravity, FK length keep, CUDA collider"

    def execute(self, context):
        ok, message = _apply_solver_step(context)
        self.report({"INFO"} if ok else {"ERROR"}, message)
        return {"FINISHED"} if ok else {"CANCELLED"}


class YURAMEKI_OT_reset_solver_state(Operator):
    bl_idname = "yurameki.reset_solver_state"
    bl_label = "Reset Solver State"
    bl_description = "Reset the solver step counter so startup gravity begins from +Y again"

    def execute(self, context):
        context.window_manager.yurameki_solver_step_index = 0
        self.report({"INFO"}, "Solver state reset")
        return {"FINISHED"}


class YURAMEKI_OT_settle_hair_to_back(Operator):
    bl_idname = "yurameki.settle_hair_to_back"
    bl_label = "Settle Hair Back"
    bl_description = "Initial groom: use CPU BVH to lay straight long hair behind the body"

    def execute(self, context):
        ok, message = _settle_hair_to_back(context)
        self.report({"INFO"} if ok else {"ERROR"}, message)
        return {"FINISHED"} if ok else {"CANCELLED"}


class YURAMEKI_OT_pick_collider(Operator):
    bl_idname = "yurameki.pick_collider"
    bl_label = "Pick Collider"
    bl_description = "Use the active mesh as the CUDA collider"

    def execute(self, context):
        obj = context.active_object
        if obj is None or obj.type != "MESH":
            self.report({"ERROR"}, "Active object must be a mesh")
            return {"CANCELLED"}
        context.window_manager.yurameki_collider_obj = obj.name
        self.report({"INFO"}, f"Collider: {obj.name}")
        return {"FINISHED"}


class YURAMEKI_OT_detect_cuda_collider(Operator):
    bl_idname = "yurameki.detect_cuda_collider"
    bl_label = "Detect CUDA Collider"
    bl_description = "Run detection-only CUDA capsule/mesh collider debug check"

    def execute(self, context):
        ok, message = _detect_cuda_collider(context)
        self.report({"INFO"} if ok else {"ERROR"}, message)
        return {"FINISHED"} if ok else {"CANCELLED"}


_classes = (
    YURAMEKI_OT_check_hair,
    YURAMEKI_OT_apply_solver_step,
    YURAMEKI_OT_reset_solver_state,
    YURAMEKI_OT_settle_hair_to_back,
    YURAMEKI_OT_pick_collider,
    YURAMEKI_OT_detect_cuda_collider,
)


_PROP_NAMES = (
    "yurameki_points_per_strand",
    "yurameki_hair_check_status",
    "yurameki_solver_sort_axis",
    "yurameki_solver_step_index",
    "yurameki_gravity_step_mm",
    "yurameki_gravity_blend_steps",
    "yurameki_groom_strands",
    "yurameki_groom_radius_mm",
    "yurameki_groom_follow_mm",
    "yurameki_groom_release_mm",
    "yurameki_cylinder_length_cm",
    "yurameki_collider_obj",
    "yurameki_collider_radius_mm",
    "yurameki_collider_substeps",
    "yurameki_collider_max_move_mm",
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
            min=2,
            max=256,
            options={"SKIP_SAVE"},
        )
        WindowManager.yurameki_hair_check_status = StringProperty(
            name="Hair Check",
            default="Hair not checked",
            options={"SKIP_SAVE"},
        )
        WindowManager.yurameki_solver_sort_axis = EnumProperty(
            name="Sort Axis",
            items=(
                ("Z", "Z", "Root ascending by Z"),
                ("Y", "Y", "Root ascending by Y"),
                ("X", "X", "Root ascending by X"),
                ("-Z", "-Z", "Root descending by Z"),
                ("-Y", "-Y", "Root descending by Y"),
            ),
            default=str(defaults.get("SOLVE_ORDER_AXIS", "Z")),
            options={"SKIP_SAVE"},
        )
        WindowManager.yurameki_solver_step_index = IntProperty(
            name="Step",
            default=0,
            min=0,
            options={"SKIP_SAVE"},
        )
        WindowManager.yurameki_gravity_step_mm = FloatProperty(
            name="Gravity Step mm",
            default=float(defaults.get("GRAVITY_STEP_MM", 1.0)),
            min=0.0,
            max=10.0,
            precision=3,
            options={"SKIP_SAVE"},
        )
        WindowManager.yurameki_gravity_blend_steps = IntProperty(
            name="Y to -Z Steps",
            default=int(defaults.get("GRAVITY_BLEND_STEPS", 12)),
            min=0,
            max=240,
            options={"SKIP_SAVE"},
        )
        WindowManager.yurameki_groom_strands = IntProperty(
            name="Groom Strands",
            description="Number of lower-Z root strands to groom; 0 means all strands",
            default=int(defaults.get("GROOM_STRANDS", 500)),
            min=0,
            max=200000,
            options={"SKIP_SAVE"},
        )
        WindowManager.yurameki_groom_radius_mm = FloatProperty(
            name="Groom Radius mm",
            default=float(defaults.get("GROOM_RADIUS_MM", 2.5)),
            min=0.1,
            max=20.0,
            precision=3,
            options={"SKIP_SAVE"},
        )
        WindowManager.yurameki_groom_follow_mm = FloatProperty(
            name="Follow mm",
            default=float(defaults.get("GROOM_FOLLOW_MM", 30.0)),
            min=1.0,
            max=200.0,
            precision=3,
            options={"SKIP_SAVE"},
        )
        WindowManager.yurameki_groom_release_mm = FloatProperty(
            name="Release Probe mm",
            default=float(defaults.get("GROOM_RELEASE_MM", 20.0)),
            min=1.0,
            max=200.0,
            precision=3,
            options={"SKIP_SAVE"},
        )
        WindowManager.yurameki_cylinder_length_cm = FloatProperty(
            name="Cylinder Length cm",
            default=float(defaults.get("CYLINDER_LENGTH_CM", 1.0)),
            min=0.1,
            max=10.0,
            precision=3,
            options={"SKIP_SAVE"},
        )
        WindowManager.yurameki_collider_obj = StringProperty(
            name="Collider",
            default=str(defaults.get("COLLIDER_OBJECT", "")),
            options={"SKIP_SAVE"},
        )
        WindowManager.yurameki_collider_radius_mm = FloatProperty(
            name="Radius mm",
            default=float(defaults.get("COLLIDER_RADIUS_MM", 0.5)),
            min=0.01,
            max=20.0,
            precision=3,
            options={"SKIP_SAVE"},
        )
        WindowManager.yurameki_collider_substeps = IntProperty(
            name="Substeps",
            default=int(defaults.get("COLLIDER_SUBSTEPS", 8)),
            min=1,
            max=128,
            options={"SKIP_SAVE"},
        )
        WindowManager.yurameki_collider_max_move_mm = FloatProperty(
            name="Max Move mm",
            default=float(defaults.get("COLLIDER_MAX_MOVE_MM", 1.0)),
            min=0.01,
            max=10.0,
            precision=3,
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
    ui.unregister()
    _clear_props()
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)
