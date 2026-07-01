"""Yurameki 0.4.x -- CUDA solver prototype interface.

This fork intentionally contains only the new Blender <-> solver interface:

* read a Curves object
* split each strand into 1 cm cylinders
* build a fixed solve-order array
* export the arrays
* apply one probe step back to Blender while preserving cylinder length
"""

from __future__ import annotations

import json
import os

import bpy
from bpy.props import EnumProperty, FloatProperty, IntProperty, StringProperty
from bpy.types import Operator, WindowManager

from . import ui


def _load_defaults():
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


def _solver_probe_kwargs(context, pps: int) -> dict:
    wm = context.window_manager
    return dict(
        points_per_strand=pps,
        sort_axis=wm.yurameki_solver_probe_sort_axis,
        target_length_m=float(wm.yurameki_cylinder_length_cm) * 1.0e-2,
        axis_step_m=float(wm.yurameki_solver_probe_axis_step_mm) * 1.0e-3,
        tip_back_m=float(wm.yurameki_solver_probe_tip_back_cm) * 1.0e-2,
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


def _export_solver_interface(context):
    from . import solver_interface as si

    obj = _find_curves_obj()
    if obj is None:
        return False, "Need exactly one Curves object"
    try:
        pps, _strands = _points_per_strand(obj)
        path = context.window_manager.yurameki_solver_probe_path.strip()
        if not path:
            path = "//yurameki_solver_probe"
        if not path.lower().endswith(".npz"):
            path += ".npz"
        npz_path = bpy.path.abspath(path)
        json_path = bpy.path.abspath(os.path.splitext(path)[0] + ".json")
        abs_npz = si.export_probe_data(obj, npz_path, **_solver_probe_kwargs(context, pps))
        abs_json = si.export_probe_json(obj, json_path, **_solver_probe_kwargs(context, pps))
    except Exception as exc:
        return False, f"Solver interface export failed: {exc!r}"
    return True, f"Exported solver interface: {abs_npz}; {abs_json}"


def _apply_solver_probe_step(context):
    from . import solver_interface as si

    obj = _find_curves_obj()
    if obj is None:
        return False, "Need exactly one Curves object"
    try:
        pps, _strands = _points_per_strand(obj)
        stats = si.apply_probe_step(obj, **_solver_probe_kwargs(context, pps))
    except Exception as exc:
        return False, f"Solver probe step failed: {exc!r}"
    return (
        True,
        "Probe step applied: "
        f"{stats['n_strands']} strands, {stats['n_cylinders']} cylinders, "
        f"max length error {stats['max_len_err_mm']:.6f} mm",
    )


def _apply_fk_root_pull(context):
    from . import solver_interface as si

    obj = _find_curves_obj()
    if obj is None:
        return False, "Need exactly one Curves object"
    try:
        wm = context.window_manager
        pps, _strands = _points_per_strand(obj)
        stats = si.apply_root_pull_fk_step(
            obj,
            points_per_strand=pps,
            sort_axis=wm.yurameki_solver_probe_sort_axis,
            target_length_m=float(wm.yurameki_cylinder_length_cm) * 1.0e-2,
            root_pull_y_m=float(wm.yurameki_fk_root_pull_y_mm) * 1.0e-3,
        )
    except Exception as exc:
        return False, f"FK root pull failed: {exc!r}"
    return (
        True,
        "FK root pull applied: "
        f"{stats['n_strands']} strands, {stats['n_cylinders']} cylinders, "
        f"len_err={stats['max_len_err_mm']:.6f}mm, "
        f"gap={stats['max_chain_gap_mm']:.6f}mm, "
        f"tip={stats['max_tip_displacement_mm']:.3f}mm",
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
            sort_axis=wm.yurameki_solver_probe_sort_axis,
            cylinder_length_m=float(wm.yurameki_cylinder_length_cm) * 1.0e-2,
        )
    except Exception as exc:
        return False, f"CUDA collider failed: {exc!r}"
    return (
        True,
        f"CUDA collider hits: {result.hit_count} / {result.n_cylinders} "
        f"cylinders, triangles={result.n_triangles}",
    )


def _apply_cuda_collider_avoidance(context):
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
        result = cuda_collider.apply_capsule_mesh_avoidance(
            obj,
            collider,
            points_per_strand=pps,
            radius_m=float(wm.yurameki_collider_radius_mm) * 1.0e-3,
            sort_axis=wm.yurameki_solver_probe_sort_axis,
            cylinder_length_m=float(wm.yurameki_cylinder_length_cm) * 1.0e-2,
            n_substeps=int(wm.yurameki_collider_substeps),
            max_move_m=float(wm.yurameki_collider_max_move_mm) * 1.0e-3,
        )
    except Exception as exc:
        return False, f"CUDA avoidance failed: {exc!r}"
    return (
        True,
        f"CUDA avoidance: hits={result.hit_count}/{result.n_cylinders}, "
        f"len_err={result.max_length_error_mm:.6f}mm, "
        f"tip_adjust={result.max_tip_adjust_mm:.3f}mm",
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


class YURAMEKI_OT_export_solver_interface(Operator):
    bl_idname = "yurameki.export_solver_interface"
    bl_label = "Export Solver Interface"
    bl_description = "Export 1 cm cylinder arrays and fixed solve order"

    def execute(self, context):
        ok, message = _export_solver_interface(context)
        self.report({"INFO"} if ok else {"ERROR"}, message)
        return {"FINISHED"} if ok else {"CANCELLED"}


class YURAMEKI_OT_apply_solver_probe_step(Operator):
    bl_idname = "yurameki.apply_solver_probe_step"
    bl_label = "Apply Probe Step"
    bl_description = "Apply the length-preserving 1 cm cylinder probe step"

    def execute(self, context):
        ok, message = _apply_solver_probe_step(context)
        self.report({"INFO"} if ok else {"ERROR"}, message)
        return {"FINISHED"} if ok else {"CANCELLED"}


class YURAMEKI_OT_apply_fk_root_pull(Operator):
    bl_idname = "yurameki.apply_fk_root_pull"
    bl_label = "Apply FK Root Pull"
    bl_description = "Move only strand roots and rebuild the 1 cm cylinder FK chain"

    def execute(self, context):
        ok, message = _apply_fk_root_pull(context)
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
    bl_description = "Run detection-only CUDA capsule/mesh collider test"

    def execute(self, context):
        ok, message = _detect_cuda_collider(context)
        self.report({"INFO"} if ok else {"ERROR"}, message)
        return {"FINISHED"} if ok else {"CANCELLED"}


class YURAMEKI_OT_apply_cuda_collider_avoidance(Operator):
    bl_idname = "yurameki.apply_cuda_collider_avoidance"
    bl_label = "Apply CUDA Avoidance"
    bl_description = "Move cylinder tips away from the collider on CUDA"

    def execute(self, context):
        ok, message = _apply_cuda_collider_avoidance(context)
        self.report({"INFO"} if ok else {"ERROR"}, message)
        return {"FINISHED"} if ok else {"CANCELLED"}


_classes = (
    YURAMEKI_OT_check_hair,
    YURAMEKI_OT_export_solver_interface,
    YURAMEKI_OT_apply_solver_probe_step,
    YURAMEKI_OT_apply_fk_root_pull,
    YURAMEKI_OT_pick_collider,
    YURAMEKI_OT_detect_cuda_collider,
    YURAMEKI_OT_apply_cuda_collider_avoidance,
)


_PROP_NAMES = (
    "yurameki_points_per_strand",
    "yurameki_hair_check_status",
    "yurameki_solver_probe_path",
    "yurameki_solver_probe_sort_axis",
    "yurameki_solver_probe_axis_step_mm",
    "yurameki_solver_probe_tip_back_cm",
    "yurameki_fk_root_pull_y_mm",
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
        WindowManager.yurameki_solver_probe_path = StringProperty(
            name="Solver Probe Path",
            default="//yurameki_solver_probe.npz",
            subtype="FILE_PATH",
            options={"SKIP_SAVE"},
        )
        WindowManager.yurameki_solver_probe_sort_axis = EnumProperty(
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
        WindowManager.yurameki_cylinder_length_cm = FloatProperty(
            name="Cylinder Length cm",
            default=float(defaults.get("CYLINDER_LENGTH_CM", 1.0)),
            min=0.1,
            max=10.0,
            precision=3,
            options={"SKIP_SAVE"},
        )
        WindowManager.yurameki_solver_probe_axis_step_mm = FloatProperty(
            name="Y Step mm",
            default=float(defaults.get("PROBE_AXIS_STEP_MM", 0.5)),
            min=0.0,
            max=10.0,
            precision=3,
            options={"SKIP_SAVE"},
        )
        WindowManager.yurameki_solver_probe_tip_back_cm = FloatProperty(
            name="Tip Back cm",
            default=float(defaults.get("PROBE_TIP_BACK_CM", 3.0)),
            min=0.0,
            max=30.0,
            precision=3,
            options={"SKIP_SAVE"},
        )
        WindowManager.yurameki_fk_root_pull_y_mm = FloatProperty(
            name="Root Pull Y mm",
            default=float(defaults.get("FK_ROOT_PULL_Y_MM", 0.3)),
            min=-50.0,
            max=50.0,
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
