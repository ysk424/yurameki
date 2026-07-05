"""Yurameki 0.6.x -- CUDA straight long-hair solver prototype."""

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


def _find_curves_obj(context=None):
    wm = context.window_manager if context is not None else bpy.context.window_manager
    name = getattr(wm, "yurameki_curves_obj", "").strip()
    if name:
        obj = bpy.data.objects.get(name)
        return obj if obj is not None and obj.type == "CURVES" else None
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
    if min(lengths) < 2:
        raise ValueError(f"{min(lengths)} points per strand is too small")
    pps = int(max(lengths))
    return pps, len(spans)


def _solver_kwargs(context, pps: int) -> dict:
    wm = context.window_manager
    return dict(
        points_per_strand=pps,
        sort_axis=wm.yurameki_solver_sort_axis,
        target_length_m=float(wm.yurameki_cylinder_length_cm) * 1.0e-2,
    )


def _source_body_collider(context):
    wm = context.window_manager
    collider = bpy.data.objects.get(wm.yurameki_collider_obj.strip())
    return collider if collider is not None and collider.type == "MESH" else None


def _source_clothes_collider(context):
    wm = context.window_manager
    name = getattr(wm, "yurameki_clothes_obj", "").strip()
    collider = bpy.data.objects.get(name)
    return collider if collider is not None and collider.type == "MESH" else None


def _compute_colliders(context):
    from . import collider_proxy

    source = _source_body_collider(context)
    if source is None:
        return []
    proxy = collider_proxy.get_valid_proxy(
        source,
        getattr(context.window_manager, "yurameki_collider_proxy_obj", ""),
    )
    colliders = [proxy if proxy is not None else source]
    clothes = _source_clothes_collider(context)
    if clothes is not None:
        colliders.append(clothes)
    return colliders


def _collider_label(colliders) -> str:
    if not colliders:
        return "collider skipped"
    return "+".join(obj.name for obj in colliders)


def _check_hair(context):
    from . import collider_proxy

    obj = _find_curves_obj(context)
    if obj is None:
        return False, "Pick one Hair Curves object"
    collider = _source_body_collider(context)
    if collider is None:
        return False, "Check failed: set a Body mesh first"
    clothes = _source_clothes_collider(context)
    try:
        pps, strands = _points_per_strand(obj)
        proxy_stats = collider_proxy.build_filled_proxy(
            collider,
            getattr(context.window_manager, "yurameki_collider_proxy_obj", ""),
        )
    except Exception as exc:
        return False, f"Check failed: {exc}"
    context.window_manager.yurameki_points_per_strand = pps
    context.window_manager.yurameki_collider_proxy_obj = proxy_stats["proxy_name"]
    lengths = sorted({length for _start, length in _curve_spans(obj.data)})
    if len(lengths) == 1:
        point_text = f"{pps} points"
    else:
        point_text = f"{lengths[0]}-{lengths[-1]} points"
    return (
        True,
        f"Check PASS: {strands} strands, {point_text}, "
        f"proxy={proxy_stats['proxy_name']}, "
        f"filled={proxy_stats['faces_added']} faces, "
        f"ears={proxy_stats.get('ear_faces_removed', 0)}, "
        f"boundary={proxy_stats['boundary_edges_after']}, "
        f"clothes={clothes.name if clothes is not None else 'none'}",
    )


def _apply_solver_step(context):
    from . import cuda_collider
    from . import solver_interface as si

    obj = _find_curves_obj(context)
    if obj is None:
        return False, "Pick one Hair Curves object"
    wm = context.window_manager
    colliders = _compute_colliders(context)
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
        if colliders:
            collider_result = cuda_collider.apply_capsule_mesh_avoidance(
                obj,
                colliders,
                points_per_strand=pps,
                radius_m=float(wm.yurameki_collider_radius_mm) * 1.0e-3,
                sort_axis=wm.yurameki_solver_sort_axis,
                cylinder_length_m=float(wm.yurameki_cylinder_length_cm) * 1.0e-2,
                n_substeps=int(wm.yurameki_collider_substeps),
                max_move_m=float(wm.yurameki_collider_max_move_mm) * 1.0e-3,
            )
            collider_text = (
                f"hits={collider_result.hit_count}/{collider_result.n_cylinders}, "
                f"tip_adjust={collider_result.max_tip_adjust_mm:.3f}mm, "
                f"colliders={_collider_label(colliders)}"
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

    obj = _find_curves_obj(context)
    if obj is None:
        return False, "Pick one Hair Curves object"
    wm = context.window_manager
    colliders = _compute_colliders(context)
    if not colliders:
        return False, "Set a Body mesh first"
    try:
        stats = initial_groom.settle_hair_back(
            obj,
            colliders,
            max_strands=0,
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
        f"root_lock={stats.get('normal_root_locks', 0)}, "
        f"turn={stats.get('angle_limited_rods', 0)}, "
        f"lower_free={stats.get('lower_free_rods', 0)}, "
        f"tip_down={stats['avg_tip_down_dot']:.3f}",
    )


def _detect_cuda_collider(context):
    from . import cuda_collider
    from . import gravity_sim

    obj = _find_curves_obj(context)
    if obj is None:
        return False, "Pick one Hair Curves object"
    wm = context.window_manager
    colliders = _compute_colliders(context)
    if not colliders:
        return False, "Set a Body mesh first"
    try:
        pps, _strands = _points_per_strand(obj)
        result = gravity_sim.prepare_gravity_sim(
            obj,
            colliders,
            points_per_strand=pps,
            radius_m=float(wm.yurameki_collider_radius_mm) * 1.0e-3,
            sort_axis=wm.yurameki_solver_sort_axis,
            cylinder_length_m=float(wm.yurameki_cylinder_length_cm) * 1.0e-2,
        )
    except Exception as exc:
        return False, f"CUDA collider failed: {exc!r}"
    return (
        True,
        f"CUDA ready: strands={result['n_strands']}, cylinders={result['n_cylinders']}, "
        f"hits={result['hit_count']}, triangles={result['n_triangles']}, "
        f"colliders={_collider_label(colliders)}",
    )


def _simulate_gravity(context):
    from . import gravity_sim

    obj = _find_curves_obj(context)
    if obj is None:
        return False, "Pick one Hair Curves object"
    wm = context.window_manager
    colliders = _compute_colliders(context)
    if not colliders:
        return False, "Set a Body mesh first"
    try:
        pps, _strands = _points_per_strand(obj)
        stats = gravity_sim.simulate_gravity_bake(
            obj,
            colliders,
            points_per_strand=pps,
            start_frame=int(wm.yurameki_sim_start_frame),
            end_frame=int(wm.yurameki_sim_end_frame),
            gravity_step_m=float(wm.yurameki_gravity_step_mm) * 1.0e-3,
            radius_m=float(wm.yurameki_collider_radius_mm) * 1.0e-3,
            collider_substeps=int(wm.yurameki_collider_substeps),
            collider_max_move_m=float(wm.yurameki_collider_max_move_mm) * 1.0e-3,
            target_segment_length_m=float(wm.yurameki_cylinder_length_cm) * 1.0e-2,
            interpolation_steps=int(wm.yurameki_sim_interpolation_steps),
            propagation_length_m=float(wm.yurameki_sim_propagation_cm) * 1.0e-2,
            memory_height_m=float(wm.yurameki_sim_memory_height_m),
            memory_strength=float(wm.yurameki_sim_memory_strength),
            bake_mode=wm.yurameki_sim_bake_mode,
        )
    except Exception as exc:
        return False, f"Simulation failed: {exc!r}"
    return (
        True,
        f"Simulate: frames={stats.start_frame}-{stats.end_frame}, "
        f"strands={stats.n_strands}, sim_points={stats.sim_points_per_strand}, "
        f"seg<={stats.max_segment_mm:.2f}mm/{stats.target_segment_cm:.2f}cm, "
        f"interp={stats.interpolation_steps}, "
        f"prop={stats.propagation_length_cm:.1f}cm, "
        f"memory>{stats.memory_height_m:.2f}m/{stats.memory_strength:.2f}, "
        f"bake={stats.bake_mode.lower()}, "
        f"max_substeps={stats.max_substeps}, root_move={stats.max_root_move_mm:.3f}mm, "
        f"hits={stats.total_hits}, time={stats.elapsed_sec:.2f}s",
    )


class YURAMEKI_OT_check_hair(Operator):
    bl_idname = "yurameki.check_hair"
    bl_label = "Check"
    bl_description = "Validate inputs and build a filled collider proxy"

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
    bl_description = "Use the active mesh as the Clothes collider"

    def execute(self, context):
        obj = context.active_object
        if obj is None or obj.type != "MESH":
            self.report({"ERROR"}, "Active object must be a mesh")
            return {"CANCELLED"}
        context.window_manager.yurameki_clothes_obj = obj.name
        self.report({"INFO"}, f"Clothes: {obj.name}")
        return {"FINISHED"}


class YURAMEKI_OT_detect_cuda_collider(Operator):
    bl_idname = "yurameki.detect_cuda_collider"
    bl_label = "Detect CUDA Collider"
    bl_description = "Run detection-only CUDA capsule/mesh collider debug check"

    def execute(self, context):
        ok, message = _detect_cuda_collider(context)
        self.report({"INFO"} if ok else {"ERROR"}, message)
        return {"FINISHED"} if ok else {"CANCELLED"}


class YURAMEKI_OT_simulate_gravity(Operator):
    bl_idname = "yurameki.simulate_gravity"
    bl_label = "Simulate"
    bl_description = "Simulate the selected frame range with the V0.6 fixed chain and bake Curves position keyframes"

    def execute(self, context):
        ok, message = _simulate_gravity(context)
        self.report({"INFO"} if ok else {"ERROR"}, message)
        return {"FINISHED"} if ok else {"CANCELLED"}


_classes = (
    YURAMEKI_OT_check_hair,
    YURAMEKI_OT_apply_solver_step,
    YURAMEKI_OT_reset_solver_state,
    YURAMEKI_OT_settle_hair_to_back,
    YURAMEKI_OT_pick_curves,
    YURAMEKI_OT_pick_collider,
    YURAMEKI_OT_pick_clothes,
    YURAMEKI_OT_detect_cuda_collider,
    YURAMEKI_OT_simulate_gravity,
)


_PROP_NAMES = (
    "yurameki_points_per_strand",
    "yurameki_hair_check_status",
    "yurameki_curves_obj",
    "yurameki_solver_sort_axis",
    "yurameki_solver_step_index",
    "yurameki_gravity_step_mm",
    "yurameki_gravity_blend_steps",
    "yurameki_sim_start_frame",
    "yurameki_sim_end_frame",
    "yurameki_sim_interpolation_steps",
    "yurameki_sim_propagation_cm",
    "yurameki_sim_memory_height_m",
    "yurameki_sim_memory_strength",
    "yurameki_sim_bake_mode",
    "yurameki_groom_radius_mm",
    "yurameki_groom_follow_mm",
    "yurameki_groom_release_mm",
    "yurameki_cylinder_length_cm",
    "yurameki_collider_obj",
    "yurameki_collider_proxy_obj",
    "yurameki_clothes_obj",
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
            name="Check",
            default="Not checked",
            options={"SKIP_SAVE"},
        )
        WindowManager.yurameki_curves_obj = StringProperty(
            name="Hair",
            default=str(defaults.get("CURVES_OBJECT", "")),
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
            default=float(defaults.get("GRAVITY_STEP_MM", 5.0)),
            min=0.0,
            max=50.0,
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
        WindowManager.yurameki_sim_interpolation_steps = IntProperty(
            name="Interpolation",
            default=int(defaults.get("SIM_INTERPOLATION_STEPS", 1)),
            min=0,
            max=8,
            options={"SKIP_SAVE"},
        )
        WindowManager.yurameki_sim_propagation_cm = FloatProperty(
            name="Propagation cm",
            default=float(defaults.get("SIM_PROPAGATION_CM", 50.0)),
            min=1.0,
            max=300.0,
            precision=2,
            options={"SKIP_SAVE"},
        )
        WindowManager.yurameki_sim_memory_height_m = FloatProperty(
            name="Memory Height m",
            default=float(defaults.get("SIM_MEMORY_HEIGHT_M", 1.5)),
            min=-10.0,
            max=10.0,
            precision=3,
            options={"SKIP_SAVE"},
        )
        WindowManager.yurameki_sim_memory_strength = FloatProperty(
            name="Memory Strength",
            default=float(defaults.get("SIM_MEMORY_STRENGTH", 0.55)),
            min=0.0,
            max=1.0,
            precision=3,
            options={"SKIP_SAVE"},
        )
        WindowManager.yurameki_sim_bake_mode = EnumProperty(
            name="Bake",
            items=(
                ("FINAL", "Final Only", "Write only the final simulated frame to the Curves data"),
                ("KEYFRAMES", "Keyframes", "Bake every simulated frame as Curves position keyframes"),
            ),
            default=str(defaults.get("SIM_BAKE_MODE", "FINAL")),
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
            default=int(defaults.get("COLLIDER_SUBSTEPS", 1)),
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
