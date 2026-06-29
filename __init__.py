"""Yurameki (揺らめき) — Warp CUDA XPBD hair simulation and bake.

Yurameki is the simulation core extracted from Tokoya. It takes an existing
Hair Curves object (planted/styled in Tokoya or by hand), simulates it against
an animated Body mesh, and bakes a frame range for playback and export.
"""
from __future__ import annotations
import json, math, os
import bpy
import numpy as np
from bpy.app.handlers import persistent
from bpy.props import (
    BoolProperty, EnumProperty, FloatProperty, FloatVectorProperty,
    IntProperty, StringProperty,
)
from bpy.types import Operator, WindowManager
from . import ui


def _load_defaults():
    path = os.path.join(os.path.dirname(__file__), "yurameki_defaults.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _snapshot_sim_params(wm):
    from . import _world_passthrough as _wp
    from . import _recording
    pps = max(3, int(wm.yurameki_points_per_strand))
    _wp.POINTS_PER_STRAND = pps
    _recording.POINTS_PER_STRAND = pps
    _wp.SPRING_KE       = 10.0 ** wm.yurameki_spring_ke
    _wp.DAMPING         = wm.yurameki_damping       / 100.0
    _wp.PARTICLE_MASS   = wm.yurameki_particle_mass / 1000.0
    _wp.GRAVITY         = tuple(wm.yurameki_gravity)
    _wp.ITERATIONS      = wm.yurameki_iterations
    _wp.SUBSTEPS        = 1
    _wp.BENDING_ENABLED = wm.yurameki_bending_enabled
    _wp.ROOT_BENDING_KE = 10.0 ** wm.yurameki_root_bending_ke
    _wp.BENDING_KE      = 10.0 ** wm.yurameki_bending_ke
    _wp.COLLISION_MARGIN = wm.yurameki_collision_margin / 1000.0
    _wp.COLLISION_SEARCH = wm.yurameki_collision_search / 1000.0
    _wp.BODY_COLLISION_TARGET = wm.yurameki_body_obj.strip()
    _wp.CLOTH_COLLISION_TARGET = wm.yurameki_cloth_obj.strip()


def _find_curves_obj():
    objs = [o for o in bpy.data.objects if o.type == "CURVES"]
    return objs[0] if len(objs) == 1 else None


def _curve_spans(curves_data):
    return [
        (int(curve.first_point_index), int(curve.points_length))
        for curve in curves_data.curves
    ]


def _repair_close_roots(obj, scene, spans, min_distance_m: float) -> int:
    if min_distance_m <= 0.0 or len(spans) < 2:
        return 0
    from math import cos, sin
    from mathutils import Vector
    from mathutils.kdtree import KDTree
    from . import _world_passthrough as _wp

    n_total = len(obj.data.points)
    repaired_pairs = 0
    for _pass in range(4):
        dg = bpy.context.evaluated_depsgraph_get()
        obj_eval = obj.evaluated_get(dg)
        eval_world = _wp._read_world(
            obj_eval.data, n_total, obj_eval.matrix_world
        )
        orig_world = _wp._read_world(
            obj.data, n_total, obj.matrix_world
        )
        if eval_world is None or orig_world is None:
            return repaired_pairs
        offset_world = eval_world - orig_world

        roots = eval_world[[start for start, _length in spans]]
        tree = KDTree(len(roots))
        for index, root in enumerate(roots):
            tree.insert(Vector(root), index)
        tree.balance()

        offsets = np.zeros_like(roots)
        pairs_this_pass = 0
        for i, root in enumerate(roots):
            for co, j, dist in tree.find_range(Vector(root), min_distance_m):
                if j <= i:
                    continue
                push = min_distance_m - float(dist)
                if push <= 0.0:
                    continue
                if dist > 1.0e-9:
                    direction = (Vector(root) - co).normalized()
                else:
                    angle = float((i + 1) * 12.9898 + (j + 1) * 78.233)
                    direction = Vector((cos(angle), sin(angle), 0.0)).normalized()
                delta = np.array(direction, dtype=np.float32) * (push * 0.5)
                offsets[i] += delta
                offsets[j] -= delta
                pairs_this_pass += 1

        if pairs_this_pass == 0:
            break
        for strand, (start, length) in enumerate(spans):
            eval_world[start:start + length] += offsets[strand]
        _wp._write_world(obj, eval_world, offset=offset_world)
        bpy.context.view_layer.update()
        repaired_pairs += pairs_this_pass
    return repaired_pairs


def _check_hair(context, repair_roots: bool = True):
    obj = _find_curves_obj()
    if obj is None:
        return False, "Need exactly one Curves object"
    wm = context.window_manager
    spans = _curve_spans(obj.data)
    if not spans:
        return False, "Curves object has no strands"
    lengths = [length for _start, length in spans]
    unique_lengths = sorted(set(lengths))
    if len(unique_lengths) != 1:
        return (
            False,
            "Hair Check failed: strands have different point counts "
            f"({unique_lengths[:8]})",
        )
    pps = unique_lengths[0]
    if pps < 3:
        return False, f"Hair Check failed: {pps} points per strand is too small"
    wm.yurameki_points_per_strand = pps

    min_distance_m = float(wm.yurameki_root_min_distance) / 1000.0
    repaired = 0
    if repair_roots:
        repaired = _repair_close_roots(obj, context.scene, spans, min_distance_m)

    message = (
        f"Hair Check PASS: {len(spans)} strands, {pps} points per strand"
    )
    if repaired:
        message += f"; repaired {repaired} close root pairs"
    return True, message


def _clear_recording_cache():
    from . import _recording
    _recording.manager.clear()


def _sync_bake_range_to_scene():
    """Pre-fill bake Start/End with the scene frame range (1..last frame)."""
    wm = getattr(bpy.context, "window_manager", None)
    scene = getattr(bpy.context, "scene", None)
    if wm is None or scene is None:
        return
    if hasattr(wm, "yurameki_bake_start"):
        wm.yurameki_bake_start = int(scene.frame_start)
    if hasattr(wm, "yurameki_bake_end"):
        wm.yurameki_bake_end = int(scene.frame_end)


def _sync_default_ui_state():
    wm = getattr(bpy.context, "window_manager", None)
    if wm is None:
        return
    if hasattr(wm, "yurameki_auto_frame_interpolation"):
        wm.yurameki_auto_frame_interpolation = True
    if hasattr(wm, "yurameki_bake_progress"):
        wm.yurameki_bake_progress = 0.0
    if hasattr(wm, "yurameki_bake_progress_text"):
        wm.yurameki_bake_progress_text = "Ready"


class YURAMEKI_OT_simulate(Operator):
    bl_idname      = "yurameki.simulate"
    bl_label       = "Simulate (current frame)"
    bl_description = "Run N steps of static Warp CUDA XPBD styling on the current frame"

    def execute(self, context):
        obj = _find_curves_obj()
        if obj is None:
            self.report({"ERROR"}, "Need exactly one Curves object"); return {"CANCELLED"}
        wm = context.window_manager
        ok, message = _check_hair(context, repair_roots=True)
        wm.yurameki_hair_check_ok = ok
        wm.yurameki_hair_check_status = message
        if not ok:
            self.report({"WARNING"}, message)
            return {"CANCELLED"}
        _snapshot_sim_params(wm)

        from . import _world_passthrough as _wp
        body_name = wm.yurameki_body_obj.strip()
        body = bpy.data.objects.get(body_name)
        if body is None or body.type != "MESH":
            self.report({"ERROR"}, "Select a Body Mesh first"); return {"CANCELLED"}
        cloth_name = wm.yurameki_cloth_obj.strip()
        if cloth_name:
            cloth = bpy.data.objects.get(cloth_name)
            if cloth is None or cloth.type != "MESH":
                self.report({"ERROR"}, "Cloth Collider must be a mesh")
                return {"CANCELLED"}
        _wp.BODY_COLLISION_TARGET = body.name
        status = _wp.run_simulation(
            obj.name, wm.yurameki_simulation_steps, context.scene
        )
        if status.startswith("ERROR"):
            self.report({"ERROR"}, status); return {"CANCELLED"}
        _clear_recording_cache()
        self.report({"INFO"}, status)
        return {"FINISHED"}


class YURAMEKI_OT_check_hair(Operator):
    bl_idname = "yurameki.check_hair"
    bl_label = "Check Hair"
    bl_description = "Validate hair strands and separate roots that are too close"

    def execute(self, context):
        ok, message = _check_hair(context, repair_roots=True)
        wm = context.window_manager
        wm.yurameki_hair_check_ok = ok
        wm.yurameki_hair_check_status = message
        self.report({"INFO"} if ok else {"WARNING"}, message)
        if ok:
            _clear_recording_cache()
            return {"FINISHED"}
        return {"CANCELLED"}


def _tag_redraw(context):
    area = getattr(context, "area", None)
    if area is not None:
        area.tag_redraw()


class _BusyOperatorMixin:
    busy_prop = ""

    def invoke(self, context, _event):
        if self.busy_prop:
            setattr(context.window_manager, self.busy_prop, True)
        _tag_redraw(context)
        self._busy_timer = context.window_manager.event_timer_add(
            0.01, window=context.window
        )
        context.window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        if event.type != "TIMER":
            return {"RUNNING_MODAL"}
        context.window_manager.event_timer_remove(self._busy_timer)
        return self._execute_busy(context)

    def execute(self, context):
        if self.busy_prop:
            setattr(context.window_manager, self.busy_prop, True)
        _tag_redraw(context)
        return self._execute_busy(context)

    def _execute_busy(self, context):
        try:
            return self._execute(context)
        finally:
            if self.busy_prop:
                setattr(context.window_manager, self.busy_prop, False)
            _tag_redraw(context)


class YURAMEKI_OT_record(Operator):
    bl_idname = "yurameki.record"
    bl_label = "REC"
    bl_description = (
        "Toggle timeline recording. Recording advances only on consecutive "
        "forward frames; reverse playback or a frame jump aborts to playback"
    )

    def execute(self, context):
        _snapshot_sim_params(context.window_manager)
        from . import _recording
        ok, message = _recording.manager.toggle(context.scene)
        if not ok:
            self.report({"ERROR"}, message)
            return {"CANCELLED"}
        self.report({"INFO"}, message)
        return {"FINISHED"}


class YURAMEKI_OT_bake_range(_BusyOperatorMixin, Operator):
    bl_idname = "yurameki.bake_range"
    bl_label = "Simulate Range"
    bl_description = (
        "Simulate the Start..End frame range in one batch and cache every "
        "frame for timeline playback and export"
    )
    busy_prop = "yurameki_bake_running"

    def _execute(self, context):
        obj = _find_curves_obj()
        if obj is None:
            self.report({"ERROR"}, "Need exactly one Curves object"); return {"CANCELLED"}
        wm = context.window_manager
        ok, message = _check_hair(context, repair_roots=True)
        wm.yurameki_hair_check_ok = ok
        wm.yurameki_hair_check_status = message
        if not ok:
            self.report({"WARNING"}, message)
            return {"CANCELLED"}
        body = bpy.data.objects.get(wm.yurameki_body_obj.strip())
        if body is None or body.type != "MESH":
            self.report({"ERROR"}, "Select a Body Mesh first"); return {"CANCELLED"}
        cloth_name = wm.yurameki_cloth_obj.strip()
        if cloth_name:
            cloth = bpy.data.objects.get(cloth_name)
            if cloth is None or cloth.type != "MESH":
                self.report({"ERROR"}, "Cloth Collider must be a mesh")
                return {"CANCELLED"}
        _snapshot_sim_params(wm)
        from . import _recording
        start = int(wm.yurameki_bake_start)
        end = int(wm.yurameki_bake_end)
        total = max(1, end - start + 1)

        wm.yurameki_bake_progress = 0.0
        wm.yurameki_bake_progress_text = f"Starting {start}-{end}"
        context.window_manager.progress_begin(0, total)

        def _progress(frame, first, last, completed):
            percent = min(100.0, max(0.0, 100.0 * completed / total))
            wm.yurameki_bake_progress = percent
            wm.yurameki_bake_progress_text = (
                f"Frame {frame}/{last}  {percent:.1f}%"
            )
            context.window_manager.progress_update(completed)
            _tag_redraw(context)
            if completed == 1 or completed == total or completed % 5 == 0:
                try:
                    bpy.ops.wm.redraw_timer(type="DRAW_WIN_SWAP", iterations=1)
                except Exception:
                    pass

        try:
            ok, message = _recording.manager.bake_range(
                context.scene,
                start,
                end,
                progress_callback=_progress,
            )
        finally:
            context.window_manager.progress_end()
        if not ok:
            wm.yurameki_bake_progress_text = message
            self.report({"ERROR"}, message); return {"CANCELLED"}
        wm.yurameki_bake_progress = 100.0
        wm.yurameki_bake_progress_text = message
        self.report({"INFO"}, message)
        return {"FINISHED"}


class YURAMEKI_OT_use_scene_range(Operator):
    bl_idname = "yurameki.use_scene_range"
    bl_label = "Use Scene Range"
    bl_description = "Set Start/End to the scene frame range"

    def execute(self, context):
        scene = context.scene
        context.window_manager.yurameki_bake_start = int(scene.frame_start)
        context.window_manager.yurameki_bake_end = int(scene.frame_end)
        return {"FINISHED"}


class YURAMEKI_OT_export_alembic(Operator):
    bl_idname = "yurameki.export_alembic"
    bl_label = "Export Alembic"
    bl_description = (
        "Export the baked hair to the Export Path as Alembic "
        "(placeholder — server-side export, not implemented in v0.1.0)"
    )

    def execute(self, context):
        path = context.window_manager.yurameki_export_path.strip()
        if not path:
            self.report({"ERROR"}, "Set an Export Path first"); return {"CANCELLED"}
        # v0.1.0 ships the UI only. Baking is migrating to a standalone server,
        # so on-device Alembic writing is intentionally left unimplemented.
        self.report(
            {"WARNING"},
            "Alembic export is not implemented in v0.1.0 "
            "(handled by the upcoming server).",
        )
        return {"CANCELLED"}


class YURAMEKI_OT_pick_body(Operator):
    bl_idname = "yurameki.pick_body"
    bl_label = "Pick Active as Body"

    def execute(self, context):
        obj = context.active_object
        if obj is None or obj.type != "MESH":
            self.report({"WARNING"}, "Active object must be a mesh")
            return {"CANCELLED"}
        context.window_manager.yurameki_body_obj = obj.name
        curves = _find_curves_obj()
        if curves is not None:
            curves.data.surface = obj
            if obj.data.uv_layers.active:
                curves.data.surface_uv_map = obj.data.uv_layers.active.name
        self.report({"INFO"}, f"Body Mesh: {obj.name!r}")
        return {"FINISHED"}


class YURAMEKI_OT_pick_cloth(Operator):
    bl_idname = "yurameki.pick_cloth"
    bl_label = "Pick Active as Cloth Collider"

    def execute(self, context):
        obj = context.active_object
        if obj is None or obj.type != "MESH":
            self.report({"WARNING"}, "Active object must be a mesh")
            return {"CANCELLED"}
        context.window_manager.yurameki_cloth_obj = obj.name
        self.report({"INFO"}, f"Cloth Collider: {obj.name!r}")
        return {"FINISHED"}


_classes = (
    YURAMEKI_OT_simulate,
    YURAMEKI_OT_check_hair,
    YURAMEKI_OT_record,
    YURAMEKI_OT_bake_range,
    YURAMEKI_OT_use_scene_range,
    YURAMEKI_OT_export_alembic,
    YURAMEKI_OT_pick_body,
    YURAMEKI_OT_pick_cloth,
)


@persistent
def _on_frame_change_post(scene, _depsgraph):
    from . import _recording
    _recording.manager.on_frame_change(scene)


@persistent
def _on_save_post(_filepath):
    from . import _recording
    _recording.manager.save_cache()


@persistent
def _on_load_post(_filepath):
    from . import _recording
    _sync_bake_range_to_scene()
    _sync_default_ui_state()
    if _recording.manager.load_cache():
        _recording.manager.restore(bpy.context.scene, bpy.context.scene.frame_current)


def _install_handlers():
    handlers = (
        (bpy.app.handlers.frame_change_post, _on_frame_change_post),
        (bpy.app.handlers.save_post, _on_save_post),
        (bpy.app.handlers.load_post, _on_load_post),
    )
    for collection, handler in handlers:
        if handler not in collection:
            collection.append(handler)


def _uninstall_handlers():
    handlers = (
        (bpy.app.handlers.frame_change_post, _on_frame_change_post),
        (bpy.app.handlers.save_post, _on_save_post),
        (bpy.app.handlers.load_post, _on_load_post),
    )
    for collection, handler in handlers:
        if handler in collection:
            collection.remove(handler)


_PROP_NAMES = (
    "yurameki_simulation_steps", "yurameki_frame_interpolation",
    "yurameki_auto_frame_interpolation", "yurameki_auto_interpolation_current",
    "yurameki_interpolation_mag", "yurameki_record_mode",
    "yurameki_body_obj",
    "yurameki_cloth_obj",
    "yurameki_bake_start", "yurameki_bake_end", "yurameki_export_path",
    "yurameki_bake_running", "yurameki_bake_progress",
    "yurameki_bake_progress_text",
    "yurameki_spring_ke", "yurameki_damping", "yurameki_particle_mass",
    "yurameki_gravity", "yurameki_iterations",
    "yurameki_collision_margin", "yurameki_collision_search",
    "yurameki_root_min_distance", "yurameki_hair_check_ok",
    "yurameki_hair_check_status", "yurameki_points_per_strand",
    "yurameki_bending_enabled", "yurameki_root_bending_ke", "yurameki_bending_ke",
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
    handlers_installed = False
    try:
        for cls in _classes:
            bpy.utils.register_class(cls)
            registered_classes.append(cls)

        WindowManager.yurameki_simulation_steps = IntProperty(
            name="Simulation Steps", description="Number of XPBD simulation steps",
            default=20, min=1, max=500, options={"SKIP_SAVE"})
        WindowManager.yurameki_frame_interpolation = IntProperty(
            name="Frame Interpolation",
            description=(
                "Physics evaluations between animation frames. Increase when "
                "fast Body motion causes tunneling"
            ),
            default=2, min=1, max=64, options={"SKIP_SAVE"})
        WindowManager.yurameki_auto_frame_interpolation = BoolProperty(
            name="Auto Frame Interpolation",
            description=(
                "Choose 1-64 interpolation steps from root motion and "
                "median root spacing"
            ),
            default=True, options={"SKIP_SAVE"})
        WindowManager.yurameki_auto_interpolation_current = IntProperty(
            name="Auto Steps", default=1, min=1, max=1024,
            options={"SKIP_SAVE"})
        WindowManager.yurameki_interpolation_mag = IntProperty(
            name="Interpolation Mag",
            description="Multiply Auto or manual Frame Interpolation by this value",
            default=int(defaults["INTERPOLATION_MAG"]),
            min=1, max=16, options={"SKIP_SAVE"})
        WindowManager.yurameki_record_mode = EnumProperty(
            name="Recording Mode",
            items=(
                ("PLAYBACK", "Playback", "Play cached frames without simulation"),
                ("RECORDING", "Recording", "Simulate and cache forward frames"),
            ),
            default="PLAYBACK",
            options={"SKIP_SAVE"},
        )
        WindowManager.yurameki_body_obj = StringProperty(
            name="Body Mesh", description="Animated surface and collision mesh",
            default="", options={"SKIP_SAVE"})
        WindowManager.yurameki_cloth_obj = StringProperty(
            name="Cloth Collider",
            description="Optional animated Alembic mesh used only for collision",
            default="", options={"SKIP_SAVE"})
        WindowManager.yurameki_bake_start = IntProperty(
            name="Start", description="First frame to simulate",
            default=1, options={"SKIP_SAVE"})
        WindowManager.yurameki_bake_end = IntProperty(
            name="End", description="Last frame to simulate",
            default=250, options={"SKIP_SAVE"})
        WindowManager.yurameki_export_path = StringProperty(
            name="Export Path", description="Alembic (.abc) output file",
            default="//hair.abc", subtype="FILE_PATH", options={"SKIP_SAVE"})
        WindowManager.yurameki_bake_running = BoolProperty(
            name="Bake Running", default=False, options={"SKIP_SAVE"})
        WindowManager.yurameki_bake_progress = FloatProperty(
            name="Progress", default=0.0, min=0.0, max=100.0,
            subtype="PERCENTAGE", options={"SKIP_SAVE"})
        WindowManager.yurameki_bake_progress_text = StringProperty(
            name="Bake Progress", default="Ready", options={"SKIP_SAVE"})
        WindowManager.yurameki_spring_ke = FloatProperty(
            name="Stiffness 10^N", default=math.log10(defaults["SPRING_KE"]),
            min=1.0, max=9.0, step=10, precision=2, options={"SKIP_SAVE"})
        WindowManager.yurameki_damping = FloatProperty(
            name="Damping /100", default=defaults["DAMPING"] * 100.0,
            min=0.0, max=50.0, step=10, precision=1, options={"SKIP_SAVE"})
        WindowManager.yurameki_particle_mass = FloatProperty(
            name="Mass /1000", default=defaults["PARTICLE_MASS"] * 1000.0,
            min=1.0, max=10000.0, step=100, precision=1, options={"SKIP_SAVE"})
        WindowManager.yurameki_gravity = FloatVectorProperty(
            name="Gravity m/s2", default=defaults["GRAVITY"],
            size=3, subtype="XYZ", min=-100.0, max=100.0,
            step=10, precision=2, options={"SKIP_SAVE"})
        WindowManager.yurameki_iterations = IntProperty(
            name="Iterations", default=int(defaults["ITERATIONS"]),
            min=1, max=64, options={"SKIP_SAVE"})
        WindowManager.yurameki_collision_margin = FloatProperty(
            name="Collision Radius mm",
            default=float(defaults.get("COLLISION_MARGIN", 0.0005)) * 1000.0,
            min=0.0, max=20.0, step=10, precision=3, options={"SKIP_SAVE"})
        WindowManager.yurameki_collision_search = FloatProperty(
            name="Collision Search mm",
            default=float(defaults.get("COLLISION_SEARCH", 0.003)) * 1000.0,
            min=0.1, max=100.0, step=10, precision=3, options={"SKIP_SAVE"})
        WindowManager.yurameki_root_min_distance = FloatProperty(
            name="Root Min Distance mm",
            default=float(defaults.get("ROOT_MIN_DISTANCE", 0.0001)) * 1000.0,
            min=0.0, max=10.0, step=10, precision=3, options={"SKIP_SAVE"})
        WindowManager.yurameki_hair_check_ok = BoolProperty(
            name="Hair Check OK", default=False, options={"SKIP_SAVE"})
        WindowManager.yurameki_hair_check_status = StringProperty(
            name="Hair Check", default="Hair not checked", options={"SKIP_SAVE"})
        WindowManager.yurameki_points_per_strand = IntProperty(
            name="Points Per Strand", default=9, min=3, max=256,
            options={"SKIP_SAVE"})
        WindowManager.yurameki_bending_enabled = BoolProperty(
            name="Bending", default=bool(defaults["BENDING_ENABLED"]),
            options={"SKIP_SAVE"})
        WindowManager.yurameki_root_bending_ke = FloatProperty(
            name="Root Stiff 10^N", default=math.log10(defaults["ROOT_BENDING_KE"]),
            min=0.0, max=7.0, step=10, precision=2, options={"SKIP_SAVE"})
        WindowManager.yurameki_bending_ke = FloatProperty(
            name="Strand Stiff 10^N", default=math.log10(defaults["BENDING_KE"]),
            min=0.0, max=6.0, step=10, precision=2, options={"SKIP_SAVE"})
        ui.register()
        ui_registered = True
        _install_handlers()
        handlers_installed = True
        _sync_bake_range_to_scene()
        _sync_default_ui_state()
        from . import _recording
        if _recording.manager.load_cache():
            scene = getattr(bpy.context, "scene", None)
            if scene is not None:
                _recording.manager.restore(scene, scene.frame_current)
    except Exception:
        if handlers_installed:
            _uninstall_handlers()
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
    _uninstall_handlers()
    from . import _recording
    _recording.manager.stop("extension unregister")
    ui.unregister()
    _clear_props()
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)
