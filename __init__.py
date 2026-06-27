"""Yurameki (揺らめき) — Warp CUDA XPBD hair simulation and bake.

Yurameki is the simulation core extracted from Tokoya. It takes an existing
Hair Curves object (planted/styled in Tokoya or by hand), simulates it against
an animated Body mesh, and bakes a frame range for playback and export.
"""
from __future__ import annotations
import json, math, os
import bpy
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


class YURAMEKI_OT_simulate(Operator):
    bl_idname      = "yurameki.simulate"
    bl_label       = "Simulate (current frame)"
    bl_description = "Run N steps of static Warp CUDA XPBD styling on the current frame"

    def execute(self, context):
        obj = _find_curves_obj()
        if obj is None:
            self.report({"ERROR"}, "Need exactly one Curves object"); return {"CANCELLED"}
        wm = context.window_manager
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
        ok, message = _recording.manager.bake_range(
            context.scene,
            int(wm.yurameki_bake_start),
            int(wm.yurameki_bake_end),
        )
        if not ok:
            self.report({"ERROR"}, message); return {"CANCELLED"}
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
    "yurameki_bake_running",
    "yurameki_spring_ke", "yurameki_damping", "yurameki_particle_mass",
    "yurameki_gravity", "yurameki_iterations",
    "yurameki_collision_margin", "yurameki_collision_search",
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
