"""Yurameki N-panel (VIEW_3D sidebar, tab 'Yurameki')."""
from __future__ import annotations
import os, tomllib
import bpy
from bpy.types import Panel


def _version():
    try:
        path = os.path.join(os.path.dirname(__file__), "blender_manifest.toml")
        with open(path, "rb") as f:
            return tomllib.load(f).get("version", "?")
    except Exception:
        return "?"


class YURAMEKI_PT_main(Panel):
    bl_idname      = "YURAMEKI_PT_main"
    bl_label       = "Yurameki"
    bl_space_type  = "VIEW_3D"
    bl_region_type = "UI"
    bl_category    = "Yurameki"

    def draw(self, context):
        layout = self.layout
        wm     = context.window_manager

        layout.label(text=f"Yurameki  v{_version()}")
        layout.separator(factor=0.3)

        # Body and solver
        box = layout.box()
        box.label(text="Setup")
        col = box.column(align=True)
        row = col.row(align=True)
        row.prop(wm, "yurameki_body_obj", text="Body")
        row.operator("yurameki.pick_body", text="", icon="EYEDROPPER")
        row = col.row(align=True)
        row.prop(wm, "yurameki_cloth_obj", text="Cloth")
        row.operator("yurameki.pick_cloth", text="", icon="EYEDROPPER")
        col.prop(wm, "yurameki_root_min_distance")
        row = col.row(align=True)
        row.operator("yurameki.check_hair", icon="CHECKMARK")
        status = getattr(wm, "yurameki_hair_check_status", "")
        if status:
            icon = "CHECKMARK" if getattr(wm, "yurameki_hair_check_ok", False) else "ERROR"
            col.label(text=status, icon=icon)

        # Static styling on the current frame
        box = layout.box()
        box.label(text="Static Styling")
        col = box.column(align=True)
        col.prop(wm, "yurameki_simulation_steps")
        col.operator("yurameki.simulate", icon="PLAY")

        # Range bake + export
        box = layout.box()
        box.label(text="Bake & Export")
        col = box.column(align=True)
        row = col.row(align=True)
        row.prop(wm, "yurameki_bake_start")
        row.prop(wm, "yurameki_bake_end")
        col.operator("yurameki.use_scene_range", icon="PREVIEW_RANGE")
        col.separator()
        col.prop(wm, "yurameki_auto_frame_interpolation")
        row = col.row(align=True)
        row.enabled = not wm.yurameki_auto_frame_interpolation
        row.prop(wm, "yurameki_frame_interpolation")
        if wm.yurameki_auto_frame_interpolation:
            col.label(text=f"Auto Steps: {wm.yurameki_auto_interpolation_current}")
        col.separator()
        row = col.row(align=True)
        baking = getattr(wm, "yurameki_bake_running", False)
        row.alert = baking
        row.operator(
            "yurameki.bake_range",
            text="Simulate Range" if not baking else "Simulate Range ●",
            icon="RENDER_ANIMATION",
            depress=baking,
        )
        col.separator()
        col.prop(wm, "yurameki_export_path", text="")
        col.operator("yurameki.export_alembic", icon="EXPORT")

        # Physics
        layout.separator(factor=0.3)
        box = layout.box()
        box.label(text="Physics (applied at Simulate / Bake)")
        col = box.column(align=True)
        col.prop(wm, "yurameki_spring_ke")
        col.prop(wm, "yurameki_damping")
        col.prop(wm, "yurameki_particle_mass")
        col.prop(wm, "yurameki_gravity")
        col.separator()
        col.prop(wm, "yurameki_iterations")
        col.prop(wm, "yurameki_interpolation_mag")
        col.prop(wm, "yurameki_collision_margin")
        col.prop(wm, "yurameki_collision_search")
        col.separator()
        col.prop(wm, "yurameki_bending_enabled")
        if getattr(wm, "yurameki_bending_enabled", False):
            col.prop(wm, "yurameki_root_bending_ke")
            col.prop(wm, "yurameki_bending_ke")


_classes = (YURAMEKI_PT_main,)


def register():
    for cls in _classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)
