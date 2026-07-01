"""Yurameki CUDA prototype N-panel."""

from __future__ import annotations

import os
import tomllib

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
    bl_idname = "YURAMEKI_PT_main"
    bl_label = "Yurameki"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Yurameki"

    def draw(self, context):
        layout = self.layout
        wm = context.window_manager

        layout.label(text=f"Yurameki v{_version()}")
        layout.separator(factor=0.4)

        box = layout.box()
        box.label(text="Input")
        col = box.column(align=True)
        col.operator("yurameki.check_hair", icon="CHECKMARK")
        col.prop(wm, "yurameki_points_per_strand")
        status = getattr(wm, "yurameki_hair_check_status", "")
        if status:
            col.label(text=status)

        box = layout.box()
        box.label(text="Cylinder Interface")
        col = box.column(align=True)
        col.prop(wm, "yurameki_cylinder_length_cm")
        col.prop(wm, "yurameki_solver_probe_sort_axis")
        col.prop(wm, "yurameki_solver_probe_axis_step_mm")
        col.prop(wm, "yurameki_solver_probe_tip_back_cm")
        col.prop(wm, "yurameki_solver_probe_path")
        row = col.row(align=True)
        row.operator("yurameki.export_solver_interface", icon="EXPORT")
        row.operator("yurameki.apply_solver_probe_step", icon="FORWARD")

        box = layout.box()
        box.label(text="FK Chain Test")
        col = box.column(align=True)
        col.prop(wm, "yurameki_fk_root_pull_y_mm")
        col.operator("yurameki.apply_fk_root_pull", icon="CONSTRAINT_BONE")

        box = layout.box()
        box.label(text="CUDA Collider")
        col = box.column(align=True)
        row = col.row(align=True)
        row.prop(wm, "yurameki_collider_obj")
        row.operator("yurameki.pick_collider", text="", icon="EYEDROPPER")
        col.prop(wm, "yurameki_collider_radius_mm")
        col.prop(wm, "yurameki_collider_substeps")
        col.prop(wm, "yurameki_collider_max_move_mm")
        col.operator("yurameki.detect_cuda_collider", icon="MOD_PHYSICS")
        col.operator("yurameki.apply_cuda_collider_avoidance", icon="FORCE_FORCE")


_classes = (YURAMEKI_PT_main,)


def register():
    for cls in _classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)
