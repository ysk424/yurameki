"""Yurameki CUDA straight long-hair N-panel."""

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
        box.label(text="Solver Step")
        col = box.column(align=True)
        col.prop(wm, "yurameki_cylinder_length_cm")
        col.prop(wm, "yurameki_solver_sort_axis")
        col.prop(wm, "yurameki_solver_step_index")
        col.prop(wm, "yurameki_gravity_step_mm")
        col.prop(wm, "yurameki_gravity_blend_steps")
        col.prop(wm, "yurameki_groom_strands")
        col.prop(wm, "yurameki_groom_radius_mm")
        col.prop(wm, "yurameki_groom_follow_mm")
        col.prop(wm, "yurameki_groom_release_mm")
        row = col.row(align=True)
        row.prop(wm, "yurameki_collider_obj")
        row.operator("yurameki.pick_collider", text="", icon="EYEDROPPER")
        col.prop(wm, "yurameki_collider_radius_mm")
        col.prop(wm, "yurameki_collider_substeps")
        col.prop(wm, "yurameki_collider_max_move_mm")
        row = col.row(align=True)
        row.operator("yurameki.apply_solver_step", icon="PLAY")
        row.operator("yurameki.reset_solver_state", icon="LOOP_BACK")
        col.operator("yurameki.settle_hair_to_back", text="Settle Hair Back", icon="MOD_CLOTH")

        box = layout.box()
        box.label(text="Debug")
        col = box.column(align=True)
        col.operator("yurameki.detect_cuda_collider", icon="MOD_PHYSICS")


_classes = (YURAMEKI_PT_main,)


def register():
    for cls in _classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)
