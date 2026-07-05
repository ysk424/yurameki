"""Yurameki NVIDIA Warp N-panel."""

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
        row = col.row(align=True)
        row.prop(wm, "yurameki_curves_obj")
        row.operator("yurameki.pick_curves", text="", icon="EYEDROPPER")
        row = col.row(align=True)
        row.prop(wm, "yurameki_collider_obj")
        row.operator("yurameki.pick_collider", text="", icon="EYEDROPPER")
        row = col.row(align=True)
        row.prop(wm, "yurameki_clothes_obj")
        row.operator("yurameki.pick_clothes", text="", icon="EYEDROPPER")
        col.operator("yurameki.check_hair", icon="CHECKMARK")
        col.prop(wm, "yurameki_points_per_strand")
        status = getattr(wm, "yurameki_hair_check_status", "")
        if status:
            col.label(text=status)

        box = layout.box()
        box.label(text="Simulate")
        col = box.column(align=True)
        col.prop(wm, "yurameki_sim_start_frame")
        col.prop(wm, "yurameki_sim_end_frame")
        col.prop(wm, "yurameki_sim_bake_mode")
        col.operator("yurameki.simulate", icon="RENDER_ANIMATION")

        box = layout.box()
        box.label(text="Warp")
        col = box.column(align=True)
        col.prop(wm, "yurameki_root_locked_points")
        col.prop(wm, "yurameki_gravity")
        col.prop(wm, "yurameki_damping")
        col.prop(wm, "yurameki_max_velocity_mps")
        col.prop(wm, "yurameki_particle_mass_kg")
        col.prop(wm, "yurameki_iterations")
        col.prop(wm, "yurameki_stretch_compliance")
        col.prop(wm, "yurameki_bend_compliance")

        box = layout.box()
        box.label(text="Collision")
        col = box.column(align=True)
        col.prop(wm, "yurameki_collision_margin_mm")
        col.prop(wm, "yurameki_collision_search_mm")
        col.prop(wm, "yurameki_collision_max_correction_mm")
        col.prop(wm, "yurameki_collision_response")
        col.prop(wm, "yurameki_collision_velocity_damping")
        col.prop(wm, "yurameki_collision_passes")
        col.prop(wm, "yurameki_post_collision_iterations")
        col.prop(wm, "yurameki_auto_substep_mm")
        col.prop(wm, "yurameki_max_substeps")


_classes = (YURAMEKI_PT_main,)


def register():
    for cls in _classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)
