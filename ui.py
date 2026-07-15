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


def _label(layout, text: str) -> None:
    layout.label(text=text)


def _prop(layout, wm, name: str) -> None:
    layout.prop(wm, name)


def _operator(layout, op_id: str, *, text: str | None = None, icon: str = "NONE"):
    kwargs = {"icon": icon}
    if text is not None:
        kwargs["text"] = text
    return layout.operator(op_id, **kwargs)


class YURAMEKI_PT_main(Panel):
    bl_idname = "YURAMEKI_PT_main"
    bl_label = "Yurameki"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Yurameki"

    def draw(self, context):
        layout = self.layout
        wm = context.window_manager

        _label(layout, f"Yurameki v{_version()}")
        layout.separator(factor=0.4)

        box = layout.box()
        _label(box, "Input")
        col = box.column(align=True)
        row = col.row(align=True)
        _prop(row, wm, "yurameki_curves_obj")
        _operator(row, "yurameki.pick_curves", text="", icon="EYEDROPPER")
        row = col.row(align=True)
        _prop(row, wm, "yurameki_collider_obj")
        _operator(row, "yurameki.pick_collider", text="", icon="EYEDROPPER")
        row = col.row(align=True)
        _prop(row, wm, "yurameki_clothes_obj")
        _operator(row, "yurameki.pick_clothes", text="", icon="EYEDROPPER")
        _prop(col, wm, "yurameki_points_per_strand")

        box = layout.box()
        _label(box, "Simulate")
        col = box.column(align=True)
        _prop(col, wm, "yurameki_sim_start_frame")
        _prop(col, wm, "yurameki_sim_end_frame")
        _prop(col, wm, "yurameki_sim_bake_mode")
        _prop(col, wm, "yurameki_guide_decimation")
        _prop(col, wm, "yurameki_keep_length")
        running = getattr(wm, "yurameki_sim_running", False)
        sim_row = col.row(align=True)
        sim_row.enabled = not running
        _operator(sim_row, "yurameki.simulate", icon="RENDER_ANIMATION")
        if running:
            _operator(col, "yurameki.stop_simulate", icon="CANCEL")
        status = getattr(wm, "yurameki_sim_status", "")
        if status:
            _label(col, status)
        _operator(col, "yurameki.bake_cache", icon="ACTION")

        box = layout.box()
        _label(box, "Elastic Rod")
        col = box.column(align=True)
        _prop(col, wm, "yurameki_root_locked_points")
        _prop(col, wm, "yurameki_adaptive_root_lock")
        _prop(col, wm, "yurameki_gravity")
        _prop(col, wm, "yurameki_damping")
        _prop(col, wm, "yurameki_internal_damping")
        _prop(col, wm, "yurameki_max_velocity_mps")
        _prop(col, wm, "yurameki_particle_mass_g")
        _prop(col, wm, "yurameki_bend_stiffness_log10")
        _prop(col, wm, "yurameki_iterations")

        box = layout.box()
        _label(box, "Collision")
        col = box.column(align=True)
        _prop(col, wm, "yurameki_collision_margin_mm")
        _prop(col, wm, "yurameki_collision_search_mm")
        _prop(col, wm, "yurameki_collision_max_correction_mm")
        _prop(col, wm, "yurameki_collision_response")
        _prop(col, wm, "yurameki_collision_smoothing")
        _prop(col, wm, "yurameki_collision_velocity_damping")
        _prop(col, wm, "yurameki_collision_passes")
        _prop(col, wm, "yurameki_post_collision_iterations")
        _prop(col, wm, "yurameki_auto_substep_mm")
        _prop(col, wm, "yurameki_max_substeps")


_classes = (YURAMEKI_PT_main,)


def register():
    for cls in _classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)
