"""events.py — Event-name constants for SliceViewerState's observer API.

Use these constants instead of string literals with
``SliceViewerState.add_listener`` so a typo fails at import / lint time
rather than silently registering a listener that never fires.
"""

from typing import Final

PRIMARY_IMAGE_DATA_CHANGED: Final = "primary_image_data_changed"
SECONDARY_IMAGE_DATA_CHANGED: Final = "secondary_image_data_changed"
BLEND_ALPHA_CHANGED: Final = "blend_alpha_changed"
SECONDARY_IMAGE_CMAP_CHANGED: Final = "secondary_image_cmap_changed"
SECONDARY_WINDOW_LEVEL_CHANGED: Final = "secondary_window_level_changed"
WINDOW_LEVEL_TARGET_CHANGED: Final = "window_level_target_changed"
PHASES_DATA_LOADED: Final = "phases_data_loaded"
PHASE_CHANGED: Final = "phase_changed"
RT_DOSE_CHANGED: Final = "rt_dose_changed"
LAYOUT_MODE_CHANGED: Final = "layout_mode_changed"
INDEX_CHANGED: Final = "index_changed"
WINDOW_LEVEL_CHANGED: Final = "window_level_changed"
CROSSHAIR_CHANGED: Final = "crosshair_changed"
CROSSHAIR_VISIBLE_CHANGED: Final = "crosshair_visible_changed"
BOUNDING_BOXES_CHANGED: Final = "bounding_boxes_changed"
BOUNDING_BOX_3D_CHANGED: Final = "bounding_box_3d_changed"
ALL_CONTOURS_CHANGED: Final = "all_contours_changed"
ACTIVE_CONTOURS_CHANGED: Final = "active_contours_changed"
OVERLAY_CONTOURS_CHANGED: Final = "overlay_contours_changed"
BRUSH_TOOL_ACTIVE_CHANGED: Final = "brush_tool_active_changed"
BRUSH_SIZE_MM_CHANGED: Final = "brush_size_mm_changed"
BRUSH_FILL_INSIDE_CHANGED: Final = "brush_fill_inside_changed"
SELECTED_ROI_CHANGED: Final = "selected_roi_changed"
CONTOUR_CACHE_BUILT: Final = "contour_cache_built"

#: Every event type SliceViewerState may broadcast; ``_notify`` rejects others.
ALL_EVENTS: Final[frozenset[str]] = frozenset(
    {
        PRIMARY_IMAGE_DATA_CHANGED,
        SECONDARY_IMAGE_DATA_CHANGED,
        BLEND_ALPHA_CHANGED,
        SECONDARY_IMAGE_CMAP_CHANGED,
        SECONDARY_WINDOW_LEVEL_CHANGED,
        WINDOW_LEVEL_TARGET_CHANGED,
        PHASES_DATA_LOADED,
        PHASE_CHANGED,
        RT_DOSE_CHANGED,
        LAYOUT_MODE_CHANGED,
        INDEX_CHANGED,
        WINDOW_LEVEL_CHANGED,
        CROSSHAIR_CHANGED,
        CROSSHAIR_VISIBLE_CHANGED,
        BOUNDING_BOXES_CHANGED,
        BOUNDING_BOX_3D_CHANGED,
        ALL_CONTOURS_CHANGED,
        ACTIVE_CONTOURS_CHANGED,
        OVERLAY_CONTOURS_CHANGED,
        BRUSH_TOOL_ACTIVE_CHANGED,
        BRUSH_SIZE_MM_CHANGED,
        BRUSH_FILL_INSIDE_CHANGED,
        SELECTED_ROI_CHANGED,
        CONTOUR_CACHE_BUILT,
    }
)
