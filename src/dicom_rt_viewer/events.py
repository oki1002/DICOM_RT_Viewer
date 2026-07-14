"""events.py — Canonical event-name constants for SliceViewerState's Observer pattern.

``SliceViewerState.add_listener`` / ``_notify`` accept plain strings, so a
typo'd event name (``"windw_level_changed"``) previously failed silently —
the listener would simply never fire. Importing these constants instead of
writing string literals turns that typo into an ``AttributeError`` /
``NameError`` at import or lint time.

This module is the single source of truth for every event name and is also
used by :meth:`SliceViewerState._notify` to validate that only known event
types are ever broadcast (see ``ALL_EVENTS`` below).
"""

from typing import Final

PRIMARY_IMAGE_DATA_CHANGED: Final = "primary_image_data_changed"
SECONDARY_IMAGE_DATA_CHANGED: Final = "secondary_image_data_changed"
BLEND_ALPHA_CHANGED: Final = "blend_alpha_changed"
SECONDARY_IMAGE_CMAP_CHANGED: Final = "secondary_image_cmap_changed"
SECONDARY_CLIM_CHANGED: Final = "secondary_clim_changed"
PHASES_DATA_LOADED: Final = "phases_data_loaded"
PHASE_CHANGED: Final = "phase_changed"
RT_DOSE_CHANGED: Final = "rt_dose_changed"
LAYOUT_MODE_CHANGED: Final = "layout_mode_changed"
INDEX_CHANGED: Final = "index_changed"
WINDOW_LEVEL_CHANGED: Final = "window_level_changed"
CROSSHAIR_CHANGED: Final = "crosshair_changed"
CROSSHAIR_VISIBLE_CHANGED: Final = "crosshair_visible_changed"
BOUNDING_BOXES_CHANGED: Final = "bounding_boxes_changed"
ALL_CONTOURS_CHANGED: Final = "all_contours_changed"
ACTIVE_CONTOURS_CHANGED: Final = "active_contours_changed"
OVERLAY_CONTOURS_CHANGED: Final = "overlay_contours_changed"
BRUSH_TOOL_ACTIVE_CHANGED: Final = "brush_tool_active_changed"
BRUSH_SIZE_MM_CHANGED: Final = "brush_size_mm_changed"
BRUSH_FILL_INSIDE_CHANGED: Final = "brush_fill_inside_changed"
SELECTED_ROI_CHANGED: Final = "selected_roi_changed"
CONTOUR_CACHE_BUILT: Final = "contour_cache_built"

#: Every event type SliceViewerState may broadcast. Used by ``_notify`` to
#: catch a typo'd event name (a string not in this set) at the point it is
#: fired, instead of silently reaching zero listeners.
ALL_EVENTS: Final[frozenset[str]] = frozenset(
    {
        PRIMARY_IMAGE_DATA_CHANGED,
        SECONDARY_IMAGE_DATA_CHANGED,
        BLEND_ALPHA_CHANGED,
        SECONDARY_IMAGE_CMAP_CHANGED,
        SECONDARY_CLIM_CHANGED,
        PHASES_DATA_LOADED,
        PHASE_CHANGED,
        RT_DOSE_CHANGED,
        LAYOUT_MODE_CHANGED,
        INDEX_CHANGED,
        WINDOW_LEVEL_CHANGED,
        CROSSHAIR_CHANGED,
        CROSSHAIR_VISIBLE_CHANGED,
        BOUNDING_BOXES_CHANGED,
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
