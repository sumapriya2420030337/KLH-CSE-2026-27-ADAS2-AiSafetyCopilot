from __future__ import annotations

from .object_state import ObjectState


def update_occlusion_state(state: ObjectState, max_missing_frames: int) -> bool:
    """Mark a missed track occluded and return whether it should remain active."""
    state.mark_missed()
    if state.missed_frames > max_missing_frames:
        state.status = "dead"
        return False
    return True
