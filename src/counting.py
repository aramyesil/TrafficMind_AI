"""Virtual counting-line logic for tracked vehicles."""

from __future__ import annotations


def crossing_direction(
    previous: tuple[float, float],
    current: tuple[float, float],
    orientation: str,
    line_position: float,
) -> str | None:
    """Return the direction when a point crosses the counting line."""
    previous_x, previous_y = previous
    current_x, current_y = current
    if orientation == "horizontal":
        if previous_y < line_position <= current_y:
            return "down"
        if previous_y > line_position >= current_y:
            return "up"
        return None
    if previous_x < line_position <= current_x:
        return "right"
    if previous_x > line_position >= current_x:
        return "left"
    return None


class LineCounter:
    """Count each track once for each direction unless unique counting is disabled."""

    def __init__(self, orientation: str, line_position: float, count_once_per_direction: bool = True) -> None:
        self.orientation = orientation
        self.line_position = line_position
        self.count_once_per_direction = count_once_per_direction
        self._previous: dict[int, tuple[float, float]] = {}
        self._counted: set[tuple[int, str]] = set()
        self.total = 0

    def update(self, track_id: int | None, point: tuple[float, float]) -> str | None:
        if track_id is None:
            return None
        previous = self._previous.get(track_id)
        self._previous[track_id] = point
        if previous is None:
            return None
        direction = crossing_direction(previous, point, self.orientation, self.line_position)
        if direction is None:
            return None
        key = (track_id, direction)
        if self.count_once_per_direction and key in self._counted:
            return None
        self._counted.add(key)
        self.total += 1
        return direction
