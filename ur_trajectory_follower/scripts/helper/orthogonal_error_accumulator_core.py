from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass
class AccumulatorStats:
    total: float = 0.0
    count: int = 0

    def update(self, value: float) -> None:
        self.total += float(value)
        self.count += 1

    @property
    def mean(self) -> float:
        if self.count <= 0:
            return 0.0
        return self.total / float(self.count)


def layer_id_from_z(z: float, reference_z: float, delta_z: float) -> int:
    if delta_z <= 0:
        raise ValueError("delta_z must be positive")
    return int(round((z - reference_z) / float(delta_z)))


def build_reference_layer_points(
    points: Sequence[Tuple[float, float, float]],
    delta_z: float,
    reference_z: Optional[float] = None,
    reference_layer_id: int = 0,
) -> Tuple[float, List[Tuple[float, float, int]]]:
    if not points:
        return 0.0, []

    if reference_z is None:
        reference_z = points[0][2]

    reference_points: List[Tuple[float, float, int]] = []
    for idx, (x, y, z) in enumerate(points):
        layer_id = layer_id_from_z(z, reference_z, delta_z)
        if layer_id == reference_layer_id:
            reference_points.append((x, y, idx))

    if not reference_points:
        closest = min(points, key=lambda p: abs(p[2] - reference_z))
        reference_z = closest[2]
        for idx, (x, y, z) in enumerate(points):
            layer_id = layer_id_from_z(z, reference_z, delta_z)
            if layer_id == reference_layer_id:
                reference_points.append((x, y, idx))

    return reference_z, reference_points


class LayeredNearestNeighborAccumulator:
    def __init__(self, reference_points: Sequence[Tuple[float, float, int]], nn_window: int = 0):
        self.reference_points = list(reference_points)
        self.nn_window = max(0, int(nn_window))
        self.stats: Dict[int, AccumulatorStats] = {}
        self.last_ref_list_index: Optional[int] = None

    def find_nearest(self, x: float, y: float) -> Optional[int]:
        if not self.reference_points:
            return None

        start = 0
        end = len(self.reference_points)
        if self.nn_window > 0 and self.last_ref_list_index is not None:
            start = max(0, self.last_ref_list_index - self.nn_window)
            end = min(len(self.reference_points), self.last_ref_list_index + self.nn_window + 1)

        best_idx = None
        best_dist = None
        for list_idx in range(start, end):
            rx, ry, _path_idx = self.reference_points[list_idx]
            dist = (rx - x) ** 2 + (ry - y) ** 2
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_idx = list_idx

        if best_idx is None:
            return None

        self.last_ref_list_index = best_idx
        return self.reference_points[best_idx][2]

    def update(self, ref_index: int, value: float) -> AccumulatorStats:
        stats = self.stats.setdefault(ref_index, AccumulatorStats())
        stats.update(value)
        return stats
