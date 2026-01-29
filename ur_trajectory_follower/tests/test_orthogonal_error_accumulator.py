import os
import sys
import unittest

SCRIPTS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "scripts")
)
sys.path.append(SCRIPTS_DIR)

from helper.orthogonal_error_accumulator_core import (  # noqa: E402
    LayeredNearestNeighborAccumulator,
    build_reference_layer_points,
)


class OrthogonalErrorAccumulatorTests(unittest.TestCase):
    def test_reference_layer_mapping_and_mean(self):
        points = [
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
            (0.1, 0.0, 0.01),
            (1.1, 0.0, 0.01),
            (2.1, 0.0, 0.01),
        ]
        _, reference_points = build_reference_layer_points(points, delta_z=0.01, reference_z=0.0)
        accumulator = LayeredNearestNeighborAccumulator(reference_points, nn_window=0)

        ref_index = accumulator.find_nearest(1.1, 0.0)
        self.assertEqual(ref_index, 1)

        stats = accumulator.update(ref_index, 0.2)
        stats = accumulator.update(ref_index, 0.4)
        self.assertAlmostEqual(stats.mean, 0.3)


if __name__ == "__main__":
    unittest.main()
