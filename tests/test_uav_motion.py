import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from uav_motion import (  # noqa: E402
    altitude_above_road_at_distance,
    is_hovering_at_frame,
    motion_intervals_at_frame,
    required_frames,
)


class UAVMotionTest(unittest.TestCase):
    def test_hover_freezes_route_progress_for_exact_duration(self):
        move_frames = 300
        hover_frames = 30

        self.assertEqual(motion_intervals_at_frame(299, move_frames, hover_frames), 299)
        self.assertEqual(motion_intervals_at_frame(300, move_frames, hover_frames), 300)
        self.assertTrue(is_hovering_at_frame(300, move_frames, hover_frames))
        self.assertEqual(motion_intervals_at_frame(329, move_frames, hover_frames), 300)
        self.assertTrue(is_hovering_at_frame(329, move_frames, hover_frames))
        self.assertEqual(motion_intervals_at_frame(330, move_frames, hover_frames), 300)
        self.assertFalse(is_hovering_at_frame(330, move_frames, hover_frames))
        self.assertEqual(motion_intervals_at_frame(331, move_frames, hover_frames), 301)

    def test_full_route_frame_count_includes_only_hovers_before_finish(self):
        self.assertEqual(required_frames(240.0, 0.8, 300, 30), 301)
        self.assertEqual(required_frames(480.0, 0.8, 300, 30), 631)

    def test_cosine_altitude_stays_in_range_and_stops_at_extrema(self):
        values = [
            altitude_above_road_at_distance(distance, 8.0, 20.0, 50.0, 80.0)
            for distance in (0.0, 160.0, 320.0, 480.0, 640.0)
        ]
        self.assertAlmostEqual(values[0], 50.0)
        self.assertAlmostEqual(values[2], 20.0)
        self.assertAlmostEqual(values[4], 50.0)
        self.assertTrue(all(20.0 <= value <= 50.0 for value in values))


if __name__ == "__main__":
    unittest.main()
