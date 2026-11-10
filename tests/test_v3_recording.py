import copy
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from recording_options import parse_args, run_directory
from uav_motion import build_motion_plan, build_attitude_schedule, scene_index


class V3MotionTest(unittest.TestCase):
    speed = dict(low_mps=4., high_mps=12., stage_seconds=20.)
    hover = dict(after_motion_seconds=30., duration_seconds=3., brake_seconds=3., recover_seconds=3.)
    disturbance = dict(enabled=True, seed=42, probability_per_second=.3,
                       duration_seconds=[3., 6.], min_gap_seconds=2., roll_deg=2., pitch_deg=2., yaw_deg=3.)

    def test_motion_clocks_hover_and_exact_endpoint(self):
        rows = build_motion_plan(1200., 10., self.speed, self.hover)
        self.assertEqual(rows[0]["speed_mps"], 4.)
        self.assertAlmostEqual(rows[300]["cruise_time_s"], 30.)
        self.assertEqual(rows[300]["motion_state"], "brake")
        self.assertEqual(rows[330]["motion_state"], "hover")
        self.assertEqual(sum(r["is_hovering"] and r["hover_index"] == 1 for r in rows), 30)
        for row in rows[330:361]:
            self.assertEqual(row["route_distance_m"], rows[330]["route_distance_m"])
            self.assertEqual(row["motion_time_s"], rows[330]["motion_time_s"])
        self.assertTrue(all(0 <= r["speed_mps"] <= 12 for r in rows))
        self.assertEqual(rows[-1]["route_distance_m"], 1200.)
        self.assertEqual({r["speed_stage"] for r in rows}, {"low", "high", "accelerate", "decelerate"})
        self.assertTrue(all(a["route_distance_m"] <= b["route_distance_m"] for a, b in zip(rows, rows[1:])))

    def test_disturbance_reproducible_bounded_and_zero_outside_events(self):
        schedule = build_attitude_schedule(2000, 10, self.disturbance)
        self.assertEqual(schedule, build_attitude_schedule(2000, 10, self.disturbance))
        self.assertTrue(schedule["events"])
        for row in schedule["frames"]:
            for axis, limit in (("roll_deg", 2), ("pitch_deg", 2), ("yaw_deg", 3)):
                self.assertLessEqual(abs(row[axis]), limit)
                if row["event_index"] is None:
                    self.assertEqual(row[axis], 0)
        for a, b in zip(schedule["events"], schedule["events"][1:]):
            self.assertGreaterEqual(b["start_frame"] - a["end_frame"], 20)
        disabled = dict(self.disturbance, enabled=False)
        off = build_attitude_schedule(2000, 10, disabled)
        self.assertFalse(off["events"])
        self.assertTrue(all(r["roll_deg"] == r["pitch_deg"] == r["yaw_deg"] == 0 for r in off["frames"]))

    def test_scenes_preserve_tail_and_cli_controls(self):
        rows = build_motion_plan(1200., 10., self.speed, self.hover, num_frames=421)
        scenes = scene_index(rows, 10.)["scenes"]
        self.assertEqual([s["num_frames"] for s in scenes], [200, 200, 21])
        self.assertEqual(scenes[-1]["end_frame_exclusive"], 421)
        args = parse_args(["--no-disturbance", "--output-root", "D:/trial", "--weather", "rain"])
        self.assertFalse(args.disturbance)
        self.assertEqual(run_directory(args.output_root, "Town05_Opt", args.weather, args.run_id).as_posix(),
                         "D:/trial/Town05_Opt_rain/run_0001")
        with self.assertRaises(ValueError):
            build_motion_plan(10., 0., self.speed, self.hover)


try:
    import carla
    import numpy as np
    from collect_uavdataset import sensor_world_transform, matrix, UAVRoute, load_yaml, CONFIG_PATH
except ImportError:
    carla = None


@unittest.skipIf(carla is None, "CARLA client dependencies not installed")
class V3GeometryTest(unittest.TestCase):
    def test_rigid_mount_preserves_extrinsic_for_tilt_and_nadir_camera(self):
        position = dict(x=0., y=.1, z=0.)
        mount = dict(pitch=-90., yaw=0., roll=0.)
        relative = matrix(carla.Transform(carla.Location(**position), carla.Rotation(**mount)))
        for pitch, yaw, roll in ((0, 0, 0), (2, 85, -2), (-2, -150, 1), (0, 90, 2)):
            platform = carla.Transform(carla.Location(x=7, y=3, z=40), carla.Rotation(pitch=pitch, yaw=yaw, roll=roll))
            actual = sensor_world_transform(platform, position, mount)
            np.testing.assert_allclose(np.linalg.inv(matrix(platform)) @ matrix(actual), relative, atol=2e-5)

    def test_route_hover_freezes_altitude_and_off_switch(self):
        config = load_yaml(CONFIG_PATH)
        uav = copy.deepcopy(config["uav"])
        uav["disturbance"]["enabled"] = False
        uav["route"] = dict(name="test", map="Town05_Opt",
                            planned_path=[dict(x=i, y=0, z=3) for i in range(1001)])
        route = UAVRoute(uav, config["route_planner"], 10.)
        a = matrix(route.pose_at_frame(330))
        for i in range(331, 360):
            np.testing.assert_allclose(matrix(route.pose_at_frame(i)), a)
        self.assertAlmostEqual(route.pose_at_frame(300).rotation.pitch, 0.)


if __name__ == "__main__":
    unittest.main()
