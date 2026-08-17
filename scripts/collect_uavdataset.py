import carla

import json
import math
import queue
import random
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from ruamel.yaml import YAML

try:
    import cv2
except ImportError:
    cv2 = None


########################## 路径：定义项目根目录和配置文件位置 ################################

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "UAVdataset.yaml"


########################## 常量：定义类别编号、坐标关系和固定参数 ################################

CLASS_TO_SEMANTIC_TAG = {
    "pedestrian": 12,
    "car": 14,
    "van": 14,
    "truck": 15,
    "bus": 16,
    "motorcycle": 18,
    "bicycle": 19,
}


SUPPORTED_CLASSES = set(CLASS_TO_SEMANTIC_TAG.keys())

# CARLA 的 instance segmentation 语义标签来自 Actor 实际组件标签。
# 数据集类别（例如 van）和 CARLA 的渲染语义标签并不保证一一对应，
# 因此动态 Actor 的实例像素匹配必须优先读取 actor.semantic_tags，
# 不能仅依赖 CLASS_TO_SEMANTIC_TAG 的数据集类别映射。
DYNAMIC_INSTANCE_THING_TAGS = frozenset(
    CLASS_TO_SEMANTIC_TAG.values()
)


STATIC_ENV_CLASS_TO_CITY_LABEL = {
    "car": carla.CityObjectLabel.Car,
    "truck": carla.CityObjectLabel.Truck,
    "bus": carla.CityObjectLabel.Bus,
    "motorcycle": carla.CityObjectLabel.Motorcycle,
    "bicycle": carla.CityObjectLabel.Bicycle,
}

BBOX_EDGES = [
    (0, 1), (1, 3), (3, 2), (2, 0),
    (0, 4), (4, 5), (5, 1), (5, 7),
    (7, 6), (6, 4), (6, 2), (7, 3),
]

T_CV_UE = np.array(
    [
        [0.0, 1.0,  0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [1.0, 0.0,  0.0, 0.0],
        [0.0, 0.0,  0.0, 1.0],
    ],
    dtype=np.float64,
)


########################## 配置读取：加载 YAML 配置并转换为程序数据 ################################

def load_yaml(path):
    yaml = YAML()
    with open(path, "r", encoding="utf-8") as f:
        return yaml.load(f)


def validate_route_schema(route_root, route_path):
    """Require the current route schema only; legacy route files are rejected."""
    if not isinstance(route_root, dict):
        raise ValueError(f"Route file {route_path} must contain a mapping at the top level.")

    route = route_root.get("route")
    if not isinstance(route, dict):
        raise ValueError(f"Route file {route_path} must contain a 'route' mapping.")

    required = ("name", "map", "anchors", "planned_path")
    missing = [key for key in required if key not in route]
    if missing:
        raise ValueError(
            f"Route file {route_path} is not in the current route format; "
            f"missing fields: {', '.join(missing)}"
        )

    extra = [key for key in route if key not in required]
    if extra:
        raise ValueError(
            f"Route file {route_path} contains unsupported route fields: "
            f"{', '.join(map(str, extra))}"
        )

    if not isinstance(route["name"], str) or not route["name"].strip():
        raise ValueError(f"Route file {route_path}: route.name must be a non-empty string.")
    if not isinstance(route["map"], str) or not route["map"].strip():
        raise ValueError(f"Route file {route_path}: route.map must be a non-empty string.")
    if not isinstance(route["anchors"], list):
        raise ValueError(f"Route file {route_path}: route.anchors must be a list.")
    if not isinstance(route["planned_path"], list):
        raise ValueError(f"Route file {route_path}: route.planned_path must be a list.")

    return route


def load_config():
    config = load_yaml(CONFIG_PATH)

    route_file = config["uav"]["route_file"]
    route_path = (CONFIG_PATH.parent / route_file).resolve()

    if not route_path.exists():
        raise FileNotFoundError(
            f"\nRoute YAML does not exist:\n{route_path}"
        )

    route_config = load_yaml(route_path)
    route = validate_route_schema(route_config, route_path)

    config["uav"]["route"] = route
    config["uav"]["route_source"] = str(route_path)

    print("\nRoute configuration:")
    print("  file :", route_path)
    print("  name :", route["name"])

    return config


def validate_config(config):
    if "recording" not in config:
        raise KeyError("Missing 'recording:' section in UAVdataset.yaml")

    fps = float(config["recording"]["fps"])
    num_frames = int(config["recording"]["num_frames"])

    if fps <= 0:
        raise ValueError("recording.fps must be > 0")

    if num_frames == 0 or num_frames < -1:
        raise ValueError(
            "recording.num_frames must be -1 or a positive integer"
        )

    speed = float(config["uav"]["speed_mps"])
    if speed <= 0:
        raise ValueError("uav.speed_mps must be > 0")

    lidar_cfg = config["sensors"]["lidar"]
    if float(lidar_cfg["points_per_second"]) <= 0:
        raise ValueError("lidar.points_per_second must be > 0")

    if float(lidar_cfg["range"]) <= 0:
        raise ValueError("lidar.range must be > 0")

    horizontal_fov = float(lidar_cfg["horizontal_fov"])
    if not (0.0 < horizontal_fov <= 360.0):
        raise ValueError("lidar.horizontal_fov must be in (0, 360]")

    lower_fov = float(lidar_cfg["lower_fov"])
    upper_fov = float(lidar_cfg["upper_fov"])

    if not (-90.0 <= lower_fov < upper_fov <= 90.0):
        raise ValueError(
            "LiDAR vertical FOV must satisfy "
            "-90 <= lower_fov < upper_fov <= 90"
        )

    traffic_cfg = config["traffic"]

    if int(traffic_cfg["num_vehicles"]) < 0:
        raise ValueError("traffic.num_vehicles must be >= 0")

    if int(traffic_cfg.get("num_pedestrians", 0)) < 0:
        raise ValueError("traffic.num_pedestrians must be >= 0")

    if float(traffic_cfg.get("warmup_seconds", 0.0)) < 0:
        raise ValueError("traffic.warmup_seconds must be >= 0")

    ann_cfg = config.get("annotations", {})
    if ann_cfg.get("enabled", False):
        classes = list(ann_cfg.get("classes", []))

        if not classes:
            raise ValueError(
                "annotations.classes must contain at least one class"
            )

        unknown = sorted(set(classes) - SUPPORTED_CLASSES)
        if unknown:
            raise ValueError(
                f"Unsupported annotation classes: {unknown}. "
                f"Supported: {sorted(SUPPORTED_CLASSES)}"
            )

        static_cfg = ann_cfg.get("static_environment", {})
        if static_cfg.get("enabled", True):
            static_classes = list(
                static_cfg.get(
                    "classes",
                    [
                        "car",
                        "truck",
                        "bus",
                        "motorcycle",
                        "bicycle",
                    ],
                )
            )

            unknown_static = sorted(
                set(static_classes)
                - set(STATIC_ENV_CLASS_TO_CITY_LABEL)
            )

            if unknown_static:
                raise ValueError(
                    "Unsupported annotations.static_environment.classes: "
                    f"{unknown_static}. Supported static map classes: "
                    f"{sorted(STATIC_ENV_CLASS_TO_CITY_LABEL)}"
                )

        if float(ann_cfg.get("max_distance_m", 120.0)) <= 0:
            raise ValueError("annotations.max_distance_m must be > 0")

        bbox2d_cfg = ann_cfg.get("bbox2d", {})

        rgb_visibility_cfg = ann_cfg.get(
            "rgb_visibility_filter",
            {},
        )
        min_rgb_visible_pixels = int(
            rgb_visibility_cfg.get(
                "min_visible_pixels",
                5,
            )
        )

        if min_rgb_visible_pixels < 1:
            raise ValueError(
                "annotations.rgb_visibility_filter.min_visible_pixels "
                "must be >= 1"
            )

        lidar_filter_cfg = ann_cfg.get("lidar_fov_filter", {})
        samples_per_axis = int(
            lidar_filter_cfg.get("bbox_samples_per_axis", 5)
        )

        if samples_per_axis < 3:
            raise ValueError(
                "annotations.lidar_fov_filter.bbox_samples_per_axis "
                "must be >= 3"
            )

        lidar_visibility_cfg = ann_cfg.get(
            "lidar_visibility",
            {},
        )
        min_lidar_points = int(
            lidar_visibility_cfg.get(
                "min_lidar_points",
                3,
            )
        )

        if min_lidar_points < 1:
            raise ValueError(
                "annotations.lidar_visibility.min_lidar_points "
                "must be >= 1"
            )


########################## 坐标变换：在 CARLA、LiDAR 和相机坐标系之间转换 ################################

def pose_dict(transform):
    return {
        "location": {
            "x": float(transform.location.x),
            "y": float(transform.location.y),
            "z": float(transform.location.z),
        },
        "rotation": {
            "pitch": float(transform.rotation.pitch),
            "yaw": float(transform.rotation.yaw),
            "roll": float(transform.rotation.roll),
        },
    }


def matrix(transform):
    return np.array(transform.get_matrix(), dtype=np.float64)


def transform_points(points_xyz, transform_matrix):
    points_xyz = np.asarray(points_xyz, dtype=np.float64)

    if points_xyz.size == 0:
        return np.empty((0, 3), dtype=np.float64)

    ones = np.ones((points_xyz.shape[0], 1), dtype=np.float64)
    points_h = np.concatenate([points_xyz, ones], axis=1)
    out = (transform_matrix @ points_h.T).T
    return out[:, :3]


def carla_locations_to_numpy(locations):
    return np.array(
        [[float(p.x), float(p.y), float(p.z)] for p in locations],
        dtype=np.float64,
    )


def matrix_to_carla_rotation(transform_matrix):
    """
    Extract CARLA pitch/yaw/roll from a proper CARLA/UE rotation matrix.

    This is used only for transforms that remain in CARLA UE handedness:
      world, lidar-local UE, camera-local UE.

    camera_cv is a handedness conversion and is intentionally not converted
    to CARLA Euler angles.
    """
    r = np.asarray(transform_matrix, dtype=np.float64)[:3, :3]

    sp = float(np.clip(r[2, 0], -1.0, 1.0))
    pitch = math.asin(sp)
    cp = math.cos(pitch)

    if abs(cp) > 1e-8:
        yaw = math.atan2(r[1, 0], r[0, 0])
        roll = math.atan2(-r[2, 1], r[2, 2])
    else:
        yaw = math.atan2(-r[0, 1], r[1, 1])
        roll = 0.0

    return {
        "pitch": math.degrees(pitch),
        "yaw": math.degrees(yaw),
        "roll": math.degrees(roll),
    }


def sensor_world_transform(
    uav_transform,
    sensor_position,
    sensor_rotation,
):
    relative_location = carla.Location(
        x=float(sensor_position["x"]),
        y=float(sensor_position["y"]),
        z=float(sensor_position["z"]),
    )

    world_location = uav_transform.transform(relative_location)

    world_rotation = carla.Rotation(
        pitch=(
            uav_transform.rotation.pitch
            + float(sensor_rotation["pitch"])
        ),
        yaw=(
            uav_transform.rotation.yaw
            + float(sensor_rotation["yaw"])
        ),
        roll=(
            uav_transform.rotation.roll
            + float(sensor_rotation["roll"])
        ),
    )

    return carla.Transform(world_location, world_rotation)


########################## 相机内参：根据图像尺寸和视场角计算投影参数 ################################

def camera_intrinsic(width, height, fov):
    focal = width / (
        2.0 * math.tan(math.radians(fov) / 2.0)
    )

    return np.array(
        [
            [focal, 0.0, width / 2.0],
            [0.0, focal, height / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


########################## 传感器队列：按帧号配对相机与 LiDAR 数据 ################################

def get_frame(sensor_queue, target_frame, name, timeout=10.0):
    deadline = time.monotonic() + timeout

    while True:
        remaining = deadline - time.monotonic()

        if remaining <= 0:
            raise TimeoutError(
                f"{name}: timeout waiting for CARLA frame {target_frame}"
            )

        try:
            data = sensor_queue.get(timeout=remaining)
        except queue.Empty as exc:
            raise TimeoutError(
                f"{name}: timeout waiting for CARLA frame {target_frame}"
            ) from exc

        if data.frame < target_frame:
            continue

        if data.frame == target_frame:
            return data

        raise RuntimeError(
            f"{name}: frame synchronization failed. "
            f"Expected {target_frame}, got {data.frame}."
        )


########################## UAV 路线：读取航点并计算无人机的移动目标 ################################


class UAVRoute:

    def __init__(self, uav_config, route_planner_config, fps):
        self.fps = float(fps)

        if self.fps <= 0:
            raise ValueError("FPS must be > 0")

        self.speed = float(uav_config["speed_mps"])

        if self.speed <= 0:
            raise ValueError("UAV speed must be > 0")

        self.distance_per_frame = self.speed / self.fps

        route = uav_config["route"]

        self.name = str(route["name"])
        self.map_name = str(route["map"])

        self.altitude_m = float(
            uav_config["altitude_above_road_m"]
        )

        if self.altitude_m <= 0:
            raise ValueError(
                "uav.altitude_above_road_m must be > 0"
            )

        self.heading_lookahead_m = float(
            uav_config["heading_lookahead_m"]
        )

        if self.heading_lookahead_m <= 0:
            raise ValueError(
                "uav.heading_lookahead_m must be > 0"
            )

        self.planner_resolution_m = float(
            route_planner_config["sampling_resolution_m"]
        )

        if self.planner_resolution_m <= 0:
            raise ValueError(
                "route_planner.sampling_resolution_m must be > 0"
            )

        self.points = []
        duplicate_count = 0

        for index, p in enumerate(route["planned_path"]):
            point = np.array(
                [
                    float(p["x"]),
                    float(p["y"]),
                    float(p["z"]),
                ],
                dtype=np.float64,
            )

            if self.points:
                # Flight progress is defined in the XY plane. Consecutive
                # points with identical XY therefore have zero route length
                # even if their road z differs slightly.
                d_xy = float(
                    np.linalg.norm(
                        point[:2] - self.points[-1][:2]
                    )
                )

                if d_xy <= 1e-6:
                    duplicate_count += 1
                    print(
                        "WARNING: removed zero-XY-length planned_path point "
                        f"#{index}: "
                        f"({point[0]:.3f}, {point[1]:.3f}, {point[2]:.3f})"
                    )
                    continue

            self.points.append(point)

        if duplicate_count > 0:
            print(
                f"Removed {duplicate_count} zero-XY-length "
                "planned_path point(s)."
            )

        if len(self.points) < 2:
            raise ValueError(
                "route.planned_path requires at least 2 unique XY points. "
                "Create and save the route with scripts/route_editor.py first."
            )

        self.segment_lengths = []
        cumulative = [0.0]

        for i in range(len(self.points) - 1):
            length = float(
                np.linalg.norm(
                    self.points[i + 1][:2] - self.points[i][:2]
                )
            )

            if length <= 1e-9:
                continue

            self.segment_lengths.append(length)
            cumulative.append(cumulative[-1] + length)

        self.cumulative_distances = np.asarray(
            cumulative,
            dtype=np.float64,
        )
        self.total_length = float(self.cumulative_distances[-1])

        if self.total_length <= 1e-6:
            raise ValueError("route.planned_path XY length is zero.")

        if len(self.segment_lengths) != len(self.points) - 1:
            raise RuntimeError(
                "Internal route preprocessing mismatch after removing "
                "zero-length points."
            )

        self.max_segment_length_m = max(self.segment_lengths)
        allowed_max_segment = max(
            10.0,
            self.planner_resolution_m * 20.0,
        )

        if self.max_segment_length_m > allowed_max_segment:
            raise ValueError(
                "route.planned_path contains an unexpected XY jump: "
                f"max_segment={self.max_segment_length_m:.3f} m, "
                f"allowed={allowed_max_segment:.3f} m. Re-open "
                "scripts/route_editor.py and inspect the route."
            )

    @staticmethod
    def _map_short_name(world_map):
        return str(world_map.name).replace("\\", "/").split("/")[-1]

    def validate_world_map(self, world_map):
        if not self.map_name:
            raise ValueError(
                "route.map is empty. Save the route with "
                "scripts/route_editor.py before recording."
            )

        current_map = self._map_short_name(world_map)

        if current_map != self.map_name:
            raise RuntimeError(
                "Route/map mismatch: route_01.yaml was planned for "
                f"'{self.map_name}', but CARLA is currently running "
                f"'{current_map}'. Re-open route_editor.py on the intended map "
                "and save the route again."
            )

    def required_frames(self):
        movement_intervals = math.ceil(
            self.total_length / self.distance_per_frame
        )
        return movement_intervals + 1

    def estimated_duration(self):
        return self.total_length / self.speed

    def distance_at_frame(self, frame_index):
        return min(
            float(frame_index) * self.distance_per_frame,
            self.total_length,
        )

    def is_finished(self, frame_index):
        return (
            self.distance_at_frame(frame_index)
            >= self.total_length - 1e-6
        )

    def road_point_at_distance(self, distance):
        distance = float(
            np.clip(distance, 0.0, self.total_length)
        )

        if distance >= self.total_length - 1e-9:
            return self.points[-1].copy()

        segment_index = int(
            np.searchsorted(
                self.cumulative_distances,
                distance,
                side="right",
            ) - 1
        )
        segment_index = int(
            np.clip(
                segment_index,
                0,
                len(self.segment_lengths) - 1,
            )
        )

        segment_start_distance = self.cumulative_distances[segment_index]
        segment_length = self.segment_lengths[segment_index]
        ratio = (distance - segment_start_distance) / segment_length
        ratio = float(np.clip(ratio, 0.0, 1.0))

        start = self.points[segment_index]
        end = self.points[segment_index + 1]

        return start + ratio * (end - start)

    def yaw_at_distance(self, distance):
        distance = float(
            np.clip(distance, 0.0, self.total_length)
        )

        forward_distance = min(
            distance + self.heading_lookahead_m,
            self.total_length,
        )

        if forward_distance - distance > 1e-6:
            start = self.road_point_at_distance(distance)
            end = self.road_point_at_distance(forward_distance)
        else:
            # At the endpoint there is no forward sample. Use a short incoming
            # chord (no longer than one dataset movement step) so the last frame
            # keeps the local arrival heading instead of jumping back to a much
            # longer backward-lookahead direction.
            backward_span = min(
                self.heading_lookahead_m,
                self.distance_per_frame,
            )
            backward_distance = max(
                0.0,
                distance - backward_span,
            )
            start = self.road_point_at_distance(backward_distance)
            end = self.road_point_at_distance(distance)

        direction = end[:2] - start[:2]

        if float(np.linalg.norm(direction)) <= 1e-9:
            # Extremely short/local degenerate geometry: fall back to the
            # nearest non-zero stored segment.
            segment_index = min(
                int(
                    np.searchsorted(
                        self.cumulative_distances,
                        distance,
                        side="right",
                    ) - 1
                ),
                len(self.segment_lengths) - 1,
            )
            segment_index = max(segment_index, 0)
            direction = (
                self.points[segment_index + 1][:2]
                - self.points[segment_index][:2]
            )

        return math.degrees(
            math.atan2(direction[1], direction[0])
        )

    def pose_at_frame(self, frame_index):
        return self.pose_at_distance(
            self.distance_at_frame(frame_index)
        )

    def pose_at_distance(self, distance):
        road_position = self.road_point_at_distance(distance)
        yaw = self.yaw_at_distance(distance)

        return carla.Transform(
            carla.Location(
                x=float(road_position[0]),
                y=float(road_position[1]),
                z=float(road_position[2] + self.altitude_m),
            ),
            carla.Rotation(
                pitch=0.0,
                yaw=float(yaw),
                roll=0.0,
            ),
        )


########################## 录制计划：决定录制帧数和结束条件 ################################

def determine_recording_frames(route, configured_num_frames):
    configured_num_frames = int(configured_num_frames)

    if configured_num_frames == -1:
        return route.required_frames()

    if configured_num_frames <= 0:
        raise ValueError(
            "recording.num_frames must be -1 or a positive integer."
        )

    return configured_num_frames


def print_recording_plan(
    route,
    fps,
    configured_num_frames,
    actual_num_frames,
    traffic_cfg,
    ann_cfg,
    lidar_cfg,
):
    print()
    print("=" * 64)
    print("UAV DATASET V3.3 RECORDING PLAN")
    print("=" * 64)

    print(f"Route name             : {route.name}")
    print(f"Route map              : {route.map_name}")
    print(f"Planned path points    : {len(route.points)}")
    print(f"Route XY length        : {route.total_length:.3f} m")
    print(f"UAV altitude / road    : {route.altitude_m:.3f} m")
    print(f"Planner resolution     : {route.planner_resolution_m:.3f} m")
    print(f"Max stored XY segment  : {route.max_segment_length_m:.3f} m")
    print(f"Heading lookahead      : {route.heading_lookahead_m:.3f} m")
    print(f"UAV XY speed           : {route.speed:.3f} m/s")
    print(f"Dataset FPS            : {fps:.3f} Hz")
    print(
        f"XY distance / frame    : "
        f"{route.distance_per_frame:.3f} m"
    )
    print(
        f"Estimated flight time  : "
        f"{route.estimated_duration():.3f} s"
    )
    print(f"Full-route frames      : {route.required_frames()}")

    recording_mode = (
        "UNTIL_ROUTE_END"
        if configured_num_frames == -1
        else "FIXED_NUM_FRAMES"
    )

    print(f"Recording mode         : {recording_mode}")
    print(f"Configured frames      : {configured_num_frames}")
    print(f"Actual frames          : {actual_num_frames}")

    if configured_num_frames > 0:
        movement_distance = max(
            configured_num_frames - 1,
            0,
        ) * route.distance_per_frame

        coverage_distance = min(
            movement_distance,
            route.total_length,
        )

        coverage_percent = (
            coverage_distance / route.total_length * 100.0
        )

        print(
            f"Route coverage         : "
            f"{coverage_distance:.3f} / {route.total_length:.3f} m "
            f"({coverage_percent:.1f}%)"
        )

        if configured_num_frames < route.required_frames():
            print(
                "\nWARNING: configured num_frames cannot cover the "
                "full route. Use num_frames: -1 for full-route recording."
            )
        elif configured_num_frames > route.required_frames():
            extra = configured_num_frames - route.required_frames()
            print(
                f"\nWARNING: {extra} extra frame(s) will be recorded "
                "at the endpoint."
            )

    print()
    print(f"Traffic vehicles       : {traffic_cfg['num_vehicles']}")
    print(
        f"Traffic pedestrians    : "
        f"{traffic_cfg.get('num_pedestrians', 0)}"
    )
    print(
        f"Traffic warmup         : "
        f"{traffic_cfg.get('warmup_seconds', 0.0)} s"
    )

    print()
    print(
        f"Annotations enabled    : "
        f"{ann_cfg.get('enabled', False)}"
    )

    if ann_cfg.get("enabled", False):
        print(
            "Annotation classes    : "
            + ", ".join(ann_cfg["classes"])
        )
        print(
            f"Annotation max range   : "
            f"{float(ann_cfg.get('max_distance_m', 120.0)):.1f} m"
        )
        print(
            f"3D bbox                : "
            f"{ann_cfg.get('bbox3d', {}).get('enabled', True)}"
        )
        print(
            f"2D bbox                : "
            f"{ann_cfg.get('bbox2d', {}).get('enabled', True)}"
        )

        lidar_filter_cfg = ann_cfg.get("lidar_fov_filter", {})
        lidar_visibility_cfg = ann_cfg.get("lidar_visibility", {})
        print(
            f"LiDAR FOV filter       : "
            f"{lidar_filter_cfg.get('enabled', True)}"
        )
        print(
            f"LiDAR visibility filter: "
            f"{lidar_visibility_cfg.get('enabled', True)}"
        )
        print(
            f"Min LiDAR bbox points  : "
            f"{int(lidar_visibility_cfg.get('min_lidar_points', 3))}"
        )

        rgb_visibility_cfg = ann_cfg.get(
            "rgb_visibility_filter",
            {},
        )
        print(
            f"RGB visibility filter  : "
            f"{rgb_visibility_cfg.get('enabled', True)}"
        )
        print(
            f"Min RGB visible pixels : "
            f"{int(rgb_visibility_cfg.get('min_visible_pixels', 5))}"
        )
        print(
            f"LiDAR sensor range     : "
            f"{float(lidar_cfg['range']):.1f} m"
        )
        print(
            f"LiDAR horizontal FOV   : "
            f"{float(lidar_cfg['horizontal_fov']):.1f} deg"
        )
        print(
            f"LiDAR vertical FOV     : "
            f"{float(lidar_cfg['lower_fov']):.1f} .. "
            f"{float(lidar_cfg['upper_fov']):.1f} deg"
        )
        print(
            f"BBox FOV samples/axis  : "
            f"{int(lidar_filter_cfg.get('bbox_samples_per_axis', 5))}"
        )

    print("=" * 64)
    print()


########################## 对象分类：把 CARLA 对象映射为数据集类别 ################################

def blueprint_base_type(blueprint):
    if blueprint.has_attribute("base_type"):
        return blueprint.get_attribute("base_type").as_str().lower()

    return ""


def classify_actor(actor):
    if actor.type_id.startswith("walker.pedestrian."):
        return "pedestrian"

    if actor.type_id.startswith("vehicle."):
        base_type = str(
            actor.attributes.get("base_type", "")
        ).lower()

        if base_type in (
            "car",
            "van",
            "truck",
            "bus",
            "motorcycle",
            "bicycle",
        ):
            return base_type

        semantic_tags = set(int(x) for x in actor.semantic_tags)

        if 15 in semantic_tags:
            return "truck"
        if 16 in semantic_tags:
            return "bus"
        if 18 in semantic_tags:
            return "motorcycle"
        if 19 in semantic_tags:
            return "bicycle"
        if 14 in semantic_tags:
            return "car"

    return None


def actor_instance_semantic_tag(
    actor,
    actor_class,
):
    """
    Resolve the semantic tag that CARLA actually renders for this dynamic
    actor in sensor.camera.instance_segmentation.

    Dataset class and rendered semantic tag are intentionally treated as
    separate concepts. For example, an actor classified as dataset class
    "van" by its base_type may still use a CARLA component tag associated
    with another vehicle semantic category.

    actor.semantic_tags is therefore authoritative when it contains one of
    the supported dynamic thing tags. The static class mapping is only a
    fallback for malformed/custom assets that expose no usable semantic tag.
    """
    actor_tags = []

    try:
        actor_tags = [
            int(tag)
            for tag in actor.semantic_tags
        ]
    except Exception:
        actor_tags = []

    for tag in actor_tags:
        if tag in DYNAMIC_INSTANCE_THING_TAGS:
            return int(tag)

    return int(
        CLASS_TO_SEMANTIC_TAG[
            actor_class
        ]
    )


########################## 静态地图对象：读取场景中不移动的车辆等对象 ################################

def collect_static_environment_objects(
    world,
    ann_cfg,
):
    """
    Load static vehicle-like level geometry once.

    CARLA 0.9.16 EnvironmentObject bounding boxes are already expressed in
    world space. These are distinct from spawned carla.Vehicle actors.

    Return:
        list of dicts:
        {
            "object": carla.EnvironmentObject,
            "class": dataset class string,
        }
    """
    static_cfg = ann_cfg.get(
        "static_environment",
        {},
    )

    if not static_cfg.get(
        "enabled",
        True,
    ):
        print("\nStatic environment annotations disabled.")
        return []

    allowed_classes = set(
        ann_cfg.get(
            "classes",
            [],
        )
    )

    requested_classes = list(
        static_cfg.get(
            "classes",
            [
                "car",
                "truck",
                "bus",
                "motorcycle",
                "bicycle",
            ],
        )
    )

    result = []
    seen_ids = set()
    counts = {}

    print("\nLoading static map vehicle objects...")

    for object_class in requested_classes:
        object_class = str(
            object_class
        ).lower()

        if object_class not in allowed_classes:
            continue

        city_label = (
            STATIC_ENV_CLASS_TO_CITY_LABEL[
                object_class
            ]
        )

        environment_objects = list(
            world.get_environment_objects(
                city_label
            )
        )

        added = 0

        for environment_object in environment_objects:
            object_id = int(
                environment_object.id
            )

            if object_id in seen_ids:
                continue

            seen_ids.add(
                object_id
            )

            result.append(
                {
                    "object": environment_object,
                    "class": object_class,
                }
            )

            added += 1

        counts[
            object_class
        ] = added

        print(
            f"  {object_class:<12}: {added}"
        )

    print(
        "  total       :",
        len(result),
    )

    if "van" in allowed_classes:
        print(
            "  note        : static map vans use CARLA's Car semantic "
            "label and are stored as class 'car'."
        )

    return result


########################## 交通：生成和管理车辆、行人及其控制器 ################################

def spawn_traffic(
    client,
    world,
    config,
    random_seed,
    allowed_classes,
):
    if not config["enabled"]:
        print("Traffic disabled.")
        return []

    rng = random.Random(int(random_seed))

    bp_lib = world.get_blueprint_library()
    all_vehicle_bps = list(bp_lib.filter("vehicle.*"))

    allowed_vehicle_classes = (
        set(allowed_classes)
        & {
            "car",
            "van",
            "truck",
            "bus",
            "motorcycle",
            "bicycle",
        }
    )

    vehicle_bps = []

    for bp in all_vehicle_bps:
        base_type = blueprint_base_type(bp)

        if not base_type or base_type in allowed_vehicle_classes:
            vehicle_bps.append(bp)

    if not vehicle_bps:
        raise RuntimeError(
            "No vehicle blueprints are available for configured classes."
        )

    spawn_points = world.get_map().get_spawn_points()
    rng.shuffle(spawn_points)

    requested_number = int(config["num_vehicles"])
    number = min(requested_number, len(spawn_points))

    print("\nSpawning traffic vehicles...")
    print("  requested vehicles :", requested_number)
    print("  available spawns   :", len(spawn_points))
    print("  spawn attempts     :", number)

    if requested_number > len(spawn_points):
        print(
            "WARNING: requested vehicle count exceeds available "
            "spawn points."
        )

    tm_port = int(config["tm_port"])
    batch = []

    for spawn in spawn_points[:number]:
        bp = rng.choice(vehicle_bps)

        if bp.has_attribute("role_name"):
            bp.set_attribute("role_name", "autopilot")

        if bp.has_attribute("color"):
            colors = bp.get_attribute("color").recommended_values
            if colors:
                bp.set_attribute("color", rng.choice(colors))

        if bp.has_attribute("driver_id"):
            drivers = bp.get_attribute("driver_id").recommended_values
            if drivers:
                bp.set_attribute("driver_id", rng.choice(drivers))

        batch.append(
            carla.command.SpawnActor(bp, spawn).then(
                carla.command.SetAutopilot(
                    carla.command.FutureActor,
                    True,
                    tm_port,
                )
            )
        )

    responses = client.apply_batch_sync(batch, True)

    ids = []
    failed = 0

    for response in responses:
        if response.error:
            failed += 1
        else:
            ids.append(response.actor_id)

    vehicles = []

    for actor_id in ids:
        actor = world.get_actor(actor_id)
        if actor is not None:
            vehicles.append(actor)

    print("  successfully spawned:", len(vehicles))
    print("  failed              :", failed)

    return vehicles


def spawn_pedestrians(
    world,
    config,
    random_seed,
):
    requested = int(config.get("num_pedestrians", 0))

    if not config["enabled"] or requested <= 0:
        return [], []

    rng = random.Random(int(random_seed) + 100003)

    try:
        world.set_pedestrians_seed(int(random_seed))
    except RuntimeError:
        pass

    bp_lib = world.get_blueprint_library()
    walker_bps = list(bp_lib.filter("walker.pedestrian.*"))

    if not walker_bps:
        print("WARNING: no pedestrian blueprints found.")
        return [], []

    walkers = []

    max_attempts = max(requested * 10, 50)
    attempts = 0

    print("\nSpawning pedestrians...")
    print("  requested pedestrians:", requested)

    while len(walkers) < requested and attempts < max_attempts:
        attempts += 1

        location = world.get_random_location_from_navigation()

        if location is None:
            continue

        bp = rng.choice(walker_bps)

        if bp.has_attribute("is_invincible"):
            bp.set_attribute("is_invincible", "false")

        transform = carla.Transform(location)

        walker = world.try_spawn_actor(bp, transform)

        if walker is not None:
            walkers.append(walker)

    if not walkers:
        print("WARNING: no pedestrians could be spawned.")
        return [], []

    controller_bp = bp_lib.find("controller.ai.walker")
    controllers = []

    for walker in walkers:
        controller = world.try_spawn_actor(
            controller_bp,
            carla.Transform(),
            attach_to=walker,
        )

        if controller is not None:
            controllers.append(controller)

    world.tick()

    speed_min = float(
        config.get("pedestrian_speed_min_mps", 1.0)
    )
    speed_max = float(
        config.get("pedestrian_speed_max_mps", 2.0)
    )

    if speed_max < speed_min:
        speed_min, speed_max = speed_max, speed_min

    active_controllers = []

    for controller in controllers:
        try:
            controller.start()

            destination = world.get_random_location_from_navigation()

            if destination is not None:
                controller.go_to_location(destination)

            controller.set_max_speed(
                rng.uniform(speed_min, speed_max)
            )

            active_controllers.append(controller)

        except RuntimeError:
            pass

    print("  walkers spawned     :", len(walkers))
    print("  controllers active  :", len(active_controllers))

    return walkers, active_controllers


def warmup_traffic(world, fps, warmup_seconds):
    warmup_seconds = float(warmup_seconds)

    if warmup_seconds <= 0:
        return

    warmup_ticks = int(math.ceil(warmup_seconds * fps))

    print(
        f"\nTraffic warmup: "
        f"{warmup_seconds:.1f} s ({warmup_ticks} ticks)"
    )

    for _ in range(warmup_ticks):
        world.tick()

    print("Traffic warmup complete.")


########################## 实例分割：读取用于计算可见框的分割图 ################################

def decode_instance_segmentation(instance_image):
    """
    CARLA raw camera data is BGRA.

    CARLA's documented RGB interpretation is:
      R = semantic tag
      G/B = instance ID

    The numeric 16-bit instance ID is decoded as:
      (B << 8) | G

    CARLA 0.9.16 uses ActorIDs for actor instances when available.
    """
    raw = np.frombuffer(
        instance_image.raw_data,
        dtype=np.uint8,
    ).reshape(
        instance_image.height,
        instance_image.width,
        4,
    )

    blue = raw[:, :, 0].astype(np.uint16)
    green = raw[:, :, 1].astype(np.uint16)
    red = raw[:, :, 2].astype(np.uint8)

    instance_ids = (
        (blue << np.uint16(8))
        | green
    ).astype(np.uint16)

    semantic_tags = red

    return semantic_tags, instance_ids


def instance_key(actor, actor_class):
    return (
        int(actor.id) & 0xFFFF,
        actor_instance_semantic_tag(
            actor,
            actor_class,
        ),
    )


def print_dynamic_semantic_tag_audit(
    actors,
    allowed_classes,
):
    """
    Print only semantic-tag overrides. This makes asset-specific differences
    visible immediately without flooding the console every frame.
    """
    overrides = defaultdict(
        lambda: {
            "count": 0,
            "actor_ids": [],
        }
    )

    for actor in actors:
        if actor is None or not actor.is_alive:
            continue

        actor_class = classify_actor(actor)

        if actor_class not in allowed_classes:
            continue

        configured_tag = int(
            CLASS_TO_SEMANTIC_TAG[
                actor_class
            ]
        )
        actual_tag = int(
            actor_instance_semantic_tag(
                actor,
                actor_class,
            )
        )

        if actual_tag == configured_tag:
            continue

        key = (
            actor_class,
            actor.type_id,
            configured_tag,
            actual_tag,
        )

        overrides[key]["count"] += 1

        if len(
            overrides[key]["actor_ids"]
        ) < 5:
            overrides[key][
                "actor_ids"
            ].append(
                int(actor.id)
            )

    if not overrides:
        print(
            "\nDynamic semantic-tag audit: "
            "no dataset-class/render-tag overrides detected."
        )
        return

    print(
        "\nDynamic semantic-tag audit:"
    )
    print(
        "  CARLA-rendered semantic tags override the old "
        "dataset-class mapping for these actor types:"
    )

    for (
        actor_class,
        type_id,
        configured_tag,
        actual_tag,
    ), info in sorted(
        overrides.items()
    ):
        print(
            f"  class={actor_class:<10} "
            f"type={type_id:<36} "
            f"old_tag={configured_tag:<3} "
            f"actual_tag={actual_tag:<3} "
            f"count={info['count']:<3} "
            f"sample_actor_ids={info['actor_ids']}"
        )


def find_instance_key_collisions(actors, allowed_classes):
    groups = defaultdict(list)

    for actor in actors:
        if actor is None or not actor.is_alive:
            continue

        actor_class = classify_actor(actor)

        if actor_class not in allowed_classes:
            continue

        groups[instance_key(actor, actor_class)].append(
            int(actor.id)
        )

    collisions = {
        key: ids
        for key, ids in groups.items()
        if len(ids) > 1
    }

    if collisions:
        print(
            "\nWARNING: instance-segmentation 16-bit ID collision(s) "
            "detected."
        )

        for key, ids in collisions.items():
            print(
                f"  instance_key={key} actor_ids={ids}"
            )

        print(
            "Visible 2D boxes for colliding actors will be disabled "
            "to avoid incorrect labels."
        )

    return set(collisions.keys())


def visible_bbox_from_instance(
    semantic_tags,
    instance_ids,
    actor,
    actor_class,
    min_visible_pixels,
    collision_keys,
):
    key = instance_key(actor, actor_class)

    if key in collision_keys:
        return None, "instance_id_collision", 0

    instance_id_16 = key[0]
    semantic_tag = key[1]

    mask = (
        (instance_ids == instance_id_16)
        & (semantic_tags == semantic_tag)
    )

    visible_pixels = int(np.count_nonzero(mask))

    if visible_pixels < int(min_visible_pixels):
        return (
            None,
            "not_visible_or_too_small",
            visible_pixels,
        )

    ys, xs = np.nonzero(mask)

    xmin = int(xs.min())
    ymin = int(ys.min())
    xmax = int(xs.max())
    ymax = int(ys.max())

    return (
        {
            "xyxy": [xmin, ymin, xmax, ymax],
            "visible_pixels": visible_pixels,
            "bbox_area_px2": int(
                (xmax - xmin + 1)
                * (ymax - ymin + 1)
            ),
        },
        None,
        visible_pixels,
    )


########################## 三维框几何：计算目标包围盒角点和坐标 ################################

def frame_bbox_dict(
    target_from_bbox,
    size_xyz,
    corners_xyz,
):
    return {
        "center_xyz_m": [
            float(v)
            for v in target_from_bbox[:3, 3]
        ],
        "size_xyz_m": [
            float(v)
            for v in size_xyz
        ],
        "rotation_deg": matrix_to_carla_rotation(
            target_from_bbox
        ),
        "orientation_matrix": (
            target_from_bbox[:3, :3].tolist()
        ),
        "transform": target_from_bbox.tolist(),
        "corners_xyz_m": corners_xyz.tolist(),
    }


def camera_cv_bbox_dict(
    camera_cv_from_bbox,
    size_xyz,
    corners_camera_cv,
):
    return {
        "center_xyz_m": [
            float(v)
            for v in camera_cv_from_bbox[:3, 3]
        ],
        "size_xyz_m": [
            float(v)
            for v in size_xyz
        ],
        "orientation_matrix": (
            camera_cv_from_bbox[:3, :3].tolist()
        ),
        "transform": camera_cv_from_bbox.tolist(),
        "corners_xyz_m": corners_camera_cv.tolist(),
    }


def count_lidar_points_in_bbox(
    lidar_points_xyz,
    lidar_from_bbox,
    extent_xyz,
):
    if lidar_points_xyz.size == 0:
        return 0

    extent_xyz = np.asarray(
        extent_xyz,
        dtype=np.float64,
    )

    local_corners = np.array(
        [
            [sx * extent_xyz[0], sy * extent_xyz[1], sz * extent_xyz[2]]
            for sx in (-1.0, 1.0)
            for sy in (-1.0, 1.0)
            for sz in (-1.0, 1.0)
        ],
        dtype=np.float64,
    )

    corners_lidar = transform_points(
        local_corners,
        lidar_from_bbox,
    )

    bb_min = corners_lidar.min(axis=0) - 1e-4
    bb_max = corners_lidar.max(axis=0) + 1e-4

    candidate_mask = np.all(
        (lidar_points_xyz >= bb_min)
        & (lidar_points_xyz <= bb_max),
        axis=1,
    )

    candidates = lidar_points_xyz[candidate_mask]

    if candidates.size == 0:
        return 0

    bbox_from_lidar = np.linalg.inv(
        lidar_from_bbox
    )

    candidates_bbox = transform_points(
        candidates,
        bbox_from_lidar,
    )

    inside = np.all(
        np.abs(candidates_bbox)
        <= extent_xyz + 1e-4,
        axis=1,
    )

    return int(np.count_nonzero(inside))



########################## LiDAR 视场筛选：判断目标是否在量程和视场内 ################################

def make_bbox_local_grid(extent_xyz, samples_per_axis=5):
    """
    Sample the complete oriented bounding-box volume in bbox-local coordinates.

    5 samples/axis -> 125 test points per object. This intentionally tests the
    bbox volume rather than only its center or 8 corners, which makes FOV-edge
    decisions much more stable for buses/trucks and other large actors.
    """
    extent_xyz = np.asarray(
        extent_xyz,
        dtype=np.float64,
    )

    samples_per_axis = max(
        3,
        int(samples_per_axis),
    )

    xs = np.linspace(
        -extent_xyz[0],
        extent_xyz[0],
        samples_per_axis,
        dtype=np.float64,
    )
    ys = np.linspace(
        -extent_xyz[1],
        extent_xyz[1],
        samples_per_axis,
        dtype=np.float64,
    )
    zs = np.linspace(
        -extent_xyz[2],
        extent_xyz[2],
        samples_per_axis,
        dtype=np.float64,
    )

    xx, yy, zz = np.meshgrid(
        xs,
        ys,
        zs,
        indexing="ij",
    )

    return np.stack(
        [
            xx.reshape(-1),
            yy.reshape(-1),
            zz.reshape(-1),
        ],
        axis=1,
    )


def bbox_lidar_fov_status(
    lidar_from_bbox,
    extent_xyz,
    lidar_cfg,
    ann_cfg,
):
    """
    Check whether any sampled part of an oriented actor bbox lies inside the
    LiDAR detection volume defined by:

      - range
      - horizontal_fov
      - lower_fov / upper_fov

    This function performs geometry-only FOV/range filtering. Occlusion and
    actual detectability are handled separately by the real LiDAR-return
    visibility filter.

    CARLA LiDAR local coordinates:
      x forward, y right, z up.
    """
    filter_cfg = ann_cfg.get(
        "lidar_fov_filter",
        {},
    )

    samples_per_axis = int(
        filter_cfg.get(
            "bbox_samples_per_axis",
            5,
        )
    )

    local_points = make_bbox_local_grid(
        extent_xyz,
        samples_per_axis,
    )

    points_lidar = transform_points(
        local_points,
        lidar_from_bbox,
    )

    x = points_lidar[:, 0]
    y = points_lidar[:, 1]
    z = points_lidar[:, 2]

    horizontal_distance = np.hypot(
        x,
        y,
    )

    distances = np.sqrt(
        x * x
        + y * y
        + z * z
    )

    azimuth_deg = np.degrees(
        np.arctan2(
            y,
            x,
        )
    )

    elevation_deg = np.degrees(
        np.arctan2(
            z,
            horizontal_distance,
        )
    )

    sensor_range = float(
        lidar_cfg["range"]
    )

    annotation_range = float(
        ann_cfg.get(
            "max_distance_m",
            sensor_range,
        )
    )

    effective_range = min(
        sensor_range,
        annotation_range,
    )

    horizontal_fov = float(
        lidar_cfg["horizontal_fov"]
    )

    lower_fov = float(
        lidar_cfg["lower_fov"]
    )

    upper_fov = float(
        lidar_cfg["upper_fov"]
    )

    range_mask = (
        distances
        <= effective_range + 1e-6
    )

    vertical_mask = (
        (elevation_deg >= lower_fov - 1e-6)
        & (elevation_deg <= upper_fov + 1e-6)
    )

    if horizontal_fov >= 359.999:
        horizontal_mask = np.ones(
            len(points_lidar),
            dtype=bool,
        )
    else:
        half_horizontal_fov = (
            horizontal_fov / 2.0
        )

        horizontal_mask = (
            np.abs(azimuth_deg)
            <= half_horizontal_fov + 1e-6
        )

    angular_mask = (
        vertical_mask
        & horizontal_mask
    )

    inside_mask = (
        range_mask
        & angular_mask
    )

    inside_count = int(
        np.count_nonzero(
            inside_mask
        )
    )

    return {
        "in_detection_volume": bool(
            inside_count > 0
        ),
        "num_bbox_test_points": int(
            len(points_lidar)
        ),
        "num_bbox_test_points_inside": int(
            inside_count
        ),
        "method": "bbox_volume_grid_sampling",
        "bbox_samples_per_axis": int(
            samples_per_axis
        ),
        "sensor_range_m": float(
            sensor_range
        ),
        "annotation_max_distance_m": float(
            annotation_range
        ),
        "effective_range_m": float(
            effective_range
        ),
        "horizontal_fov_deg": float(
            horizontal_fov
        ),
        "lower_fov_deg": float(
            lower_fov
        ),
        "upper_fov_deg": float(
            upper_fov
        ),
    }


########################## 二维投影：把三维包围盒投影到相机图像 ################################

def project_cv_points(points_cv, K):
    points_cv = np.asarray(points_cv, dtype=np.float64)

    projected = (K @ points_cv.T).T
    projected[:, 0] /= projected[:, 2]
    projected[:, 1] /= projected[:, 2]

    return projected[:, :2]


def clip_cuboid_to_near_plane(
    corners_cv,
    near_clip_m,
):
    """
    Return cuboid points after clipping edges against z=near_clip.
    This gives a stable projected 2D box even if part of the 3D box is
    behind the camera.
    """
    corners_cv = np.asarray(
        corners_cv,
        dtype=np.float64,
    )

    points = []

    for p in corners_cv:
        if p[2] >= near_clip_m:
            points.append(p)

    for i0, i1 in BBOX_EDGES:
        p0 = corners_cv[i0]
        p1 = corners_cv[i1]

        z0_front = p0[2] >= near_clip_m
        z1_front = p1[2] >= near_clip_m

        if z0_front == z1_front:
            continue

        dz = p1[2] - p0[2]

        if abs(dz) <= 1e-12:
            continue

        t = (near_clip_m - p0[2]) / dz

        if 0.0 <= t <= 1.0:
            p = p0 + t * (p1 - p0)
            p[2] = near_clip_m
            points.append(p)

    if not points:
        return np.empty((0, 3), dtype=np.float64)

    return np.asarray(points, dtype=np.float64)


def projected_bbox_from_corners(
    corners_camera_cv,
    K,
    image_width,
    image_height,
    near_clip_m=0.05,
):
    clipped_3d = clip_cuboid_to_near_plane(
        corners_camera_cv,
        near_clip_m,
    )

    if clipped_3d.shape[0] == 0:
        return None

    pixels = project_cv_points(
        clipped_3d,
        K,
    )

    xmin = float(np.min(pixels[:, 0]))
    ymin = float(np.min(pixels[:, 1]))
    xmax = float(np.max(pixels[:, 0]))
    ymax = float(np.max(pixels[:, 1]))

    if (
        xmax < 0.0
        or ymax < 0.0
        or xmin > image_width - 1
        or ymin > image_height - 1
    ):
        return None

    cxmin = float(
        np.clip(xmin, 0.0, image_width - 1.0)
    )
    cymin = float(
        np.clip(ymin, 0.0, image_height - 1.0)
    )
    cxmax = float(
        np.clip(xmax, 0.0, image_width - 1.0)
    )
    cymax = float(
        np.clip(ymax, 0.0, image_height - 1.0)
    )

    if cxmax <= cxmin or cymax <= cymin:
        return None

    truncated = bool(
        xmin < 0.0
        or ymin < 0.0
        or xmax > image_width - 1
        or ymax > image_height - 1
        or np.any(
            np.asarray(corners_camera_cv)[:, 2]
            < near_clip_m
        )
    )

    return {
        "xyxy_unclipped": [
            xmin,
            ymin,
            xmax,
            ymax,
        ],
        "xyxy": [
            cxmin,
            cymin,
            cxmax,
            cymax,
        ],
        "truncated": truncated,
        "bbox_area_px2": float(
            (cxmax - cxmin)
            * (cymax - cymin)
        ),
    }


########################## 标注：生成目标三维框、二维框和诊断信息 ################################

def build_object_annotation(
    actor,
    actor_class,
    camera_transform,
    lidar_transform,
    lidar_points_xyz,
    semantic_tags,
    instance_ids,
    K,
    image_width,
    image_height,
    ann_cfg,
    lidar_cfg,
    collision_keys,
):
    if actor is None or not actor.is_alive:
        return None, "invalid_actor"

    actor_transform = actor.get_transform()
    bbox = actor.bounding_box

    world_from_actor = matrix(actor_transform)
    actor_from_bbox = matrix(
        carla.Transform(
            bbox.location,
            bbox.rotation,
        )
    )

    world_from_bbox = (
        world_from_actor
        @ actor_from_bbox
    )

    corners_world = carla_locations_to_numpy(
        bbox.get_world_vertices(
            actor_transform
        )
    )

    world_from_lidar = matrix(
        lidar_transform
    )
    world_from_camera = matrix(
        camera_transform
    )

    lidar_from_world = np.linalg.inv(
        world_from_lidar
    )
    camera_ue_from_world = np.linalg.inv(
        world_from_camera
    )

    lidar_from_bbox = (
        lidar_from_world
        @ world_from_bbox
    )

    camera_ue_from_bbox = (
        camera_ue_from_world
        @ world_from_bbox
    )

    camera_cv_from_bbox = (
        T_CV_UE
        @ camera_ue_from_bbox
    )

    corners_lidar = transform_points(
        corners_world,
        lidar_from_world,
    )

    corners_camera_ue = transform_points(
        corners_world,
        camera_ue_from_world,
    )

    corners_camera_cv = transform_points(
        corners_world,
        T_CV_UE @ camera_ue_from_world,
    )

    extent_xyz = np.array(
        [
            float(bbox.extent.x),
            float(bbox.extent.y),
            float(bbox.extent.z),
        ],
        dtype=np.float64,
    )

    size_xyz = 2.0 * extent_xyz

    center_world = world_from_bbox[:3, 3]
    lidar_origin_world = world_from_lidar[:3, 3]

    center_distance = float(
        np.linalg.norm(
            center_world - lidar_origin_world
        )
    )

    bounding_radius = float(
        np.linalg.norm(extent_xyz)
    )

    nearest_box_distance = max(
        0.0,
        center_distance - bounding_radius,
    )

    max_distance = float(
        ann_cfg.get(
            "max_distance_m",
            float(lidar_cfg["range"]),
        )
    )

    lidar_filter_cfg = ann_cfg.get(
        "lidar_fov_filter",
        {},
    )

    lidar_filter_enabled = bool(
        lidar_filter_cfg.get(
            "enabled",
            True,
        )
    )

    lidar_fov_status = None

    if lidar_filter_enabled:
        lidar_fov_status = bbox_lidar_fov_status(
            lidar_from_bbox=lidar_from_bbox,
            extent_xyz=extent_xyz,
            lidar_cfg=lidar_cfg,
            ann_cfg=ann_cfg,
        )

        if not lidar_fov_status[
            "in_detection_volume"
        ]:
            return None, "outside_lidar_fov"

    else:
        if nearest_box_distance > max_distance:
            return None, "outside_annotation_distance"

    ########################## LiDAR 可见性：使用真实点云作为目标保留的硬门槛 ################################

    lidar_visibility_cfg = ann_cfg.get(
        "lidar_visibility",
        {},
    )
    lidar_visibility_enabled = bool(
        lidar_visibility_cfg.get(
            "enabled",
            True,
        )
    )
    min_lidar_points = int(
        lidar_visibility_cfg.get(
            "min_lidar_points",
            3,
        )
    )

    num_lidar_points = count_lidar_points_in_bbox(
        lidar_points_xyz,
        lidar_from_bbox,
        extent_xyz,
    )
    lidar_visible = (
        num_lidar_points
        >= min_lidar_points
    )

    if (
        lidar_visibility_enabled
        and not lidar_visible
    ):
        return None, "insufficient_lidar_points"

    bbox3d_cfg = ann_cfg.get(
        "bbox3d",
        {},
    )
    bbox2d_cfg = ann_cfg.get(
        "bbox2d",
        {},
    )

    ########################## RGB 可见性：动态 Actor 必须通过真实实例像素硬门槛 ################################

    rgb_visibility_cfg = ann_cfg.get(
        "rgb_visibility_filter",
        {},
    )
    rgb_visibility_enabled = bool(
        rgb_visibility_cfg.get(
            "enabled",
            True,
        )
    )
    min_rgb_visible_pixels = int(
        rgb_visibility_cfg.get(
            "min_visible_pixels",
            5,
        )
    )

    visible_bbox = None
    visible_status = None
    rgb_visible_pixels = None
    rgb_visibility_evaluated = False

    need_rgb_measurement = bool(
        rgb_visibility_enabled
        or (
            bbox2d_cfg.get(
                "enabled",
                True,
            )
            and bbox2d_cfg.get(
                "visible",
                True,
            )
        )
    )

    if need_rgb_measurement:
        if (
            semantic_tags is not None
            and instance_ids is not None
        ):
            (
                visible_bbox,
                visible_status,
                rgb_visible_pixels,
            ) = visible_bbox_from_instance(
                semantic_tags,
                instance_ids,
                actor,
                actor_class,
                min_rgb_visible_pixels,
                collision_keys,
            )
            rgb_visibility_evaluated = True
        else:
            visible_status = (
                "instance_segmentation_unavailable"
            )

    if rgb_visibility_enabled:
        if not rgb_visibility_evaluated:
            return None, "rgb_visibility_unavailable"

        if int(rgb_visible_pixels) < min_rgb_visible_pixels:
            return None, "insufficient_rgb_visible_pixels"

    object_data = {
        "actor_id": int(actor.id),
        "instance_segmentation_id_16bit": (
            int(actor.id) & 0xFFFF
        ),
        "class": actor_class,
        "type_id": actor.type_id,
        "semantic_tag": int(
            actor_instance_semantic_tag(
                actor,
                actor_class,
            )
        ),
        "distance_to_bbox_center_m": center_distance,
        "distance_to_bbox_surface_approx_m": nearest_box_distance,
        "actor_pose_world": pose_dict(
            actor_transform
        ),
        "num_lidar_points": int(
            num_lidar_points
        ),
        "num_rgb_visible_pixels": (
            int(rgb_visible_pixels)
            if rgb_visible_pixels is not None
            else None
        ),
        "visibility": {
            "lidar": (
                bool(lidar_visible)
                if lidar_visibility_enabled
                else None
            ),
            "lidar_status": (
                "visible"
                if lidar_visibility_enabled
                else "filter_disabled"
            ),
            "rgb": (
                bool(
                    rgb_visibility_evaluated
                    and int(rgb_visible_pixels)
                    >= min_rgb_visible_pixels
                )
                if rgb_visibility_evaluated
                else None
            ),
            "rgb_status": (
                "visible"
                if (
                    rgb_visibility_evaluated
                    and int(rgb_visible_pixels)
                    >= min_rgb_visible_pixels
                )
                else (
                    visible_status
                    or "not_evaluated"
                )
            ),
        },
    }

    if lidar_filter_enabled:
        object_data["lidar_fov"] = (
            lidar_fov_status
        )

    if actor.type_id.startswith(
        "vehicle."
    ):
        object_data["base_type"] = str(
            actor.attributes.get(
                "base_type",
                "",
            )
        )

    ########################## 三维框：生成三维包围盒数据 ################################

    if bbox3d_cfg.get(
        "enabled",
        True,
    ):
        bbox3d = {
            "size_xyz_m": size_xyz.tolist(),
        }

        if bbox3d_cfg.get(
            "save_world",
            True,
        ):
            bbox3d["world"] = frame_bbox_dict(
                world_from_bbox,
                size_xyz,
                corners_world,
            )

        if bbox3d_cfg.get(
            "save_lidar",
            True,
        ):
            bbox3d["lidar"] = frame_bbox_dict(
                lidar_from_bbox,
                size_xyz,
                corners_lidar,
            )

        if bbox3d_cfg.get(
            "save_camera",
            True,
        ):
            bbox3d["camera_ue"] = frame_bbox_dict(
                camera_ue_from_bbox,
                size_xyz,
                corners_camera_ue,
            )

            bbox3d["camera_cv"] = camera_cv_bbox_dict(
                camera_cv_from_bbox,
                size_xyz,
                corners_camera_cv,
            )

        object_data["bbox3d"] = bbox3d

    ########################## 二维框：生成图像中的目标框 ################################

    if bbox2d_cfg.get(
        "enabled",
        True,
    ):
        bbox2d = {
            "projected": None,
            "visible": None,
            "visible_status": None,
        }

        if bbox2d_cfg.get(
            "projected",
            True,
        ):
            bbox2d["projected"] = (
                projected_bbox_from_corners(
                    corners_camera_cv,
                    K,
                    image_width,
                    image_height,
                    near_clip_m=float(
                        bbox2d_cfg.get(
                            "near_clip_m",
                            0.05,
                        )
                    ),
                )
            )

        if bbox2d_cfg.get(
            "visible",
            True,
        ):
            bbox2d["visible"] = (
                visible_bbox
                if rgb_visibility_evaluated
                else None
            )
            bbox2d["visible_status"] = (
                visible_status
                if visible_status is not None
                else (
                    None
                    if rgb_visibility_evaluated
                    else "instance_segmentation_unavailable"
                )
            )

        object_data["bbox2d"] = bbox2d

    return object_data, None



def build_environment_object_annotation(
    environment_object,
    object_class,
    camera_transform,
    lidar_transform,
    lidar_points_xyz,
    K,
    image_width,
    image_height,
    ann_cfg,
    lidar_cfg,
):
    """
    Build the same 3D/projected-2D annotation structure for static map
    EnvironmentObjects.

    Important:
      - EnvironmentObject.bounding_box is already in world space in CARLA.
      - There is no reliable ActorID mapping for these static level objects in
        the instance-segmentation stream, so bbox2d.visible is intentionally
        left unavailable. bbox2d.projected and the RGB 3D cuboid are saved.
      - LiDAR FOV filtering is identical to dynamic actors.
      - Real LiDAR returns inside the oriented 3D bbox are a hard retention
        filter when annotations.lidar_visibility.enabled is true.
    """
    if environment_object is None:
        return None, "invalid_environment_object"

    bbox = environment_object.bounding_box

    world_from_bbox = matrix(
        carla.Transform(
            bbox.location,
            bbox.rotation,
        )
    )

    corners_world = carla_locations_to_numpy(
        bbox.get_world_vertices(
            carla.Transform()
        )
    )

    world_from_lidar = matrix(
        lidar_transform
    )

    world_from_camera = matrix(
        camera_transform
    )

    lidar_from_world = np.linalg.inv(
        world_from_lidar
    )

    camera_ue_from_world = np.linalg.inv(
        world_from_camera
    )

    lidar_from_bbox = (
        lidar_from_world
        @ world_from_bbox
    )

    camera_ue_from_bbox = (
        camera_ue_from_world
        @ world_from_bbox
    )

    camera_cv_from_bbox = (
        T_CV_UE
        @ camera_ue_from_bbox
    )

    corners_lidar = transform_points(
        corners_world,
        lidar_from_world,
    )

    corners_camera_ue = transform_points(
        corners_world,
        camera_ue_from_world,
    )

    corners_camera_cv = transform_points(
        corners_world,
        T_CV_UE @ camera_ue_from_world,
    )

    extent_xyz = np.array(
        [
            float(bbox.extent.x),
            float(bbox.extent.y),
            float(bbox.extent.z),
        ],
        dtype=np.float64,
    )

    size_xyz = (
        2.0
        * extent_xyz
    )

    center_world = (
        world_from_bbox[
            :3,
            3,
        ]
    )

    lidar_origin_world = (
        world_from_lidar[
            :3,
            3,
        ]
    )

    center_distance = float(
        np.linalg.norm(
            center_world
            - lidar_origin_world
        )
    )

    bounding_radius = float(
        np.linalg.norm(
            extent_xyz
        )
    )

    nearest_box_distance = max(
        0.0,
        center_distance
        - bounding_radius,
    )

    lidar_filter_cfg = ann_cfg.get(
        "lidar_fov_filter",
        {},
    )

    lidar_filter_enabled = bool(
        lidar_filter_cfg.get(
            "enabled",
            True,
        )
    )

    lidar_fov_status = None

    if lidar_filter_enabled:
        effective_range = min(
            float(
                lidar_cfg[
                    "range"
                ]
            ),
            float(
                ann_cfg.get(
                    "max_distance_m",
                    lidar_cfg[
                        "range"
                    ],
                )
            ),
        )

        if (
            nearest_box_distance
            > effective_range
        ):
            return None, "outside_lidar_fov"

        lidar_fov_status = (
            bbox_lidar_fov_status(
                lidar_from_bbox=(
                    lidar_from_bbox
                ),
                extent_xyz=(
                    extent_xyz
                ),
                lidar_cfg=(
                    lidar_cfg
                ),
                ann_cfg=(
                    ann_cfg
                ),
            )
        )

        if not lidar_fov_status[
            "in_detection_volume"
        ]:
            return None, "outside_lidar_fov"

    else:
        max_distance = float(
            ann_cfg.get(
                "max_distance_m",
                lidar_cfg[
                    "range"
                ],
            )
        )

        if (
            nearest_box_distance
            > max_distance
        ):
            return (
                None,
                "outside_annotation_distance",
            )

    ########################## LiDAR 可见性：静态地图对象同样使用真实点云硬筛选 ################################

    lidar_visibility_cfg = ann_cfg.get(
        "lidar_visibility",
        {},
    )
    lidar_visibility_enabled = bool(
        lidar_visibility_cfg.get(
            "enabled",
            True,
        )
    )
    min_lidar_points = int(
        lidar_visibility_cfg.get(
            "min_lidar_points",
            3,
        )
    )

    num_lidar_points = count_lidar_points_in_bbox(
        lidar_points_xyz,
        lidar_from_bbox,
        extent_xyz,
    )
    lidar_visible = (
        num_lidar_points
        >= min_lidar_points
    )

    if (
        lidar_visibility_enabled
        and not lidar_visible
    ):
        return None, "insufficient_lidar_points"

    bbox3d_cfg = ann_cfg.get(
        "bbox3d",
        {},
    )

    bbox2d_cfg = ann_cfg.get(
        "bbox2d",
        {},
    )

    object_data = {
        "source": "static_environment",
        "environment_object_id": int(
            environment_object.id
        ),
        "environment_object_name": str(
            environment_object.name
        ),
        "class": object_class,
        "type_id": (
            f"environment.{object_class}"
        ),
        "semantic_tag": int(
            CLASS_TO_SEMANTIC_TAG[
                object_class
            ]
        ),
        "distance_to_bbox_center_m": (
            center_distance
        ),
        "distance_to_bbox_surface_approx_m": (
            nearest_box_distance
        ),
        "environment_pose_world": (
            pose_dict(
                environment_object.transform
            )
        ),
        "num_lidar_points": int(
            num_lidar_points
        ),
        "num_rgb_visible_pixels": None,
        "visibility": {
            "lidar": (
                bool(lidar_visible)
                if lidar_visibility_enabled
                else None
            ),
            "lidar_status": (
                "visible"
                if lidar_visibility_enabled
                else "filter_disabled"
            ),
            "rgb": None,
            "rgb_status": (
                "not_available_for_static_environment_object"
            ),
        },
    }

    if lidar_filter_enabled:
        object_data[
            "lidar_fov"
        ] = lidar_fov_status

    ########################## 三维框：生成三维包围盒数据 ################################

    if bbox3d_cfg.get(
        "enabled",
        True,
    ):
        bbox3d = {
            "size_xyz_m": (
                size_xyz.tolist()
            ),
        }

        if bbox3d_cfg.get(
            "save_world",
            True,
        ):
            bbox3d[
                "world"
            ] = frame_bbox_dict(
                world_from_bbox,
                size_xyz,
                corners_world,
            )

        if bbox3d_cfg.get(
            "save_lidar",
            True,
        ):
            bbox3d[
                "lidar"
            ] = frame_bbox_dict(
                lidar_from_bbox,
                size_xyz,
                corners_lidar,
            )

        if bbox3d_cfg.get(
            "save_camera",
            True,
        ):
            bbox3d[
                "camera_ue"
            ] = frame_bbox_dict(
                camera_ue_from_bbox,
                size_xyz,
                corners_camera_ue,
            )

            bbox3d[
                "camera_cv"
            ] = camera_cv_bbox_dict(
                camera_cv_from_bbox,
                size_xyz,
                corners_camera_cv,
            )

        object_data[
            "bbox3d"
        ] = bbox3d

    ########################## 二维框：生成图像中的目标框 ################################

    if bbox2d_cfg.get(
        "enabled",
        True,
    ):
        bbox2d = {
            "projected": None,
            "visible": None,
            "visible_status": (
                "not_available_for_static_environment_object"
            ),
        }

        if bbox2d_cfg.get(
            "projected",
            True,
        ):
            bbox2d[
                "projected"
            ] = projected_bbox_from_corners(
                corners_camera_cv,
                K,
                image_width,
                image_height,
                near_clip_m=float(
                    bbox2d_cfg.get(
                        "near_clip_m",
                        0.05,
                    )
                ),
            )

        object_data[
            "bbox2d"
        ] = bbox2d

    return object_data, None


def build_frame_annotations(
    actors,
    environment_objects,
    camera_transform,
    lidar_transform,
    lidar_points_xyz,
    instance_image,
    K,
    image_width,
    image_height,
    ann_cfg,
    lidar_cfg,
    collision_keys,
):
    allowed_classes = set(
        ann_cfg["classes"]
    )

    semantic_tags = None
    instance_ids = None

    bbox2d_cfg = ann_cfg.get(
        "bbox2d",
        {},
    )

    rgb_visibility_cfg = ann_cfg.get(
        "rgb_visibility_filter",
        {},
    )

    need_instance_visibility = bool(
        rgb_visibility_cfg.get(
            "enabled",
            True,
        )
        or (
            bbox2d_cfg.get(
                "enabled",
                True,
            )
            and bbox2d_cfg.get(
                "visible",
                True,
            )
        )
    )

    if (
        need_instance_visibility
        and instance_image is not None
    ):
        semantic_tags, instance_ids = (
            decode_instance_segmentation(
                instance_image
            )
        )

    objects = []

    stats = {
        "candidate_actors": 0,
        "candidate_environment_objects": 0,
        "saved_dynamic_objects": 0,
        "saved_static_objects": 0,
        "saved_objects": 0,
        "filtered_outside_lidar_fov": 0,
        "filtered_outside_annotation_distance": 0,
        "filtered_insufficient_lidar_points": 0,
        "filtered_insufficient_rgb_visible_pixels": 0,
        "filtered_rgb_visibility_unavailable": 0,
    }

    ########################## 动态对象：处理会移动的交通对象 ################################

    for actor in actors:
        if (
            actor is None
            or not actor.is_alive
        ):
            continue

        actor_class = classify_actor(
            actor
        )

        if (
            actor_class
            not in allowed_classes
        ):
            continue

        stats[
            "candidate_actors"
        ] += 1

        annotation, reject_reason = (
            build_object_annotation(
                actor=actor,
                actor_class=actor_class,
                camera_transform=(
                    camera_transform
                ),
                lidar_transform=(
                    lidar_transform
                ),
                lidar_points_xyz=(
                    lidar_points_xyz
                ),
                semantic_tags=(
                    semantic_tags
                ),
                instance_ids=(
                    instance_ids
                ),
                K=K,
                image_width=(
                    image_width
                ),
                image_height=(
                    image_height
                ),
                ann_cfg=ann_cfg,
                lidar_cfg=lidar_cfg,
                collision_keys=(
                    collision_keys
                ),
            )
        )

        if annotation is not None:
            annotation[
                "source"
            ] = "dynamic_actor"

            objects.append(
                annotation
            )

            stats[
                "saved_dynamic_objects"
            ] += 1

            stats[
                "saved_objects"
            ] += 1

        elif (
            reject_reason
            == "outside_lidar_fov"
        ):
            stats[
                "filtered_outside_lidar_fov"
            ] += 1

        elif (
            reject_reason
            == "outside_annotation_distance"
        ):
            stats[
                "filtered_outside_annotation_distance"
            ] += 1

        elif (
            reject_reason
            == "insufficient_lidar_points"
        ):
            stats[
                "filtered_insufficient_lidar_points"
            ] += 1

        elif (
            reject_reason
            == "insufficient_rgb_visible_pixels"
        ):
            stats[
                "filtered_insufficient_rgb_visible_pixels"
            ] += 1

        elif (
            reject_reason
            == "rgb_visibility_unavailable"
        ):
            stats[
                "filtered_rgb_visibility_unavailable"
            ] += 1

    ########################## 静态地图对象：处理地图中的固定对象 ################################

    for item in environment_objects:
        environment_object = item[
            "object"
        ]

        object_class = item[
            "class"
        ]

        if (
            object_class
            not in allowed_classes
        ):
            continue

        stats[
            "candidate_environment_objects"
        ] += 1

        annotation, reject_reason = (
            build_environment_object_annotation(
                environment_object=(
                    environment_object
                ),
                object_class=(
                    object_class
                ),
                camera_transform=(
                    camera_transform
                ),
                lidar_transform=(
                    lidar_transform
                ),
                lidar_points_xyz=(
                    lidar_points_xyz
                ),
                K=K,
                image_width=(
                    image_width
                ),
                image_height=(
                    image_height
                ),
                ann_cfg=ann_cfg,
                lidar_cfg=lidar_cfg,
            )
        )

        if annotation is not None:
            objects.append(
                annotation
            )

            stats[
                "saved_static_objects"
            ] += 1

            stats[
                "saved_objects"
            ] += 1

        elif (
            reject_reason
            == "outside_lidar_fov"
        ):
            stats[
                "filtered_outside_lidar_fov"
            ] += 1

        elif (
            reject_reason
            == "outside_annotation_distance"
        ):
            stats[
                "filtered_outside_annotation_distance"
            ] += 1

        elif (
            reject_reason
            == "insufficient_lidar_points"
        ):
            stats[
                "filtered_insufficient_lidar_points"
            ] += 1

    return objects, stats


########################## 调试可视化：生成检查投影结果的辅助图像 ################################

def image_to_bgr(image):
    raw = np.frombuffer(
        image.raw_data,
        dtype=np.uint8,
    ).reshape(
        image.height,
        image.width,
        4,
    )

    return raw[:, :, :3].copy()


def draw_debug_bboxes(
    image,
    objects,
    output_path,
):
    if cv2 is None:
        return

    canvas = image_to_bgr(image)

    for obj in objects:
        bbox2d = obj.get("bbox2d")

        if not bbox2d:
            continue

        projected = bbox2d.get("projected")
        visible = bbox2d.get("visible")
        rgb_visibility = obj.get(
            "visibility",
            {},
        ).get(
            "rgb",
            None,
        )

        if (
            projected is not None
            and rgb_visibility is not False
        ):
            x1, y1, x2, y2 = projected["xyxy"]

            cv2.rectangle(
                canvas,
                (int(round(x1)), int(round(y1))),
                (int(round(x2)), int(round(y2))),
                (255, 0, 0),
                1,
            )

        if visible is not None:
            x1, y1, x2, y2 = visible["xyxy"]

            cv2.rectangle(
                canvas,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (0, 255, 0),
                2,
            )

            display_id = obj.get(
                "actor_id",
                obj.get(
                    "environment_object_id",
                    "?",
                ),
            )

            text = (
                f"{obj['class']} "
                f"id={display_id} "
                f"pts={obj.get('num_lidar_points', '-')}"
            )

            text_y = max(int(y1) - 5, 12)

            cv2.putText(
                canvas,
                text,
                (int(x1), text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

    cv2.imwrite(
        str(output_path),
        canvas,
    )


########################## 传感器蓝图：创建 RGB、分割相机和 LiDAR 参数 ################################

def configure_camera_blueprint(
    bp,
    camera_cfg,
):
    bp.set_attribute(
        "image_size_x",
        str(camera_cfg["width"]),
    )
    bp.set_attribute(
        "image_size_y",
        str(camera_cfg["height"]),
    )
    bp.set_attribute(
        "fov",
        str(camera_cfg["fov"]),
    )
    bp.set_attribute(
        "sensor_tick",
        "0.0",
    )

    if bp.has_attribute("lens_k"):
        bp.set_attribute("lens_k", "0.0")

    if bp.has_attribute("lens_kcube"):
        bp.set_attribute("lens_kcube", "0.0")


########################## 程序入口：读取配置并启动工具 ################################

def main():
    config = load_config()
    validate_config(config)

    sim_cfg = config["simulation"]
    recording_cfg = config["recording"]
    traffic_cfg = config["traffic"]
    uav_cfg = config["uav"]
    camera_cfg = config["sensors"]["camera"]
    lidar_cfg = config["sensors"]["lidar"]
    output_cfg = config["output"]
    ann_cfg = config.get("annotations", {})

    FPS = float(recording_cfg["fps"])
    CONFIGURED_NUM_FRAMES = int(
        recording_cfg["num_frames"]
    )
    RANDOM_SEED = int(
        sim_cfg["random_seed"]
    )

    route = UAVRoute(
        uav_cfg,
        config["route_planner"],
        FPS,
    )

    NUM_FRAMES = determine_recording_frames(
        route,
        CONFIGURED_NUM_FRAMES,
    )

    print_recording_plan(
        route,
        FPS,
        CONFIGURED_NUM_FRAMES,
        NUM_FRAMES,
        traffic_cfg,
        ann_cfg,
        lidar_cfg,
    )

    client = carla.Client(
        "localhost",
        2000,
    )
    client.set_timeout(20.0)

    world = client.get_world()
    route.validate_world_map(world.get_map())
    original_settings = world.get_settings()

    server_version = client.get_server_version()
    client_version = client.get_client_version()

    print("CARLA client version :", client_version)
    print("CARLA server version :", server_version)

    if not str(server_version).startswith("0.9.16"):
        print(
            "WARNING: this v3.3 collector is designed for CARLA 0.9.16. "
            "Visible 2D instance-to-ActorID matching may not be reliable "
            "on older releases."
        )

    tm_port = int(
        traffic_cfg["tm_port"]
    )

    traffic_manager = (
        client.get_trafficmanager(
            tm_port
        )
    )

    camera = None
    lidar = None
    instance_camera = None

    vehicles = []
    walkers = []
    walker_controllers = []

    tm_sync_enabled = False

    ########################## 输出：准备数据集目录和文件 ################################

    output_root = (
        PROJECT_ROOT
        / str(output_cfg["root"])
    )

    scene_name = datetime.now().strftime(
        "scene_%Y%m%d_%H%M%S"
    )

    scene_dir = output_root / scene_name
    rgb_dir = scene_dir / "rgb"
    lidar_dir = scene_dir / "lidar"
    pose_dir = scene_dir / "pose"
    labels_dir = scene_dir / "labels"
    debug_dir = scene_dir / "debug_bbox"

    scene_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    SAVE_RGB = bool(
        output_cfg.get("save_rgb", True)
    )
    SAVE_LIDAR = bool(
        output_cfg.get("save_lidar", True)
    )
    SAVE_POSE = bool(
        output_cfg.get("save_pose", True)
    )
    SAVE_LABELS = bool(
        output_cfg.get("save_labels", True)
    )

    ANNOTATIONS_ENABLED = bool(
        ann_cfg.get("enabled", False)
    )

    vis_cfg = ann_cfg.get(
        "visualization",
        {},
    )

    SAVE_DEBUG = bool(
        ANNOTATIONS_ENABLED
        and vis_cfg.get(
            "save_debug_images",
            True,
        )
    )

    if SAVE_DEBUG and cv2 is None:
        print(
            "WARNING: OpenCV (cv2) is not installed. "
            "debug_bbox images are disabled. "
            "Annotation JSON generation is unaffected."
        )
        SAVE_DEBUG = False

    if SAVE_RGB:
        rgb_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    if SAVE_LIDAR:
        lidar_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    if SAVE_POSE:
        pose_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    if ANNOTATIONS_ENABLED and SAVE_LABELS:
        labels_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    if SAVE_DEBUG:
        debug_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    try:
        ########################## 同步模式：让模拟器和传感器按同一帧运行 ################################

        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / FPS

        world.apply_settings(settings)

        traffic_manager.set_synchronous_mode(
            True
        )
        tm_sync_enabled = True

        traffic_manager.set_random_device_seed(
            RANDOM_SEED
        )

        print(
            f"\nCARLA synchronous mode: "
            f"{FPS:.1f} Hz"
        )

        ########################## 交通：生成和管理车辆、行人及其控制器 ################################

        allowed_classes = set(
            ann_cfg.get(
                "classes",
                SUPPORTED_CLASSES,
            )
        )

        vehicles = spawn_traffic(
            client=client,
            world=world,
            config=traffic_cfg,
            random_seed=RANDOM_SEED,
            allowed_classes=allowed_classes,
        )

        if (
            traffic_cfg.get(
                "num_pedestrians",
                0,
            ) > 0
        ):
            walkers, walker_controllers = (
                spawn_pedestrians(
                    world=world,
                    config=traffic_cfg,
                    random_seed=RANDOM_SEED,
                )
            )

        print(
            f"\nTraffic ready: "
            f"{len(vehicles)} vehicles, "
            f"{len(walkers)} pedestrians"
        )

        warmup_traffic(
            world,
            FPS,
            traffic_cfg.get(
                "warmup_seconds",
                0.0,
            ),
        )

        tracked_actors = (
            list(vehicles)
            + list(walkers)
        )

        static_environment_objects = []

        if ANNOTATIONS_ENABLED:
            static_environment_objects = (
                collect_static_environment_objects(
                    world=world,
                    ann_cfg=ann_cfg,
                )
            )

        collision_keys = set()

        if ANNOTATIONS_ENABLED:
            print_dynamic_semantic_tag_audit(
                tracked_actors,
                allowed_classes,
            )

            collision_keys = (
                find_instance_key_collisions(
                    tracked_actors,
                    allowed_classes,
                )
            )

            print(
                "\nAnnotation sources ready: "
                f"{len(tracked_actors)} dynamic actors, "
                f"{len(static_environment_objects)} "
                "static environment objects"
            )

        ########################## 初始 UAV：放置无人机并设置姿态 ################################

        initial_uav = route.pose_at_frame(0)

        camera_tf = sensor_world_transform(
            initial_uav,
            camera_cfg["position"],
            camera_cfg["rotation"],
        )

        lidar_tf = sensor_world_transform(
            initial_uav,
            lidar_cfg["position"],
            lidar_cfg["rotation"],
        )

        bp_lib = world.get_blueprint_library()

        ########################## RGB 相机：创建记录彩色图像的相机 ################################

        camera_bp = bp_lib.find(
            "sensor.camera.rgb"
        )

        configure_camera_blueprint(
            camera_bp,
            camera_cfg,
        )

        camera = world.spawn_actor(
            camera_bp,
            camera_tf,
        )


        bbox2d_cfg = ann_cfg.get(
            "bbox2d",
            {},
        )

        rgb_visibility_cfg = ann_cfg.get(
            "rgb_visibility_filter",
            {},
        )

        NEED_INSTANCE_CAMERA = bool(
            ANNOTATIONS_ENABLED
            and (
                rgb_visibility_cfg.get(
                    "enabled",
                    True,
                )
                or (
                    bbox2d_cfg.get(
                        "enabled",
                        True,
                    )
                    and bbox2d_cfg.get(
                        "visible",
                        True,
                    )
                )
            )
        )

        if NEED_INSTANCE_CAMERA:
            instance_bp = bp_lib.find(
                "sensor.camera.instance_segmentation"
            )

            configure_camera_blueprint(
                instance_bp,
                camera_cfg,
            )

            instance_camera = (
                world.spawn_actor(
                    instance_bp,
                    camera_tf,
                )
            )

        ########################## LiDAR：保存三维点云 ################################

        lidar_bp = bp_lib.find(
            "sensor.lidar.ray_cast"
        )

        lidar_attributes = {
            "channels": lidar_cfg["channels"],
            "range": lidar_cfg["range"],
            "points_per_second": (
                lidar_cfg[
                    "points_per_second"
                ]
            ),
            "rotation_frequency": FPS,
            "horizontal_fov": (
                lidar_cfg[
                    "horizontal_fov"
                ]
            ),
            "upper_fov": lidar_cfg["upper_fov"],
            "lower_fov": lidar_cfg["lower_fov"],
            "sensor_tick": 0.0,
        }

        for key, value in lidar_attributes.items():
            lidar_bp.set_attribute(
                key,
                str(value),
            )

        if lidar_bp.has_attribute(
            "noise_stddev"
        ):
            lidar_bp.set_attribute(
                "noise_stddev",
                "0.0",
            )

        lidar = world.spawn_actor(
            lidar_bp,
            lidar_tf,
        )

        ########################## 队列：接收并暂存传感器数据 ################################

        camera_queue = queue.Queue()
        lidar_queue = queue.Queue()
        instance_queue = queue.Queue()

        camera.listen(
            camera_queue.put
        )
        lidar.listen(
            lidar_queue.put
        )

        if instance_camera is not None:
            instance_camera.listen(
                instance_queue.put
            )

        ########################## 标定：保存传感器之间的内外参 ################################

        image_width = int(
            camera_cfg["width"]
        )
        image_height = int(
            camera_cfg["height"]
        )

        K = camera_intrinsic(
            image_width,
            image_height,
            float(camera_cfg["fov"]),
        )

        T_world_camera = matrix(camera_tf)
        T_world_lidar = matrix(lidar_tf)

        T_camera_ue_lidar = (
            np.linalg.inv(
                T_world_camera
            )
            @ T_world_lidar
        )

        T_camera_cv_lidar = (
            T_CV_UE
            @ T_camera_ue_lidar
        )

        calibration = {
            "carla_client_version": client_version,
            "carla_server_version": server_version,
            "dataset_fps_hz": float(FPS),
            "uav_speed_mps": float(route.speed),
            "route_name": route.name,
            "route_map": route.map_name,
            "route_length_m": float(
                route.total_length
            ),
            "planned_path_points": int(
                len(route.points)
            ),
            "planner_sampling_resolution_m": float(
                route.planner_resolution_m
            ),
            "max_planned_path_segment_m": float(
                route.max_segment_length_m
            ),
            "uav_altitude_above_road_m": float(
                route.altitude_m
            ),
            "heading_lookahead_m": float(
                route.heading_lookahead_m
            ),
            "K": K.tolist(),
            "camera_resolution": [
                image_width,
                image_height,
            ],
            "camera_fov_deg": float(
                camera_cfg["fov"]
            ),
            "lidar_rotation_frequency_hz": float(
                FPS
            ),
            "T_camera_ue_from_lidar": (
                T_camera_ue_lidar.tolist()
            ),
            "T_camera_cv_from_lidar": (
                T_camera_cv_lidar.tolist()
            ),
            "T_camera_cv_from_camera_ue": (
                T_CV_UE.tolist()
            ),
            "camera_lidar_distance_m": float(
                np.linalg.norm(
                    T_world_camera[:3, 3]
                    - T_world_lidar[:3, 3]
                )
            ),
            "coordinate_systems": {
                "world": (
                    "CARLA/UE: x forward/reference, "
                    "y right, z up"
                ),
                "lidar": (
                    "LiDAR local CARLA/UE: "
                    "x forward, y right, z up"
                ),
                "camera_ue": (
                    "Camera local CARLA/UE: "
                    "x forward, y right, z up"
                ),
                "camera_cv": (
                    "OpenCV camera: "
                    "x right, y down, z forward"
                ),
            },
        }

        with open(
            scene_dir / "calibration.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                calibration,
                f,
                indent=4,
            )

        print(
            "\nCamera-LiDAR distance:",
            calibration[
                "camera_lidar_distance_m"
            ],
            "m",
        )

        ########################## 元数据：记录场景、传感器和录制设置 ################################

        metadata = {
            "scene_name": scene_name,
            "collector_version": "uav_v3.3",
            "carla_client_version": client_version,
            "carla_server_version": server_version,
            "dataset_fps_hz": float(FPS),
            "configured_num_frames": int(
                CONFIGURED_NUM_FRAMES
            ),
            "actual_num_frames": int(
                NUM_FRAMES
            ),
            "recording_mode": (
                "until_route_end"
                if CONFIGURED_NUM_FRAMES == -1
                else "fixed_num_frames"
            ),
            "uav_speed_mps": float(
                route.speed
            ),
            "distance_per_frame_m": float(
                route.distance_per_frame
            ),
            "route_name": route.name,
            "route_source": (
                uav_cfg["route_source"]
            ),
            "route_length_m": float(
                route.total_length
            ),
            "planned_path_points": int(
                len(route.points)
            ),
            "route_map": route.map_name,
            "planner_sampling_resolution_m": float(
                route.planner_resolution_m
            ),
            "max_planned_path_segment_m": float(
                route.max_segment_length_m
            ),
            "uav_altitude_above_road_m": float(
                route.altitude_m
            ),
            "heading_lookahead_m": float(
                route.heading_lookahead_m
            ),
            "traffic_requested": {
                "vehicles": int(
                    traffic_cfg[
                        "num_vehicles"
                    ]
                ),
                "pedestrians": int(
                    traffic_cfg.get(
                        "num_pedestrians",
                        0,
                    )
                ),
            },
            "traffic_spawned": {
                "vehicles": int(
                    len(vehicles)
                ),
                "pedestrians": int(
                    len(walkers)
                ),
            },
            "traffic_warmup_seconds": float(
                traffic_cfg.get(
                    "warmup_seconds",
                    0.0,
                )
            ),
            "static_environment": {
                "enabled": bool(
                    ann_cfg.get(
                        "static_environment",
                        {},
                    ).get(
                        "enabled",
                        True,
                    )
                ),
                "configured_classes": list(
                    ann_cfg.get(
                        "static_environment",
                        {},
                    ).get(
                        "classes",
                        [
                            "car",
                            "truck",
                            "bus",
                            "motorcycle",
                            "bicycle",
                        ],
                    )
                ),
                "objects_loaded": int(
                    len(
                        static_environment_objects
                    )
                ),
                "source": (
                    "World.get_environment_objects"
                ),
                "bbox_space": "world",
                "visible_2d_note": (
                    "Static EnvironmentObjects do not use the "
                    "dynamic ActorID instance-segmentation mapping; "
                    "bbox2d.visible is unavailable and bbox2d.projected "
                    "is retained."
                ),
                "static_van_note": (
                    "CARLA CityObjectLabel has no separate Van label; "
                    "static map vans tagged as Car are stored as class car."
                ),
            },
            "annotations": {
                "enabled": ANNOTATIONS_ENABLED,
                "classes": list(
                    ann_cfg.get(
                        "classes",
                        [],
                    )
                ),
                "class_to_semantic_tag": (
                    CLASS_TO_SEMANTIC_TAG
                ),
                "max_distance_m": (
                    float(
                        ann_cfg.get(
                            "max_distance_m",
                            120.0,
                        )
                    )
                    if ANNOTATIONS_ENABLED
                    else None
                ),
                "bbox2d_format": "xyxy",
                "visible_bbox_source": (
                    "sensor.camera.instance_segmentation"
                ),
                "instance_id_decoding": (
                    "(B << 8) | G from raw BGRA; "
                    "CARLA 0.9.16 ActorID when available; semantic tag resolved from actor.semantic_tags"
                ),
                "van_note": (
                    "Dataset class 'van' comes from vehicle "
                    "base_type; dynamic CARLA semantic tag is resolved from actor.semantic_tags."
                ),
                "lidar_fov_filter": {
                    "enabled": bool(
                        ann_cfg.get(
                            "lidar_fov_filter",
                            {},
                        ).get(
                            "enabled",
                            True,
                        )
                    ),
                    "method": "bbox_volume_grid_sampling",
                    "bbox_samples_per_axis": int(
                        ann_cfg.get(
                            "lidar_fov_filter",
                            {},
                        ).get(
                            "bbox_samples_per_axis",
                            5,
                        )
                    ),
                    "sensor_range_m": float(
                        lidar_cfg["range"]
                    ),
                    "horizontal_fov_deg": float(
                        lidar_cfg["horizontal_fov"]
                    ),
                    "lower_fov_deg": float(
                        lidar_cfg["lower_fov"]
                    ),
                    "upper_fov_deg": float(
                        lidar_cfg["upper_fov"]
                    ),
                    "uses_occlusion": False,
                    "role": "geometry_prefilter_only",
                },
                "lidar_visibility": {
                    "enabled": bool(
                        ann_cfg.get(
                            "lidar_visibility",
                            {},
                        ).get(
                            "enabled",
                            True,
                        )
                    ),
                    "method": (
                        "actual_lidar_returns_inside_oriented_bbox"
                    ),
                    "min_lidar_points": int(
                        ann_cfg.get(
                            "lidar_visibility",
                            {},
                        ).get(
                            "min_lidar_points",
                            3,
                        )
                    ),
                    "uses_actual_lidar_returns": True,
                    "acts_as_hard_retention_filter": True,
                },
                "rgb_visibility_filter": {
                    "enabled": bool(
                        ann_cfg.get(
                            "rgb_visibility_filter",
                            {},
                        ).get(
                            "enabled",
                            True,
                        )
                    ),
                    "source": (
                        "sensor.camera.instance_segmentation"
                    ),
                    "min_visible_pixels": int(
                        ann_cfg.get(
                            "rgb_visibility_filter",
                            {},
                        ).get(
                            "min_visible_pixels",
                            5,
                        )
                    ),
                    "dynamic_actor_policy": (
                        "Dynamic actors must pass both LiDAR and RGB "
                        "visibility thresholds to be written to labels."
                    ),
                    "static_environment_policy": (
                        "Static EnvironmentObjects use the LiDAR hard "
                        "threshold only because reliable ActorID RGB pixel "
                        "ownership is unavailable."
                    ),
                },
            },
        }

        with open(
            scene_dir / "metadata.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                metadata,
                f,
                indent=4,
            )

        spectator = world.get_spectator()

        ########################## 录制：逐帧推进模拟器并写入数据集 ################################

        print()
        print("=" * 64)
        print("RECORDING START")
        print("=" * 64)
        print(f"Frames to record: {NUM_FRAMES}")
        print()

        for i in range(NUM_FRAMES):
            ########################## UAV 和传感器姿态：记录每帧位姿 ################################

            uav_tf = route.pose_at_frame(i)

            route_distance = (
                route.distance_at_frame(i)
            )

            route_progress = (
                route_distance / route.total_length
            )

            camera_tf = sensor_world_transform(
                uav_tf,
                camera_cfg["position"],
                camera_cfg["rotation"],
            )

            lidar_tf = sensor_world_transform(
                uav_tf,
                lidar_cfg["position"],
                lidar_cfg["rotation"],
            )

            camera.set_transform(camera_tf)
            lidar.set_transform(lidar_tf)

            if instance_camera is not None:
                instance_camera.set_transform(
                    camera_tf
                )

            spectator.set_transform(
                carla.Transform(
                    carla.Location(
                        x=uav_tf.location.x,
                        y=uav_tf.location.y,
                        z=uav_tf.location.z + 5.0,
                    ),
                    carla.Rotation(
                        pitch=-60.0,
                        yaw=uav_tf.rotation.yaw,
                        roll=0.0,
                    ),
                )
            )

            ########################## 模拟器步进：推进一帧并等待传感器 ################################

            carla_frame = world.tick()

            image = get_frame(
                camera_queue,
                carla_frame,
                "Camera",
            )

            cloud = get_frame(
                lidar_queue,
                carla_frame,
                "LiDAR",
            )

            instance_image = None

            if instance_camera is not None:
                instance_image = get_frame(
                    instance_queue,
                    carla_frame,
                    "InstanceCamera",
                )

            if image.frame != cloud.frame:
                raise RuntimeError(
                    "Camera-LiDAR frame mismatch: "
                    f"camera={image.frame}, "
                    f"lidar={cloud.frame}"
                )

            if (
                instance_image is not None
                and instance_image.frame
                != image.frame
            ):
                raise RuntimeError(
                    "RGB-instance frame mismatch: "
                    f"rgb={image.frame}, "
                    f"instance={instance_image.frame}"
                )

            ########################## RGB：保存彩色图像 ################################

            if SAVE_RGB:
                image.save_to_disk(
                    str(
                        rgb_dir
                        / f"{i:06d}.png"
                    )
                )

            ########################## LiDAR：保存点云 ################################

            points = np.frombuffer(
                cloud.raw_data,
                dtype=np.float32,
            ).reshape(-1, 4)

            if SAVE_LIDAR:
                points.tofile(
                    str(
                        lidar_dir
                        / f"{i:06d}.bin"
                    )
                )

            ########################## 三维和二维标注：保存目标框 ################################

            objects = []
            annotation_stats = {
                "candidate_actors": 0,
                "candidate_environment_objects": 0,
                "saved_dynamic_objects": 0,
                "saved_static_objects": 0,
                "saved_objects": 0,
                "filtered_outside_lidar_fov": 0,
                "filtered_outside_annotation_distance": 0,
                "filtered_insufficient_lidar_points": 0,
                "filtered_insufficient_rgb_visible_pixels": 0,
                "filtered_rgb_visibility_unavailable": 0,
            }

            if ANNOTATIONS_ENABLED:
                objects, annotation_stats = build_frame_annotations(
                    actors=tracked_actors,
                    environment_objects=static_environment_objects,
                    camera_transform=image.transform,
                    lidar_transform=cloud.transform,
                    lidar_points_xyz=points[:, :3],
                    instance_image=instance_image,
                    K=K,
                    image_width=image_width,
                    image_height=image_height,
                    ann_cfg=ann_cfg,
                    lidar_cfg=lidar_cfg,
                    collision_keys=collision_keys,
                )

                if SAVE_LABELS:
                    label_info = {
                        "sample_index": int(i),
                        "carla_frame": int(
                            carla_frame
                        ),
                        "timestamp": float(
                            image.timestamp
                        ),
                        "image_size": [
                            image_width,
                            image_height,
                        ],
                        "bbox2d_format": "xyxy",
                        "num_objects": int(
                            len(objects)
                        ),
                        "annotation_stats": annotation_stats,
                        "objects": objects,
                    }

                    with open(
                        labels_dir
                        / f"{i:06d}.json",
                        "w",
                        encoding="utf-8",
                    ) as f:
                        json.dump(
                            label_info,
                            f,
                            indent=4,
                        )

                if SAVE_DEBUG:
                    draw_debug_bboxes(
                        image=image,
                        objects=objects,
                        output_path=(
                            debug_dir
                            / f"{i:06d}.png"
                        ),
                    )

            ########################## 姿态：保存 UAV 位姿 ################################

            dataset_time = float(i) / FPS

            frame_info = {
                "sample_index": int(i),
                "dataset_time_s": float(
                    dataset_time
                ),
                "carla_frame": int(
                    carla_frame
                ),
                "carla_timestamp": float(
                    image.timestamp
                ),
                "dataset_fps_hz": float(FPS),
                "uav_speed_mps": float(
                    route.speed
                ),
                "route_distance_m": float(
                    route_distance
                ),
                "route_progress": float(
                    route_progress
                ),
                "route_finished": bool(
                    route.is_finished(i)
                ),
                "uav": pose_dict(uav_tf),
                "camera": pose_dict(
                    image.transform
                ),
                "lidar": pose_dict(
                    cloud.transform
                ),
                "T_world_camera": matrix(
                    image.transform
                ).tolist(),
                "T_world_lidar": matrix(
                    cloud.transform
                ).tolist(),
                "lidar_points": int(
                    points.shape[0]
                ),
                "annotation_objects": int(
                    len(objects)
                ),
                "annotation_stats": annotation_stats,
            }

            if SAVE_POSE:
                with open(
                    pose_dir
                    / f"{i:06d}.json",
                    "w",
                    encoding="utf-8",
                ) as f:
                    json.dump(
                        frame_info,
                        f,
                        indent=4,
                    )

            visible_2d_count = sum(
                1
                for obj in objects
                if (
                    obj.get("bbox2d")
                    and obj["bbox2d"].get(
                        "visible"
                    )
                    is not None
                )
            )

            print(
                f"[{i + 1:04d}/{NUM_FRAMES:04d}] "
                f"frame={carla_frame} "
                f"lidar={points.shape[0]} "
                f"objects={len(objects)} "
                f"dyn={annotation_stats['saved_dynamic_objects']} "
                f"static={annotation_stats['saved_static_objects']} "
                f"fov_drop="
                f"{annotation_stats['filtered_outside_lidar_fov']} "
                f"lidar_drop="
                f"{annotation_stats['filtered_insufficient_lidar_points']} "
                f"rgb_drop="
                f"{annotation_stats['filtered_insufficient_rgb_visible_pixels']} "
                f"rgb_na="
                f"{annotation_stats['filtered_rgb_visibility_unavailable']} "
                f"visible2d={visible_2d_count} "
                f"distance="
                f"{route_distance:.1f}/"
                f"{route.total_length:.1f}m "
                f"progress="
                f"{route_progress * 100.0:.1f}% "
                f"UAV=("
                f"{uav_tf.location.x:.1f}, "
                f"{uav_tf.location.y:.1f}, "
                f"{uav_tf.location.z:.1f})"
            )

        ########################## 完成：输出本次录制的结果摘要 ################################

        print()
        print("=" * 64)
        print("RECORDING DONE")
        print("=" * 64)
        print("Scene:")
        print(scene_dir)

        if CONFIGURED_NUM_FRAMES == -1:
            final_distance = (
                route.distance_at_frame(
                    NUM_FRAMES - 1
                )
            )

            print()
            print(
                "Final route distance:"
            )
            print(
                f"{final_distance:.3f} / "
                f"{route.total_length:.3f} m"
            )
            print(
                "Final route endpoint reached:",
                route.is_finished(
                    NUM_FRAMES - 1
                ),
            )

    finally:
        ########################## 清理：停止控制器和传感器并恢复模拟器设置 ################################

        print("\nCleaning up...")

        for sensor in (
            camera,
            lidar,
            instance_camera,
        ):
            if sensor is not None:
                try:
                    sensor.stop()
                except RuntimeError:
                    pass

        for controller in walker_controllers:
            if controller is not None:
                try:
                    controller.stop()
                except RuntimeError:
                    pass

        destroy_ids = []

        for actor in (
            [camera, lidar, instance_camera]
            + walker_controllers
            + walkers
            + vehicles
        ):
            if actor is not None:
                try:
                    destroy_ids.append(
                        actor.id
                    )
                except RuntimeError:
                    pass

        if destroy_ids:
            try:
                client.apply_batch(
                    [
                        carla.command.DestroyActor(
                            actor_id
                        )
                        for actor_id
                        in destroy_ids
                    ]
                )
            except RuntimeError as exc:
                print(
                    "WARNING: actor cleanup failed:",
                    exc,
                )

        if tm_sync_enabled:
            try:
                traffic_manager.set_synchronous_mode(
                    False
                )
            except RuntimeError:
                pass

        try:
            world.apply_settings(
                original_settings
            )
        except RuntimeError:
            pass

        print("Cleanup complete.")


if __name__ == "__main__":
    main()
