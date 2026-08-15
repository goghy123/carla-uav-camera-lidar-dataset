import carla

import json
import math
import queue
import random
import time
from pathlib import Path
from datetime import datetime

import numpy as np
from ruamel.yaml import YAML


# ============================================================
# PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "UAVdataset.yaml"
)


# ============================================================
# YAML
# ============================================================

def load_yaml(path):

    yaml = YAML()

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        return yaml.load(f)


def load_config():

    # ========================================================
    # 主配置
    # ========================================================

    config = load_yaml(
        CONFIG_PATH
    )

    # ========================================================
    # Route 配置
    # ========================================================

    route_file = config["uav"]["route_file"]

    route_path = (
        CONFIG_PATH.parent
        / route_file
    ).resolve()

    if not route_path.exists():

        raise FileNotFoundError(
            f"\nRoute YAML does not exist:\n"
            f"{route_path}"
        )

    route_config = load_yaml(
        route_path
    )

    if "route" not in route_config:

        raise KeyError(
            f"\nMissing 'route:' in:\n"
            f"{route_path}"
        )

    # 注入 route
    config["uav"]["route"] = (
        route_config["route"]
    )

    config["uav"]["route_source"] = str(
        route_path
    )

    print("\nRoute configuration:")
    print("  file :", route_path)

    print(
        "  name :",
        route_config["route"].get(
            "name",
            "unnamed"
        )
    )

    print(
        "  mode :",
        route_config["route"].get(
            "mode",
            "unknown"
        )
    )

    return config


# ============================================================
# CONFIG VALIDATION
# ============================================================

def validate_config(config):

    # --------------------------------------------------------
    # Recording
    # --------------------------------------------------------

    if "recording" not in config:
        raise KeyError(
            "Missing 'recording:' section "
            "in UAVdataset.yaml"
        )

    fps = float(
        config["recording"]["fps"]
    )

    num_frames = int(
        config["recording"]["num_frames"]
    )

    if fps <= 0:

        raise ValueError(
            "recording.fps must be > 0"
        )

    if num_frames == 0 or num_frames < -1:

        raise ValueError(
            "recording.num_frames must be "
            "-1 or a positive integer"
        )

    # --------------------------------------------------------
    # UAV
    # --------------------------------------------------------

    speed = float(
        config["uav"]["speed_mps"]
    )

    if speed <= 0:

        raise ValueError(
            "uav.speed_mps must be > 0"
        )

    # --------------------------------------------------------
    # LiDAR
    # --------------------------------------------------------

    lidar_cfg = (
        config["sensors"]["lidar"]
    )

    if float(
        lidar_cfg["points_per_second"]
    ) <= 0:

        raise ValueError(
            "lidar.points_per_second "
            "must be > 0"
        )

    # --------------------------------------------------------
    # Traffic
    # --------------------------------------------------------

    traffic_cfg = config["traffic"]

    if int(
        traffic_cfg["num_vehicles"]
    ) < 0:

        raise ValueError(
            "traffic.num_vehicles "
            "must be >= 0"
        )

    if float(
        traffic_cfg.get(
            "warmup_seconds",
            0.0
        )
    ) < 0:

        raise ValueError(
            "traffic.warmup_seconds "
            "must be >= 0"
        )


# ============================================================
# TRANSFORM
# ============================================================

def pose_dict(transform):

    return {

        "location": {
            "x": transform.location.x,
            "y": transform.location.y,
            "z": transform.location.z
        },

        "rotation": {
            "pitch": transform.rotation.pitch,
            "yaw": transform.rotation.yaw,
            "roll": transform.rotation.roll
        }
    }


def matrix(transform):

    return np.array(
        transform.get_matrix(),
        dtype=np.float64
    )


# ============================================================
# UAV -> SENSOR
#
# 当前假设：
#
# UAV:
#   pitch = 0
#   roll  = 0
#
# 飞行过程中主要变化：
#
#   x / y / z / yaw
#
# 因此：
#
# sensor world yaw
# =
# UAV yaw + sensor relative yaw
# ============================================================

def sensor_world_transform(
    uav_transform,
    sensor_position,
    sensor_rotation
):

    relative_location = carla.Location(

        x=float(sensor_position["x"]),
        y=float(sensor_position["y"]),
        z=float(sensor_position["z"])
    )

    # UAV local -> world
    world_location = (
        uav_transform.transform(
            relative_location
        )
    )

    world_rotation = carla.Rotation(

        pitch=(
            uav_transform.rotation.pitch
            +
            float(
                sensor_rotation["pitch"]
            )
        ),

        yaw=(
            uav_transform.rotation.yaw
            +
            float(
                sensor_rotation["yaw"]
            )
        ),

        roll=(
            uav_transform.rotation.roll
            +
            float(
                sensor_rotation["roll"]
            )
        )
    )

    return carla.Transform(
        world_location,
        world_rotation
    )


# ============================================================
# CAMERA INTRINSIC
# ============================================================

def camera_intrinsic(
    width,
    height,
    fov
):

    focal = width / (

        2.0
        *
        math.tan(
            math.radians(fov) / 2.0
        )
    )

    K = np.array([

        [
            focal,
            0.0,
            width / 2.0
        ],

        [
            0.0,
            focal,
            height / 2.0
        ],

        [
            0.0,
            0.0,
            1.0
        ]

    ], dtype=np.float64)

    return K


# ============================================================
# SENSOR QUEUE
# ============================================================

def get_frame(
    sensor_queue,
    target_frame,
    name,
    timeout=10.0
):

    deadline = (
        time.monotonic()
        + timeout
    )

    while True:

        remaining_time = (
            deadline
            - time.monotonic()
        )

        if remaining_time <= 0:

            raise TimeoutError(
                f"{name}: timeout waiting "
                f"for CARLA frame "
                f"{target_frame}"
            )

        try:

            data = sensor_queue.get(
                timeout=remaining_time
            )

        except queue.Empty:

            raise TimeoutError(
                f"{name}: timeout waiting "
                f"for CARLA frame "
                f"{target_frame}"
            )

        # 老数据直接丢弃
        if data.frame < target_frame:
            continue

        # 正确帧
        if data.frame == target_frame:
            return data

        # 已经跳到未来帧，说明同步出了问题
        raise RuntimeError(

            f"{name}: frame synchronization "
            f"failed. Expected "
            f"{target_frame}, "
            f"but received "
            f"{data.frame}."
        )


# ============================================================
# UAV ROUTE
# ============================================================

class UAVRoute:

    def __init__(
        self,
        uav_config,
        fps
    ):

        self.fps = float(fps)

        if self.fps <= 0:
            raise ValueError(
                "FPS must be > 0"
            )

        # ----------------------------------------------------
        # UAV speed
        # ----------------------------------------------------

        self.speed = float(
            uav_config[
                "speed_mps"
            ]
        )

        if self.speed <= 0:
            raise ValueError(
                "UAV speed must be > 0"
            )

        # 每个数据帧 UAV 移动距离
        self.distance_per_frame = (
            self.speed
            / self.fps
        )

        # ----------------------------------------------------
        # Initial pose
        #
        # static 模式或 waypoint 不合法时使用。
        #
        # waypoint 模式真正的起点为：
        #
        # route.waypoints[0]
        # ----------------------------------------------------

        initial = uav_config[
            "initial_pose"
        ]

        self.initial = carla.Transform(

            carla.Location(

                x=float(
                    initial["x"]
                ),

                y=float(
                    initial["y"]
                ),

                z=float(
                    initial["z"]
                )
            ),

            carla.Rotation(

                pitch=float(
                    initial["pitch"]
                ),

                yaw=float(
                    initial["yaw"]
                ),

                roll=float(
                    initial["roll"]
                )
            )
        )

        # ----------------------------------------------------
        # Route
        # ----------------------------------------------------

        route = uav_config["route"]

        self.name = str(
            route.get(
                "name",
                "unnamed"
            )
        )

        self.mode = str(
            route["mode"]
        ).lower()

        self.heading_mode = str(
            route.get(
                "heading_mode",
                "follow_route"
            )
        ).lower()

        if self.mode not in (
            "static",
            "waypoints"
        ):

            raise ValueError(
                f"Unsupported route mode: "
                f"{self.mode}"
            )

        if (
            self.mode == "waypoints"
            and self.heading_mode
            != "follow_route"
        ):

            raise ValueError(
                "Currently only "
                "'heading_mode: follow_route' "
                "is supported."
            )

        # ----------------------------------------------------
        # Waypoints
        #
        # 自动删除连续重复点
        # ----------------------------------------------------

        self.points = []

        duplicate_count = 0

        raw_waypoints = (
            route.get(
                "waypoints",
                []
            )
        )

        for index, p in enumerate(
            raw_waypoints
        ):

            point = np.array(
                [
                    float(p["x"]),
                    float(p["y"]),
                    float(p["z"])
                ],
                dtype=np.float64
            )

            if self.points:

                distance = np.linalg.norm(
                    point
                    - self.points[-1]
                )

                if distance <= 1e-6:

                    duplicate_count += 1

                    print(
                        "WARNING: removed "
                        "duplicated waypoint "
                        f"#{index}: "
                        f"({point[0]:.3f}, "
                        f"{point[1]:.3f}, "
                        f"{point[2]:.3f})"
                    )

                    continue

            self.points.append(
                point
            )

        if duplicate_count > 0:

            print(
                f"Removed "
                f"{duplicate_count} "
                f"duplicated waypoint(s)."
            )

        # ----------------------------------------------------
        # Segment length
        # ----------------------------------------------------

        self.segment_lengths = []

        self.total_length = 0.0

        if len(self.points) >= 2:

            for i in range(
                len(self.points) - 1
            ):

                length = float(

                    np.linalg.norm(
                        self.points[i + 1]
                        -
                        self.points[i]
                    )
                )

                self.segment_lengths.append(
                    length
                )

                self.total_length += (
                    length
                )

        # ----------------------------------------------------
        # Validate waypoint route
        # ----------------------------------------------------

        if self.mode == "waypoints":

            if len(self.points) < 2:

                raise ValueError(
                    "Waypoint route requires "
                    "at least 2 unique "
                    "waypoints."
                )

            if self.total_length <= 1e-6:

                raise ValueError(
                    "Route length is zero."
                )

    # ========================================================
    # Recording properties
    # ========================================================

    def required_frames(self):

        """
        完整覆盖路线需要的帧数。

        frame 0:
            distance = 0

        frame N:
            distance = N * speed / fps

        因此需要：

            ceil(total_length /
                 distance_per_frame)
            + 1

        +1 是为了包含起点 frame 0。
        """

        if self.mode != "waypoints":
            return None

        movement_intervals = math.ceil(

            self.total_length
            /
            self.distance_per_frame
        )

        return (
            movement_intervals
            + 1
        )

    def estimated_duration(self):

        if self.mode != "waypoints":
            return None

        return (
            self.total_length
            / self.speed
        )

    def distance_at_frame(
        self,
        frame_index
    ):

        if self.mode != "waypoints":
            return 0.0

        distance = (

            float(frame_index)
            *
            self.distance_per_frame
        )

        return min(
            distance,
            self.total_length
        )

    def is_finished(
        self,
        frame_index
    ):

        if self.mode != "waypoints":
            return False

        return (
            self.distance_at_frame(
                frame_index
            )
            >=
            self.total_length
            - 1e-6
        )

    # ========================================================
    # Pose
    # ========================================================

    def pose_at_frame(
        self,
        frame_index
    ):

        # ----------------------------------------------------
        # Static
        # ----------------------------------------------------

        if self.mode == "static":
            return self.initial

        distance = (
            self.distance_at_frame(
                frame_index
            )
        )

        return self.pose_at_distance(
            distance
        )

    def pose_at_distance(
        self,
        distance
    ):

        # ----------------------------------------------------
        # Static
        # ----------------------------------------------------

        if self.mode == "static":
            return self.initial

        distance = float(
            np.clip(
                distance,
                0.0,
                self.total_length
            )
        )

        remaining = distance

        selected_segment = (
            len(self.segment_lengths)
            - 1
        )

        segment_remaining = (
            self.segment_lengths[
                selected_segment
            ]
        )

        # ----------------------------------------------------
        # Find segment
        # ----------------------------------------------------

        for i, length in enumerate(
            self.segment_lengths
        ):

            if remaining <= length:

                selected_segment = i

                segment_remaining = (
                    remaining
                )

                break

            remaining -= length

        # ----------------------------------------------------
        # Interpolation
        # ----------------------------------------------------

        start = self.points[
            selected_segment
        ]

        end = self.points[
            selected_segment + 1
        ]

        length = self.segment_lengths[
            selected_segment
        ]

        if length <= 1e-9:
            ratio = 0.0
        else:
            ratio = (
                segment_remaining
                / length
            )

        ratio = float(
            np.clip(
                ratio,
                0.0,
                1.0
            )
        )

        position = (

            start
            +
            ratio
            *
            (end - start)
        )

        # ----------------------------------------------------
        # Heading
        # ----------------------------------------------------

        direction = (
            end
            - start
        )

        yaw = math.degrees(

            math.atan2(
                direction[1],
                direction[0]
            )
        )

        return carla.Transform(

            carla.Location(

                x=float(
                    position[0]
                ),

                y=float(
                    position[1]
                ),

                z=float(
                    position[2]
                )
            ),

            carla.Rotation(

                pitch=0.0,

                yaw=float(yaw),

                roll=0.0
            )
        )


# ============================================================
# RECORDING PLAN
# ============================================================

def determine_recording_frames(
    route,
    configured_num_frames
):

    configured_num_frames = int(
        configured_num_frames
    )

    # ========================================================
    # AUTO MODE
    #
    # -1:
    # 录制到最后 waypoint
    # ========================================================

    if configured_num_frames == -1:

        if route.mode != "waypoints":

            raise ValueError(
                "recording.num_frames = -1 "
                "requires "
                "'route.mode: waypoints'."
            )

        return route.required_frames()

    # ========================================================
    # FIXED MODE
    # ========================================================

    if configured_num_frames <= 0:

        raise ValueError(
            "recording.num_frames must be "
            "-1 or a positive integer."
        )

    return configured_num_frames


def print_recording_plan(
    route,
    fps,
    configured_num_frames,
    actual_num_frames,
    traffic_cfg
):

    print()
    print(
        "=" * 58
    )

    print(
        "UAV DATASET RECORDING PLAN"
    )

    print(
        "=" * 58
    )

    print(
        f"Route name           : "
        f"{route.name}"
    )

    print(
        f"Route mode           : "
        f"{route.mode}"
    )

    if route.mode == "waypoints":

        print(
            f"Unique waypoints     : "
            f"{len(route.points)}"
        )

        print(
            f"Route length         : "
            f"{route.total_length:.3f} m"
        )

        print(
            f"UAV speed            : "
            f"{route.speed:.3f} m/s"
        )

        print(
            f"Dataset FPS          : "
            f"{fps:.3f} Hz"
        )

        print(
            f"Distance / frame     : "
            f"{route.distance_per_frame:.3f} m"
        )

        print(
            f"Estimated flight time: "
            f"{route.estimated_duration():.3f} s"
        )

        print(
            f"Full-route frames    : "
            f"{route.required_frames()}"
        )

    else:

        print(
            f"Dataset FPS          : "
            f"{fps:.3f} Hz"
        )

    if configured_num_frames == -1:

        print(
            "Recording mode       : "
            "UNTIL_ROUTE_END"
        )

    else:

        print(
            "Recording mode       : "
            "FIXED_NUM_FRAMES"
        )

    print(
        f"Configured frames    : "
        f"{configured_num_frames}"
    )

    print(
        f"Actual frames        : "
        f"{actual_num_frames}"
    )

    # --------------------------------------------------------
    # Coverage warning
    # --------------------------------------------------------

    if (
        route.mode == "waypoints"
        and configured_num_frames > 0
    ):

        movement_distance = max(
            configured_num_frames - 1,
            0
        ) * route.distance_per_frame

        coverage_distance = min(
            movement_distance,
            route.total_length
        )

        coverage_percent = (

            coverage_distance
            /
            route.total_length
            *
            100.0
        )

        print(
            f"Route coverage       : "
            f"{coverage_distance:.3f} / "
            f"{route.total_length:.3f} m "
            f"({coverage_percent:.1f}%)"
        )

        if (
            configured_num_frames
            <
            route.required_frames()
        ):

            print()
            print(
                "WARNING:"
            )

            print(
                "Configured num_frames "
                "cannot cover the full route."
            )

            print(
                "Use num_frames: -1 "
                "to automatically record "
                "until the final waypoint."
            )

        elif (
            configured_num_frames
            >
            route.required_frames()
        ):

            stationary_frames = (

                configured_num_frames
                -
                route.required_frames()
            )

            print()
            print(
                "WARNING:"
            )

            print(
                "Configured num_frames is "
                "longer than the route."
            )

            print(
                f"Approximately "
                f"{stationary_frames} "
                f"extra frame(s) will be "
                f"recorded at the endpoint."
            )

    # --------------------------------------------------------
    # Traffic
    # --------------------------------------------------------

    print()

    print(
        f"Traffic enabled      : "
        f"{traffic_cfg['enabled']}"
    )

    print(
        f"Traffic requested    : "
        f"{traffic_cfg['num_vehicles']}"
    )

    print(
        f"Traffic warmup       : "
        f"{traffic_cfg.get('warmup_seconds', 0.0)} s"
    )

    print(
        "=" * 58
    )

    print()


# ============================================================
# TRAFFIC
# ============================================================

def spawn_traffic(
    client,
    world,
    config,
    random_seed
):

    if not config["enabled"]:

        print(
            "Traffic disabled."
        )

        return []

    random.seed(
        int(random_seed)
    )

    bp_lib = (
        world
        .get_blueprint_library()
    )

    vehicle_bps = list(
        bp_lib.filter(
            "vehicle.*"
        )
    )

    if len(vehicle_bps) == 0:

        raise RuntimeError(
            "No vehicle blueprints found."
        )

    spawn_points = (

        world
        .get_map()
        .get_spawn_points()
    )

    random.shuffle(
        spawn_points
    )

    requested_number = int(
        config["num_vehicles"]
    )

    available_number = len(
        spawn_points
    )

    number = min(
        requested_number,
        available_number
    )

    print(
        "\nSpawning traffic..."
    )

    print(
        "  requested vehicles :",
        requested_number
    )

    print(
        "  available spawns   :",
        available_number
    )

    print(
        "  spawn attempts     :",
        number
    )

    if requested_number > available_number:

        print(
            "WARNING: requested vehicle "
            "count exceeds available "
            "spawn points."
        )

    tm_port = int(
        config["tm_port"]
    )

    batch = []

    for spawn in spawn_points[
        :number
    ]:

        bp = random.choice(
            vehicle_bps
        )

        # ----------------------------------------------------
        # Role
        # ----------------------------------------------------

        if bp.has_attribute(
            "role_name"
        ):

            bp.set_attribute(
                "role_name",
                "autopilot"
            )

        # ----------------------------------------------------
        # Random vehicle color
        # ----------------------------------------------------

        if bp.has_attribute(
            "color"
        ):

            colors = (
                bp.get_attribute(
                    "color"
                )
                .recommended_values
            )

            if colors:

                bp.set_attribute(
                    "color",
                    random.choice(
                        colors
                    )
                )

        # ----------------------------------------------------
        # Random driver
        # ----------------------------------------------------

        if bp.has_attribute(
            "driver_id"
        ):

            drivers = (
                bp.get_attribute(
                    "driver_id"
                )
                .recommended_values
            )

            if drivers:

                bp.set_attribute(
                    "driver_id",
                    random.choice(
                        drivers
                    )
                )

        batch.append(

            carla.command.SpawnActor(
                bp,
                spawn
            ).then(

                carla.command.SetAutopilot(

                    carla.command.FutureActor,

                    True,

                    tm_port
                )
            )
        )

    responses = (
        client.apply_batch_sync(
            batch,
            True
        )
    )

    ids = []

    failed = 0

    for response in responses:

        if response.error:

            failed += 1

        else:

            ids.append(
                response.actor_id
            )

    vehicles = []

    for actor_id in ids:

        actor = world.get_actor(
            actor_id
        )

        if actor is not None:
            vehicles.append(
                actor
            )

    print(
        "  successfully spawned:",
        len(vehicles)
    )

    if failed > 0:

        print(
            "  failed              :",
            failed
        )

    return vehicles


def warmup_traffic(
    world,
    fps,
    warmup_seconds
):

    warmup_seconds = float(
        warmup_seconds
    )

    if warmup_seconds <= 0:
        return

    warmup_ticks = int(
        math.ceil(
            warmup_seconds
            * fps
        )
    )

    print()
    print(
        f"Traffic warmup: "
        f"{warmup_seconds:.1f} s "
        f"({warmup_ticks} ticks)"
    )

    for _ in range(
        warmup_ticks
    ):

        world.tick()

    print(
        "Traffic warmup complete."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    config = load_config()

    validate_config(
        config
    )

    # --------------------------------------------------------
    # Configuration
    # --------------------------------------------------------

    sim_cfg = config[
        "simulation"
    ]

    recording_cfg = config[
        "recording"
    ]

    traffic_cfg = config[
        "traffic"
    ]

    uav_cfg = config[
        "uav"
    ]

    camera_cfg = (
        config[
            "sensors"
        ][
            "camera"
        ]
    )

    lidar_cfg = (
        config[
            "sensors"
        ][
            "lidar"
        ]
    )

    output_cfg = config[
        "output"
    ]

    # --------------------------------------------------------
    # Single global FPS
    # --------------------------------------------------------

    FPS = float(
        recording_cfg["fps"]
    )

    CONFIGURED_NUM_FRAMES = int(
        recording_cfg[
            "num_frames"
        ]
    )

    RANDOM_SEED = int(
        sim_cfg[
            "random_seed"
        ]
    )

    # --------------------------------------------------------
    # Route can be analyzed before CARLA recording
    # --------------------------------------------------------

    route = UAVRoute(
        uav_cfg,
        FPS
    )

    NUM_FRAMES = (
        determine_recording_frames(
            route,
            CONFIGURED_NUM_FRAMES
        )
    )

    print_recording_plan(

        route,

        FPS,

        CONFIGURED_NUM_FRAMES,

        NUM_FRAMES,

        traffic_cfg
    )

    # --------------------------------------------------------
    # CARLA
    # --------------------------------------------------------

    client = carla.Client(
        "localhost",
        2000
    )

    client.set_timeout(
        20.0
    )

    world = (
        client.get_world()
    )

    original_settings = (
        world.get_settings()
    )

    tm_port = int(
        traffic_cfg[
            "tm_port"
        ]
    )

    traffic_manager = (
        client.get_trafficmanager(
            tm_port
        )
    )

    camera = None
    lidar = None

    vehicles = []

    tm_sync_enabled = False

    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------

    output_root = (

        PROJECT_ROOT
        /
        str(
            output_cfg[
                "root"
            ]
        )
    )

    scene_name = (

        datetime.now()
        .strftime(
            "scene_%Y%m%d_%H%M%S"
        )
    )

    scene_dir = (
        output_root
        / scene_name
    )

    rgb_dir = (
        scene_dir
        / "rgb"
    )

    lidar_dir = (
        scene_dir
        / "lidar"
    )

    pose_dir = (
        scene_dir
        / "pose"
    )

    scene_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    SAVE_RGB = bool(
        output_cfg.get(
            "save_rgb",
            True
        )
    )

    SAVE_LIDAR = bool(
        output_cfg.get(
            "save_lidar",
            True
        )
    )

    SAVE_POSE = bool(
        output_cfg.get(
            "save_pose",
            True
        )
    )

    if SAVE_RGB:

        rgb_dir.mkdir(
            parents=True,
            exist_ok=True
        )

    if SAVE_LIDAR:

        lidar_dir.mkdir(
            parents=True,
            exist_ok=True
        )

    if SAVE_POSE:

        pose_dir.mkdir(
            parents=True,
            exist_ok=True
        )

    try:

        # ====================================================
        # SYNCHRONOUS MODE
        # ====================================================

        settings = (
            world.get_settings()
        )

        settings.synchronous_mode = True

        settings.fixed_delta_seconds = (
            1.0 / FPS
        )

        world.apply_settings(
            settings
        )

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

        # ====================================================
        # TRAFFIC
        # ====================================================

        vehicles = spawn_traffic(

            client,

            world,

            traffic_cfg,

            RANDOM_SEED
        )

        print(
            "Vehicles in dataset:",
            len(vehicles)
        )

        # ----------------------------------------------------
        # Traffic warmup
        #
        # Sensors 尚未创建，因此 warmup 不会污染传感器队列。
        # ----------------------------------------------------

        warmup_traffic(

            world,

            FPS,

            traffic_cfg.get(
                "warmup_seconds",
                0.0
            )
        )

        # ====================================================
        # INITIAL UAV
        # ====================================================

        initial_uav = (
            route.pose_at_frame(
                0
            )
        )

        # ====================================================
        # SENSOR TRANSFORMS
        # ====================================================

        camera_tf = (
            sensor_world_transform(

                initial_uav,

                camera_cfg[
                    "position"
                ],

                camera_cfg[
                    "rotation"
                ]
            )
        )

        lidar_tf = (
            sensor_world_transform(

                initial_uav,

                lidar_cfg[
                    "position"
                ],

                lidar_cfg[
                    "rotation"
                ]
            )
        )

        # ====================================================
        # BLUEPRINT LIBRARY
        # ====================================================

        bp_lib = (
            world
            .get_blueprint_library()
        )

        # ====================================================
        # CAMERA
        #
        # sensor_tick = 0:
        # 每个 CARLA tick 产生一帧。
        #
        # CARLA 本身：
        # fixed_delta_seconds = 1 / FPS
        #
        # 所以 camera = global FPS。
        # ====================================================

        camera_bp = (
            bp_lib.find(
                "sensor.camera.rgb"
            )
        )

        camera_bp.set_attribute(
            "image_size_x",
            str(
                camera_cfg[
                    "width"
                ]
            )
        )

        camera_bp.set_attribute(
            "image_size_y",
            str(
                camera_cfg[
                    "height"
                ]
            )
        )

        camera_bp.set_attribute(
            "fov",
            str(
                camera_cfg[
                    "fov"
                ]
            )
        )

        camera_bp.set_attribute(
            "sensor_tick",
            "0.0"
        )

        if camera_bp.has_attribute(
            "lens_k"
        ):

            camera_bp.set_attribute(
                "lens_k",
                "0.0"
            )

        if camera_bp.has_attribute(
            "lens_kcube"
        ):

            camera_bp.set_attribute(
                "lens_kcube",
                "0.0"
            )

        camera = (
            world.spawn_actor(
                camera_bp,
                camera_tf
            )
        )

        # ====================================================
        # LiDAR
        #
        # rotation_frequency 直接绑定 global FPS。
        #
        # FPS = 10 Hz
        #
        # => LiDAR = 10 Hz
        #
        # 每一个 world tick 对应完整一圈 LiDAR。
        # ====================================================

        lidar_bp = (
            bp_lib.find(
                "sensor.lidar.ray_cast"
            )
        )

        lidar_attributes = {

            "channels":
                lidar_cfg[
                    "channels"
                ],

            "range":
                lidar_cfg[
                    "range"
                ],

            "points_per_second":
                lidar_cfg[
                    "points_per_second"
                ],

            # 全局 FPS
            "rotation_frequency":
                FPS,

            "horizontal_fov":
                lidar_cfg[
                    "horizontal_fov"
                ],

            "upper_fov":
                lidar_cfg[
                    "upper_fov"
                ],

            "lower_fov":
                lidar_cfg[
                    "lower_fov"
                ],

            # 每个 simulation tick
            "sensor_tick":
                0.0
        }

        for key, value in (
            lidar_attributes.items()
        ):

            lidar_bp.set_attribute(
                key,
                str(value)
            )

        if lidar_bp.has_attribute(
            "noise_stddev"
        ):

            lidar_bp.set_attribute(
                "noise_stddev",
                "0.0"
            )

        lidar = (
            world.spawn_actor(
                lidar_bp,
                lidar_tf
            )
        )

        # ====================================================
        # SENSOR QUEUES
        # ====================================================

        camera_queue = (
            queue.Queue()
        )

        lidar_queue = (
            queue.Queue()
        )

        camera.listen(
            camera_queue.put
        )

        lidar.listen(
            lidar_queue.put
        )

        # ====================================================
        # CALIBRATION
        # ====================================================

        K = camera_intrinsic(

            int(
                camera_cfg[
                    "width"
                ]
            ),

            int(
                camera_cfg[
                    "height"
                ]
            ),

            float(
                camera_cfg[
                    "fov"
                ]
            )
        )

        T_world_camera = (
            matrix(
                camera_tf
            )
        )

        T_world_lidar = (
            matrix(
                lidar_tf
            )
        )

        # ----------------------------------------------------
        # LiDAR local UE
        # ->
        # Camera local UE
        # ----------------------------------------------------

        T_camera_ue_lidar = (

            np.linalg.inv(
                T_world_camera
            )

            @

            T_world_lidar
        )

        # ----------------------------------------------------
        # CARLA camera coordinates
        # ->
        # OpenCV camera coordinates
        #
        # UE:
        #
        #   x forward
        #   y right
        #   z up
        #
        # CV:
        #
        #   x right
        #   y down
        #   z forward
        #
        # UE (x,y,z)
        # ->
        # CV (y,-z,x)
        # ----------------------------------------------------

        T_cv_ue = np.array([

            [0, 1,  0, 0],

            [0, 0, -1, 0],

            [1, 0,  0, 0],

            [0, 0,  0, 1]

        ], dtype=np.float64)

        T_camera_cv_lidar = (

            T_cv_ue

            @

            T_camera_ue_lidar
        )

        calibration = {

            "dataset_fps_hz":
                float(FPS),

            "uav_speed_mps":
                float(
                    route.speed
                ),

            "route_name":
                route.name,

            "route_length_m":
                float(
                    route.total_length
                ),

            "K":
                K.tolist(),

            "camera_resolution": [

                int(
                    camera_cfg[
                        "width"
                    ]
                ),

                int(
                    camera_cfg[
                        "height"
                    ]
                )
            ],

            "camera_fov_deg":
                float(
                    camera_cfg[
                        "fov"
                    ]
                ),

            "lidar_rotation_frequency_hz":
                float(FPS),

            "T_camera_ue_from_lidar":
                T_camera_ue_lidar.tolist(),

            "T_camera_cv_from_lidar":
                T_camera_cv_lidar.tolist(),

            "camera_lidar_distance_m":
                float(

                    np.linalg.norm(

                        T_world_camera[
                            :3,
                            3
                        ]

                        -

                        T_world_lidar[
                            :3,
                            3
                        ]
                    )
                )
        }

        with open(

            scene_dir
            / "calibration.json",

            "w",
            encoding="utf-8"

        ) as f:

            json.dump(
                calibration,
                f,
                indent=4
            )

        print()
        print(
            "Camera-LiDAR distance:",
            calibration[
                "camera_lidar_distance_m"
            ],
            "m"
        )

        # ====================================================
        # DATASET METADATA
        # ====================================================

        metadata = {

            "scene_name":
                scene_name,

            "dataset_fps_hz":
                float(FPS),

            "configured_num_frames":
                int(
                    CONFIGURED_NUM_FRAMES
                ),

            "actual_num_frames":
                int(
                    NUM_FRAMES
                ),

            "recording_mode":
                (
                    "until_route_end"
                    if CONFIGURED_NUM_FRAMES
                    == -1
                    else
                    "fixed_num_frames"
                ),

            "uav_speed_mps":
                float(
                    route.speed
                ),

            "distance_per_frame_m":
                float(
                    route.distance_per_frame
                ),

            "route_name":
                route.name,

            "route_source":
                uav_cfg[
                    "route_source"
                ],

            "route_length_m":
                float(
                    route.total_length
                ),

            "unique_waypoints":
                int(
                    len(
                        route.points
                    )
                ),

            "traffic_requested":
                int(
                    traffic_cfg[
                        "num_vehicles"
                    ]
                ),

            "traffic_spawned":
                int(
                    len(
                        vehicles
                    )
                ),

            "traffic_warmup_seconds":
                float(
                    traffic_cfg.get(
                        "warmup_seconds",
                        0.0
                    )
                )
        }

        with open(

            scene_dir
            / "metadata.json",

            "w",
            encoding="utf-8"

        ) as f:

            json.dump(
                metadata,
                f,
                indent=4
            )

        # ====================================================
        # SPECTATOR
        # ====================================================

        spectator = (
            world.get_spectator()
        )

        # ====================================================
        # RECORD
        # ====================================================

        print()
        print(
            "=" * 58
        )

        print(
            "RECORDING START"
        )

        print(
            "=" * 58
        )

        print(
            f"Frames to record: "
            f"{NUM_FRAMES}"
        )

        print()

        for i in range(
            NUM_FRAMES
        ):

            # ----------------------------------------------
            # UAV pose
            # ----------------------------------------------

            uav_tf = (
                route.pose_at_frame(
                    i
                )
            )

            route_distance = (
                route.distance_at_frame(
                    i
                )
            )

            if (
                route.mode
                == "waypoints"
            ):

                route_progress = (

                    route_distance

                    /
                    route.total_length
                )

            else:

                route_progress = 0.0

            # ----------------------------------------------
            # Sensor poses
            # ----------------------------------------------

            camera_tf = (
                sensor_world_transform(

                    uav_tf,

                    camera_cfg[
                        "position"
                    ],

                    camera_cfg[
                        "rotation"
                    ]
                )
            )

            lidar_tf = (
                sensor_world_transform(

                    uav_tf,

                    lidar_cfg[
                        "position"
                    ],

                    lidar_cfg[
                        "rotation"
                    ]
                )
            )

            camera.set_transform(
                camera_tf
            )

            lidar.set_transform(
                lidar_tf
            )

            # ----------------------------------------------
            # Spectator
            # ----------------------------------------------

            spectator.set_transform(

                carla.Transform(

                    carla.Location(

                        x=(
                            uav_tf.location.x
                        ),

                        y=(
                            uav_tf.location.y
                        ),

                        z=(
                            uav_tf.location.z
                            + 5.0
                        )
                    ),

                    carla.Rotation(

                        pitch=-60.0,

                        yaw=(
                            uav_tf
                            .rotation
                            .yaw
                        ),

                        roll=0.0
                    )
                )
            )

            # ----------------------------------------------
            # CARLA tick
            # ----------------------------------------------

            carla_frame = (
                world.tick()
            )

            # ----------------------------------------------
            # Wait synchronized sensors
            # ----------------------------------------------

            image = get_frame(

                camera_queue,

                carla_frame,

                "Camera"
            )

            cloud = get_frame(

                lidar_queue,

                carla_frame,

                "LiDAR"
            )

            # ----------------------------------------------
            # Synchronization safety check
            # ----------------------------------------------

            if (
                image.frame
                != cloud.frame
            ):

                raise RuntimeError(

                    "Camera-LiDAR frame "
                    "mismatch: "
                    f"camera={image.frame}, "
                    f"lidar={cloud.frame}"
                )

            # ----------------------------------------------
            # RGB
            # ----------------------------------------------

            if SAVE_RGB:

                image.save_to_disk(

                    str(
                        rgb_dir
                        /
                        f"{i:06d}.png"
                    )
                )

            # ----------------------------------------------
            # LiDAR
            # ----------------------------------------------

            points = np.frombuffer(

                cloud.raw_data,

                dtype=np.float32

            ).reshape(
                -1,
                4
            )

            if SAVE_LIDAR:

                points.tofile(

                    str(
                        lidar_dir
                        /
                        f"{i:06d}.bin"
                    )
                )

            # ----------------------------------------------
            # Pose
            # ----------------------------------------------

            dataset_time = (
                float(i)
                / FPS
            )

            route_finished = (
                route.is_finished(
                    i
                )
            )

            frame_info = {

                "sample_index":
                    int(i),

                "dataset_time_s":
                    float(
                        dataset_time
                    ),

                "carla_frame":
                    int(
                        carla_frame
                    ),

                "carla_timestamp":
                    float(
                        image.timestamp
                    ),

                "dataset_fps_hz":
                    float(FPS),

                "uav_speed_mps":
                    float(
                        route.speed
                    ),

                "route_distance_m":
                    float(
                        route_distance
                    ),

                "route_progress":
                    float(
                        route_progress
                    ),

                "route_finished":
                    bool(
                        route_finished
                    ),

                "uav":
                    pose_dict(
                        uav_tf
                    ),

                "camera":
                    pose_dict(
                        image.transform
                    ),

                "lidar":
                    pose_dict(
                        cloud.transform
                    ),

                "T_world_camera":
                    matrix(
                        image.transform
                    ).tolist(),

                "T_world_lidar":
                    matrix(
                        cloud.transform
                    ).tolist(),

                "lidar_points":
                    int(
                        points.shape[0]
                    )
            }

            if SAVE_POSE:

                with open(

                    pose_dir
                    /
                    f"{i:06d}.json",

                    "w",
                    encoding="utf-8"

                ) as f:

                    json.dump(
                        frame_info,
                        f,
                        indent=4
                    )

            # ----------------------------------------------
            # Console
            # ----------------------------------------------

            print(

                f"[{i + 1:04d}/"
                f"{NUM_FRAMES:04d}] "

                f"frame="
                f"{carla_frame} "

                f"points="
                f"{points.shape[0]} "

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

        # ====================================================
        # DONE
        # ====================================================

        print()
        print(
            "=" * 58
        )

        print(
            "RECORDING DONE"
        )

        print(
            "=" * 58
        )

        print(
            "Scene:"
        )

        print(
            scene_dir
        )

        if (
            CONFIGURED_NUM_FRAMES
            == -1
            and route.mode
            == "waypoints"
        ):

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
                "Final waypoint reached:",
                route.is_finished(
                    NUM_FRAMES - 1
                )
            )

    finally:

        # ====================================================
        # CLEANUP
        # ====================================================

        print()
        print(
            "Cleaning up..."
        )

        if camera is not None:

            try:

                camera.stop()

            except RuntimeError:

                pass

        if lidar is not None:

            try:

                lidar.stop()

            except RuntimeError:

                pass

        destroy_ids = []

        if camera is not None:

            destroy_ids.append(
                camera.id
            )

        if lidar is not None:

            destroy_ids.append(
                lidar.id
            )

        destroy_ids += [

            vehicle.id

            for vehicle in vehicles

            if vehicle is not None
        ]

        if destroy_ids:

            try:

                client.apply_batch([

                    carla.command.DestroyActor(
                        actor_id
                    )

                    for actor_id
                    in destroy_ids
                ])

            except RuntimeError as e:

                print(
                    "WARNING: actor cleanup "
                    "failed:",
                    e
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

        print(
            "Cleanup complete."
        )


if __name__ == "__main__":
    main()
