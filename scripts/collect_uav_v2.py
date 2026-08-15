import carla

import json
import math
import queue
import random
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
    # 读取主配置
    # ========================================================

    config = load_yaml(
        CONFIG_PATH
    )

    # ========================================================
    # 找到路线文件
    #
    # UAVdataset.yaml:
    #
    # route_file: "routes/route_01.yaml"
    #
    # 实际：
    #
    # configs/routes/route_01.yaml
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

    # ========================================================
    # 读取路线 YAML
    # ========================================================

    route_config = load_yaml(
        route_path
    )

    if "route" not in route_config:

        raise KeyError(
            f"\nMissing 'route:' in:\n"
            f"{route_path}"
        )

    # ========================================================
    # 注入 config
    #
    # 这样你之前写好的 UAVRoute 类完全不用改。
    # ========================================================

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
        route_config["route"]["mode"]
    )

    return config



# ============================================================
# TRANSFORM
# ============================================================

def make_transform(position, rotation):

    return carla.Transform(

        carla.Location(
            x=float(position["x"]),
            y=float(position["y"]),
            z=float(position["z"])
        ),

        carla.Rotation(
            pitch=float(rotation["pitch"]),
            yaw=float(rotation["yaw"]),
            roll=float(rotation["roll"])
        )
    )


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
# 第一阶段假设：
#
# UAV 始终水平：
# pitch = 0
# roll  = 0
#
# UAV 飞行过程中主要改变：
#
# x
# y
# z
# yaw
#
# 这样足够我们建立数据集。
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

    # 将 UAV 局部安装位置转成世界坐标
    world_location = uav_transform.transform(
        relative_location
    )

    # 当前版本 UAV 保持水平，因此传感器姿态非常直观：
    #
    # sensor yaw =
    # UAV yaw + relative yaw

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
        )
    )

    return carla.Transform(
        world_location,
        world_rotation
    )


# ============================================================
# CAMERA K
# ============================================================

def camera_intrinsic(
    width,
    height,
    fov
):

    focal = width / (
        2.0
        * math.tan(
            math.radians(fov) / 2.0
        )
    )

    K = np.array([

        [focal, 0.0, width / 2.0],

        [0.0, focal, height / 2.0],

        [0.0, 0.0, 1.0]

    ], dtype=np.float64)

    return K


# ============================================================
# SENSOR QUEUE
# ============================================================

def get_frame(
    sensor_queue,
    target_frame,
    name
):

    while True:

        data = sensor_queue.get(
            timeout=10.0
        )

        if data.frame == target_frame:
            return data

        if data.frame > target_frame:

            raise RuntimeError(
                f"{name}: expected "
                f"{target_frame}, "
                f"got {data.frame}"
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

        self.fps = fps

        initial = uav_config[
            "initial_pose"
        ]

        self.initial = carla.Transform(

            carla.Location(
                x=float(initial["x"]),
                y=float(initial["y"]),
                z=float(initial["z"])
            ),

            carla.Rotation(
                pitch=float(initial["pitch"]),
                yaw=float(initial["yaw"]),
                roll=float(initial["roll"])
            )
        )

        route = uav_config["route"]

        self.mode = str(
            route["mode"]
        ).lower()

        self.speed = float(
            route["speed_mps"]
        )

        self.loop = bool(
            route["loop"]
        )

        self.points = []

        for p in route["waypoints"]:

            self.points.append(
                np.array([
                    float(p["x"]),
                    float(p["y"]),
                    float(p["z"])
                ])
            )

        self.segment_lengths = []
        self.total_length = 0.0

        if len(self.points) >= 2:

            for i in range(
                len(self.points) - 1
            ):

                length = np.linalg.norm(
                    self.points[i + 1]
                    - self.points[i]
                )

                self.segment_lengths.append(
                    float(length)
                )

                self.total_length += float(
                    length
                )

    def pose_at_frame(
        self,
        frame_index
    ):

        # ================================================
        # STATIC
        # ================================================

        if self.mode == "static":

            return self.initial

        # ================================================
        # WAYPOINT ROUTE
        # ================================================

        if len(self.points) < 2:

            print(
                "WARNING: not enough waypoints. "
                "Using static UAV."
            )

            return self.initial

        t = frame_index / self.fps

        distance = (
            t * self.speed
        )

        if self.loop:

            distance = (
                distance
                % self.total_length
            )

        else:

            distance = min(
                distance,
                self.total_length
            )

        remaining = distance

        segment = 0

        for i, length in enumerate(
            self.segment_lengths
        ):

            if remaining <= length:

                segment = i
                break

            remaining -= length

        else:

            segment = (
                len(self.segment_lengths)
                - 1
            )

            remaining = (
                self.segment_lengths[
                    segment
                ]
            )

        start = self.points[
            segment
        ]

        end = self.points[
            segment + 1
        ]

        length = self.segment_lengths[
            segment
        ]

        if length <= 1e-6:

            ratio = 0.0

        else:

            ratio = (
                remaining / length
            )

        pos = (
            start
            + ratio * (end - start)
        )

        direction = end - start

        # CARLA:
        #
        # x forward axis reference
        # yaw rotates in XY plane

        yaw = math.degrees(
            math.atan2(
                direction[1],
                direction[0]
            )
        )

        return carla.Transform(

            carla.Location(
                x=float(pos[0]),
                y=float(pos[1]),
                z=float(pos[2])
            ),

            carla.Rotation(
                pitch=0.0,
                yaw=yaw,
                roll=0.0
            )
        )


# ============================================================
# TRAFFIC
# ============================================================

def spawn_traffic(
    client,
    world,
    config
):

    if not config["enabled"]:

        return []

    random.seed(
        int(
            CONFIG["simulation"][
                "random_seed"
            ]
        )
    )

    bp_lib = (
        world.get_blueprint_library()
    )

    vehicle_bps = list(
        bp_lib.filter(
            "vehicle.*"
        )
    )

    spawn_points = (
        world.get_map()
        .get_spawn_points()
    )

    random.shuffle(
        spawn_points
    )

    number = min(
        int(config["num_vehicles"]),
        len(spawn_points)
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

        bp.set_attribute(
            "role_name",
            "autopilot"
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

    ids = [

        r.actor_id

        for r in responses

        if not r.error
    ]

    return [

        world.get_actor(i)

        for i in ids

        if world.get_actor(i)
        is not None
    ]


# ============================================================
# MAIN
# ============================================================

def main():

    global CONFIG

    CONFIG = load_config()

    # --------------------------------------------------------
    # Configuration
    # --------------------------------------------------------

    sim_cfg = CONFIG["simulation"]
    traffic_cfg = CONFIG["traffic"]
    uav_cfg = CONFIG["uav"]

    camera_cfg = (
        CONFIG["sensors"]["camera"]
    )

    lidar_cfg = (
        CONFIG["sensors"]["lidar"]
    )

    FPS = int(
        sim_cfg["fps"]
    )

    NUM_FRAMES = int(
        sim_cfg["num_frames"]
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

    world = client.get_world()

    original_settings = (
        world.get_settings()
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
    vehicles = []

    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------

    output_root = (
        PROJECT_ROOT
        / str(
            CONFIG["output"]["root"]
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
        scene_dir / "rgb"
    )

    lidar_dir = (
        scene_dir / "lidar"
    )

    pose_dir = (
        scene_dir / "pose"
    )

    rgb_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    lidar_dir.mkdir(
        parents=True,
        exist_ok=True
    )

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

        traffic_manager.set_random_device_seed(
            int(
                sim_cfg[
                    "random_seed"
                ]
            )
        )

        # ====================================================
        # TRAFFIC
        # ====================================================

        vehicles = spawn_traffic(
            client,
            world,
            traffic_cfg
        )

        print(
            "Vehicles:",
            len(vehicles)
        )

        # ====================================================
        # ROUTE
        # ====================================================

        route = UAVRoute(
            uav_cfg,
            FPS
        )

        initial_uav = (
            route.pose_at_frame(0)
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
        # CAMERA
        # ====================================================

        bp_lib = (
            world
            .get_blueprint_library()
        )

        camera_bp = (
            bp_lib.find(
                "sensor.camera.rgb"
            )
        )

        camera_bp.set_attribute(
            "image_size_x",
            str(camera_cfg["width"])
        )

        camera_bp.set_attribute(
            "image_size_y",
            str(camera_cfg["height"])
        )

        camera_bp.set_attribute(
            "fov",
            str(camera_cfg["fov"])
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

        camera = world.spawn_actor(
            camera_bp,
            camera_tf
        )

        # ====================================================
        # LIDAR
        # ====================================================

        lidar_bp = (
            bp_lib.find(
                "sensor.lidar.ray_cast"
            )
        )

        attributes = {

            "channels":
                lidar_cfg["channels"],

            "range":
                lidar_cfg["range"],

            "points_per_second":
                lidar_cfg[
                    "points_per_second"
                ],

            "rotation_frequency":
                lidar_cfg[
                    "rotation_frequency"
                ],

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

            "sensor_tick":
                0.0
        }

        for key, value in (
            attributes.items()
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

        lidar = world.spawn_actor(
            lidar_bp,
            lidar_tf
        )

        # ====================================================
        # QUEUES
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

            int(camera_cfg["width"]),

            int(camera_cfg["height"]),

            float(camera_cfg["fov"])
        )

        T_world_camera = matrix(
            camera_tf
        )

        T_world_lidar = matrix(
            lidar_tf
        )

        # lidar local UE
        # ->
        # camera local UE

        T_camera_ue_lidar = (

            np.linalg.inv(
                T_world_camera
            )

            @ T_world_lidar
        )

        # CARLA Camera coordinates
        # ->
        # OpenCV Camera coordinates
        #
        # UE (x,y,z)
        # ->
        # CV (y,-z,x)

        T_cv_ue = np.array([

            [0, 1,  0, 0],

            [0, 0, -1, 0],

            [1, 0,  0, 0],

            [0, 0,  0, 1]

        ], dtype=np.float64)

        T_camera_cv_lidar = (

            T_cv_ue
            @ T_camera_ue_lidar
        )

        calibration = {

            "K":
                K.tolist(),

            "camera_resolution": [
                int(camera_cfg["width"]),
                int(camera_cfg["height"])
            ],

            "camera_fov_deg":
                float(camera_cfg["fov"]),

            "T_camera_ue_from_lidar":
                T_camera_ue_lidar.tolist(),

            "T_camera_cv_from_lidar":
                T_camera_cv_lidar.tolist(),

            "camera_lidar_distance_m":
                float(
                    np.linalg.norm(
                        T_world_camera[:3, 3]
                        -
                        T_world_lidar[:3, 3]
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

        print(
            "\nCamera-LiDAR distance:",
            calibration[
                "camera_lidar_distance_m"
            ],
            "m"
        )

        # ====================================================
        # RECORD
        # ====================================================

        print(
            "\nRecording..."
        )

        for i in range(
            NUM_FRAMES
        ):

            # ----------------------------------------------
            # UAV pose
            # ----------------------------------------------

            uav_tf = (
                route.pose_at_frame(i)
            )

            # ----------------------------------------------
            # Sensors follow UAV
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

            # Optional:
            # spectator follows UAV
            #
            # 为了方便你观察。

            spectator = (
                world.get_spectator()
            )

            spectator.set_transform(

                carla.Transform(

                    carla.Location(
                        x=uav_tf.location.x,
                        y=uav_tf.location.y,
                        z=uav_tf.location.z + 5
                    ),

                    carla.Rotation(
                        pitch=-60,
                        yaw=uav_tf.rotation.yaw,
                        roll=0
                    )
                )
            )

            # ----------------------------------------------
            # Tick
            # ----------------------------------------------

            carla_frame = (
                world.tick()
            )

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
            # RGB
            # ----------------------------------------------

            image.save_to_disk(

                str(
                    rgb_dir
                    / f"{i:06d}.png"
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

            points.tofile(

                lidar_dir
                / f"{i:06d}.bin"
            )

            # ----------------------------------------------
            # Pose
            # ----------------------------------------------

            frame_info = {

                "sample_index": i,

                "carla_frame":
                    int(carla_frame),

                "timestamp":
                    float(
                        image.timestamp
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

            with open(

                pose_dir
                / f"{i:06d}.json",

                "w",
                encoding="utf-8"

            ) as f:

                json.dump(
                    frame_info,
                    f,
                    indent=4
                )

            print(
                f"[{i+1:04d}/"
                f"{NUM_FRAMES}] "
                f"frame={carla_frame} "
                f"points={points.shape[0]} "
                f"UAV=("
                f"{uav_tf.location.x:.1f}, "
                f"{uav_tf.location.y:.1f}, "
                f"{uav_tf.location.z:.1f})"
            )

        print(
            "\nDONE:"
        )

        print(
            scene_dir
        )

    finally:

        print(
            "\nCleaning up..."
        )

        if camera:

            try:
                camera.stop()
            except RuntimeError:
                pass

        if lidar:

            try:
                lidar.stop()
            except RuntimeError:
                pass

        destroy_ids = []

        if camera:
            destroy_ids.append(
                camera.id
            )

        if lidar:
            destroy_ids.append(
                lidar.id
            )

        destroy_ids += [

            vehicle.id

            for vehicle
            in vehicles

            if vehicle is not None
        ]

        client.apply_batch([

            carla.command.DestroyActor(i)

            for i in destroy_ids
        ])

        traffic_manager.set_synchronous_mode(
            False
        )

        world.apply_settings(
            original_settings
        )

        print(
            "Cleanup complete."
        )


if __name__ == "__main__":
    main()
