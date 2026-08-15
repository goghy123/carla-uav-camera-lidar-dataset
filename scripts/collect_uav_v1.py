import carla

import json
import math
import queue
import random
from datetime import datetime
from pathlib import Path

import numpy as np


# ============================================================
# 配置区域 —— 目前你主要改这里
# ============================================================

HOST = "localhost"
PORT = 2000
TM_PORT = 8000

# 数据集
FPS = 10
NUM_FRAMES = 100          # 先录100帧 = 10秒
NUM_VEHICLES = 40

# RGB Camera
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FOV = 90.0

# LiDAR
LIDAR_CHANNELS = 64
LIDAR_RANGE = 120.0
LIDAR_POINTS_PER_SECOND = 600000
LIDAR_ROTATION_FREQUENCY = FPS

# 无人机向下 LiDAR
# -89° 接近正下方
# -25° 向外围扩展
LIDAR_UPPER_FOV = -25.0
LIDAR_LOWER_FOV = -89.0

# 随机种子
RANDOM_SEED = 42


# ============================================================
# 路径
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = PROJECT_ROOT / "dataset"

scene_name = datetime.now().strftime("scene_%Y%m%d_%H%M%S")
SCENE_DIR = DATASET_ROOT / scene_name

RGB_DIR = SCENE_DIR / "rgb"
LIDAR_DIR = SCENE_DIR / "lidar"
POSE_DIR = SCENE_DIR / "pose"

RGB_DIR.mkdir(parents=True, exist_ok=True)
LIDAR_DIR.mkdir(parents=True, exist_ok=True)
POSE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 工具函数
# ============================================================

def transform_to_dict(transform):
    """把 CARLA Transform 转成 JSON 可保存格式。"""

    return {
        "location": {
            "x": transform.location.x,
            "y": transform.location.y,
            "z": transform.location.z,
        },
        "rotation": {
            "pitch": transform.rotation.pitch,
            "yaw": transform.rotation.yaw,
            "roll": transform.rotation.roll,
        },
    }


def transform_to_matrix(transform):
    """CARLA local -> world 的 4x4 齐次变换矩阵。"""

    return np.array(
        transform.get_matrix(),
        dtype=np.float64
    )


def build_camera_intrinsic(width, height, fov):
    """
    根据 CARLA Camera 的水平 FOV 构造相机内参 K。
    """

    focal = width / (
        2.0 * math.tan(math.radians(fov) / 2.0)
    )

    K = np.array([
        [focal, 0.0, width / 2.0],
        [0.0, focal, height / 2.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)

    return K


def get_sensor_data(sensor_queue, target_frame, sensor_name):
    """
    从 Queue 中取得指定 CARLA frame 的数据。
    """

    while True:
        data = sensor_queue.get(timeout=10.0)

        if data.frame == target_frame:
            return data

        if data.frame > target_frame:
            raise RuntimeError(
                f"{sensor_name} 跳过了 frame "
                f"{target_frame}，当前是 {data.frame}"
            )


# ============================================================
# 主函数
# ============================================================

def main():

    random.seed(RANDOM_SEED)

    client = carla.Client(HOST, PORT)
    client.set_timeout(20.0)

    world = client.get_world()

    print("=" * 70)
    print("Connected to CARLA")
    print("Map:", world.get_map().name)
    print("=" * 70)

    # --------------------------------------------------------
    # 保存原来的世界设置
    # --------------------------------------------------------

    original_settings = world.get_settings()

    vehicles = []
    camera = None
    lidar = None

    traffic_manager = client.get_trafficmanager(TM_PORT)

    try:

        # ====================================================
        # 1. 开启同步模式
        # ====================================================

        settings = world.get_settings()

        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / FPS

        world.apply_settings(settings)

        traffic_manager.set_synchronous_mode(True)
        traffic_manager.set_random_device_seed(RANDOM_SEED)

        print(f"Simulation FPS: {FPS}")

        # ====================================================
        # 2. 生成交通车辆
        # ====================================================

        blueprint_library = world.get_blueprint_library()

        vehicle_blueprints = list(
            blueprint_library.filter("vehicle.*")
        )

        spawn_points = world.get_map().get_spawn_points()

        random.shuffle(spawn_points)

        number_to_spawn = min(
            NUM_VEHICLES,
            len(spawn_points)
        )

        batch = []

        for transform in spawn_points[:number_to_spawn]:

            bp = random.choice(vehicle_blueprints)

            if bp.has_attribute("color"):
                colors = bp.get_attribute(
                    "color"
                ).recommended_values

                if colors:
                    bp.set_attribute(
                        "color",
                        random.choice(colors)
                    )

            bp.set_attribute(
                "role_name",
                "autopilot"
            )

            batch.append(
                carla.command.SpawnActor(
                    bp,
                    transform
                ).then(
                    carla.command.SetAutopilot(
                        carla.command.FutureActor,
                        True,
                        TM_PORT
                    )
                )
            )

        responses = client.apply_batch_sync(
            batch,
            True
        )

        vehicle_ids = []

        for response in responses:
            if not response.error:
                vehicle_ids.append(
                    response.actor_id
                )

        vehicles = [
            world.get_actor(actor_id)
            for actor_id in vehicle_ids
        ]

        vehicles = [
            v for v in vehicles
            if v is not None
        ]

        print(
            f"Spawned vehicles: {len(vehicles)}"
        )

        # ====================================================
        # 3. 获取 Spectator 当前的位置
        # ====================================================

        spectator = world.get_spectator()

        spectator_transform = spectator.get_transform()

        print("\nSpectator / UAV Camera position:")
        print(spectator_transform)

        # Camera 完全使用你当前 Spectator 的视角
        camera_transform = spectator_transform

        # LiDAR 与 Camera 放在同一位置
        # 但 LiDAR 保持水平坐标系
        # 通过 vertical FOV 向下发射激光
        lidar_transform = carla.Transform(
            carla.Location(
                x=spectator_transform.location.x,
                y=spectator_transform.location.y,
                z=spectator_transform.location.z
            ),
            carla.Rotation(
                pitch=0.0,
                yaw=spectator_transform.rotation.yaw,
                roll=0.0
            )
        )

        # ====================================================
        # 4. 创建 RGB Camera
        # ====================================================

        camera_bp = blueprint_library.find(
            "sensor.camera.rgb"
        )

        camera_bp.set_attribute(
            "image_size_x",
            str(CAMERA_WIDTH)
        )

        camera_bp.set_attribute(
            "image_size_y",
            str(CAMERA_HEIGHT)
        )

        camera_bp.set_attribute(
            "fov",
            str(CAMERA_FOV)
        )

        camera_bp.set_attribute(
            "sensor_tick",
            "0.0"
        )

        # 尽量使用理想 pinhole Camera
        if camera_bp.has_attribute("lens_k"):
            camera_bp.set_attribute(
                "lens_k",
                "0.0"
            )

        if camera_bp.has_attribute("lens_kcube"):
            camera_bp.set_attribute(
                "lens_kcube",
                "0.0"
            )

        camera = world.spawn_actor(
            camera_bp,
            camera_transform
        )

        # ====================================================
        # 5. 创建向下 LiDAR
        # ====================================================

        lidar_bp = blueprint_library.find(
            "sensor.lidar.ray_cast"
        )

        lidar_bp.set_attribute(
            "channels",
            str(LIDAR_CHANNELS)
        )

        lidar_bp.set_attribute(
            "range",
            str(LIDAR_RANGE)
        )

        lidar_bp.set_attribute(
            "points_per_second",
            str(LIDAR_POINTS_PER_SECOND)
        )

        lidar_bp.set_attribute(
            "rotation_frequency",
            str(LIDAR_ROTATION_FREQUENCY)
        )

        lidar_bp.set_attribute(
            "horizontal_fov",
            "360.0"
        )

        lidar_bp.set_attribute(
            "upper_fov",
            str(LIDAR_UPPER_FOV)
        )

        lidar_bp.set_attribute(
            "lower_fov",
            str(LIDAR_LOWER_FOV)
        )

        lidar_bp.set_attribute(
            "sensor_tick",
            "0.0"
        )

        # 第一版先关闭噪声
        if lidar_bp.has_attribute("noise_stddev"):
            lidar_bp.set_attribute(
                "noise_stddev",
                "0.0"
            )

        # 第一版先不要随机丢点
        if lidar_bp.has_attribute(
            "dropoff_general_rate"
        ):
            lidar_bp.set_attribute(
                "dropoff_general_rate",
                "0.0"
            )

        if lidar_bp.has_attribute(
            "dropoff_zero_intensity"
        ):
            lidar_bp.set_attribute(
                "dropoff_zero_intensity",
                "0.0"
            )

        lidar = world.spawn_actor(
            lidar_bp,
            lidar_transform
        )

        print("\nRGB Camera created.")
        print("LiDAR created.")

        # ====================================================
        # 6. 创建同步 Queue
        # ====================================================

        camera_queue = queue.Queue()
        lidar_queue = queue.Queue()

        camera.listen(camera_queue.put)
        lidar.listen(lidar_queue.put)

        # ====================================================
        # 7. 保存 Calibration
        # ====================================================

        K = build_camera_intrinsic(
            CAMERA_WIDTH,
            CAMERA_HEIGHT,
            CAMERA_FOV
        )

        T_world_from_camera_ue = \
            transform_to_matrix(
                camera.get_transform()
            )

        T_world_from_lidar_ue = \
            transform_to_matrix(
                lidar.get_transform()
            )

        # lidar local UE -> camera local UE
        T_camera_ue_from_lidar_ue = (
            np.linalg.inv(
                T_world_from_camera_ue
            )
            @ T_world_from_lidar_ue
        )

        # CARLA / Unreal camera coordinates:
        #
        # x = forward
        # y = right
        # z = up
        #
        # OpenCV Camera:
        #
        # x = right
        # y = down
        # z = forward
        #
        # (x,y,z)_UE -> (y,-z,x)_CV

        T_camera_cv_from_camera_ue = np.array([
            [0.0, 1.0,  0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [1.0, 0.0,  0.0, 0.0],
            [0.0, 0.0,  0.0, 1.0]
        ])

        # raw LiDAR UE coordinates -> OpenCV Camera
        T_camera_cv_from_lidar_ue = (
            T_camera_cv_from_camera_ue
            @ T_camera_ue_from_lidar_ue
        )

        calibration = {

            "camera": {

                "width": CAMERA_WIDTH,
                "height": CAMERA_HEIGHT,
                "horizontal_fov_deg": CAMERA_FOV,

                "K": K.tolist(),

                "distortion": [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0
                ]
            },

            "lidar": {

                "channels": LIDAR_CHANNELS,
                "range_m": LIDAR_RANGE,
                "points_per_second":
                    LIDAR_POINTS_PER_SECOND,

                "rotation_frequency_hz":
                    LIDAR_ROTATION_FREQUENCY,

                "horizontal_fov_deg":
                    360.0,

                "upper_fov_deg":
                    LIDAR_UPPER_FOV,

                "lower_fov_deg":
                    LIDAR_LOWER_FOV
            },

            "coordinate_system": {

                "carla_ue":
                    "x-forward, y-right, z-up",

                "opencv_camera":
                    "x-right, y-down, z-forward"
            },

            "T_world_from_camera_ue":
                T_world_from_camera_ue.tolist(),

            "T_world_from_lidar_ue":
                T_world_from_lidar_ue.tolist(),

            "T_camera_ue_from_lidar_ue":
                T_camera_ue_from_lidar_ue.tolist(),

            "T_camera_cv_from_lidar_ue":
                T_camera_cv_from_lidar_ue.tolist()
        }

        with open(
            SCENE_DIR / "calibration.json",
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                calibration,
                f,
                indent=4
            )

        # ====================================================
        # 8. Metadata
        # ====================================================

        metadata = {

            "map": world.get_map().name,
            "fps": FPS,
            "num_frames": NUM_FRAMES,
            "num_vehicles": len(vehicles),

            "camera_transform":
                transform_to_dict(
                    camera.get_transform()
                ),

            "lidar_transform":
                transform_to_dict(
                    lidar.get_transform()
                )
        }

        with open(
            SCENE_DIR / "metadata.json",
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                metadata,
                f,
                indent=4
            )

        # ====================================================
        # 9. 开始采集
        # ====================================================

        print("\n" + "=" * 70)
        print("START RECORDING")
        print("Output:", SCENE_DIR)
        print("=" * 70)

        for sample_index in range(NUM_FRAMES):

            # 推进一步仿真
            carla_frame = world.tick()

            # 获取完全相同 CARLA frame
            image = get_sensor_data(
                camera_queue,
                carla_frame,
                "Camera"
            )

            lidar_data = get_sensor_data(
                lidar_queue,
                carla_frame,
                "LiDAR"
            )

            # ------------------------------------------------
            # 保存 RGB
            # ------------------------------------------------

            image_path = (
                RGB_DIR
                / f"{sample_index:06d}.png"
            )

            image.save_to_disk(
                str(image_path)
            )

            # ------------------------------------------------
            # 保存 LiDAR XYZI
            # ------------------------------------------------

            lidar_points = np.frombuffer(
                lidar_data.raw_data,
                dtype=np.float32
            )

            lidar_points = lidar_points.reshape(
                (-1, 4)
            )

            lidar_path = (
                LIDAR_DIR
                / f"{sample_index:06d}.bin"
            )

            lidar_points.tofile(
                lidar_path
            )

            # ------------------------------------------------
            # 保存每帧 Pose
            # ------------------------------------------------

            pose_data = {

                "sample_index": sample_index,

                "carla_frame":
                    int(carla_frame),

                "timestamp":
                    float(image.timestamp),

                "num_lidar_points":
                    int(lidar_points.shape[0]),

                "camera_transform":
                    transform_to_dict(
                        image.transform
                    ),

                "lidar_transform":
                    transform_to_dict(
                        lidar_data.transform
                    ),

                "T_world_from_camera_ue":
                    transform_to_matrix(
                        image.transform
                    ).tolist(),

                "T_world_from_lidar_ue":
                    transform_to_matrix(
                        lidar_data.transform
                    ).tolist()
            }

            pose_path = (
                POSE_DIR
                / f"{sample_index:06d}.json"
            )

            with open(
                pose_path,
                "w",
                encoding="utf-8"
            ) as f:

                json.dump(
                    pose_data,
                    f,
                    indent=4
                )

            print(
                f"[{sample_index + 1:03d}"
                f"/{NUM_FRAMES}] "
                f"CARLA frame={carla_frame} | "
                f"LiDAR points="
                f"{lidar_points.shape[0]}"
            )

        print("\nRecording finished!")

        print("\nDataset saved to:")
        print(SCENE_DIR)

    finally:

        print("\nCleaning up...")

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

        actors_to_destroy = []

        if camera is not None:
            actors_to_destroy.append(camera.id)

        if lidar is not None:
            actors_to_destroy.append(lidar.id)

        actors_to_destroy.extend(
            [v.id for v in vehicles if v is not None]
        )

        if actors_to_destroy:

            client.apply_batch([
                carla.command.DestroyActor(actor_id)
                for actor_id in actors_to_destroy
            ])

        try:
            traffic_manager.set_synchronous_mode(
                False
            )
        except RuntimeError:
            pass

        # 恢复 CARLA 原来的世界设置
        world.apply_settings(
            original_settings
        )

        print("Cleanup complete.")


if __name__ == "__main__":
    main()
