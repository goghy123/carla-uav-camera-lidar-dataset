from pathlib import Path
import argparse
import math
import time

from concurrent.futures import ThreadPoolExecutor

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from PIL import Image


# ============================================================
# EXAMPLES
# ============================================================
#
# 15 FPS
#
# python scripts/play_dataset.py ^
#     --scene dataset/scene_20260814_222401 ^
#     --fps 15
#
#
# 5 FPS 慢慢看
#
# python scripts/play_dataset.py ^
#     --scene dataset/scene_20260814_222401 ^
#     --fps 5
#
#
# 点云显示范围 ±60 m
#
# python scripts/play_dataset.py ^
#     --scene dataset/scene_20260814_222401 ^
#     --fps 15 ^
#     --range 60
#
#
# 只显示最多 10000 个点
#
# python scripts/play_dataset.py ^
#     --scene dataset/scene_20260814_222401 ^
#     --fps 15 ^
#     --max-points 10000
#
#
# 显示全部点
#
# python scripts/play_dataset.py ^
#     --scene dataset/scene_20260814_222401 ^
#     --fps 5 ^
#     --max-points 0
#
#
# 增加预加载帧数
#
# python scripts/play_dataset.py ^
#     --scene dataset/scene_20260814_222401 ^
#     --fps 15 ^
#     --prefetch 12
#
#
# 循环播放
#
# python scripts/play_dataset.py ^
#     --scene dataset/scene_20260814_222401 ^
#     --fps 15 ^
#     --loop
#
# ============================================================


# ============================================================
# PROJECT
# ============================================================

SCRIPT_DIR = (
    Path(__file__)
    .resolve()
    .parent
)

PROJECT_DIR = (
    SCRIPT_DIR.parent
)


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Play synchronized RGB "
            "and CARLA LiDAR frames."
        )
    )

    parser.add_argument(
        "--scene",
        type=str,
        required=True,
        help=(
            "Dataset scene path. Example: "
            "dataset/scene_20260814_205000"
        )
    )

    parser.add_argument(
        "--fps",
        type=float,
        default=10.0,
        help="Target playback FPS. Default: 10"
    )

    parser.add_argument(
        "--range",
        type=float,
        default=80.0,
        help=(
            "LiDAR visualization range. "
            "Axes are [-range, +range]."
        )
    )

    parser.add_argument(
        "--max-points",
        type=int,
        default=15000,
        help=(
            "Maximum displayed LiDAR points "
            "per frame. "
            "Original BIN is unchanged. "
            "Use 0 for all points. "
            "Default: 15000"
        )
    )

    parser.add_argument(
        "--point-size",
        type=float,
        default=0.3,
        help="LiDAR point size. Default: 0.3"
    )

    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start from this sample index."
    )

    parser.add_argument(
        "--prefetch",
        type=int,
        default=8,
        help=(
            "Number of future frames "
            "to preload. Default: 8"
        )
    )

    parser.add_argument(
        "--io-workers",
        type=int,
        default=2,
        help=(
            "Background loading workers. "
            "Default: 2"
        )
    )

    parser.add_argument(
        "--loop",
        action="store_true",
        help="Loop playback."
    )

    return parser.parse_args()


# ============================================================
# VALIDATE ARGUMENTS
# ============================================================

def validate_args(args):

    if args.fps <= 0:

        raise ValueError(
            "--fps must be > 0"
        )

    if args.range <= 0:

        raise ValueError(
            "--range must be > 0"
        )

    if args.point_size <= 0:

        raise ValueError(
            "--point-size must be > 0"
        )

    if args.start < 0:

        raise ValueError(
            "--start must be >= 0"
        )

    if args.max_points < 0:

        raise ValueError(
            "--max-points must be >= 0"
        )

    if args.prefetch < 1:

        raise ValueError(
            "--prefetch must be >= 1"
        )

    if args.io_workers < 1:

        raise ValueError(
            "--io-workers must be >= 1"
        )


# ============================================================
# PATH
# ============================================================

def resolve_scene(scene_arg):

    path = Path(
        scene_arg
    )

    if not path.is_absolute():

        path = (
            PROJECT_DIR
            /
            path
        )

    path = path.resolve()

    if not path.exists():

        raise FileNotFoundError(
            f"\nScene does not exist:\n"
            f"{path}"
        )

    if not path.is_dir():

        raise NotADirectoryError(
            f"\nScene is not a directory:\n"
            f"{path}"
        )

    return path


# ============================================================
# FIND SYNCHRONIZED FRAMES
# ============================================================

def find_frames(scene_dir):

    rgb_dir = (
        scene_dir
        /
        "rgb"
    )

    lidar_dir = (
        scene_dir
        /
        "lidar"
    )

    if not rgb_dir.exists():

        raise FileNotFoundError(
            f"RGB directory missing:\n"
            f"{rgb_dir}"
        )

    if not lidar_dir.exists():

        raise FileNotFoundError(
            f"LiDAR directory missing:\n"
            f"{lidar_dir}"
        )

    # --------------------------------------------------------
    # Example:
    #
    # 000001.png
    # 000001.bin
    #
    # stem == 000001
    # --------------------------------------------------------

    rgb_files = {

        p.stem: p

        for p in rgb_dir.glob(
            "*.png"
        )
    }

    lidar_files = {

        p.stem: p

        for p in lidar_dir.glob(
            "*.bin"
        )
    }

    # 只播放 RGB / LiDAR 同时存在的 frame

    common = (
        set(
            rgb_files.keys()
        )
        &
        set(
            lidar_files.keys()
        )
    )

    if len(common) == 0:

        raise RuntimeError(
            "No synchronized RGB/LiDAR "
            "frame names found."
        )

    # 尽量按照数字编号排序

    def sort_key(name):

        try:

            return (
                0,
                int(name)
            )

        except ValueError:

            return (
                1,
                name
            )

    frame_ids = sorted(
        common,
        key=sort_key
    )

    frames = []

    for frame_id in frame_ids:

        frames.append(
            {
                "id": frame_id,

                "rgb":
                    rgb_files[
                        frame_id
                    ],

                "lidar":
                    lidar_files[
                        frame_id
                    ]
            }
        )

    return frames


# ============================================================
# LOAD RGB
# ============================================================

def load_rgb(path):

    # 使用 context manager，
    # 确保 PNG 文件句柄及时关闭。
    #
    # np.array(copy=True) 保证返回数组
    # 不再依赖 PIL Image 对象。

    with Image.open(path) as image:

        image = image.convert(
            "RGB"
        )

        array = np.array(
            image,
            dtype=np.uint8,
            copy=True
        )

    return array


# ============================================================
# LOAD CARLA BIN
#
# float32:
#
# x y z intensity
# ============================================================

def load_lidar(path):

    cloud = np.fromfile(
        path,
        dtype=np.float32
    )

    if cloud.size % 4 != 0:

        raise RuntimeError(
            f"Invalid XYZI BIN file:\n"
            f"{path}\n"
            f"float count = {cloud.size}"
        )

    cloud = cloud.reshape(
        -1,
        4
    )

    return cloud


# ============================================================
# DISPLAY DOWNSAMPLING
#
# 只影响显示。
# 不会修改 BIN 文件。
# ============================================================

def display_sample(
    points,
    max_points
):

    if max_points <= 0:

        return points

    count = len(
        points
    )

    if count <= max_points:

        return points

    # --------------------------------------------------------
    # 均匀 stride 降采样。
    #
    # 比每帧 random choice 更快，
    # 同时不会制造额外随机数和索引数组。
    # --------------------------------------------------------

    step = math.ceil(
        count
        /
        max_points
    )

    return points[
        ::step
    ]


# ============================================================
# LOAD COMPLETE FRAME
#
# 这个函数会在后台线程执行。
#
# 返回：
#
# RGB
# 原始点云点数
# 用于显示的 XYZ
# ============================================================

def load_frame_data(
    frame,
    max_points
):

    # --------------------------------------------------------
    # RGB
    # --------------------------------------------------------

    rgb = load_rgb(
        frame["rgb"]
    )

    # --------------------------------------------------------
    # LiDAR
    # --------------------------------------------------------

    cloud = load_lidar(
        frame["lidar"]
    )

    original_count = len(
        cloud
    )

    show_cloud = display_sample(
        cloud,
        max_points
    )

    # --------------------------------------------------------
    # Matplotlib 显示只需要 XYZ。
    #
    # 转成连续 float32 内存：
    #
    # 1. 减少后续 stride view 带来的开销
    # 2. 后台缓存不需要保存 intensity
    # 3. 减少缓存内存占用
    # --------------------------------------------------------

    xyz = np.ascontiguousarray(
        show_cloud[
            :,
            :3
        ],
        dtype=np.float32
    )

    return (
        rgb,
        original_count,
        xyz
    )


# ============================================================
# FRAME PREFETCHER
#
# 当前 Matplotlib 在画 frame N 时，
# 后台线程可以同时：
#
# 读取 PNG N+1
# 解压 PNG N+1
# 读取 BIN N+1
# 降采样 N+1
#
# 从而减少 update() 在 I/O 上等待。
# ============================================================

class FramePrefetcher:

    def __init__(
        self,
        frames,
        max_points,
        lookahead=8,
        workers=2
    ):

        self.frames = frames

        self.max_points = max_points

        self.lookahead = max(
            1,
            int(lookahead)
        )

        self.executor = ThreadPoolExecutor(
            max_workers=max(
                1,
                int(workers)
            ),
            thread_name_prefix="dataset-loader"
        )

        # index -> Future

        self.jobs = {}

        # 可以手工放入已经读取好的帧，
        # 比如第一帧。

        self.cache = {}

        self.closed = False

    # --------------------------------------------------------
    # WORKER
    # --------------------------------------------------------

    def _load(
        self,
        index
    ):

        return load_frame_data(
            self.frames[
                index
            ],
            self.max_points
        )

    # --------------------------------------------------------
    # ADD READY FRAME
    # --------------------------------------------------------

    def prime(
        self,
        index,
        data
    ):

        if self.closed:

            return

        self.cache[
            index
        ] = data

    # --------------------------------------------------------
    # SUBMIT
    # --------------------------------------------------------

    def submit(
        self,
        index
    ):

        if self.closed:

            return

        if index < 0:

            return

        if index >= len(
            self.frames
        ):

            return

        if index in self.cache:

            return

        if index in self.jobs:

            return

        self.jobs[
            index
        ] = self.executor.submit(
            self._load,
            index
        )

    # --------------------------------------------------------
    # PREFETCH
    # --------------------------------------------------------

    def prefetch(
        self,
        start_index
    ):

        if self.closed:

            return

        end_index = min(
            len(self.frames),
            start_index
            +
            self.lookahead
        )

        for index in range(
            start_index,
            end_index
        ):

            self.submit(
                index
            )

    # --------------------------------------------------------
    # GET
    # --------------------------------------------------------

    def get(
        self,
        index
    ):

        if self.closed:

            raise RuntimeError(
                "FramePrefetcher is closed."
            )

        # 已经准备好的缓存帧

        if index in self.cache:

            data = self.cache.pop(
                index
            )

        else:

            # 确保当前帧至少已经提交

            self.submit(
                index
            )

            future = self.jobs.pop(
                index
            )

            # 如果后台线程已经读完，
            # result() 几乎立即返回。
            #
            # 如果没读完，
            # 这里才会真正等待 I/O。

            data = future.result()

        # 当前帧拿到以后继续准备未来帧

        self.prefetch(
            index + 1
        )

        return data

    # --------------------------------------------------------
    # CLOSE
    # --------------------------------------------------------

    def close(
        self
    ):

        if self.closed:

            return

        self.closed = True

        for future in self.jobs.values():

            future.cancel()

        self.jobs.clear()
        self.cache.clear()

        try:

            self.executor.shutdown(
                wait=False,
                cancel_futures=True
            )

        except TypeError:

            # 兼容较老 Python

            self.executor.shutdown(
                wait=False
            )


# ============================================================
# MAIN
# ============================================================

def main():

    # ========================================================
    # ARGUMENTS
    # ========================================================

    args = parse_args()

    validate_args(
        args
    )

    # ========================================================
    # SCENE
    # ========================================================

    scene_dir = resolve_scene(
        args.scene
    )

    frames = find_frames(
        scene_dir
    )

    if args.start >= len(
        frames
    ):

        raise ValueError(
            f"--start={args.start}, "
            f"but only {len(frames)} "
            f"frames exist."
        )

    frames = frames[
        args.start:
    ]

    # ========================================================
    # INFORMATION
    # ========================================================

    print()
    print(
        "========================================"
    )

    print(
        "Dataset Player"
    )

    print(
        "========================================"
    )

    print(
        "Scene:",
        scene_dir
    )

    print(
        "Frames:",
        len(frames)
    )

    print(
        "Target FPS:",
        args.fps
    )

    print(
        "Point display limit:",
        args.max_points
    )

    print(
        "Prefetch:",
        args.prefetch
    )

    print(
        "I/O workers:",
        args.io_workers
    )

    print()
    print(
        "Controls:"
    )

    print(
        "SPACE : pause / resume"
    )

    print(
        "Q     : quit"
    )

    print(
        "ESC   : quit"
    )

    print()

    # ========================================================
    # PREFETCHER
    # ========================================================

    prefetcher = FramePrefetcher(
        frames=frames,
        max_points=args.max_points,
        lookahead=args.prefetch,
        workers=args.io_workers
    )

    # ========================================================
    # LOAD FIRST FRAME
    # ========================================================

    first_load_start = (
        time.perf_counter()
    )

    first_data = load_frame_data(
        frames[0],
        args.max_points
    )

    first_load_ms = (
        (
            time.perf_counter()
            -
            first_load_start
        )
        *
        1000.0
    )

    (
        first_rgb,
        first_cloud_count,
        first_xyz
    ) = first_data

    # 第一帧已经加载过，
    # 放到 prefetcher cache，
    # 防止 FuncAnimation update(0)
    # 再读一次磁盘。

    prefetcher.prime(
        0,
        first_data
    )

    # 马上在后台读取后面的帧。

    prefetcher.prefetch(
        1
    )

    # ========================================================
    # FIGURE
    # ========================================================

    fig = plt.figure(
        figsize=(
            16,
            8
        )
    )

    try:

        fig.canvas.manager.set_window_title(
            "UAV RGB + LiDAR Dataset Player"
        )

    except (
        AttributeError,
        NotImplementedError
    ):

        pass

    # ========================================================
    # LEFT: RGB
    # ========================================================

    ax_rgb = fig.add_subplot(
        1,
        2,
        1
    )

    # interpolation="nearest"
    #
    # 对数据播放器来说通常足够，
    # 同时可以减少 RGB 重采样开销。

    image_artist = ax_rgb.imshow(
        first_rgb,
        interpolation="nearest"
    )

    ax_rgb.axis(
        "off"
    )

    rgb_title = ax_rgb.set_title(
        (
            f"RGB | Frame "
            f"{frames[0]['id']}"
        )
    )

    # ========================================================
    # RIGHT: LiDAR
    # ========================================================

    ax_pc = fig.add_subplot(
        1,
        2,
        2,
        projection="3d"
    )

    R = float(
        args.range
    )

    # --------------------------------------------------------
    # CARLA LiDAR 坐标保持原样：
    #
    # X = forward
    # Y = right
    # Z = up
    #
    # 不做坐标转换。
    # --------------------------------------------------------

    x = first_xyz[
        :,
        0
    ]

    y = first_xyz[
        :,
        1
    ]

    z = first_xyz[
        :,
        2
    ]

    # ========================================================
    # CLOUD
    # ========================================================

    cloud_artist = ax_pc.scatter(
        x,
        y,
        z,

        c=z,

        s=args.point_size,

        marker=".",

        cmap="viridis",

        vmin=-R,
        vmax=R,

        depthshade=False,

        edgecolors="none",

        linewidths=0,

        antialiaseds=False
    )

    # ========================================================
    # ORIGIN
    # ========================================================

    ax_pc.scatter(
        [0.0],
        [0.0],
        [0.0],

        marker="x",

        s=80,

        depthshade=False
    )

    # ========================================================
    # FIXED AXES
    #
    # 固定 [-R, +R]。
    #
    # 非常重要：
    # 不允许 Matplotlib 每帧重新 autoscale。
    # ========================================================

    ax_pc.set_xlim(
        -R,
        R
    )

    ax_pc.set_ylim(
        -R,
        R
    )

    ax_pc.set_zlim(
        -R,
        R
    )

    ax_pc.set_autoscale_on(
        False
    )

    ax_pc.set_box_aspect(
        (
            1,
            1,
            1
        )
    )

    # ========================================================
    # REMOVE XYZ DASHED GRID
    # ========================================================

    ax_pc.grid(
        False
    )

    # 某些 Matplotlib 版本 / backend 下，
    # 3D grid(False) 偶尔仍然可能看到残留网格。
    #
    # 下面作为额外保险。
    #
    # _axinfo 是 Matplotlib mplot3d 内部属性，
    # 所以放在 try 中。

    try:

        for axis in (
            ax_pc.xaxis,
            ax_pc.yaxis,
            ax_pc.zaxis
        ):

            axis._axinfo[
                "grid"
            ][
                "linewidth"
            ] = 0.0

    except (
        AttributeError,
        KeyError,
        TypeError
    ):

        pass

    # ========================================================
    # PROJECTION
    # ========================================================

    # 正交投影对于这种点云播放器
    # 通常视觉上更稳定。

    try:

        ax_pc.set_proj_type(
            "ortho"
        )

    except AttributeError:

        pass

    # ========================================================
    # VIEW
    # ========================================================

    ax_pc.view_init(
        elev=25,
        azim=-60
    )

    # ========================================================
    # LABELS
    # ========================================================

    ax_pc.set_xlabel(
        "X forward [m]"
    )

    ax_pc.set_ylabel(
        "Y right [m]"
    )

    ax_pc.set_zlabel(
        "Z up [m]"
    )

    pc_title = ax_pc.set_title(
        (
            f"LiDAR | Frame "
            f"{frames[0]['id']} | "
            f"{first_cloud_count:,} points"
        )
    )

    # ========================================================
    # STATUS
    # ========================================================

    status_text = fig.text(
        0.5,
        0.02,
        (
            f"1/{len(frames)}"
            f"    |    "
            f"target: {args.fps:.1f} FPS"
            f"    |    "
            f"actual: 0.0 FPS"
            f"    |    "
            f"displayed: {len(first_xyz):,}"
            f"    |    "
            f"fetch: {first_load_ms:.1f} ms"
        ),
        ha="center"
    )

    # ========================================================
    # PLAYER STATE
    # ========================================================

    state = {

        "paused": False,

        # 上一帧 update() 的时间
        "last_frame_time": None,

        # 平滑后的真实 FPS
        "actual_fps": 0.0,

        # 当前帧等待后台数据的时间
        "fetch_ms": 0.0,

        # 用于避免关闭时重复 close
        "closed": False
    }

    # ========================================================
    # FPS CALCULATION
    # ========================================================

    def update_actual_fps():

        now = time.perf_counter()

        previous = state[
            "last_frame_time"
        ]

        state[
            "last_frame_time"
        ] = now

        if previous is None:

            return

        delta = (
            now
            -
            previous
        )

        if delta <= 0:

            return

        fps_now = (
            1.0
            /
            delta
        )

        old_fps = state[
            "actual_fps"
        ]

        # EMA 平滑：
        #
        # 当前值权重 15%
        # 历史值权重 85%
        #
        # 防止数字快速闪烁。

        if old_fps <= 0:

            state[
                "actual_fps"
            ] = fps_now

        else:

            state[
                "actual_fps"
            ] = (
                old_fps
                *
                0.85

                +

                fps_now
                *
                0.15
            )

    # ========================================================
    # UPDATE FRAME
    # ========================================================

    def update(index):

        frame = frames[
            index
        ]

        # ----------------------------------------------------
        # REAL FPS
        # ----------------------------------------------------

        update_actual_fps()

        # ----------------------------------------------------
        # GET PRELOADED DATA
        # ----------------------------------------------------

        fetch_start = (
            time.perf_counter()
        )

        (
            rgb,
            cloud_count,
            xyz
        ) = prefetcher.get(
            index
        )

        fetch_ms = (
            (
                time.perf_counter()
                -
                fetch_start
            )
            *
            1000.0
        )

        state[
            "fetch_ms"
        ] = fetch_ms

        # ----------------------------------------------------
        # RGB
        # ----------------------------------------------------

        image_artist.set_data(
            rgb
        )

        rgb_title.set_text(
            (
                f"RGB | Frame "
                f"{frame['id']}"
            )
        )

        # ----------------------------------------------------
        # LIDAR
        # ----------------------------------------------------

        x = xyz[
            :,
            0
        ]

        y = xyz[
            :,
            1
        ]

        z = xyz[
            :,
            2
        ]

        # Matplotlib mplot3d 并没有公开的高性能
        # scatter position update API。
        #
        # _offsets3d 是最常用的更新方式。

        cloud_artist._offsets3d = (
            x,
            y,
            z
        )

        # 根据当前 Z 重新着色

        cloud_artist.set_array(
            z
        )

        # ----------------------------------------------------
        # TITLE
        # ----------------------------------------------------

        pc_title.set_text(
            (
                f"LiDAR | Frame "
                f"{frame['id']} | "
                f"{cloud_count:,} points"
            )
        )

        # ----------------------------------------------------
        # STATUS
        # ----------------------------------------------------

        actual_fps = state[
            "actual_fps"
        ]

        status_text.set_text(
            (
                f"{index + 1}/{len(frames)}"

                f"    |    "

                f"target: "
                f"{args.fps:.1f} FPS"

                f"    |    "

                f"actual: "
                f"{actual_fps:.1f} FPS"

                f"    |    "

                f"displayed: "
                f"{len(xyz):,}"

                f"    |    "

                f"fetch: "
                f"{fetch_ms:.1f} ms"
            )
        )

        return (
            image_artist,
            cloud_artist,
            rgb_title,
            pc_title,
            status_text
        )

    # ========================================================
    # ANIMATION
    # ========================================================

    interval_ms = (
        1000.0
        /
        args.fps
    )

    animation = FuncAnimation(
        fig,
        update,

        frames=len(
            frames
        ),

        interval=interval_ms,

        repeat=args.loop,

        cache_frame_data=False,

        # 3D scatter 不适合使用 blit。
        blit=False
    )

    # ========================================================
    # CLOSE
    # ========================================================

    def close_player():

        if state[
            "closed"
        ]:

            return

        state[
            "closed"
        ] = True

        prefetcher.close()

    # ========================================================
    # KEYBOARD
    # ========================================================

    def on_key(event):

        key = (
            event.key
            or ""
        ).lower()

        # ----------------------------------------------------
        # QUIT
        # ----------------------------------------------------

        if key in (
            "q",
            "escape"
        ):

            close_player()

            plt.close(
                fig
            )

            return

        # ----------------------------------------------------
        # PAUSE / RESUME
        # ----------------------------------------------------

        if key == " ":

            if state[
                "paused"
            ]:

                animation.resume()

                state[
                    "paused"
                ] = False

                # 暂停期间经过了很久，
                # 所以恢复以后重置 FPS 计时，
                # 避免把 pause 时间算进 FPS。

                state[
                    "last_frame_time"
                ] = None

                print(
                    "Playback resumed."
                )

            else:

                animation.pause()

                state[
                    "paused"
                ] = True

                print(
                    "Playback paused."
                )

    # ========================================================
    # WINDOW CLOSE EVENT
    # ========================================================

    def on_close(event):

        close_player()

    # ========================================================
    # EVENTS
    # ========================================================

    fig.canvas.mpl_connect(
        "key_press_event",
        on_key
    )

    fig.canvas.mpl_connect(
        "close_event",
        on_close
    )

    # ========================================================
    # LAYOUT
    # ========================================================

    plt.tight_layout(
        rect=[
            0.0,
            0.05,
            1.0,
            1.0
        ]
    )

    # ========================================================
    # SHOW
    # ========================================================

    try:

        plt.show()

    finally:

        close_player()


# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":

    main()
