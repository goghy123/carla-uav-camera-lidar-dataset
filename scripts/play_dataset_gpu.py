from pathlib import Path
from dataclasses import dataclass
import argparse
import math
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

from vispy import use


# python scripts/play_dataset_gpu.py `
#     --scene dataset/scene_20260815_124421 `
#     --fps 15 `
#     --max-points 0 `
#     --prefetch 12 `
#     --io-workers 2


# Force VisPy to use Qt + desktop OpenGL.
use(app="PyQt6", gl="gl2")

from vispy import scene
from vispy.app import use_app
from vispy.color import get_colormap

from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtWidgets import QMainWindow


# ============================================================
# PROJECT
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="GPU player for synchronized RGB and CARLA LiDAR frames."
    )
    parser.add_argument(
        "--scene",
        type=str,
        required=True,
        help="Dataset scene path, e.g. dataset/scene_20260814_222401",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=15.0,
        help="Target playback FPS. Default: 15",
    )
    parser.add_argument(
        "--range",
        type=float,
        default=80.0,
        help="LiDAR visualization range in meters. Default: 80",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=0,
        help="Maximum displayed LiDAR points. 0 = all visible points.",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=2.0,
        help="GPU point size in screen pixels. Default: 2.0",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start from this synchronized sample index.",
    )
    parser.add_argument(
        "--prefetch",
        type=int,
        default=8,
        help="Number of future frames to preload. Default: 8",
    )
    parser.add_argument(
        "--io-workers",
        type=int,
        default=2,
        help="Background load/decode workers. Default: 2",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Loop playback.",
    )
    parser.add_argument(
        "--vsync",
        action="store_true",
        help="Enable OpenGL VSync. Normally leave disabled for dataset playback.",
    )
    return parser.parse_args()


def validate_args(args):
    if args.fps <= 0:
        raise ValueError("--fps must be > 0")
    if args.range <= 0:
        raise ValueError("--range must be > 0")
    if args.max_points < 0:
        raise ValueError("--max-points must be >= 0")
    if args.point_size <= 0:
        raise ValueError("--point-size must be > 0")
    if args.start < 0:
        raise ValueError("--start must be >= 0")
    if args.prefetch < 1:
        raise ValueError("--prefetch must be >= 1")
    if args.io_workers < 1:
        raise ValueError("--io-workers must be >= 1")


# ============================================================
# DATASET DISCOVERY
# ============================================================

def resolve_scene(scene_arg):
    path = Path(scene_arg)
    if not path.is_absolute():
        path = PROJECT_DIR / path
    path = path.resolve()

    if not path.exists():
        raise FileNotFoundError(f"\nScene does not exist:\n{path}")
    if not path.is_dir():
        raise NotADirectoryError(f"\nScene is not a directory:\n{path}")

    return path


def find_frames(scene_dir):
    rgb_dir = scene_dir / "rgb"
    lidar_dir = scene_dir / "lidar"

    if not rgb_dir.exists():
        raise FileNotFoundError(f"RGB directory missing:\n{rgb_dir}")
    if not lidar_dir.exists():
        raise FileNotFoundError(f"LiDAR directory missing:\n{lidar_dir}")

    rgb_files = {p.stem: p for p in rgb_dir.glob("*.png")}
    lidar_files = {p.stem: p for p in lidar_dir.glob("*.bin")}

    common = set(rgb_files) & set(lidar_files)
    if not common:
        raise RuntimeError("No synchronized RGB/LiDAR frame names found.")

    def sort_key(name):
        try:
            return 0, int(name)
        except ValueError:
            return 1, name

    frame_ids = sorted(common, key=sort_key)

    return [
        {
            "id": frame_id,
            "rgb": rgb_files[frame_id],
            "lidar": lidar_files[frame_id],
        }
        for frame_id in frame_ids
    ]


# ============================================================
# LOAD / PREPROCESS
# ============================================================

def load_rgb(path):
    # PNG decode is intentionally kept on CPU and hidden behind prefetch.
    with Image.open(path) as image:
        image = image.convert("RGB")
        return np.array(image, dtype=np.uint8, copy=True)


def load_lidar(path):
    cloud = np.fromfile(path, dtype=np.float32)

    if cloud.size % 4 != 0:
        raise RuntimeError(
            f"Invalid XYZI BIN file:\n"
            f"{path}\n"
            f"float count = {cloud.size}"
        )

    return cloud.reshape(-1, 4)


def display_sample(points, max_points):
    # Display-only stride sampling. Original BIN is never modified.
    if max_points <= 0 or len(points) <= max_points:
        return points

    step = math.ceil(len(points) / max_points)
    return points[::step]


def build_viridis_lut():
    # A 256-entry LUT avoids running a full colormap interpolation every frame.
    values = np.linspace(0.0, 1.0, 256, dtype=np.float32)
    lut = np.asarray(
        get_colormap("viridis").map(values),
        dtype=np.float32,
    ).reshape(256, 4)
    lut[:, 3] = 1.0
    return np.ascontiguousarray(lut)


@dataclass(slots=True)
class FrameData:
    rgb: np.ndarray
    xyz: np.ndarray
    colors: np.ndarray
    original_count: int
    visible_count: int


def load_frame_data(frame, max_points, visual_range, color_lut):
    rgb = load_rgb(frame["rgb"])
    cloud = load_lidar(frame["lidar"])

    original_count = len(cloud)
    xyz = cloud[:, :3]

    # Remove NaN/Inf before rendering.
    finite_mask = np.isfinite(xyz).all(axis=1)
    if not finite_mask.all():
        xyz = xyz[finite_mask]

    # Anything outside the visible cube cannot appear on screen.
    # Removing it here reduces CPU -> GPU traffic.
    R = float(visual_range)
    mask = (
        (xyz[:, 0] >= -R)
        & (xyz[:, 0] <= R)
        & (xyz[:, 1] >= -R)
        & (xyz[:, 1] <= R)
        & (xyz[:, 2] >= -R)
        & (xyz[:, 2] <= R)
    )
    xyz = xyz[mask]

    visible_count = len(xyz)
    xyz = display_sample(xyz, max_points)
    xyz = np.ascontiguousarray(xyz, dtype=np.float32)

    # Z-height -> viridis. This work happens in background prefetch threads.
    if len(xyz):
        normalized_z = (xyz[:, 2] + R) / (2.0 * R)
        indices = np.clip(normalized_z * 255.0, 0.0, 255.0).astype(np.uint8)
        colors = np.ascontiguousarray(color_lut[indices], dtype=np.float32)
    else:
        colors = np.empty((0, 4), dtype=np.float32)

    return FrameData(
        rgb=rgb,
        xyz=xyz,
        colors=colors,
        original_count=original_count,
        visible_count=visible_count,
    )


# ============================================================
# BACKGROUND PREFETCH
# ============================================================

class FramePrefetcher:
    def __init__(
        self,
        frames,
        max_points,
        visual_range,
        color_lut,
        lookahead=8,
        workers=2,
        loop=False,
    ):
        self.frames = frames
        self.max_points = max_points
        self.visual_range = visual_range
        self.color_lut = color_lut
        self.lookahead = max(1, int(lookahead))
        self.loop = bool(loop)

        self.executor = ThreadPoolExecutor(
            max_workers=max(1, int(workers)),
            thread_name_prefix="dataset-loader",
        )
        self.jobs = {}
        self.closed = False

    def _load(self, index):
        return load_frame_data(
            self.frames[index],
            self.max_points,
            self.visual_range,
            self.color_lut,
        )

    def submit(self, index):
        if self.closed:
            return
        if not 0 <= index < len(self.frames):
            return
        if index in self.jobs:
            return

        self.jobs[index] = self.executor.submit(self._load, index)

    def prefetch(self, start_index):
        if self.closed or not self.frames:
            return

        frame_count = len(self.frames)
        count = min(self.lookahead, frame_count)

        for offset in range(count):
            index = start_index + offset

            if self.loop:
                index %= frame_count
            elif index >= frame_count:
                break

            self.submit(index)

    def get(self, index):
        if self.closed:
            raise RuntimeError("FramePrefetcher is closed.")

        self.submit(index)
        future = self.jobs.pop(index)
        data = future.result()

        self.prefetch(index + 1)
        return data

    def close(self):
        if self.closed:
            return

        self.closed = True

        for future in self.jobs.values():
            future.cancel()

        self.jobs.clear()

        try:
            self.executor.shutdown(wait=False, cancel_futures=True)
        except TypeError:
            self.executor.shutdown(wait=False)


# ============================================================
# PLAYER WINDOW
# ============================================================

class DatasetPlayerWindow(QMainWindow):
    def __init__(self, args, frames):
        super().__init__()

        self.args = args
        self.frames = frames

        self.current_index = 0
        self.paused = False
        self.ended = False
        self.closed = False

        self.actual_fps = 0.0
        self.last_present_time = None
        self.frame_pending_draw = False

        self.last_fetch_ms = 0.0
        self.last_submit_ms = 0.0

        self.color_lut = build_viridis_lut()

        # First frame is loaded synchronously so the window opens with content.
        t0 = time.perf_counter()
        self.current_data = load_frame_data(
            self.frames[0],
            self.args.max_points,
            self.args.range,
            self.color_lut,
        )
        self.last_fetch_ms = (time.perf_counter() - t0) * 1000.0

        # Future frames load/decode in parallel with rendering.
        self.prefetcher = FramePrefetcher(
            frames=self.frames,
            max_points=self.args.max_points,
            visual_range=self.args.range,
            color_lut=self.color_lut,
            lookahead=self.args.prefetch,
            workers=self.args.io_workers,
            loop=self.args.loop,
        )
        self.prefetcher.prefetch(1)

        self.setWindowTitle("UAV RGB + LiDAR Dataset Player - VisPy GPU")
        self.resize(1600, 850)

        # One OpenGL canvas contains both RGB and LiDAR views.
        self.canvas = scene.SceneCanvas(
            title="UAV RGB + LiDAR Dataset Player",
            size=(1600, 850),
            bgcolor="#111318",
            keys=None,
            show=False,
            create_native=True,
            vsync=self.args.vsync,
        )
        self.setCentralWidget(self.canvas.native)
        self.canvas.events.draw.connect(self.on_canvas_draw)

        grid = self.canvas.central_widget.add_grid(margin=6, spacing=6)

        # ----------------------- Titles -----------------------

        self.rgb_title = scene.Label(
            f"RGB | Frame {self.frames[0]['id']}",
            color="white",
            font_size=12,
        )
        self.rgb_title.height_max = 34
        grid.add_widget(self.rgb_title, row=0, col=0)

        self.pc_title = scene.Label(
            self.make_pc_title(0, self.current_data),
            color="white",
            font_size=12,
        )
        self.pc_title.height_max = 34
        grid.add_widget(self.pc_title, row=0, col=1)

        # ----------------------- RGB --------------------------

        self.rgb_view = grid.add_view(
            row=1,
            col=0,
            border_color="#3a3f4b",
            bgcolor="black",
        )

        self.rgb_image = scene.visuals.Image(
            self.current_data.rgb,
            interpolation="nearest",
            method="subdivide",
            texture_format="auto",
            parent=self.rgb_view.scene,
        )

        self.rgb_camera = scene.PanZoomCamera(aspect=1)
        self.rgb_camera.flip = (0, 1, 0)
        self.rgb_view.camera = self.rgb_camera

        self.last_rgb_shape = self.current_data.rgb.shape
        self.reset_rgb_camera()

        # ----------------------- LiDAR ------------------------

        self.pc_view = grid.add_view(
            row=1,
            col=1,
            border_color="#3a3f4b",
            bgcolor="#080a0d",
        )

        # fov=0 gives an orthographic view similar to the old Matplotlib player.
        self.pc_camera = scene.TurntableCamera(
            fov=0.0,
            elevation=25.0,
            azimuth=-60.0,
            roll=0.0,
            center=(0.0, 0.0, 0.0),
            up="+z",
        )
        self.pc_view.camera = self.pc_camera

        # GL_POINTS fast path: positions/colors are uploaded to GPU buffers.
        self.cloud_visual = scene.visuals.Markers(
            method="points",
            scaling="fixed",
            antialias=0,
            spherical=False,
            parent=self.pc_view.scene,
        )
        self.cloud_visual.set_data(
            pos=self.current_data.xyz,
            face_color=self.current_data.colors,
            edge_width=0,
            size=self.args.point_size,
            symbol="disc",
        )

        # Fixed LiDAR origin.
        self.origin_visual = scene.visuals.Markers(
            method="points",
            scaling="fixed",
            antialias=0,
            spherical=False,
            parent=self.pc_view.scene,
        )
        self.origin_visual.set_data(
            pos=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
            face_color="white",
            edge_color="white",
            edge_width=0,
            size=10.0,
            symbol="x",
        )

        # Small XYZ orientation indicator. No 3D grid lines are created.
        self.xyz_axis = scene.visuals.XYZAxis(parent=self.pc_view.scene)
        axis_size = float(self.args.range) * 0.25
        self.xyz_axis.transform = scene.transforms.STTransform(
            scale=(axis_size, axis_size, axis_size)
        )

        self.reset_pc_camera()

        # ----------------------- Status -----------------------

        self.status_label = scene.Label(
            "",
            color="#d8dbe2",
            font_size=10,
        )
        self.status_label.height_max = 32
        grid.add_widget(self.status_label, row=2, col=0, col_span=2)
        self.update_status_text()

        # ----------------------- Controls ---------------------

        self.shortcuts = []
        self.add_shortcut("Space", self.toggle_pause)
        self.add_shortcut("R", self.reset_pc_camera)
        self.add_shortcut("Q", self.close)
        self.add_shortcut("Esc", self.close)

        # ----------------------- Timer ------------------------

        self.timer = QTimer(self)
        self.timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.timer.timeout.connect(self.advance_frame)

        interval_ms = max(1, int(round(1000.0 / self.args.fps)))
        self.timer.setInterval(interval_ms)

        # Track the first real OpenGL presentation.
        self.frame_pending_draw = True
        self.timer.start()

    # ========================================================
    # UI HELPERS
    # ========================================================

    def add_shortcut(self, sequence, callback):
        shortcut = QShortcut(QKeySequence(sequence), self)
        shortcut.activated.connect(callback)
        self.shortcuts.append(shortcut)

    def make_pc_title(self, index, data):
        return (
            f"LiDAR | Frame {self.frames[index]['id']} | "
            f"{data.original_count:,} raw | "
            f"{len(data.xyz):,} displayed"
        )

    def update_status_text(self):
        state = "PAUSED" if self.paused else "PLAYING"
        if self.ended:
            state = "END"

        self.status_label.text = (
            f"{self.current_index + 1}/{len(self.frames)}"
            f"    |    {state}"
            f"    |    target: {self.args.fps:.1f} FPS"
            f"    |    actual draw: {self.actual_fps:.1f} FPS"
            f"    |    fetch wait: {self.last_fetch_ms:.2f} ms"
            f"    |    CPU submit: {self.last_submit_ms:.2f} ms"
            f"    |    SPACE pause/resume"
            f"    |    R reset view"
            f"    |    Q/ESC quit"
        )

    # ========================================================
    # CAMERAS
    # ========================================================

    def reset_rgb_camera(self):
        height, width = self.current_data.rgb.shape[:2]
        self.rgb_camera.set_range(
            x=(0, width),
            y=(0, height),
            margin=0.0,
        )

    def reset_pc_camera(self):
        R = float(self.args.range)

        self.pc_camera.fov = 0.0
        self.pc_camera.elevation = 25.0
        self.pc_camera.azimuth = -60.0
        self.pc_camera.roll = 0.0
        self.pc_camera.center = (0.0, 0.0, 0.0)

        self.pc_camera.set_range(
            x=(-R, R),
            y=(-R, R),
            z=(-R, R),
            margin=0.02,
        )
        self.canvas.update()

    # ========================================================
    # FPS MEASUREMENT
    # ========================================================

    def on_canvas_draw(self, event):
        # Ignore redraws caused only by mouse camera movement.
        if not self.frame_pending_draw:
            return

        now = time.perf_counter()
        previous = self.last_present_time

        self.last_present_time = now
        self.frame_pending_draw = False

        if previous is None:
            return

        dt = now - previous
        if dt <= 0:
            return

        fps_now = 1.0 / dt

        # Exponential moving average to make the displayed value readable.
        if self.actual_fps <= 0:
            self.actual_fps = fps_now
        else:
            self.actual_fps = self.actual_fps * 0.85 + fps_now * 0.15

    # ========================================================
    # PLAYBACK
    # ========================================================

    def advance_frame(self):
        if self.paused or self.closed:
            return

        next_index = self.current_index + 1

        if next_index >= len(self.frames):
            if self.args.loop:
                next_index = 0
            else:
                self.ended = True
                self.timer.stop()
                self.update_status_text()
                self.canvas.update()
                return

        # If prefetch kept up, this should normally be close to zero.
        t0 = time.perf_counter()
        data = self.prefetcher.get(next_index)
        self.last_fetch_ms = (time.perf_counter() - t0) * 1000.0

        # CPU-side VisPy submission. Actual GL work completes during draw.
        t0 = time.perf_counter()

        self.current_index = next_index
        self.current_data = data

        self.rgb_image.set_data(data.rgb)

        if data.rgb.shape != self.last_rgb_shape:
            self.last_rgb_shape = data.rgb.shape
            self.reset_rgb_camera()

        self.rgb_title.text = f"RGB | Frame {self.frames[next_index]['id']}"

        self.cloud_visual.set_data(
            pos=data.xyz,
            face_color=data.colors,
            edge_width=0,
            size=self.args.point_size,
            symbol="disc",
        )

        self.pc_title.text = self.make_pc_title(next_index, data)

        self.last_submit_ms = (time.perf_counter() - t0) * 1000.0

        self.frame_pending_draw = True
        self.update_status_text()
        self.canvas.update()

    def toggle_pause(self):
        if self.ended:
            return

        if self.paused:
            self.paused = False
            self.last_present_time = None
            self.frame_pending_draw = False
            self.timer.start()
            print("Playback resumed.")
        else:
            self.paused = True
            self.timer.stop()
            print("Playback paused.")

        self.update_status_text()
        self.canvas.update()

    # ========================================================
    # CLOSE
    # ========================================================

    def closeEvent(self, event):
        if not self.closed:
            self.closed = True
            self.timer.stop()
            self.prefetcher.close()
            self.canvas.close()

        event.accept()


# ============================================================
# MAIN
# ============================================================

def main():
    args = parse_args()
    validate_args(args)

    scene_dir = resolve_scene(args.scene)
    frames = find_frames(scene_dir)

    if args.start >= len(frames):
        raise ValueError(
            f"--start={args.start}, but only "
            f"{len(frames)} synchronized frames exist."
        )

    frames = frames[args.start:]

    print()
    print("========================================")
    print("VisPy GPU Dataset Player")
    print("========================================")
    print("Scene:", scene_dir)
    print("Frames:", len(frames))
    print("Target FPS:", args.fps)
    print("LiDAR range:", args.range)
    print("Point display limit:", args.max_points)
    print("Prefetch:", args.prefetch)
    print("I/O workers:", args.io_workers)
    print("VSync:", args.vsync)
    print()
    print("Controls:")
    print("SPACE : pause / resume")
    print("R     : reset LiDAR camera")
    print("Q     : quit")
    print("ESC   : quit")
    print()

    # VisPy owns/creates the Qt application so there is only one QApplication.
    app = use_app()
    app.create()

    window = DatasetPlayerWindow(
        args=args,
        frames=frames,
    )
    window.show()

    app.run()


if __name__ == "__main__":
    main()
