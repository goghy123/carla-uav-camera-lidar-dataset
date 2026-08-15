from pathlib import Path
from dataclasses import dataclass
import argparse
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image
from vispy import use

use(app="PyQt6", gl="gl2")

from vispy import scene
from vispy.app import use_app
from vispy.color import get_colormap
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtWidgets import QMainWindow


# python scripts/play_uavdataset.py `
#     --scene dataset\scene_20260815_184328 `
#     --fps 15 `
#     --range 80 `
#     --max-points 0 `
#     --prefetch 12 `
#     --io-workers 2


# ============================================================
# PROJECT
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent


# ============================================================
# DETECTION CLASSES / COLORS
# ============================================================

CLASSES = (
    "car",
    "van",
    "truck",
    "bus",
    "motorcycle",
    "bicycle",
    "pedestrian",
)

CLASS_COLORS = {
    "car": (0.20, 0.55, 1.00, 1.00),
    "van": (0.10, 0.90, 0.90, 1.00),
    "truck": (1.00, 0.55, 0.10, 1.00),
    "bus": (1.00, 0.85, 0.10, 1.00),
    "motorcycle": (0.95, 0.25, 0.95, 1.00),
    "bicycle": (0.20, 0.95, 0.35, 1.00),
    "pedestrian": (1.00, 0.25, 0.25, 1.00),
}

BBOX_EDGES = (
    (0, 1), (1, 3), (3, 2), (2, 0),
    (0, 4), (4, 5), (5, 1), (5, 7),
    (7, 6), (6, 4), (6, 2), (7, 3),
)

BBOX_MODES = ("visible", "projected", "off")
POINT_COLOR_MODES = ("rgb", "height")


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "GPU player for UAV RGB + LiDAR + 2D/3D labels, "
            "with RGB-colorized point cloud."
        )
    )

    parser.add_argument(
        "--scene",
        type=str,
        required=True,
        help="Dataset scene path.",
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
        help=(
            "LiDAR visualization cube half-range in meters. "
            "3D boxes on the right must intersect this cube. Default: 80"
        ),
    )

    parser.add_argument(
        "--max-points",
        type=int,
        default=0,
        help="Maximum displayed LiDAR points. 0 = all.",
    )

    parser.add_argument(
        "--point-size",
        type=float,
        default=2.0,
        help="GPU point size. Default: 2.0",
    )

    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start sample index.",
    )

    parser.add_argument(
        "--prefetch",
        type=int,
        default=8,
        help="Frames to preload. Default: 8",
    )

    parser.add_argument(
        "--io-workers",
        type=int,
        default=2,
        help="Background I/O workers. Default: 2",
    )

    parser.add_argument(
        "--loop",
        action="store_true",
        help="Loop playback.",
    )

    parser.add_argument(
        "--vsync",
        action="store_true",
        help="Enable OpenGL VSync.",
    )

    parser.add_argument(
        "--classes",
        nargs="+",
        default=["all"],
        help="all, or e.g. --classes car truck bus",
    )

    parser.add_argument(
        "--min-lidar-points",
        type=int,
        default=1,
        help=(
            "Only display objects containing at least this many "
            "LiDAR points. Default: 1"
        ),
    )

    parser.add_argument(
        "--max-object-distance",
        type=float,
        default=120.0,
        help=(
            "Maximum saved bbox-center distance for displayed objects. "
            "Default: 120 m"
        ),
    )

    parser.add_argument(
        "--frustum-depth",
        type=float,
        default=20.0,
        help="Camera frustum visualization depth in meters. Default: 20",
    )

    return parser.parse_args()


def normalize_classes(values):
    tokens = []

    for value in values:
        tokens.extend(
            token.strip().lower()
            for token in value.split(",")
            if token.strip()
        )

    if not tokens or "all" in tokens:
        return tuple(CLASSES)

    unknown = sorted(
        set(tokens) - set(CLASSES)
    )

    if unknown:
        raise ValueError(
            f"Unknown classes: {unknown}. "
            f"Supported: {', '.join(CLASSES)}"
        )

    selected = set(tokens)

    return tuple(
        cls
        for cls in CLASSES
        if cls in selected
    )


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

    if args.min_lidar_points < 0:
        raise ValueError("--min-lidar-points must be >= 0")

    if args.max_object_distance <= 0:
        raise ValueError("--max-object-distance must be > 0")

    if args.frustum_depth <= 0:
        raise ValueError("--frustum-depth must be > 0")

    args.classes = normalize_classes(
        args.classes
    )


# ============================================================
# DATASET DISCOVERY
# ============================================================

def resolve_scene(scene_arg):
    path = Path(scene_arg)

    if not path.is_absolute():
        path = PROJECT_DIR / path

    path = path.resolve()

    if not path.exists():
        raise FileNotFoundError(
            f"Scene does not exist:\n{path}"
        )

    if not path.is_dir():
        raise NotADirectoryError(
            f"Scene is not a directory:\n{path}"
        )

    return path


def load_calibration(scene_dir):
    path = scene_dir / "calibration.json"

    if not path.exists():
        raise FileNotFoundError(
            f"Missing calibration.json:\n{path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    K = np.asarray(
        data["K"],
        dtype=np.float64,
    )

    if K.shape != (3, 3):
        raise ValueError(
            f"Invalid K shape: {K.shape}"
        )

    T_camera_cv_from_lidar = np.asarray(
        data["T_camera_cv_from_lidar"],
        dtype=np.float64,
    )

    if T_camera_cv_from_lidar.shape != (4, 4):
        raise ValueError(
            "Invalid T_camera_cv_from_lidar shape: "
            f"{T_camera_cv_from_lidar.shape}"
        )

    resolution = data.get(
        "camera_resolution",
        None,
    )

    if (
        not isinstance(resolution, list)
        or len(resolution) != 2
    ):
        raise ValueError(
            "calibration.json must contain "
            "camera_resolution: [width, height]"
        )

    image_width = int(resolution[0])
    image_height = int(resolution[1])

    return (
        data,
        K,
        T_camera_cv_from_lidar,
        image_width,
        image_height,
    )


def find_frames(scene_dir):
    rgb_dir = scene_dir / "rgb"
    lidar_dir = scene_dir / "lidar"
    label_dir = scene_dir / "labels"

    for path, name in (
        (rgb_dir, "rgb"),
        (lidar_dir, "lidar"),
        (label_dir, "labels"),
    ):
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {name} directory:\n{path}"
            )

    rgb = {
        p.stem: p
        for p in rgb_dir.glob("*.png")
    }

    lidar = {
        p.stem: p
        for p in lidar_dir.glob("*.bin")
    }

    labels = {
        p.stem: p
        for p in label_dir.glob("*.json")
    }

    common = (
        set(rgb)
        & set(lidar)
        & set(labels)
    )

    if not common:
        raise RuntimeError(
            "No synchronized RGB/LiDAR/label frames found."
        )

    def sort_key(name):
        try:
            return 0, int(name)
        except ValueError:
            return 1, name

    ids = sorted(
        common,
        key=sort_key,
    )

    if not (
        len(rgb)
        == len(lidar)
        == len(labels)
        == len(common)
    ):
        print(
            "WARNING: RGB/LiDAR/labels counts are not identical."
        )
        print("  RGB         :", len(rgb))
        print("  LiDAR       :", len(lidar))
        print("  labels      :", len(labels))
        print("  synchronized:", len(common))

    return [
        {
            "id": frame_id,
            "rgb": rgb[frame_id],
            "lidar": lidar[frame_id],
            "label": labels[frame_id],
        }
        for frame_id in ids
    ]


# ============================================================
# BASIC LOAD
# ============================================================

def load_rgb(path):
    with Image.open(path) as image:
        return np.array(
            image.convert("RGB"),
            dtype=np.uint8,
            copy=True,
        )


def load_lidar(path):
    cloud = np.fromfile(
        path,
        dtype=np.float32,
    )

    if cloud.size % 4 != 0:
        raise RuntimeError(
            f"Invalid XYZI BIN:\n"
            f"{path}\n"
            f"float count={cloud.size}"
        )

    return cloud.reshape(-1, 4)


def load_json(path):
    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


# ============================================================
# POINT CLOUD PREPROCESS
# ============================================================

def display_sample(
    points,
    max_points,
):
    if (
        max_points <= 0
        or len(points) <= max_points
    ):
        return points

    step = math.ceil(
        len(points) / max_points
    )

    return points[::step]


def build_viridis_lut():
    values = np.linspace(
        0.0,
        1.0,
        256,
        dtype=np.float32,
    )

    lut = np.asarray(
        get_colormap("viridis").map(values),
        dtype=np.float32,
    ).reshape(256, 4)

    lut[:, 3] = 1.0

    return np.ascontiguousarray(
        lut
    )


def make_height_colors(
    xyz,
    visual_range,
    color_lut,
):
    if len(xyz) == 0:
        return np.empty(
            (0, 4),
            dtype=np.float32,
        )

    R = float(
        visual_range
    )

    normalized_z = (
        xyz[:, 2] + R
    ) / (2.0 * R)

    indices = np.clip(
        normalized_z * 255.0,
        0.0,
        255.0,
    ).astype(
        np.uint8
    )

    return np.ascontiguousarray(
        color_lut[indices],
        dtype=np.float32,
    )


def rgb_colorize_lidar_points(
    xyz_lidar,
    rgb,
    K,
    T_camera_cv_from_lidar,
):
    """
    Project displayed LiDAR points into the synchronized RGB image.

    Points inside camera FOV:
        RGB image color.

    Points outside camera FOV / behind camera:
        dark gray.
    """
    count = len(
        xyz_lidar
    )

    if count == 0:
        return np.empty(
            (0, 4),
            dtype=np.float32,
        )

    colors = np.empty(
        (count, 4),
        dtype=np.float32,
    )

    colors[:, 0] = 0.16
    colors[:, 1] = 0.16
    colors[:, 2] = 0.16
    colors[:, 3] = 1.0

    points_h = np.concatenate(
        [
            xyz_lidar.astype(
                np.float64,
                copy=False,
            ),
            np.ones(
                (count, 1),
                dtype=np.float64,
            ),
        ],
        axis=1,
    )

    camera_cv = (
        T_camera_cv_from_lidar
        @ points_h.T
    ).T[:, :3]

    z = camera_cv[:, 2]

    valid = (
        np.isfinite(
            camera_cv
        ).all(axis=1)
        & (z > 1e-4)
    )

    if not np.any(valid):
        return colors

    valid_indices = np.flatnonzero(
        valid
    )

    points = camera_cv[
        valid_indices
    ]

    projected = (
        K @ points.T
    ).T

    u = (
        projected[:, 0]
        / projected[:, 2]
    )

    v = (
        projected[:, 1]
        / projected[:, 2]
    )

    height, width = rgb.shape[:2]

    inside = (
        (u >= 0.0)
        & (u <= width - 1)
        & (v >= 0.0)
        & (v <= height - 1)
        & np.isfinite(u)
        & np.isfinite(v)
    )

    if not np.any(inside):
        return colors

    target_indices = valid_indices[
        inside
    ]

    px = np.rint(
        u[inside]
    ).astype(
        np.int32
    )

    py = np.rint(
        v[inside]
    ).astype(
        np.int32
    )

    px = np.clip(
        px,
        0,
        width - 1,
    )

    py = np.clip(
        py,
        0,
        height - 1,
    )

    sampled_rgb = (
        rgb[py, px]
        .astype(
            np.float32
        )
        / 255.0
    )

    colors[
        target_indices,
        :3
    ] = sampled_rgb

    return np.ascontiguousarray(
        colors
    )


# ============================================================
# FILTERING
# ============================================================

def object_passes_filter(
    obj,
    selected_classes,
    min_lidar_points,
    max_object_distance,
):
    cls = str(
        obj.get(
            "class",
            "",
        )
    ).lower()

    if cls not in selected_classes:
        return False

    if int(
        obj.get(
            "num_lidar_points",
            0,
        )
    ) < min_lidar_points:
        return False

    distance = obj.get(
        "distance_to_bbox_center_m",
        None,
    )

    if (
        distance is not None
        and float(distance)
        > max_object_distance
    ):
        return False

    return True


def bbox_intersects_display_cube(
    corners_lidar,
    visual_range,
):
    corners = np.asarray(
        corners_lidar,
        dtype=np.float64,
    )

    if corners.shape != (8, 3):
        return False

    box_min = corners.min(
        axis=0
    )

    box_max = corners.max(
        axis=0
    )

    R = float(
        visual_range
    )

    return bool(
        np.all(
            box_max >= -R
        )
        and np.all(
            box_min <= R
        )
    )


# ============================================================
# 2D GEOMETRY
# ============================================================

def rectangle_segments(
    xyxy,
    width,
    height,
):
    if xyxy is None:
        return np.empty(
            (0, 2),
            dtype=np.float32,
        )

    x1, y1, x2, y2 = map(
        float,
        xyxy,
    )

    x1 = float(
        np.clip(
            x1,
            0.0,
            width - 1.0,
        )
    )

    y1 = float(
        np.clip(
            y1,
            0.0,
            height - 1.0,
        )
    )

    x2 = float(
        np.clip(
            x2,
            0.0,
            width - 1.0,
        )
    )

    y2 = float(
        np.clip(
            y2,
            0.0,
            height - 1.0,
        )
    )

    if x2 <= x1 or y2 <= y1:
        return np.empty(
            (0, 2),
            dtype=np.float32,
        )

    return np.asarray(
        [
            [x1, y1], [x2, y1],
            [x2, y1], [x2, y2],
            [x2, y2], [x1, y2],
            [x1, y2], [x1, y1],
        ],
        dtype=np.float32,
    )


def clip_segment_near_plane(
    p0,
    p1,
    near_z=0.05,
):
    p0 = np.asarray(
        p0,
        dtype=np.float64,
    ).copy()

    p1 = np.asarray(
        p1,
        dtype=np.float64,
    ).copy()

    front0 = (
        p0[2] >= near_z
    )

    front1 = (
        p1[2] >= near_z
    )

    if front0 and front1:
        return p0, p1

    if (
        not front0
        and not front1
    ):
        return None

    dz = (
        p1[2] - p0[2]
    )

    if abs(dz) < 1e-12:
        return None

    t = (
        near_z - p0[2]
    ) / dz

    if not 0.0 <= t <= 1.0:
        return None

    q = (
        p0
        + t * (p1 - p0)
    )

    q[2] = near_z

    if front0:
        return p0, q

    return q, p1


def project_camera_point(
    point,
    K,
):
    uvw = (
        K
        @ np.asarray(
            point,
            dtype=np.float64,
        )
    )

    if abs(
        uvw[2]
    ) < 1e-12:
        return None

    return np.asarray(
        [
            uvw[0] / uvw[2],
            uvw[1] / uvw[2],
        ],
        dtype=np.float64,
    )


def liang_barsky_clip(
    a,
    b,
    width,
    height,
):
    x0, y0 = map(
        float,
        a,
    )

    x1, y1 = map(
        float,
        b,
    )

    dx = x1 - x0
    dy = y1 - y0

    p = (
        -dx,
        dx,
        -dy,
        dy,
    )

    q = (
        x0,
        (width - 1.0) - x0,
        y0,
        (height - 1.0) - y0,
    )

    u1 = 0.0
    u2 = 1.0

    for pi, qi in zip(
        p,
        q,
    ):
        if abs(pi) < 1e-12:
            if qi < 0.0:
                return None
            continue

        t = qi / pi

        if pi < 0.0:
            if t > u2:
                return None
            u1 = max(
                u1,
                t,
            )
        else:
            if t < u1:
                return None
            u2 = min(
                u2,
                t,
            )

    return (
        np.asarray(
            [
                x0 + u1 * dx,
                y0 + u1 * dy,
            ],
            dtype=np.float64,
        ),
        np.asarray(
            [
                x0 + u2 * dx,
                y0 + u2 * dy,
            ],
            dtype=np.float64,
        ),
    )


def projected_cuboid_segments(
    corners_camera_cv,
    K,
    width,
    height,
):
    corners = np.asarray(
        corners_camera_cv,
        dtype=np.float64,
    )

    if corners.shape != (8, 3):
        return np.empty(
            (0, 2),
            dtype=np.float32,
        )

    output = []

    for i0, i1 in BBOX_EDGES:
        clipped = clip_segment_near_plane(
            corners[i0],
            corners[i1],
        )

        if clipped is None:
            continue

        a = project_camera_point(
            clipped[0],
            K,
        )

        b = project_camera_point(
            clipped[1],
            K,
        )

        if a is None or b is None:
            continue

        clipped_2d = liang_barsky_clip(
            a,
            b,
            width,
            height,
        )

        if clipped_2d is None:
            continue

        output.extend(
            [
                clipped_2d[0],
                clipped_2d[1],
            ]
        )

    if not output:
        return np.empty(
            (0, 2),
            dtype=np.float32,
        )

    return np.asarray(
        output,
        dtype=np.float32,
    )


# ============================================================
# 3D LIDAR GEOMETRY
# ============================================================

def lidar_cuboid_segments(
    corners_lidar,
):
    corners = np.asarray(
        corners_lidar,
        dtype=np.float32,
    )

    if corners.shape != (8, 3):
        return np.empty(
            (0, 3),
            dtype=np.float32,
        )

    output = []

    for i0, i1 in BBOX_EDGES:
        output.extend(
            [
                corners[i0],
                corners[i1],
            ]
        )

    return np.asarray(
        output,
        dtype=np.float32,
    )


def camera_frustum_segments_lidar(
    K,
    T_camera_cv_from_lidar,
    image_width,
    image_height,
    depth,
):
    """
    Create a camera frustum in LiDAR coordinates.

    Camera CV:
        x right
        y down
        z forward
    """
    fx = float(
        K[0, 0]
    )

    fy = float(
        K[1, 1]
    )

    cx = float(
        K[0, 2]
    )

    cy = float(
        K[1, 2]
    )

    z = float(
        depth
    )

    pixels = (
        (0.0, 0.0),
        (image_width - 1.0, 0.0),
        (image_width - 1.0, image_height - 1.0),
        (0.0, image_height - 1.0),
    )

    camera_points = [
        np.array(
            [0.0, 0.0, 0.0, 1.0],
            dtype=np.float64,
        )
    ]

    for u, v in pixels:
        x = (
            (u - cx)
            / fx
            * z
        )

        y = (
            (v - cy)
            / fy
            * z
        )

        camera_points.append(
            np.array(
                [x, y, z, 1.0],
                dtype=np.float64,
            )
        )

    T_lidar_from_camera_cv = (
        np.linalg.inv(
            T_camera_cv_from_lidar
        )
    )

    lidar_points = []

    for p in camera_points:
        q = (
            T_lidar_from_camera_cv
            @ p
        )

        lidar_points.append(
            q[:3]
        )

    center = lidar_points[0]
    corners = lidar_points[1:]

    output = []

    # Rays from camera center.
    for corner in corners:
        output.extend(
            [center, corner]
        )

    # Far image rectangle.
    for i in range(4):
        output.extend(
            [
                corners[i],
                corners[
                    (i + 1) % 4
                ],
            ]
        )

    return np.asarray(
        output,
        dtype=np.float32,
    )


# ============================================================
# TEXT / CONCAT
# ============================================================

def concatenate_segments(
    parts,
    dimensions,
):
    parts = [
        p
        for p in parts
        if p is not None
        and len(p) > 0
    ]

    if not parts:
        return np.empty(
            (0, dimensions),
            dtype=np.float32,
        )

    return np.ascontiguousarray(
        np.concatenate(
            parts,
            axis=0,
        ),
        dtype=np.float32,
    )


def text_anchor(
    obj,
    rgb_cuboid,
    width,
    height,
):
    bbox2d = obj.get(
        "bbox2d",
        {},
    )

    for key in (
        "visible",
        "projected",
    ):
        item = bbox2d.get(
            key
        )

        if (
            item is not None
            and item.get(
                "xyxy"
            ) is not None
        ):
            x, y = map(
                float,
                item["xyxy"][:2],
            )

            return (
                float(
                    np.clip(
                        x,
                        0.0,
                        width - 1.0,
                    )
                ),
                float(
                    np.clip(
                        y - 5.0,
                        10.0,
                        height - 1.0,
                    )
                ),
            )

    if len(
        rgb_cuboid
    ) > 0:
        return (
            float(
                np.clip(
                    rgb_cuboid[
                        :, 0
                    ].min(),
                    0.0,
                    width - 1.0,
                )
            ),
            float(
                np.clip(
                    rgb_cuboid[
                        :, 1
                    ].min() - 5.0,
                    10.0,
                    height - 1.0,
                )
            ),
        )

    return None


# ============================================================
# BUILD OVERLAYS
# ============================================================

def build_overlays(
    label_data,
    K,
    width,
    height,
    selected_classes,
    min_lidar_points,
    max_object_distance,
    visual_range,
):
    visible_lists = {
        cls: []
        for cls in CLASSES
    }

    projected_lists = {
        cls: []
        for cls in CLASSES
    }

    rgb_3d_lists = {
        cls: []
        for cls in CLASSES
    }

    lidar_3d_lists = {
        cls: []
        for cls in CLASSES
    }

    text_lists = {
        cls: []
        for cls in CLASSES
    }

    text_positions = {
        cls: []
        for cls in CLASSES
    }

    raw_objects = label_data.get(
        "objects",
        [],
    )

    kept_objects = 0
    lidar_boxes_in_display_range = 0
    lidar_boxes_outside_display_range = 0

    for obj in raw_objects:
        if not object_passes_filter(
            obj,
            selected_classes,
            min_lidar_points,
            max_object_distance,
        ):
            continue

        cls = str(
            obj.get(
                "class",
                "",
            )
        ).lower()

        if cls not in CLASSES:
            continue

        kept_objects += 1

        bbox2d = obj.get(
            "bbox2d",
            {},
        )

        visible = bbox2d.get(
            "visible"
        )

        if (
            visible is not None
            and visible.get(
                "xyxy"
            ) is not None
        ):
            segments = rectangle_segments(
                visible["xyxy"],
                width,
                height,
            )

            if len(
                segments
            ):
                visible_lists[
                    cls
                ].append(
                    segments
                )

        projected = bbox2d.get(
            "projected"
        )

        if (
            projected is not None
            and projected.get(
                "xyxy"
            ) is not None
        ):
            segments = rectangle_segments(
                projected["xyxy"],
                width,
                height,
            )

            if len(
                segments
            ):
                projected_lists[
                    cls
                ].append(
                    segments
                )

        bbox3d = obj.get(
            "bbox3d",
            {},
        )

        camera_cv = bbox3d.get(
            "camera_cv",
            {},
        )

        corners_cv = camera_cv.get(
            "corners_xyz_m",
            None,
        )

        rgb_cuboid = np.empty(
            (0, 2),
            dtype=np.float32,
        )

        if corners_cv is not None:
            rgb_cuboid = (
                projected_cuboid_segments(
                    corners_cv,
                    K,
                    width,
                    height,
                )
            )

            if len(
                rgb_cuboid
            ):
                rgb_3d_lists[
                    cls
                ].append(
                    rgb_cuboid
                )

        lidar_box = bbox3d.get(
            "lidar",
            {},
        )

        corners_lidar = lidar_box.get(
            "corners_xyz_m",
            None,
        )

        if corners_lidar is not None:
            if bbox_intersects_display_cube(
                corners_lidar,
                visual_range,
            ):
                segments = (
                    lidar_cuboid_segments(
                        corners_lidar
                    )
                )

                if len(
                    segments
                ):
                    lidar_3d_lists[
                        cls
                    ].append(
                        segments
                    )

                    lidar_boxes_in_display_range += 1
            else:
                lidar_boxes_outside_display_range += 1

        anchor = text_anchor(
            obj,
            rgb_cuboid,
            width,
            height,
        )

        if anchor is not None:
            text_lists[
                cls
            ].append(
                f"{cls} "
                f"id={obj.get('actor_id', '?')} "
                f"pts={obj.get('num_lidar_points', 0)}"
            )

            text_positions[
                cls
            ].append(
                anchor
            )

    visible_2d = {
        cls: concatenate_segments(
            visible_lists[cls],
            2,
        )
        for cls in CLASSES
    }

    projected_2d = {
        cls: concatenate_segments(
            projected_lists[cls],
            2,
        )
        for cls in CLASSES
    }

    rgb_3d = {
        cls: concatenate_segments(
            rgb_3d_lists[cls],
            2,
        )
        for cls in CLASSES
    }

    lidar_3d = {
        cls: concatenate_segments(
            lidar_3d_lists[cls],
            3,
        )
        for cls in CLASSES
    }

    text_pos_arrays = {}

    for cls in CLASSES:
        if text_positions[
            cls
        ]:
            text_pos_arrays[
                cls
            ] = np.asarray(
                text_positions[cls],
                dtype=np.float32,
            )
        else:
            text_pos_arrays[
                cls
            ] = np.empty(
                (0, 2),
                dtype=np.float32,
            )

    stats = {
        "raw_objects": int(
            len(raw_objects)
        ),
        "kept_objects": int(
            kept_objects
        ),
        "lidar_boxes_in_display_range": int(
            lidar_boxes_in_display_range
        ),
        "lidar_boxes_outside_display_range": int(
            lidar_boxes_outside_display_range
        ),
    }

    return (
        visible_2d,
        projected_2d,
        rgb_3d,
        lidar_3d,
        text_lists,
        text_pos_arrays,
        stats,
    )


# ============================================================
# FRAME DATA
# ============================================================

@dataclass(slots=True)
class FrameData:
    rgb: np.ndarray

    xyz: np.ndarray

    height_colors: np.ndarray
    rgb_colors: np.ndarray

    original_count: int
    visible_count: int

    visible_2d: dict
    projected_2d: dict

    rgb_3d: dict
    lidar_3d: dict

    text: dict
    text_pos: dict

    stats: dict


def load_frame_data(
    frame,
    max_points,
    visual_range,
    color_lut,
    K,
    T_camera_cv_from_lidar,
    selected_classes,
    min_lidar_points,
    max_object_distance,
):
    rgb = load_rgb(
        frame["rgb"]
    )

    cloud = load_lidar(
        frame["lidar"]
    )

    label = load_json(
        frame["label"]
    )

    original_count = len(
        cloud
    )

    xyz = cloud[:, :3]

    finite = np.isfinite(
        xyz
    ).all(axis=1)

    xyz = xyz[
        finite
    ]

    # --------------------------------------------------------
    # Display range
    # --------------------------------------------------------

    R = float(
        visual_range
    )

    mask = (
        (xyz[:, 0] >= -R)
        & (xyz[:, 0] <= R)
        & (xyz[:, 1] >= -R)
        & (xyz[:, 1] <= R)
        & (xyz[:, 2] >= -R)
        & (xyz[:, 2] <= R)
    )

    xyz = xyz[
        mask
    ]

    visible_count = len(
        xyz
    )

    xyz = display_sample(
        xyz,
        max_points,
    )

    xyz = np.ascontiguousarray(
        xyz,
        dtype=np.float32,
    )

    # --------------------------------------------------------
    # Point colors
    # --------------------------------------------------------

    height_colors = (
        make_height_colors(
            xyz,
            visual_range,
            color_lut,
        )
    )

    rgb_colors = (
        rgb_colorize_lidar_points(
            xyz,
            rgb,
            K,
            T_camera_cv_from_lidar,
        )
    )

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    height, width = (
        rgb.shape[:2]
    )

    (
        visible_2d,
        projected_2d,
        rgb_3d,
        lidar_3d,
        text,
        text_pos,
        stats,
    ) = build_overlays(
        label_data=label,
        K=K,
        width=width,
        height=height,
        selected_classes=selected_classes,
        min_lidar_points=min_lidar_points,
        max_object_distance=max_object_distance,
        visual_range=visual_range,
    )

    return FrameData(
        rgb=rgb,
        xyz=xyz,
        height_colors=height_colors,
        rgb_colors=rgb_colors,
        original_count=original_count,
        visible_count=visible_count,
        visible_2d=visible_2d,
        projected_2d=projected_2d,
        rgb_3d=rgb_3d,
        lidar_3d=lidar_3d,
        text=text,
        text_pos=text_pos,
        stats=stats,
    )


# ============================================================
# PREFETCH
# ============================================================

class FramePrefetcher:

    def __init__(
        self,
        frames,
        max_points,
        visual_range,
        color_lut,
        K,
        T_camera_cv_from_lidar,
        selected_classes,
        min_lidar_points,
        max_object_distance,
        lookahead,
        workers,
        loop,
    ):
        self.frames = frames

        self.max_points = (
            max_points
        )

        self.visual_range = (
            visual_range
        )

        self.color_lut = (
            color_lut
        )

        self.K = K

        self.T_camera_cv_from_lidar = (
            T_camera_cv_from_lidar
        )

        self.selected_classes = tuple(
            selected_classes
        )

        self.min_lidar_points = int(
            min_lidar_points
        )

        self.max_object_distance = float(
            max_object_distance
        )

        self.lookahead = max(
            1,
            int(lookahead),
        )

        self.loop = bool(
            loop
        )

        self.executor = (
            ThreadPoolExecutor(
                max_workers=max(
                    1,
                    int(workers),
                ),
                thread_name_prefix=(
                    "dataset-loader"
                ),
            )
        )

        self.jobs = {}
        self.closed = False

    def _load(
        self,
        index,
    ):
        return load_frame_data(
            frame=self.frames[index],
            max_points=self.max_points,
            visual_range=self.visual_range,
            color_lut=self.color_lut,
            K=self.K,
            T_camera_cv_from_lidar=(
                self.T_camera_cv_from_lidar
            ),
            selected_classes=(
                self.selected_classes
            ),
            min_lidar_points=(
                self.min_lidar_points
            ),
            max_object_distance=(
                self.max_object_distance
            ),
        )

    def submit(
        self,
        index,
    ):
        if self.closed:
            return

        if not 0 <= index < len(
            self.frames
        ):
            return

        if index in self.jobs:
            return

        self.jobs[
            index
        ] = self.executor.submit(
            self._load,
            index,
        )

    def prefetch(
        self,
        start_index,
    ):
        if (
            self.closed
            or not self.frames
        ):
            return

        frame_count = len(
            self.frames
        )

        for offset in range(
            min(
                self.lookahead,
                frame_count,
            )
        ):
            index = (
                start_index
                + offset
            )

            if self.loop:
                index %= frame_count
            elif index >= frame_count:
                break

            self.submit(
                index
            )

    def get(
        self,
        index,
    ):
        if self.closed:
            raise RuntimeError(
                "FramePrefetcher is closed."
            )

        self.submit(
            index
        )

        future = self.jobs.pop(
            index
        )

        data = future.result()

        self.prefetch(
            index + 1
        )

        return data

    def close(self):
        if self.closed:
            return

        self.closed = True

        for future in self.jobs.values():
            future.cancel()

        self.jobs.clear()

        try:
            self.executor.shutdown(
                wait=False,
                cancel_futures=True,
            )
        except TypeError:
            self.executor.shutdown(
                wait=False
            )


# ============================================================
# WINDOW
# ============================================================

class DatasetPlayerWindow(
    QMainWindow
):

    def __init__(
        self,
        args,
        frames,
        K,
        T_camera_cv_from_lidar,
        camera_frustum,
    ):
        super().__init__()

        self.args = args
        self.frames = frames

        self.K = K

        self.T_camera_cv_from_lidar = (
            T_camera_cv_from_lidar
        )

        self.camera_frustum = (
            camera_frustum
        )

        self.current_index = 0

        self.paused = False
        self.ended = False
        self.closed = False

        # ----------------------------------------------------
        # Overlay modes
        # ----------------------------------------------------

        self.bbox_mode_index = 0

        self.show_rgb_3d = True
        self.show_lidar_3d = True
        self.show_text = True
        self.show_frustum = True

        # Start with RGB-colorized points.
        self.point_color_mode_index = 0

        # ----------------------------------------------------
        # Performance
        # ----------------------------------------------------

        self.actual_fps = 0.0
        self.last_present_time = None
        self.frame_pending_draw = False

        self.last_fetch_ms = 0.0
        self.last_submit_ms = 0.0

        self.color_lut = (
            build_viridis_lut()
        )

        # ----------------------------------------------------
        # First frame
        # ----------------------------------------------------

        t0 = time.perf_counter()

        self.current_data = (
            load_frame_data(
                frame=self.frames[0],
                max_points=(
                    self.args.max_points
                ),
                visual_range=(
                    self.args.range
                ),
                color_lut=(
                    self.color_lut
                ),
                K=self.K,
                T_camera_cv_from_lidar=(
                    self.T_camera_cv_from_lidar
                ),
                selected_classes=(
                    self.args.classes
                ),
                min_lidar_points=(
                    self.args.min_lidar_points
                ),
                max_object_distance=(
                    self.args.max_object_distance
                ),
            )
        )

        self.last_fetch_ms = (
            time.perf_counter() - t0
        ) * 1000.0

        # ----------------------------------------------------
        # Prefetch
        # ----------------------------------------------------

        self.prefetcher = (
            FramePrefetcher(
                frames=self.frames,
                max_points=(
                    self.args.max_points
                ),
                visual_range=(
                    self.args.range
                ),
                color_lut=(
                    self.color_lut
                ),
                K=self.K,
                T_camera_cv_from_lidar=(
                    self.T_camera_cv_from_lidar
                ),
                selected_classes=(
                    self.args.classes
                ),
                min_lidar_points=(
                    self.args.min_lidar_points
                ),
                max_object_distance=(
                    self.args.max_object_distance
                ),
                lookahead=(
                    self.args.prefetch
                ),
                workers=(
                    self.args.io_workers
                ),
                loop=(
                    self.args.loop
                ),
            )
        )

        self.prefetcher.prefetch(
            1
        )

        # ----------------------------------------------------
        # Main window / canvas
        # ----------------------------------------------------

        self.setWindowTitle(
            "UAV RGB + LiDAR Detection Player - RGB Fusion"
        )

        self.resize(
            1600,
            850,
        )

        self.canvas = (
            scene.SceneCanvas(
                title=(
                    "UAV RGB + LiDAR Detection Player"
                ),
                size=(
                    1600,
                    850,
                ),
                bgcolor="#111318",
                keys=None,
                show=False,
                create_native=True,
                vsync=self.args.vsync,
            )
        )

        self.setCentralWidget(
            self.canvas.native
        )

        self.canvas.events.draw.connect(
            self.on_canvas_draw
        )

        grid = (
            self.canvas
            .central_widget
            .add_grid(
                margin=6,
                spacing=6,
            )
        )

        # ====================================================
        # TITLES
        # ====================================================

        self.rgb_title = scene.Label(
            "",
            color="white",
            font_size=12,
        )

        self.rgb_title.height_max = 34

        grid.add_widget(
            self.rgb_title,
            row=0,
            col=0,
        )

        self.pc_title = scene.Label(
            "",
            color="white",
            font_size=12,
        )

        self.pc_title.height_max = 34

        grid.add_widget(
            self.pc_title,
            row=0,
            col=1,
        )

        # ====================================================
        # RGB VIEW
        # ====================================================

        self.rgb_view = grid.add_view(
            row=1,
            col=0,
            border_color="#3a3f4b",
            bgcolor="black",
        )

        self.rgb_image = (
            scene.visuals.Image(
                self.current_data.rgb,
                interpolation="nearest",
                method="subdivide",
                texture_format="auto",
                parent=(
                    self.rgb_view.scene
                ),
            )
        )

        self.rgb_camera = (
            scene.PanZoomCamera(
                aspect=1
            )
        )

        self.rgb_camera.flip = (
            0,
            1,
            0,
        )

        self.rgb_view.camera = (
            self.rgb_camera
        )

        self.last_rgb_shape = (
            self.current_data.rgb.shape
        )

        self.rgb_2d_lines = {}
        self.rgb_3d_lines = {}
        self.rgb_text = {}

        for cls in CLASSES:
            self.rgb_2d_lines[
                cls
            ] = scene.visuals.Line(
                pos=np.zeros(
                    (2, 2),
                    dtype=np.float32,
                ),
                color=CLASS_COLORS[
                    cls
                ],
                width=2.0,
                connect="segments",
                method="gl",
                parent=(
                    self.rgb_view.scene
                ),
            )

            self.rgb_2d_lines[
                cls
            ].visible = False

            self.rgb_3d_lines[
                cls
            ] = scene.visuals.Line(
                pos=np.zeros(
                    (2, 2),
                    dtype=np.float32,
                ),
                color=CLASS_COLORS[
                    cls
                ],
                width=1.5,
                connect="segments",
                method="gl",
                parent=(
                    self.rgb_view.scene
                ),
            )

            self.rgb_3d_lines[
                cls
            ].visible = False

            text_visual = (
                scene.visuals.Text(
                    text="",
                    pos=(0, 0),
                    color=CLASS_COLORS[
                        cls
                    ],
                    font_size=8,
                    anchor_x="left",
                    anchor_y="bottom",
                    parent=(
                        self.rgb_view.scene
                    ),
                )
            )

            text_visual.visible = False

            self.rgb_text[
                cls
            ] = text_visual

        self.reset_rgb_camera()

        # ====================================================
        # LIDAR VIEW
        # ====================================================

        self.pc_view = grid.add_view(
            row=1,
            col=1,
            border_color="#3a3f4b",
            bgcolor="#080a0d",
        )

        self.pc_camera = (
            scene.TurntableCamera(
                fov=0.0,
                elevation=25.0,
                azimuth=-60.0,
                roll=0.0,
                center=(
                    0.0,
                    0.0,
                    0.0,
                ),
                up="+z",
            )
        )

        self.pc_view.camera = (
            self.pc_camera
        )

        self.cloud_visual = (
            scene.visuals.Markers(
                method="points",
                scaling="fixed",
                antialias=0,
                spherical=False,
                parent=(
                    self.pc_view.scene
                ),
            )
        )

        self.cloud_visual.set_data(
            pos=(
                self.current_data.xyz
            ),
            face_color=(
                self.current_data.rgb_colors
            ),
            edge_width=0,
            size=(
                self.args.point_size
            ),
            symbol="disc",
        )

        self.lidar_3d_lines = {}

        for cls in CLASSES:
            self.lidar_3d_lines[
                cls
            ] = scene.visuals.Line(
                pos=np.zeros(
                    (2, 3),
                    dtype=np.float32,
                ),
                color=CLASS_COLORS[
                    cls
                ],
                width=2.0,
                connect="segments",
                method="gl",
                parent=(
                    self.pc_view.scene
                ),
            )

            self.lidar_3d_lines[
                cls
            ].visible = False

        # Camera frustum in LiDAR coordinates.
        self.frustum_visual = (
            scene.visuals.Line(
                pos=self.camera_frustum,
                color=(
                    1.0,
                    1.0,
                    1.0,
                    0.9,
                ),
                width=1.5,
                connect="segments",
                method="gl",
                parent=(
                    self.pc_view.scene
                ),
            )
        )

        self.frustum_visual.visible = (
            self.show_frustum
        )

        # LiDAR origin.
        self.origin_visual = (
            scene.visuals.Markers(
                method="points",
                scaling="fixed",
                antialias=0,
                spherical=False,
                parent=(
                    self.pc_view.scene
                ),
            )
        )

        self.origin_visual.set_data(
            pos=np.array(
                [
                    [
                        0.0,
                        0.0,
                        0.0,
                    ]
                ],
                dtype=np.float32,
            ),
            face_color="white",
            edge_color="white",
            edge_width=0,
            size=10.0,
            symbol="x",
        )

        self.xyz_axis = (
            scene.visuals.XYZAxis(
                parent=(
                    self.pc_view.scene
                )
            )
        )

        axis_size = (
            float(
                self.args.range
            )
            * 0.25
        )

        self.xyz_axis.transform = (
            scene.transforms.STTransform(
                scale=(
                    axis_size,
                    axis_size,
                    axis_size,
                )
            )
        )

        self.reset_pc_camera()

        # ====================================================
        # STATUS
        # ====================================================

        self.status_label = scene.Label(
            "",
            color="#d8dbe2",
            font_size=9,
        )

        self.status_label.height_max = 48

        grid.add_widget(
            self.status_label,
            row=2,
            col=0,
            col_span=2,
        )

        # ====================================================
        # CONTROLS
        # ====================================================

        self.shortcuts = []

        self.add_shortcut(
            "Space",
            self.toggle_pause,
        )

        self.add_shortcut(
            "R",
            self.reset_pc_camera,
        )

        self.add_shortcut(
            "B",
            self.toggle_bbox_mode,
        )

        self.add_shortcut(
            "C",
            self.toggle_rgb_3d,
        )

        self.add_shortcut(
            "L",
            self.toggle_lidar_3d,
        )

        self.add_shortcut(
            "T",
            self.toggle_text,
        )

        self.add_shortcut(
            "P",
            self.toggle_point_color_mode,
        )

        self.add_shortcut(
            "F",
            self.toggle_frustum,
        )

        self.add_shortcut(
            "Q",
            self.close,
        )

        self.add_shortcut(
            "Esc",
            self.close,
        )

        self.update_frame_visuals(
            self.current_data,
            reset_rgb=False,
        )

        # ====================================================
        # TIMER
        # ====================================================

        self.timer = QTimer(
            self
        )

        self.timer.setTimerType(
            Qt.TimerType.PreciseTimer
        )

        self.timer.timeout.connect(
            self.advance_frame
        )

        self.timer.setInterval(
            max(
                1,
                int(
                    round(
                        1000.0
                        / self.args.fps
                    )
                ),
            )
        )

        self.frame_pending_draw = True

        self.timer.start()

    # ========================================================
    # PROPERTIES
    # ========================================================

    @property
    def bbox_mode(
        self
    ):
        return BBOX_MODES[
            self.bbox_mode_index
        ]

    @property
    def point_color_mode(
        self
    ):
        return POINT_COLOR_MODES[
            self.point_color_mode_index
        ]

    # ========================================================
    # HELPERS
    # ========================================================

    def add_shortcut(
        self,
        sequence,
        callback,
    ):
        shortcut = QShortcut(
            QKeySequence(
                sequence
            ),
            self,
        )

        shortcut.activated.connect(
            callback
        )

        self.shortcuts.append(
            shortcut
        )

    def reset_rgb_camera(
        self
    ):
        height, width = (
            self.current_data
            .rgb
            .shape[:2]
        )

        self.rgb_camera.set_range(
            x=(
                0,
                width,
            ),
            y=(
                0,
                height,
            ),
            margin=0.0,
        )

    def reset_pc_camera(
        self
    ):
        R = float(
            self.args.range
        )

        self.pc_camera.fov = 0.0
        self.pc_camera.elevation = 25.0
        self.pc_camera.azimuth = -60.0
        self.pc_camera.roll = 0.0
        self.pc_camera.center = (
            0.0,
            0.0,
            0.0,
        )

        self.pc_camera.set_range(
            x=(-R, R),
            y=(-R, R),
            z=(-R, R),
            margin=0.02,
        )

        self.canvas.update()

    @staticmethod
    def set_line(
        visual,
        positions,
        enabled,
    ):
        has_data = (
            positions is not None
            and len(positions) > 0
        )

        if has_data:
            visual.set_data(
                pos=positions,
                connect="segments",
            )

        visual.visible = bool(
            enabled
            and has_data
        )

    def current_point_colors(
        self
    ):
        if (
            self.point_color_mode
            == "rgb"
        ):
            return (
                self.current_data
                .rgb_colors
            )

        return (
            self.current_data
            .height_colors
        )

    # ========================================================
    # OVERLAY UPDATES
    # ========================================================

    def update_overlays(
        self
    ):
        for cls in CLASSES:
            if (
                self.bbox_mode
                == "visible"
            ):
                box_pos = (
                    self.current_data
                    .visible_2d[
                        cls
                    ]
                )

            elif (
                self.bbox_mode
                == "projected"
            ):
                box_pos = (
                    self.current_data
                    .projected_2d[
                        cls
                    ]
                )

            else:
                box_pos = np.empty(
                    (0, 2),
                    dtype=np.float32,
                )

            self.set_line(
                self.rgb_2d_lines[
                    cls
                ],
                box_pos,
                self.bbox_mode
                != "off",
            )

            self.set_line(
                self.rgb_3d_lines[
                    cls
                ],
                self.current_data
                .rgb_3d[
                    cls
                ],
                self.show_rgb_3d,
            )

            self.set_line(
                self.lidar_3d_lines[
                    cls
                ],
                self.current_data
                .lidar_3d[
                    cls
                ],
                self.show_lidar_3d,
            )

            texts = (
                self.current_data
                .text[
                    cls
                ]
            )

            positions = (
                self.current_data
                .text_pos[
                    cls
                ]
            )

            visual = (
                self.rgb_text[
                    cls
                ]
            )

            if (
                self.show_text
                and len(texts) > 0
                and len(positions) > 0
            ):
                visual.text = texts
                visual.pos = (
                    positions
                )
                visual.visible = True
            else:
                visual.visible = False

        self.frustum_visual.visible = bool(
            self.show_frustum
        )

    def update_titles(
        self
    ):
        stats = (
            self.current_data
            .stats
        )

        self.rgb_title.text = (
            f"RGB | Frame "
            f"{self.frames[self.current_index]['id']} | "
            f"objects "
            f"{stats['kept_objects']}/"
            f"{stats['raw_objects']} | "
            f"2D {self.bbox_mode} | "
            f"3D-proj "
            f"{'ON' if self.show_rgb_3d else 'OFF'}"
        )

        self.pc_title.text = (
            f"LiDAR | Frame "
            f"{self.frames[self.current_index]['id']} | "
            f"{self.current_data.original_count:,} raw | "
            f"{len(self.current_data.xyz):,} displayed | "
            f"color={self.point_color_mode} | "
            f"3D boxes="
            f"{stats['lidar_boxes_in_display_range']} | "
            f"outside-range="
            f"{stats['lidar_boxes_outside_display_range']}"
        )

    def update_status_text(
        self
    ):
        if self.ended:
            state = "END"

        elif self.paused:
            state = "PAUSED"

        else:
            state = "PLAYING"

        self.status_label.text = (
            f"{self.current_index + 1}/"
            f"{len(self.frames)}"
            f" | {state}"
            f" | target {self.args.fps:.1f} FPS"
            f" | draw {self.actual_fps:.1f} FPS"
            f" | fetch {self.last_fetch_ms:.2f} ms"
            f" | submit {self.last_submit_ms:.2f} ms"
            f" | minPts {self.args.min_lidar_points}"
            f" | range ±{self.args.range:.0f}m"
            f" | B 2D | C RGB-3D | L LiDAR-3D"
            f" | P point-color | F frustum | T text"
        )

    def update_frame_visuals(
        self,
        data,
        reset_rgb=False,
    ):
        self.rgb_image.set_data(
            data.rgb
        )

        if reset_rgb:
            self.reset_rgb_camera()

        self.cloud_visual.set_data(
            pos=data.xyz,
            face_color=(
                self.current_point_colors()
            ),
            edge_width=0,
            size=self.args.point_size,
            symbol="disc",
        )

        self.update_overlays()
        self.update_titles()
        self.update_status_text()

    # ========================================================
    # DRAW FPS
    # ========================================================

    def on_canvas_draw(
        self,
        event,
    ):
        if not self.frame_pending_draw:
            return

        now = time.perf_counter()

        previous = (
            self.last_present_time
        )

        self.last_present_time = now
        self.frame_pending_draw = False

        if previous is None:
            return

        dt = now - previous

        if dt <= 0:
            return

        fps_now = (
            1.0 / dt
        )

        if self.actual_fps <= 0:
            self.actual_fps = (
                fps_now
            )
        else:
            self.actual_fps = (
                self.actual_fps
                * 0.85
                + fps_now
                * 0.15
            )

    # ========================================================
    # PLAYBACK
    # ========================================================

    def advance_frame(
        self
    ):
        if (
            self.paused
            or self.closed
        ):
            return

        next_index = (
            self.current_index
            + 1
        )

        if next_index >= len(
            self.frames
        ):
            if self.args.loop:
                next_index = 0

            else:
                self.ended = True
                self.timer.stop()
                self.update_status_text()
                self.canvas.update()
                return

        t0 = time.perf_counter()

        data = self.prefetcher.get(
            next_index
        )

        self.last_fetch_ms = (
            time.perf_counter()
            - t0
        ) * 1000.0

        t0 = time.perf_counter()

        self.current_index = (
            next_index
        )

        self.current_data = (
            data
        )

        shape_changed = (
            data.rgb.shape
            != self.last_rgb_shape
        )

        if shape_changed:
            self.last_rgb_shape = (
                data.rgb.shape
            )

        self.update_frame_visuals(
            data,
            reset_rgb=shape_changed,
        )

        self.last_submit_ms = (
            time.perf_counter()
            - t0
        ) * 1000.0

        self.frame_pending_draw = True

        self.canvas.update()

    # ========================================================
    # CONTROLS
    # ========================================================

    def toggle_pause(
        self
    ):
        if self.ended:
            return

        self.paused = (
            not self.paused
        )

        if self.paused:
            self.timer.stop()
            print(
                "Playback paused."
            )

        else:
            self.last_present_time = None
            self.frame_pending_draw = False
            self.timer.start()
            print(
                "Playback resumed."
            )

        self.update_status_text()
        self.canvas.update()

    def toggle_bbox_mode(
        self
    ):
        self.bbox_mode_index = (
            self.bbox_mode_index
            + 1
        ) % len(
            BBOX_MODES
        )

        print(
            "2D bbox mode:",
            self.bbox_mode,
        )

        self.update_overlays()
        self.update_titles()
        self.update_status_text()
        self.canvas.update()

    def toggle_rgb_3d(
        self
    ):
        self.show_rgb_3d = (
            not self.show_rgb_3d
        )

        print(
            "RGB projected 3D cuboids:",
            (
                "ON"
                if self.show_rgb_3d
                else "OFF"
            ),
        )

        self.update_overlays()
        self.update_titles()
        self.update_status_text()
        self.canvas.update()

    def toggle_lidar_3d(
        self
    ):
        self.show_lidar_3d = (
            not self.show_lidar_3d
        )

        print(
            "LiDAR 3D boxes:",
            (
                "ON"
                if self.show_lidar_3d
                else "OFF"
            ),
        )

        self.update_overlays()
        self.update_titles()
        self.update_status_text()
        self.canvas.update()

    def toggle_text(
        self
    ):
        self.show_text = (
            not self.show_text
        )

        print(
            "RGB text:",
            (
                "ON"
                if self.show_text
                else "OFF"
            ),
        )

        self.update_overlays()
        self.update_status_text()
        self.canvas.update()

    def toggle_point_color_mode(
        self
    ):
        self.point_color_mode_index = (
            self.point_color_mode_index
            + 1
        ) % len(
            POINT_COLOR_MODES
        )

        print(
            "LiDAR point color mode:",
            self.point_color_mode,
        )

        self.cloud_visual.set_data(
            pos=(
                self.current_data
                .xyz
            ),
            face_color=(
                self.current_point_colors()
            ),
            edge_width=0,
            size=self.args.point_size,
            symbol="disc",
        )

        self.update_titles()
        self.update_status_text()
        self.canvas.update()

    def toggle_frustum(
        self
    ):
        self.show_frustum = (
            not self.show_frustum
        )

        print(
            "Camera frustum:",
            (
                "ON"
                if self.show_frustum
                else "OFF"
            ),
        )

        self.frustum_visual.visible = (
            self.show_frustum
        )

        self.update_status_text()
        self.canvas.update()

    # ========================================================
    # CLOSE
    # ========================================================

    def closeEvent(
        self,
        event,
    ):
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

    validate_args(
        args
    )

    scene_dir = resolve_scene(
        args.scene
    )

    (
        calibration,
        K,
        T_camera_cv_from_lidar,
        image_width,
        image_height,
    ) = load_calibration(
        scene_dir
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
            f"synchronized frames exist."
        )

    frames = frames[
        args.start:
    ]

    frustum = (
        camera_frustum_segments_lidar(
            K=K,
            T_camera_cv_from_lidar=(
                T_camera_cv_from_lidar
            ),
            image_width=image_width,
            image_height=image_height,
            depth=args.frustum_depth,
        )
    )

    print()
    print("=" * 68)
    print("VisPy GPU Detection Dataset Player - RGB/LiDAR Fusion")
    print("=" * 68)

    print("Scene:", scene_dir)
    print("Frames:", len(frames))
    print("Target FPS:", args.fps)
    print("LiDAR display range:", args.range, "m")
    print("Point display limit:", args.max_points)
    print("Prefetch:", args.prefetch)
    print("I/O workers:", args.io_workers)
    print("VSync:", args.vsync)

    print()
    print(
        "Classes:",
        ", ".join(
            args.classes
        ),
    )

    print(
        "Min LiDAR points:",
        args.min_lidar_points,
    )

    print(
        "Max object distance:",
        args.max_object_distance,
        "m",
    )

    print(
        "Camera frustum depth:",
        args.frustum_depth,
        "m",
    )

    if (
        "carla_server_version"
        in calibration
    ):
        print(
            "Dataset CARLA version:",
            calibration[
                "carla_server_version"
            ],
        )

    print()
    print("Initial display:")
    print("  2D bbox              : visible")
    print("  RGB projected 3D     : ON")
    print("  LiDAR 3D boxes       : ON")
    print("  LiDAR point colors   : RGB")
    print("  Camera frustum       : ON")
    print("  Object text          : ON")

    print()
    print("Important:")
    print(
        "  Right-side 3D boxes are drawn only if their "
        "LiDAR bbox intersects the current --range cube."
    )
    print(
        "  Objects with fewer than --min-lidar-points "
        "are not displayed; labels on disk are never modified."
    )
    print(
        "  RGB point coloring uses synchronized RGB + "
        "T_camera_cv_from_lidar + K."
    )
    print(
        "  LiDAR points outside the camera FOV are shown dark gray."
    )

    print()
    print("Controls:")
    print("  SPACE : pause / resume")
    print("  B     : visible -> projected -> off 2D bbox")
    print("  C     : toggle projected 3D cuboids on RGB")
    print("  L     : toggle LiDAR 3D boxes")
    print("  P     : RGB <-> height point-cloud coloring")
    print("  F     : toggle camera frustum")
    print("  T     : toggle class / actor-id / LiDAR-point text")
    print("  R     : reset LiDAR camera")
    print("  Q/ESC : quit")

    print("=" * 68)
    print()

    app = use_app()
    app.create()

    window = DatasetPlayerWindow(
        args=args,
        frames=frames,
        K=K,
        T_camera_cv_from_lidar=(
            T_camera_cv_from_lidar
        ),
        camera_frustum=(
            frustum
        ),
    )

    window.show()

    app.run()


if __name__ == "__main__":
    main()
