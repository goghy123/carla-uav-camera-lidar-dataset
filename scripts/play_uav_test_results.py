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


CLASSES = (
    "car",
    "van",
    "truck",
    "bus",
    "motorcycle",
    "bicycle",
    "pedestrian",
)

# Keep the exact class colors used by the original play_uavdataset.py.
CLASS_COLORS = {
    "car": (0.20, 0.55, 1.00, 1.00),
    "van": (0.10, 0.90, 0.90, 1.00),
    "truck": (1.00, 0.55, 0.10, 1.00),
    "bus": (1.00, 0.85, 0.10, 1.00),
    "motorcycle": (0.95, 0.25, 0.95, 1.00),
    "bicycle": (0.20, 0.95, 0.35, 1.00),
    "pedestrian": (1.00, 0.25, 0.25, 1.00),
}

# In BOTH mode, source is encoded by hue and class by brightness.
_BRIGHTNESS = {
    "car": 1.00,
    "van": 0.90,
    "truck": 0.80,
    "bus": 0.70,
    "motorcycle": 0.60,
    "bicycle": 0.50,
    "pedestrian": 0.42,
}
BOTH_GT_COLORS = {
    cls: (0.10 * v, 1.00 * v, 0.18 * v, 1.0)
    for cls, v in _BRIGHTNESS.items()
}
BOTH_PRED_COLORS = {
    cls: (1.00 * v, 0.45 * v, 0.05 * v, 1.0)
    for cls, v in _BRIGHTNESS.items()
}

BBOX_EDGES = (
    (0, 1), (1, 3), (3, 2), (2, 0),
    (0, 4), (4, 5), (5, 1), (5, 7),
    (7, 6), (6, 4), (6, 2), (7, 3),
)
MMDET_BBOX_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
)
POINT_COLOR_MODES = ("rgb", "height")


def parse_args():
    parser = argparse.ArgumentParser(
        description="播放 UAVDataset test 的 GT、BEVFusion 预测结果，或二者叠加。"
    )
    parser.add_argument("--scene", required=True, help="原始 CARLA UAV 场景目录，例如 dataset\\Town07_Opt")
    parser.add_argument(
        "--pred-dir",
        required=True,
        help="WSL2 中 tools/test_uav_predictions.py 生成并复制到 Windows 的结果目录",
    )
    parser.add_argument(
        "--mode", choices=("gt", "pred", "both"), default="both",
        help="显示模式：gt / pred / both。默认 both",
    )
    parser.add_argument("--score-threshold", type=float, default=0.1)
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--range", type=float, default=80.0)
    parser.add_argument("--max-points", type=int, default=0)
    parser.add_argument("--point-size", type=float, default=2.0)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--prefetch", type=int, default=8)
    parser.add_argument("--io-workers", type=int, default=2)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--vsync", action="store_true")
    parser.add_argument("--classes", nargs="+", default=["all"])
    parser.add_argument("--min-lidar-points", type=int, default=3)
    parser.add_argument("--max-object-distance", type=float, default=120.0)
    parser.add_argument("--frustum-depth", type=float, default=20.0)
    return parser.parse_args()


def normalize_classes(values):
    tokens = []
    for value in values:
        tokens.extend(v.strip().lower() for v in value.split(",") if v.strip())
    if not tokens or "all" in tokens:
        return tuple(CLASSES)
    unknown = sorted(set(tokens) - set(CLASSES))
    if unknown:
        raise ValueError("Unknown classes: {}. Supported: {}".format(unknown, ", ".join(CLASSES)))
    selected = set(tokens)
    return tuple(cls for cls in CLASSES if cls in selected)


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
    if args.score_threshold < 0:
        raise ValueError("--score-threshold must be >= 0")
    if args.min_lidar_points < 0:
        raise ValueError("--min-lidar-points must be >= 0")
    if args.max_object_distance <= 0:
        raise ValueError("--max-object-distance must be > 0")
    if args.frustum_depth <= 0:
        raise ValueError("--frustum-depth must be > 0")
    args.classes = normalize_classes(args.classes)


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def load_rgb(path):
    with Image.open(path) as image:
        return np.array(image.convert("RGB"), dtype=np.uint8, copy=True)


def load_lidar(path):
    cloud = np.fromfile(path, dtype=np.float32)
    if cloud.size % 4 != 0:
        raise RuntimeError("Invalid raw CARLA XYZI bin: {}".format(path))
    return cloud.reshape(-1, 4)


def load_calibration(scene_dir):
    data = load_json(scene_dir / "calibration.json")
    K = np.asarray(data["K"], dtype=np.float64)
    T = np.asarray(data["T_camera_cv_from_lidar"], dtype=np.float64)
    resolution = data["camera_resolution"]
    if K.shape != (3, 3) or T.shape != (4, 4):
        raise ValueError("Invalid calibration matrix shape")
    return data, K, T, int(resolution[0]), int(resolution[1])


def resolve_test_frames(scene_dir, pred_dir):
    manifest_path = pred_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("未找到预测清单 manifest.json: {}".format(manifest_path))
    manifest = load_json(manifest_path)
    scene_name = scene_dir.name
    matching = [s for s in manifest.get("samples", []) if str(s.get("scene_name")) == scene_name]
    if not matching:
        available = sorted({str(s.get("scene_name")) for s in manifest.get("samples", [])})
        raise RuntimeError(
            "manifest.json 中没有场景 {!r} 的 test 预测。可用场景: {}".format(
                scene_name, ", ".join(available) if available else "<none>"
            )
        )

    frames = []
    for sample in matching:
        frame_index = int(sample["frame_index"])
        stem = "{:06d}".format(frame_index)
        frame = {
            "id": stem,
            "frame_index": frame_index,
            "rgb": scene_dir / "rgb" / (stem + ".png"),
            "lidar": scene_dir / "lidar" / (stem + ".bin"),
            "label": scene_dir / "labels" / (stem + ".json"),
            "pose": scene_dir / "pose" / (stem + ".json"),
            "prediction": pred_dir / sample["prediction_file"],
        }
        missing = [str(p) for k, p in frame.items() if k not in ("id", "frame_index") and not p.exists()]
        if missing:
            raise FileNotFoundError("Missing files for test frame {}:\n{}".format(stem, "\n".join(missing)))
        frames.append(frame)
    frames.sort(key=lambda f: f["frame_index"])
    return frames


def reference_height_m(pose, altitude_m):
    uav_z = float(pose["uav"]["location"]["z"])
    lidar_z = float(pose["lidar"]["location"]["z"])
    route_ground_z = uav_z - float(altitude_m)
    return lidar_z - route_ground_z


def reference_corners_to_raw_lidar(corners_reference, height_m):
    corners = np.asarray(corners_reference, dtype=np.float64).reshape(8, 3).copy()
    corners[:, 1] *= -1.0
    corners[:, 2] -= float(height_m)
    return corners


def transform_points(points, transform):
    points = np.asarray(points, dtype=np.float64)
    hom = np.concatenate([points, np.ones((len(points), 1), dtype=np.float64)], axis=1)
    return (transform @ hom.T).T[:, :3]


def display_sample(points, max_points):
    if max_points <= 0 or len(points) <= max_points:
        return points
    return points[:: math.ceil(len(points) / max_points)]


def build_viridis_lut():
    values = np.linspace(0.0, 1.0, 256, dtype=np.float32)
    lut = np.asarray(get_colormap("viridis").map(values), dtype=np.float32).reshape(256, 4)
    lut[:, 3] = 1.0
    return np.ascontiguousarray(lut)


def make_height_colors(xyz, visual_range, color_lut):
    if len(xyz) == 0:
        return np.empty((0, 4), dtype=np.float32)
    normalized_z = (xyz[:, 2] + float(visual_range)) / (2.0 * float(visual_range))
    indices = np.clip(normalized_z * 255.0, 0.0, 255.0).astype(np.uint8)
    return np.ascontiguousarray(color_lut[indices], dtype=np.float32)


def rgb_colorize_lidar_points(xyz_lidar, rgb, K, T_camera_cv_from_lidar):
    count = len(xyz_lidar)
    if count == 0:
        return np.empty((0, 4), dtype=np.float32)
    colors = np.empty((count, 4), dtype=np.float32)
    colors[:, :3] = 0.16
    colors[:, 3] = 1.0
    camera_cv = transform_points(xyz_lidar, T_camera_cv_from_lidar)
    z = camera_cv[:, 2]
    valid = np.isfinite(camera_cv).all(axis=1) & (z > 1e-4)
    if not np.any(valid):
        return colors
    valid_indices = np.flatnonzero(valid)
    points = camera_cv[valid_indices]
    projected = (K @ points.T).T
    u = projected[:, 0] / projected[:, 2]
    v = projected[:, 1] / projected[:, 2]
    height, width = rgb.shape[:2]
    inside = (
        (u >= 0.0) & (u <= width - 1) & (v >= 0.0) & (v <= height - 1)
        & np.isfinite(u) & np.isfinite(v)
    )
    if not np.any(inside):
        return colors
    target_indices = valid_indices[inside]
    px = np.clip(np.rint(u[inside]).astype(np.int32), 0, width - 1)
    py = np.clip(np.rint(v[inside]).astype(np.int32), 0, height - 1)
    colors[target_indices, :3] = rgb[py, px].astype(np.float32) / 255.0
    return np.ascontiguousarray(colors)


def rectangle_segments(xyxy, width, height):
    if xyxy is None:
        return np.empty((0, 2), dtype=np.float32)
    x1, y1, x2, y2 = map(float, xyxy)
    x1 = float(np.clip(x1, 0.0, width - 1.0)); x2 = float(np.clip(x2, 0.0, width - 1.0))
    y1 = float(np.clip(y1, 0.0, height - 1.0)); y2 = float(np.clip(y2, 0.0, height - 1.0))
    if x2 <= x1 or y2 <= y1:
        return np.empty((0, 2), dtype=np.float32)
    return np.asarray(
        [[x1,y1],[x2,y1],[x2,y1],[x2,y2],[x2,y2],[x1,y2],[x1,y2],[x1,y1]],
        dtype=np.float32,
    )


def clip_segment_near_plane(p0, p1, near_z=0.05):
    p0 = np.asarray(p0, dtype=np.float64).copy(); p1 = np.asarray(p1, dtype=np.float64).copy()
    front0 = p0[2] >= near_z; front1 = p1[2] >= near_z
    if front0 and front1:
        return p0, p1
    if not front0 and not front1:
        return None
    dz = p1[2] - p0[2]
    if abs(dz) < 1e-12:
        return None
    t = (near_z - p0[2]) / dz
    if not 0.0 <= t <= 1.0:
        return None
    q = p0 + t * (p1 - p0); q[2] = near_z
    return (p0, q) if front0 else (q, p1)


def project_camera_point(point, K):
    uvw = K @ np.asarray(point, dtype=np.float64)
    if abs(uvw[2]) < 1e-12:
        return None
    return np.asarray([uvw[0] / uvw[2], uvw[1] / uvw[2]], dtype=np.float64)


def liang_barsky_clip(a, b, width, height):
    x0, y0 = map(float, a); x1, y1 = map(float, b)
    dx = x1 - x0; dy = y1 - y0
    p = (-dx, dx, -dy, dy)
    q = (x0, (width - 1.0) - x0, y0, (height - 1.0) - y0)
    u1 = 0.0; u2 = 1.0
    for pi, qi in zip(p, q):
        if abs(pi) < 1e-12:
            if qi < 0.0:
                return None
            continue
        t = qi / pi
        if pi < 0.0:
            if t > u2:
                return None
            u1 = max(u1, t)
        else:
            if t < u1:
                return None
            u2 = min(u2, t)
    return (
        np.asarray([x0 + u1 * dx, y0 + u1 * dy], dtype=np.float64),
        np.asarray([x0 + u2 * dx, y0 + u2 * dy], dtype=np.float64),
    )


def projected_cuboid_segments(corners_camera_cv, K, width, height, edges=BBOX_EDGES):
    corners = np.asarray(corners_camera_cv, dtype=np.float64)
    if corners.shape != (8, 3):
        return np.empty((0, 2), dtype=np.float32)
    output = []
    for i0, i1 in edges:
        clipped = clip_segment_near_plane(corners[i0], corners[i1])
        if clipped is None:
            continue
        a = project_camera_point(clipped[0], K); b = project_camera_point(clipped[1], K)
        if a is None or b is None:
            continue
        clipped_2d = liang_barsky_clip(a, b, width, height)
        if clipped_2d is not None:
            output.extend([clipped_2d[0], clipped_2d[1]])
    return np.asarray(output, dtype=np.float32) if output else np.empty((0, 2), dtype=np.float32)


def projected_bbox_from_segments(segments, width, height):
    if segments is None or len(segments) == 0:
        return np.empty((0, 2), dtype=np.float32)
    x1, y1 = np.min(segments, axis=0)
    x2, y2 = np.max(segments, axis=0)
    return rectangle_segments((x1, y1, x2, y2), width, height)


def lidar_cuboid_segments(corners_lidar, edges=BBOX_EDGES):
    corners = np.asarray(corners_lidar, dtype=np.float32)
    if corners.shape != (8, 3):
        return np.empty((0, 3), dtype=np.float32)
    output = []
    for i0, i1 in edges:
        output.extend([corners[i0], corners[i1]])
    return np.asarray(output, dtype=np.float32)


def bbox_intersects_display_cube(corners_lidar, visual_range):
    corners = np.asarray(corners_lidar, dtype=np.float64)
    if corners.shape != (8, 3):
        return False
    box_min = corners.min(axis=0); box_max = corners.max(axis=0); r = float(visual_range)
    return bool(np.all(box_max >= -r) and np.all(box_min <= r))


def camera_frustum_segments_lidar(K, T_camera_cv_from_lidar, image_width, image_height, depth):
    fx, fy = float(K[0,0]), float(K[1,1]); cx, cy = float(K[0,2]), float(K[1,2]); z = float(depth)
    pixels = ((0.0,0.0),(image_width-1.0,0.0),(image_width-1.0,image_height-1.0),(0.0,image_height-1.0))
    camera_points = [np.array([0.0,0.0,0.0,1.0], dtype=np.float64)]
    for u, v in pixels:
        camera_points.append(np.array([(u-cx)/fx*z, (v-cy)/fy*z, z, 1.0], dtype=np.float64))
    T_lidar_from_camera = np.linalg.inv(T_camera_cv_from_lidar)
    lidar_points = [(T_lidar_from_camera @ p)[:3] for p in camera_points]
    center, corners = lidar_points[0], lidar_points[1:]
    output = []
    for corner in corners:
        output.extend([center, corner])
    for i in range(4):
        output.extend([corners[i], corners[(i+1)%4]])
    return np.asarray(output, dtype=np.float32)


def concatenate_segments(values, dims):
    nonempty = [np.asarray(v, dtype=np.float32).reshape(-1, dims) for v in values if v is not None and len(v)]
    return np.concatenate(nonempty, axis=0) if nonempty else np.empty((0, dims), dtype=np.float32)


def object_passes_filter(obj, selected_classes, min_lidar_points, max_object_distance):
    cls = str(obj.get("class", "")).lower()
    if cls not in selected_classes:
        return False
    if obj.get("visibility", {}).get("lidar", None) is False:
        return False
    if int(obj.get("num_lidar_points", 0)) < min_lidar_points:
        return False
    distance = obj.get("distance_to_bbox_center_m")
    return distance is None or float(distance) <= max_object_distance


def _empty_overlay_dict(dims):
    return {cls: np.empty((0, dims), dtype=np.float32) for cls in CLASSES}


def _empty_text_dict():
    return {cls: [] for cls in CLASSES}, {cls: np.empty((0, 2), dtype=np.float32) for cls in CLASSES}


def build_gt_overlays(label_data, K, width, height, selected_classes, min_lidar_points, max_object_distance, visual_range, text_prefix=""):
    visible_lists = {cls: [] for cls in CLASSES}; projected_lists = {cls: [] for cls in CLASSES}
    rgb3d_lists = {cls: [] for cls in CLASSES}; lidar3d_lists = {cls: [] for cls in CLASSES}
    texts = {cls: [] for cls in CLASSES}; text_pos = {cls: [] for cls in CLASSES}
    raw_objects = label_data.get("objects", []); kept = 0; in_range = 0; outside = 0
    for obj in raw_objects:
        if not object_passes_filter(obj, selected_classes, min_lidar_points, max_object_distance):
            continue
        cls = str(obj.get("class", "")).lower()
        if cls not in CLASSES:
            continue
        kept += 1
        rgb_allowed = obj.get("visibility", {}).get("rgb", None) is not False
        current_projected = np.empty((0, 2), dtype=np.float32)
        if rgb_allowed:
            bbox2d = obj.get("bbox2d", {})
            vis = bbox2d.get("visible", {}).get("xyxy") if bbox2d.get("visible") else None
            proj = bbox2d.get("projected", {}).get("xyxy") if bbox2d.get("projected") else None
            if vis is not None:
                visible_lists[cls].append(rectangle_segments(vis, width, height))
            if proj is not None:
                current_projected = rectangle_segments(proj, width, height)
                projected_lists[cls].append(current_projected)
        bbox3d = obj.get("bbox3d", {})
        rgb_segments = np.empty((0,2), dtype=np.float32)
        corners_cv = bbox3d.get("camera_cv", {}).get("corners_xyz_m")
        if rgb_allowed and corners_cv is not None:
            rgb_segments = projected_cuboid_segments(corners_cv, K, width, height)
            if len(rgb_segments):
                rgb3d_lists[cls].append(rgb_segments)
        corners_lidar = bbox3d.get("lidar", {}).get("corners_xyz_m")
        if corners_lidar is not None:
            if bbox_intersects_display_cube(corners_lidar, visual_range):
                lidar3d_lists[cls].append(lidar_cuboid_segments(corners_lidar)); in_range += 1
            else:
                outside += 1
        anchor_segments = concatenate_segments([current_projected, rgb_segments], 2)
        if len(anchor_segments):
            anchor = np.min(anchor_segments, axis=0)
            texts[cls].append(
                "{}{} id={} pts={} rgbpx={}".format(
                    text_prefix, cls,
                    obj.get("actor_id", obj.get("environment_object_id", "?")),
                    obj.get("num_lidar_points", 0),
                    obj.get("num_rgb_visible_pixels", "-"),
                )
            )
            text_pos[cls].append(anchor)
    return {
        "visible_2d": {cls: concatenate_segments(visible_lists[cls], 2) for cls in CLASSES},
        "projected_2d": {cls: concatenate_segments(projected_lists[cls], 2) for cls in CLASSES},
        "rgb_3d": {cls: concatenate_segments(rgb3d_lists[cls], 2) for cls in CLASSES},
        "lidar_3d": {cls: concatenate_segments(lidar3d_lists[cls], 3) for cls in CLASSES},
        "text": texts,
        "text_pos": {cls: np.asarray(text_pos[cls], dtype=np.float32).reshape(-1,2) if text_pos[cls] else np.empty((0,2), dtype=np.float32) for cls in CLASSES},
        "stats": {"raw": len(raw_objects), "kept": kept, "in_range": in_range, "outside": outside},
    }


def build_pred_overlays(prediction_data, height_m, K, T_camera_cv_from_lidar, width, height, selected_classes, score_threshold, visual_range):
    projected_lists = {cls: [] for cls in CLASSES}; rgb3d_lists = {cls: [] for cls in CLASSES}; lidar3d_lists = {cls: [] for cls in CLASSES}
    texts = {cls: [] for cls in CLASSES}; text_pos = {cls: [] for cls in CLASSES}
    raw_detections = prediction_data.get("detections", []); kept = 0; in_range = 0; outside = 0
    for det in raw_detections:
        cls = str(det.get("class", "")).lower(); score = float(det.get("score", 0.0))
        if cls not in selected_classes or cls not in CLASSES or score < score_threshold:
            continue
        corners_ref = det.get("corners_3d_reference")
        if corners_ref is None:
            continue
        corners_lidar = reference_corners_to_raw_lidar(corners_ref, height_m)
        corners_cv = transform_points(corners_lidar, T_camera_cv_from_lidar)
        rgb_segments = projected_cuboid_segments(corners_cv, K, width, height, edges=MMDET_BBOX_EDGES)
        box2d_segments = projected_bbox_from_segments(rgb_segments, width, height)
        if len(rgb_segments):
            rgb3d_lists[cls].append(rgb_segments)
        if len(box2d_segments):
            projected_lists[cls].append(box2d_segments)
            text_pos[cls].append(np.min(box2d_segments, axis=0))
            texts[cls].append("P {} {:.2f}".format(cls, score))
        if bbox_intersects_display_cube(corners_lidar, visual_range):
            lidar3d_lists[cls].append(lidar_cuboid_segments(corners_lidar, edges=MMDET_BBOX_EDGES)); in_range += 1
        else:
            outside += 1
        kept += 1
    return {
        "visible_2d": _empty_overlay_dict(2),
        "projected_2d": {cls: concatenate_segments(projected_lists[cls], 2) for cls in CLASSES},
        "rgb_3d": {cls: concatenate_segments(rgb3d_lists[cls], 2) for cls in CLASSES},
        "lidar_3d": {cls: concatenate_segments(lidar3d_lists[cls], 3) for cls in CLASSES},
        "text": texts,
        "text_pos": {cls: np.asarray(text_pos[cls], dtype=np.float32).reshape(-1,2) if text_pos[cls] else np.empty((0,2), dtype=np.float32) for cls in CLASSES},
        "stats": {"raw": len(raw_detections), "kept": kept, "in_range": in_range, "outside": outside},
    }


@dataclass
class FrameData:
    rgb: np.ndarray
    xyz: np.ndarray
    height_colors: np.ndarray
    rgb_colors: np.ndarray
    original_count: int
    visible_count: int
    gt: dict
    pred: dict
    inference_ms: float
    reference_height_m: float


def load_frame_data(frame, args, color_lut, K, T_camera_cv_from_lidar, altitude_m):
    rgb = load_rgb(frame["rgb"])
    cloud = load_lidar(frame["lidar"])
    label = load_json(frame["label"])
    prediction = load_json(frame["prediction"])
    pose = load_json(frame["pose"])
    height_m = reference_height_m(pose, altitude_m)

    original_count = len(cloud)
    xyz = cloud[:, :3]
    xyz = xyz[np.isfinite(xyz).all(axis=1)]
    r = float(args.range)
    mask = np.all((xyz >= -r) & (xyz <= r), axis=1)
    xyz = xyz[mask]
    visible_count = len(xyz)
    xyz = np.ascontiguousarray(display_sample(xyz, args.max_points), dtype=np.float32)
    height_colors = make_height_colors(xyz, args.range, color_lut)
    rgb_colors = rgb_colorize_lidar_points(xyz, rgb, K, T_camera_cv_from_lidar)
    image_height, image_width = rgb.shape[:2]

    gt = build_gt_overlays(
        label, K, image_width, image_height, args.classes,
        args.min_lidar_points, args.max_object_distance, args.range,
        text_prefix="GT " if args.mode == "both" else "",
    )
    pred = build_pred_overlays(
        prediction, height_m, K, T_camera_cv_from_lidar,
        image_width, image_height, args.classes, args.score_threshold, args.range,
    )
    return FrameData(
        rgb=rgb, xyz=xyz, height_colors=height_colors, rgb_colors=rgb_colors,
        original_count=original_count, visible_count=visible_count,
        gt=gt, pred=pred,
        inference_ms=float(prediction.get("inference_ms", float("nan"))),
        reference_height_m=float(height_m),
    )


class FramePrefetcher:
    def __init__(self, frames, args, color_lut, K, T, altitude_m):
        self.frames = frames; self.args = args; self.color_lut = color_lut; self.K = K; self.T = T; self.altitude_m = altitude_m
        self.executor = ThreadPoolExecutor(max_workers=args.io_workers)
        self.futures = {}; self.closed = False

    def _submit(self, index):
        if self.closed or index in self.futures or not 0 <= index < len(self.frames):
            return
        self.futures[index] = self.executor.submit(
            load_frame_data, self.frames[index], self.args, self.color_lut, self.K, self.T, self.altitude_m
        )

    def prefetch(self, current):
        for offset in range(1, self.args.prefetch + 1):
            index = current + offset
            if index >= len(self.frames):
                if not self.args.loop:
                    break
                index %= len(self.frames)
            self._submit(index)

    def get(self, index):
        self._submit(index)
        future = self.futures.pop(index)
        data = future.result()
        self.prefetch(index)
        return data

    def close(self):
        self.closed = True
        for future in self.futures.values():
            future.cancel()
        self.futures.clear()
        self.executor.shutdown(wait=False, cancel_futures=True)


class UAVResultPlayer(QMainWindow):
    def __init__(self, args, frames, K, T_camera_cv_from_lidar, camera_frustum, altitude_m):
        super().__init__()
        self.args = args; self.frames = frames; self.K = K; self.T = T_camera_cv_from_lidar; self.camera_frustum = camera_frustum
        self.current_index = 0; self.paused = False; self.ended = False; self.closed = False
        self.show_gt = args.mode in ("gt", "both"); self.show_pred = args.mode in ("pred", "both")
        self.show_rgb_3d = True; self.show_lidar_3d = True; self.show_text = True; self.show_frustum = True
        self.point_color_mode_index = 0
        self.bbox_modes = ("visible", "projected", "off") if args.mode == "gt" else ("projected", "off")
        self.bbox_mode_index = 0
        self.actual_fps = 0.0; self.last_present_time = None; self.frame_pending_draw = False
        self.last_fetch_ms = 0.0; self.last_submit_ms = 0.0
        self.color_lut = build_viridis_lut()

        t0 = time.perf_counter()
        self.current_data = load_frame_data(frames[0], args, self.color_lut, K, T_camera_cv_from_lidar, altitude_m)
        self.last_fetch_ms = (time.perf_counter() - t0) * 1000.0
        self.prefetcher = FramePrefetcher(frames, args, self.color_lut, K, T_camera_cv_from_lidar, altitude_m)
        self.prefetcher.prefetch(0)

        self.setWindowTitle("UAV BEVFusion 测试结果播放器")
        self.resize(1600, 850)
        self.canvas = scene.SceneCanvas(
            title="UAV BEVFusion 测试结果播放器", size=(1600,850), bgcolor="#111318",
            keys=None, show=False, create_native=True, vsync=args.vsync,
        )
        self.setCentralWidget(self.canvas.native)
        self.canvas.events.draw.connect(self.on_canvas_draw)
        grid = self.canvas.central_widget.add_grid(margin=6, spacing=6)
        self.rgb_title = scene.Label("", color="white", font_size=12); self.rgb_title.height_max = 34; grid.add_widget(self.rgb_title, row=0, col=0)
        self.pc_title = scene.Label("", color="white", font_size=12); self.pc_title.height_max = 34; grid.add_widget(self.pc_title, row=0, col=1)

        self.rgb_view = grid.add_view(row=1, col=0, border_color="#3a3f4b", bgcolor="black")
        self.rgb_image = scene.visuals.Image(self.current_data.rgb, interpolation="nearest", method="subdivide", texture_format="auto", parent=self.rgb_view.scene)
        self.rgb_camera = scene.PanZoomCamera(aspect=1); self.rgb_camera.flip=(0,1,0); self.rgb_view.camera=self.rgb_camera
        self.last_rgb_shape = self.current_data.rgb.shape

        self.pc_view = grid.add_view(row=1, col=1, border_color="#3a3f4b", bgcolor="#080a0d")
        self.pc_camera = scene.TurntableCamera(fov=0.0, elevation=25.0, azimuth=-60.0, roll=0.0, center=(0,0,0), up="+z")
        self.pc_view.camera = self.pc_camera
        self.cloud_visual = scene.visuals.Markers(method="points", scaling="fixed", antialias=0, spherical=False, parent=self.pc_view.scene)

        self.visuals = {source: {"2d": {}, "rgb3d": {}, "lidar3d": {}, "text": {}} for source in ("gt", "pred")}
        for source in ("gt", "pred"):
            for cls in CLASSES:
                color = self.source_color(source, cls)
                self.visuals[source]["2d"][cls] = scene.visuals.Line(
                    pos=np.zeros((2,2), dtype=np.float32), color=color, width=2.0,
                    connect="segments", method="gl", parent=self.rgb_view.scene,
                )
                self.visuals[source]["rgb3d"][cls] = scene.visuals.Line(
                    pos=np.zeros((2,2), dtype=np.float32), color=color, width=1.5,
                    connect="segments", method="gl", parent=self.rgb_view.scene,
                )
                self.visuals[source]["lidar3d"][cls] = scene.visuals.Line(
                    pos=np.zeros((2,3), dtype=np.float32), color=color, width=2.0,
                    connect="segments", method="gl", parent=self.pc_view.scene,
                )
                self.visuals[source]["text"][cls] = scene.visuals.Text(
                    text="", pos=(0,0), color=color, font_size=8, anchor_x="left", anchor_y="bottom", parent=self.rgb_view.scene,
                )
                for kind in ("2d", "rgb3d", "lidar3d", "text"):
                    self.visuals[source][kind][cls].visible = False

        self.frustum_visual = scene.visuals.Line(pos=self.camera_frustum, color=(1,1,1,0.9), width=1.5, connect="segments", method="gl", parent=self.pc_view.scene)
        self.origin_visual = scene.visuals.Markers(method="points", scaling="fixed", antialias=0, spherical=False, parent=self.pc_view.scene)
        self.origin_visual.set_data(pos=np.array([[0,0,0]], dtype=np.float32), face_color="white", edge_color="white", edge_width=0, size=10.0, symbol="x")
        self.xyz_axis = scene.visuals.XYZAxis(parent=self.pc_view.scene)
        axis_size = float(args.range) * 0.25
        self.xyz_axis.transform = scene.transforms.STTransform(scale=(axis_size,axis_size,axis_size))

        self.status_label = scene.Label("", color="#d8dbe2", font_size=9); self.status_label.height_max = 48; grid.add_widget(self.status_label, row=2, col=0, col_span=2)
        self.shortcuts=[]
        for key, callback in (
            ("Space", self.toggle_pause), ("R", self.reset_pc_camera), ("B", self.toggle_bbox_mode),
            ("C", self.toggle_rgb_3d), ("L", self.toggle_lidar_3d), ("T", self.toggle_text),
            ("P", self.toggle_point_color), ("F", self.toggle_frustum), ("G", self.toggle_gt),
            ("D", self.toggle_pred), ("Q", self.close), ("Esc", self.close),
        ):
            self.add_shortcut(key, callback)

        self.reset_rgb_camera(); self.reset_pc_camera(); self.update_frame_visuals(self.current_data)
        self.timer = QTimer(self); self.timer.setTimerType(Qt.TimerType.PreciseTimer); self.timer.timeout.connect(self.advance_frame)
        self.timer.setInterval(max(1, int(round(1000.0 / args.fps)))); self.frame_pending_draw=True; self.timer.start()

    @property
    def bbox_mode(self): return self.bbox_modes[self.bbox_mode_index]
    @property
    def point_color_mode(self): return POINT_COLOR_MODES[self.point_color_mode_index]

    def source_color(self, source, cls):
        if self.args.mode == "both":
            return BOTH_GT_COLORS[cls] if source == "gt" else BOTH_PRED_COLORS[cls]
        return CLASS_COLORS[cls]

    def add_shortcut(self, sequence, callback):
        shortcut = QShortcut(QKeySequence(sequence), self); shortcut.activated.connect(callback); self.shortcuts.append(shortcut)

    def reset_rgb_camera(self):
        h,w = self.current_data.rgb.shape[:2]; self.rgb_camera.set_range(x=(0,w), y=(0,h), margin=0.0)

    def reset_pc_camera(self):
        r=float(self.args.range); self.pc_camera.fov=0.0; self.pc_camera.elevation=25.0; self.pc_camera.azimuth=-60.0; self.pc_camera.roll=0.0; self.pc_camera.center=(0,0,0)
        self.pc_camera.set_range(x=(-r,r), y=(-r,r), z=(-r,r), margin=0.02); self.canvas.update()

    @staticmethod
    def set_line(visual, positions, enabled):
        has_data = positions is not None and len(positions)>0
        if has_data: visual.set_data(pos=positions, connect="segments")
        visual.visible = bool(enabled and has_data)

    def current_point_colors(self): return self.current_data.rgb_colors if self.point_color_mode=="rgb" else self.current_data.height_colors

    def update_source(self, source, enabled):
        data = self.current_data.gt if source=="gt" else self.current_data.pred
        for cls in CLASSES:
            if self.bbox_mode == "off":
                box_pos = np.empty((0,2), dtype=np.float32)
            elif source == "gt" and self.args.mode == "gt" and self.bbox_mode == "visible":
                box_pos = data["visible_2d"][cls]
            else:
                box_pos = data["projected_2d"][cls]
            self.set_line(self.visuals[source]["2d"][cls], box_pos, enabled and self.bbox_mode!="off")
            self.set_line(self.visuals[source]["rgb3d"][cls], data["rgb_3d"][cls], enabled and self.show_rgb_3d)
            self.set_line(self.visuals[source]["lidar3d"][cls], data["lidar_3d"][cls], enabled and self.show_lidar_3d)
            text_visual = self.visuals[source]["text"][cls]
            texts=data["text"][cls]; positions=data["text_pos"][cls]
            if enabled and self.show_text and len(texts) and len(positions):
                text_visual.text=texts; text_visual.pos=positions; text_visual.visible=True
            else:
                text_visual.visible=False

    def update_overlays(self):
        self.update_source("gt", self.show_gt); self.update_source("pred", self.show_pred); self.frustum_visual.visible=bool(self.show_frustum)

    def update_titles(self):
        gt=self.current_data.gt["stats"]; pred=self.current_data.pred["stats"]; fid=self.frames[self.current_index]["id"]
        self.rgb_title.text = (
            "RGB | Frame {} | mode={} | GT {}/{} | Pred {}/{} @score>={:.2f} | 2D {} | RGB-3D {}".format(
                fid, self.args.mode, gt["kept"], gt["raw"], pred["kept"], pred["raw"], self.args.score_threshold,
                self.bbox_mode, "ON" if self.show_rgb_3d else "OFF"
            )
        )
        self.pc_title.text = (
            "LiDAR | Frame {} | {:,} raw | {:,} displayed | color={} | GT boxes={} | Pred boxes={} | infer={:.2f} ms".format(
                fid, self.current_data.original_count, len(self.current_data.xyz), self.point_color_mode,
                gt["in_range"], pred["in_range"], self.current_data.inference_ms
            )
        )

    def update_status(self):
        state="END" if self.ended else ("PAUSED" if self.paused else "PLAYING")
        self.status_label.text = (
            "{}/{} | {} | target {:.1f} FPS | draw {:.1f} FPS | fetch {:.2f} ms | submit {:.2f} ms | "
            "GT={} Pred={} | B 2D | C RGB-3D | L LiDAR-3D | G GT | D Pred | P point-color | F frustum | T text".format(
                self.current_index+1, len(self.frames), state, self.args.fps, self.actual_fps,
                self.last_fetch_ms, self.last_submit_ms, "ON" if self.show_gt else "OFF", "ON" if self.show_pred else "OFF"
            )
        )

    def update_frame_visuals(self, data, reset_rgb=False):
        self.rgb_image.set_data(data.rgb)
        if reset_rgb: self.reset_rgb_camera()
        self.cloud_visual.set_data(pos=data.xyz, face_color=self.current_point_colors(), edge_width=0, size=self.args.point_size, symbol="disc")
        self.update_overlays(); self.update_titles(); self.update_status()

    def on_canvas_draw(self, event):
        if not self.frame_pending_draw: return
        now=time.perf_counter(); previous=self.last_present_time; self.last_present_time=now; self.frame_pending_draw=False
        if previous is None: return
        dt=now-previous
        if dt<=0: return
        fps_now=1.0/dt; self.actual_fps=fps_now if self.actual_fps<=0 else self.actual_fps*0.85+fps_now*0.15

    def advance_frame(self):
        if self.paused or self.closed: return
        next_index=self.current_index+1
        if next_index>=len(self.frames):
            if self.args.loop: next_index=0
            else:
                self.ended=True; self.timer.stop(); self.update_status(); self.canvas.update(); return
        t0=time.perf_counter(); data=self.prefetcher.get(next_index); self.last_fetch_ms=(time.perf_counter()-t0)*1000.0
        t0=time.perf_counter(); self.current_index=next_index; self.current_data=data
        shape_changed=data.rgb.shape!=self.last_rgb_shape
        if shape_changed: self.last_rgb_shape=data.rgb.shape
        self.update_frame_visuals(data, reset_rgb=shape_changed); self.last_submit_ms=(time.perf_counter()-t0)*1000.0
        self.frame_pending_draw=True; self.canvas.update()

    def toggle_pause(self):
        if self.ended: return
        self.paused=not self.paused
        if self.paused: self.timer.stop()
        else: self.last_present_time=None; self.frame_pending_draw=False; self.timer.start()
        self.update_status(); self.canvas.update()

    def toggle_bbox_mode(self):
        self.bbox_mode_index=(self.bbox_mode_index+1)%len(self.bbox_modes); self.update_overlays(); self.update_titles(); self.canvas.update()

    def toggle_rgb_3d(self): self.show_rgb_3d=not self.show_rgb_3d; self.update_overlays(); self.update_titles(); self.canvas.update()
    def toggle_lidar_3d(self): self.show_lidar_3d=not self.show_lidar_3d; self.update_overlays(); self.update_titles(); self.canvas.update()
    def toggle_text(self): self.show_text=not self.show_text; self.update_overlays(); self.update_status(); self.canvas.update()
    def toggle_gt(self): self.show_gt=not self.show_gt; self.update_overlays(); self.update_status(); self.canvas.update()
    def toggle_pred(self): self.show_pred=not self.show_pred; self.update_overlays(); self.update_status(); self.canvas.update()

    def toggle_point_color(self):
        self.point_color_mode_index=(self.point_color_mode_index+1)%len(POINT_COLOR_MODES)
        self.cloud_visual.set_data(pos=self.current_data.xyz, face_color=self.current_point_colors(), edge_width=0, size=self.args.point_size, symbol="disc")
        self.update_titles(); self.canvas.update()

    def toggle_frustum(self): self.show_frustum=not self.show_frustum; self.frustum_visual.visible=self.show_frustum; self.update_status(); self.canvas.update()

    def closeEvent(self, event):
        if not self.closed:
            self.closed=True; self.timer.stop(); self.prefetcher.close(); self.canvas.close()
        event.accept()



def print_export_summary(pred_dir):
    """打印 WSL2 测试脚本已经计算好的计时和整体评估指标。"""
    summary_path = pred_dir / "summary.json"
    metrics_path = pred_dir / "metrics.json"

    print()
    print("=" * 78)
    print("BEVFusion 测试结果摘要")
    print("=" * 78)

    if summary_path.exists():
        try:
            summary = load_json(summary_path)
            timing = summary.get("timing", {})
            if timing:
                print("测试样本数       :", timing.get("samples", "-"))
                total_ms = timing.get("total_inference_ms")
                if total_ms is not None:
                    print("模型推理总时间   : {:.3f} s".format(float(total_ms) / 1000.0))
                avg_ms = timing.get("average_inference_ms")
                if avg_ms is not None:
                    print("平均单帧推理时间 : {:.3f} ms".format(float(avg_ms)))
                median_ms = timing.get("median_inference_ms")
                if median_ms is not None:
                    print("中位单帧推理时间 : {:.3f} ms".format(float(median_ms)))
                fps = timing.get("inference_fps")
                if fps is not None:
                    print("模型推理 FPS      : {:.3f}".format(float(fps)))

            wall = summary.get("whole_test_wall_time_s")
            if wall is not None:
                print("完整 test 循环时间: {:.3f} s".format(float(wall)))

            evaluation = summary.get("evaluation", {})
            if evaluation:
                print(
                    "评估阈值         : IoU={}  Score={}".format(
                        evaluation.get("iou_threshold", "-"),
                        evaluation.get("score_threshold", "-"),
                    )
                )
        except Exception as exc:
            print("读取 summary.json 失败:", exc)
    else:
        print("未找到 summary.json（不影响播放）。")

    if metrics_path.exists():
        try:
            metrics = load_json(metrics_path)
            if metrics:
                print()
                print("整体评估指标：")
                for key in sorted(metrics):
                    value = metrics[key]
                    if isinstance(value, (int, float)):
                        print("  {:<36} {:.6f}".format(str(key), float(value)))
                    else:
                        print("  {:<36} {}".format(str(key), value))
        except Exception as exc:
            print("读取 metrics.json 失败:", exc)
    else:
        print("未找到 metrics.json（不影响播放）。")

    print("=" * 78)
    print()


def main():
    args=parse_args(); validate_args(args)
    scene_dir=Path(args.scene).expanduser().resolve(); pred_dir=Path(args.pred_dir).expanduser().resolve()
    if not scene_dir.is_dir(): raise NotADirectoryError(scene_dir)
    if not pred_dir.is_dir(): raise NotADirectoryError(pred_dir)
    metadata_path = scene_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError("场景缺少 metadata.json: {}".format(metadata_path))
    metadata = load_json(metadata_path)
    if "uav_altitude_above_road_m" not in metadata:
        raise KeyError(
            "metadata.json 缺少 uav_altitude_above_road_m；"
            "无法把 BEVFusion ground-reference 预测框转换回 CARLA LiDAR 坐标。"
        )
    altitude_m = float(metadata["uav_altitude_above_road_m"])
    _,K,T,width,height=load_calibration(scene_dir)
    frames=resolve_test_frames(scene_dir, pred_dir)
    if args.start>=len(frames): raise ValueError("--start {} is outside {} test frames".format(args.start, len(frames)))
    frames=frames[args.start:]
    frustum=camera_frustum_segments_lidar(K,T,width,height,args.frustum_depth)
    print()
    print("=" * 78)
    print("UAV BEVFusion 测试结果播放器（Windows）")
    print("=" * 78)
    print("原始场景目录     : {}".format(scene_dir))
    print("预测结果目录     : {}".format(pred_dir))
    print("显示模式         : {}".format(args.mode))
    print("Test keyframes   : {}".format(len(frames)))
    print("预测分数阈值     : {:.3f}".format(args.score_threshold))
    print("显示类别         : {}".format(", ".join(args.classes)))
    print()
    print("快捷键：")
    print("  Space  暂停 / 继续")
    print("  B      2D 框模式切换")
    print("  C      RGB 上的 3D 框开 / 关")
    print("  L      LiDAR 3D 框开 / 关")
    print("  T      文字开 / 关")
    print("  P      点云 RGB / Height 着色")
    print("  F      相机视锥开 / 关")
    print("  G      GT 开 / 关")
    print("  D      Prediction 开 / 关")
    print("  R      重置 LiDAR 视角")
    print("  Q/Esc  退出")
    print("=" * 78)
    print()

    print_export_summary(pred_dir)
    app = use_app()
    app.create()
    window = UAVResultPlayer(args, frames, K, T, frustum, altitude_m)
    window.show()
    app.run()


if __name__ == "__main__":
    main()
