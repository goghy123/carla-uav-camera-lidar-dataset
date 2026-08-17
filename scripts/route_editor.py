import carla

import argparse
import copy
import math
import re
import time
import msvcrt

from pathlib import Path
from ruamel.yaml import YAML

from carla_route_planner import GlobalRoutePlanner


########################## 路径：定义项目根目录和配置文件位置 ################################

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "UAVdataset.yaml"


########################## 配置读取：加载 YAML 配置并转换为程序数据 ################################

yaml = YAML()
yaml.preserve_quotes = True
yaml.indent(mapping=2, sequence=4, offset=2)


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.load(f)


def save_yaml(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f)


def normalize_route_name(value):
    name = str(value).strip()
    if name.lower().endswith(".yaml"):
        name = name[:-5]

    if not name:
        raise ValueError("Route name cannot be empty.")

    # Route files stay flat under configs/routes. Existing files with the same
    # name are intentionally reused/overwritten without an overwrite prompt.
    if name in {".", ".."} or re.search(r'[<>:"/\\|?*]', name):
        raise ValueError(
            "Route name contains invalid filename characters: " + name
        )

    return name


def route_path_for_name(route_name):
    return (CONFIG_PATH.parent / "routes" / f"{route_name}.yaml").resolve()


def route_name_from_config(config):
    route_file = str(config["uav"]["route_file"])
    return normalize_route_name(Path(route_file).stem)


def make_empty_route_config(route_name):
    return {
        "route": {
            "name": route_name,
            "map": None,
            "anchors": [],
            "planned_path": [],
        }
    }


########################## 路线数据：道路锚点和规划路径 ################################


def map_short_name(world_map):
    return str(world_map.name).replace("\\", "/").split("/")[-1]


def make_location(point, z_offset=0.0):
    return carla.Location(
        x=float(point["x"]),
        y=float(point["y"]),
        z=float(point["z"]) + float(z_offset),
    )


def xy_distance(a, b):
    return math.hypot(
        float(b["x"]) - float(a["x"]),
        float(b["y"]) - float(a["y"]),
    )


def polyline_xy_length(points):
    return sum(
        xy_distance(points[i], points[i + 1])
        for i in range(len(points) - 1)
    )


def validate_planned_path_geometry(points, sampling_resolution_m):
    if len(points) < 2:
        return

    max_segment = max(
        xy_distance(points[i], points[i + 1])
        for i in range(len(points) - 1)
    )
    allowed_max = max(10.0, float(sampling_resolution_m) * 20.0)

    if max_segment > allowed_max:
        raise RuntimeError(
            "Planned path contains an unexpected XY jump: "
            f"max_segment={max_segment:.3f} m, "
            f"allowed={allowed_max:.3f} m. Inspect the selected lanes/anchors "
            "before saving."
        )


def validate_route_schema(route_root, route_path):
    """Require the current route schema only when editing an existing route."""
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
    if not isinstance(route["map"], str):
        raise ValueError(f"Route file {route_path}: route.map must be a string.")
    if not isinstance(route["anchors"], list):
        raise ValueError(f"Route file {route_path}: route.anchors must be a list.")
    if not isinstance(route["planned_path"], list):
        raise ValueError(f"Route file {route_path}: route.planned_path must be a list.")

    return route


def waypoint_to_anchor(waypoint):
    location = waypoint.transform.location
    return {
        "x": round(float(location.x), 3),
        "y": round(float(location.y), 3),
        "z": round(float(location.z), 3),
        "road_id": int(waypoint.road_id),
        "section_id": int(waypoint.section_id),
        "lane_id": int(waypoint.lane_id),
        "s": round(float(waypoint.s), 3),
    }


def waypoint_to_path_point(waypoint):
    location = waypoint.transform.location
    return {
        "x": round(float(location.x), 3),
        "y": round(float(location.y), 3),
        "z": round(float(location.z), 3),
    }


def anchor_to_path_point(anchor):
    return {
        "x": round(float(anchor["x"]), 3),
        "y": round(float(anchor["y"]), 3),
        "z": round(float(anchor["z"]), 3),
    }


def append_unique_point(points, point, xy_epsilon=1e-4, z_epsilon=1e-4):
    if points:
        last = points[-1]
        if (
            xy_distance(last, point) <= xy_epsilon
            and abs(float(last["z"]) - float(point["z"])) <= z_epsilon
        ):
            return
    points.append(point)


def plan_full_route(planner, anchors):
    """Plan all A0->A1->... segments and return one immutable road polyline."""
    if len(anchors) < 2:
        return []

    planned_path = []

    for segment_index in range(len(anchors) - 1):
        start_anchor = anchors[segment_index]
        end_anchor = anchors[segment_index + 1]
        start_location = make_location(start_anchor)
        end_location = make_location(end_anchor)

        try:
            trace = planner.trace_route(start_location, end_location)
        except Exception as exc:
            raise RuntimeError(
                f"Planning failed for A{segment_index:02d} -> "
                f"A{segment_index + 1:02d}: {exc}"
            ) from exc

        if not trace:
            raise RuntimeError(
                f"Planning failed for A{segment_index:02d} -> "
                f"A{segment_index + 1:02d}: empty route returned."
            )

        # trace_route may stop within its sampling tolerance of the destination.
        # Explicitly keep both projected anchors so the stored path passes every
        # required anchor exactly.
        append_unique_point(planned_path, anchor_to_path_point(start_anchor))

        for waypoint, _road_option in trace:
            append_unique_point(planned_path, waypoint_to_path_point(waypoint))

        append_unique_point(planned_path, anchor_to_path_point(end_anchor))

    if len(planned_path) < 2 or polyline_xy_length(planned_path) <= 1e-6:
        raise RuntimeError("Planned path is empty or has zero XY length.")

    return planned_path


def decimate_polyline(points, min_spacing_m):
    if len(points) <= 2:
        return list(points)

    result = [points[0]]
    accumulated = 0.0

    for i in range(1, len(points)):
        accumulated += xy_distance(points[i - 1], points[i])
        if accumulated >= min_spacing_m:
            result.append(points[i])
            accumulated = 0.0

    if result[-1] is not points[-1]:
        last = points[-1]
        if xy_distance(result[-1], last) > 1e-6:
            result.append(last)

    return result


########################## 路线绘制：地面 Anchor + 空中 UAV 实际轨迹 ################################


def draw_route(world, anchors, planned_path, altitude_m, life_time=1.0):
    debug = world.debug

    anchor_color = carla.Color(0, 255, 0)
    route_color = carla.Color(0, 180, 255)
    text_color = carla.Color(255, 255, 255)
    arrow_color = carla.Color(255, 180, 0)

    for i, anchor in enumerate(anchors):
        ground_location = make_location(anchor, z_offset=0.35)

        debug.draw_point(
            ground_location,
            size=0.25,
            color=anchor_color,
            life_time=life_time,
        )

        debug.draw_string(
            carla.Location(
                x=ground_location.x,
                y=ground_location.y,
                z=ground_location.z + 0.8,
            ),
            f"A{i:02d}",
            draw_shadow=True,
            color=text_color,
            life_time=life_time,
        )

    if len(planned_path) < 2:
        return

    # The saved path remains dense (normally 0.5 m). Only the debug drawing is
    # decimated so long routes do not flood CARLA with thousands of draw calls.
    display_path = decimate_polyline(planned_path, min_spacing_m=2.0)

    for i in range(len(display_path) - 1):
        start = make_location(display_path[i], z_offset=altitude_m)
        end = make_location(display_path[i + 1], z_offset=altitude_m)

        debug.draw_line(
            start,
            end,
            thickness=0.06,
            color=route_color,
            life_time=life_time,
        )

    # Direction arrows every ~20 m along the UAV path.
    distance_since_arrow = 1e9
    for i in range(len(display_path) - 1):
        segment_length = xy_distance(display_path[i], display_path[i + 1])
        distance_since_arrow += segment_length

        if distance_since_arrow < 20.0:
            continue

        start = make_location(display_path[i], z_offset=altitude_m)
        end = make_location(display_path[i + 1], z_offset=altitude_m)

        debug.draw_arrow(
            start,
            end,
            thickness=0.08,
            arrow_size=0.25,
            color=arrow_color,
            life_time=life_time,
        )
        distance_since_arrow = 0.0


########################## 打印路线：输出锚点、规划点数和路线长度 ################################


def print_route(
    route_name,
    route_map,
    anchors,
    planned_path,
    altitude_m,
    sampling_resolution,
    lookahead_m,
):
    print("\n============================================")
    print("Route:", route_name)
    print("Map:", route_map or "not_saved_yet")
    print("Anchors:", len(anchors))
    print("Planned path points:", len(planned_path))
    print("Planned XY length:", f"{polyline_xy_length(planned_path):.3f} m")
    print("Planner resolution:", f"{sampling_resolution:.3f} m")
    print("UAV altitude above road:", f"{altitude_m:.3f} m")
    print("Heading lookahead:", f"{lookahead_m:.3f} m")
    print("============================================")

    for i, anchor in enumerate(anchors):
        print(
            f"A{i:02d}: "
            f"x={float(anchor['x']):8.3f}  "
            f"y={float(anchor['y']):8.3f}  "
            f"z={float(anchor['z']):7.3f}  "
            f"road={int(anchor['road_id'])}  "
            f"section={int(anchor['section_id'])}  "
            f"lane={int(anchor['lane_id'])}  "
            f"s={float(anchor['s']):.3f}"
        )

    print()


########################## 保存：仅保存完整有效路线 ################################


def save_current_route(
    route_path,
    config,
    route_name,
    anchors,
    planned_path,
    current_map_name,
    sampling_resolution,
):
    if len(anchors) < 2:
        print("\nCannot save: at least 2 anchors are required.")
        return False

    if len(planned_path) < 2 or polyline_xy_length(planned_path) <= 1e-6:
        print("\nCannot save: planned_path is empty or invalid.")
        return False

    try:
        validate_planned_path_geometry(
            planned_path,
            sampling_resolution,
        )
    except RuntimeError as exc:
        print(f"\nCannot save: {exc}")
        return False

    # Route YAML stores route-specific data only. Shared UAV/planner parameters
    # are read from configs/UAVdataset.yaml.
    route_config = {
        "route": {
            "name": route_name,
            "map": current_map_name,
            "anchors": copy.deepcopy(anchors),
            "planned_path": copy.deepcopy(planned_path),
        }
    }
    save_yaml(route_path, route_config)

    # The collector should use the route that was just saved.
    config["uav"]["route_file"] = f"routes/{route_name}.yaml"
    save_yaml(CONFIG_PATH, config)

    print("\nRoute saved:")
    print(route_path)
    print(
        f"anchors={len(anchors)}, "
        f"planned_path_points={len(planned_path)}, "
        f"xy_length={polyline_xy_length(planned_path):.3f} m"
    )
    return True


########################## 程序入口：连接 CARLA、规划并固化路线 ################################


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Edit the current CARLA UAV route, or pass a name to start a "
            "new configs/routes/<name>.yaml route."
        )
    )
    parser.add_argument(
        "route_name",
        nargs="?",
        help=(
            "New route name, e.g. route_02. If omitted, edit the route "
            "currently selected in UAVdataset.yaml. A same-name file is "
            "overwritten on save without confirmation."
        ),
    )
    args = parser.parse_args()

    config = load_yaml(CONFIG_PATH)

    if args.route_name is None:
        # No explicit name: edit the currently selected route.
        route_name = route_name_from_config(config)
        route_path = route_path_for_name(route_name)

        if route_path.exists():
            route_config = load_yaml(route_path)
            route = validate_route_schema(route_config, route_path)
            if route["name"] != route_name:
                raise ValueError(
                    f"Route file name and route.name do not match: "
                    f"{route_path.name} vs {route['name']}"
                )
        else:
            route_config = make_empty_route_config(route_name)
            route = route_config["route"]
    else:
        # Explicit name: start a NEW empty route. If a file with the same name
        # already exists, S/Q overwrites it directly with no confirmation.
        route_name = normalize_route_name(args.route_name)
        route_path = route_path_for_name(route_name)
        route_config = make_empty_route_config(route_name)
        route = route_config["route"]

    uav_cfg = config["uav"]
    planner_cfg = config["route_planner"]

    altitude_m = float(uav_cfg["altitude_above_road_m"])
    sampling_resolution = float(planner_cfg["sampling_resolution_m"])
    lookahead_m = float(uav_cfg["heading_lookahead_m"])

    if altitude_m <= 0:
        raise ValueError("uav.altitude_above_road_m must be > 0")
    if sampling_resolution <= 0:
        raise ValueError(
            "route_planner.sampling_resolution_m must be > 0"
        )
    if lookahead_m <= 0:
        raise ValueError("uav.heading_lookahead_m must be > 0")

    anchors = copy.deepcopy(route.get("anchors") or [])
    planned_path = copy.deepcopy(route.get("planned_path") or [])

    client = carla.Client("localhost", 2000)
    client.set_timeout(20.0)

    world = client.get_world()
    world_map = world.get_map()
    spectator = world.get_spectator()
    current_map_name = map_short_name(world_map)

    saved_map_name = route.get("map")
    if saved_map_name and str(saved_map_name) != current_map_name:
        print(
            "\nWARNING: route file was saved for map "
            f"'{saved_map_name}', but CARLA is currently on "
            f"'{current_map_name}'. The working anchors/planned_path were "
            "cleared to prevent accidentally re-labeling a path from another map."
        )
        print(
            "The YAML file on disk is unchanged until you press S or Q."
        )
        anchors = []
        planned_path = []

    print("\nBuilding CARLA GlobalRoutePlanner...")
    planner = GlobalRoutePlanner(world_map, sampling_resolution)
    print("GlobalRoutePlanner ready.")

    print("\n============================================")
    print("UAV ROAD ROUTE EDITOR")
    print("============================================")
    print("Route name:", route_name)
    print("Route file:")
    print(route_path)
    print("Current CARLA map:", current_map_name)
    print("Planner resolution:", f"{sampling_resolution:.3f} m")
    print("UAV altitude:", f"road_z + {altitude_m:.3f} m")
    print("Heading lookahead:", f"{lookahead_m:.3f} m")
    print("\n在 CARLA 窗口移动 Spectator 到目标道路附近，然后切回终端按键：")
    print()
    print("A     将 Spectator 投影到最近 Driving lane，并添加为 Anchor")
    print("U     撤销最后一个 Anchor，并重新规划")
    print("P     打印当前 Anchor / planned_path 信息")
    print("S     保存 anchors + 完整 planned_path")
    print("C     清空当前路线")
    print("Q     保存并退出")
    print("ESC   不保存退出")
    print()
    print("绿色 Axx = 地面道路 Anchor")
    print("蓝色线    = 实际 UAV 路径（road_z + altitude）")
    print("橙色箭头  = 飞行方向")
    print()
    print("当前 Anchor 数量:", len(anchors))
    print("当前 planned_path 点数:", len(planned_path))
    print("\n等待操作...")

    last_draw_time = 0.0
    running = True

    while running:
        current_time = time.time()

        if current_time - last_draw_time > 0.45:
            draw_route(
                world,
                anchors,
                planned_path,
                altitude_m=altitude_m,
                life_time=0.9,
            )
            last_draw_time = current_time

        if not msvcrt.kbhit():
            time.sleep(0.02)
            continue

        key = msvcrt.getwch().lower()

        ########################## 添加 Anchor：吸附到 Driving lane 并立即重新规划 ################################

        if key == "a":
            spectator_location = spectator.get_transform().location

            waypoint = world_map.get_waypoint(
                spectator_location,
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )

            if waypoint is None:
                print("\nNo Driving waypoint found near the Spectator.")
                continue

            new_anchor = waypoint_to_anchor(waypoint)
            candidate_anchors = anchors + [new_anchor]

            try:
                candidate_path = plan_full_route(planner, candidate_anchors)
                validate_planned_path_geometry(
                    candidate_path,
                    sampling_resolution,
                )
            except RuntimeError as exc:
                print(f"\n{exc}")
                print("Anchor was NOT added.")
                continue

            anchors = candidate_anchors
            planned_path = candidate_path

            print("\nAdded:")
            print(
                f"A{len(anchors)-1:02d}  "
                f"x={new_anchor['x']:.3f}  "
                f"y={new_anchor['y']:.3f}  "
                f"z={new_anchor['z']:.3f}  "
                f"road={new_anchor['road_id']}  "
                f"section={new_anchor['section_id']}  "
                f"lane={new_anchor['lane_id']}  "
                f"s={new_anchor['s']:.3f}"
            )

            if len(anchors) >= 2:
                print(
                    f"Replanned: {len(planned_path)} points, "
                    f"{polyline_xy_length(planned_path):.3f} m"
                )

        ########################## 撤销：删除最近 Anchor 并立即重新规划 ################################

        elif key == "u":
            if not anchors:
                print("\nNo anchor to remove.")
                continue

            removed = anchors[-1]
            candidate_anchors = anchors[:-1]

            try:
                candidate_path = plan_full_route(planner, candidate_anchors)
                validate_planned_path_geometry(
                    candidate_path,
                    sampling_resolution,
                )
            except RuntimeError as exc:
                print(f"\nUnexpected re-planning failure: {exc}")
                print("Anchor was NOT removed.")
                continue

            anchors = candidate_anchors
            planned_path = candidate_path

            print("\nRemoved:")
            print(removed)
            print(
                f"Remaining anchors={len(anchors)}, "
                f"planned_path_points={len(planned_path)}"
            )

        ########################## 打印 ################################

        elif key == "p":
            print_route(
                route_name,
                route.get("map"),
                anchors,
                planned_path,
                altitude_m,
                sampling_resolution,
                lookahead_m,
            )

        ########################## 保存 ################################

        elif key == "s":
            save_current_route(
                route_path,
                config,
                route_name,
                anchors,
                planned_path,
                current_map_name,
                sampling_resolution,
            )

        ########################## 清空 ################################

        elif key == "c":
            anchors.clear()
            planned_path.clear()
            print("\nAll anchors and planned_path cleared in the editor.")

        ########################## 保存并退出 ################################

        elif key == "q":
            if save_current_route(
                route_path,
                config,
                route_name,
                anchors,
                planned_path,
                current_map_name,
                sampling_resolution,
            ):
                running = False

        ########################## 不保存退出 ################################

        elif ord(key) == 27:
            print("\nExit without saving.")
            running = False

    print("\nRoute editor closed.")


if __name__ == "__main__":
    main()
