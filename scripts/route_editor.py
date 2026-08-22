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
        raise ValueError("路线名称不能为空。")

    # Route files stay flat under configs/routes. Existing files with the same
    # name are intentionally reused/overwritten without an overwrite prompt.
    if name in {".", ".."} or re.search(r'[<>:"/\\|?*]', name):
        raise ValueError(
            "路线名称包含不能用于文件名的字符: " + name
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
        print("\n无法保存：至少需要 2 个 Anchor。")
        return False

    if len(planned_path) < 2 or polyline_xy_length(planned_path) <= 1e-6:
        print("\n无法保存：planned_path 为空或无效。")
        return False

    try:
        validate_planned_path_geometry(
            planned_path,
            sampling_resolution,
        )
    except RuntimeError as exc:
        print(f"\n无法保存：{exc}")
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

    print("\n路线已保存：")
    print(route_path)
    print(
        f"anchors={len(anchors)}, "
        f"planned_path_points={len(planned_path)}, "
        f"xy_length={polyline_xy_length(planned_path):.3f} m"
    )
    return True


########################## 地图与路线管理：切换地图、新建路线、加载路线 ################################


def map_name_from_value(value):
    """从 CARLA 地图完整路径或短名称中提取 TownXX/TownXX_Opt。"""
    return str(value).replace("\\", "/").split("/")[-1]


def get_available_map_entries(client):
    """返回 [(short_name, full_name), ...]，按短名称排序并去重。"""
    entries = []
    seen = set()

    for full_name in client.get_available_maps():
        short_name = map_name_from_value(full_name)
        key = short_name.lower()

        if key in seen:
            continue

        seen.add(key)
        entries.append((short_name, str(full_name)))

    entries.sort(key=lambda item: item[0].lower())
    return entries


def find_map_identifier(client, requested_map_name):
    """把 route.map 的短名称解析成 client.load_world() 可用的地图标识。"""
    requested = map_name_from_value(requested_map_name).lower()

    for short_name, full_name in get_available_map_entries(client):
        if short_name.lower() == requested:
            return full_name

    return None


def choose_map(client, current_map_name, allow_cancel=True):
    """中文地图选择菜单；支持序号、短名称、完整名称。"""
    entries = get_available_map_entries(client)

    if not entries:
        print("\nCARLA 服务器没有返回可用地图列表。")
        return None

    print("\n================ 可用 CARLA 地图 ================")
    for index, (short_name, _full_name) in enumerate(entries, start=1):
        marker = "  <- 当前" if short_name == current_map_name else ""
        print(f"[{index:02d}] {short_name}{marker}")
    print("==================================================")
    print("当前地图:", current_map_name)

    while True:
        if allow_cancel:
            raw = input(
                "\n请输入地图序号或名称（直接回车取消/保持当前地图）："
            ).strip()
        else:
            raw = input(
                "\n请输入地图序号或名称（直接回车保持当前地图）："
            ).strip()

        if not raw:
            return None

        if raw.isdigit():
            index = int(raw)
            if 1 <= index <= len(entries):
                return entries[index - 1]
            print("序号超出范围，请重新输入。")
            continue

        raw_lower = raw.lower()
        for short_name, full_name in entries:
            if (
                short_name.lower() == raw_lower
                or full_name.lower() == raw_lower
            ):
                return short_name, full_name

        print("没有找到该地图，请输入列表中的序号或地图名称。")


def prompt_yes_no(message, default=False):
    """读取中文 Y/N 确认。"""
    suffix = " [Y/n]：" if default else " [y/N]："

    while True:
        answer = input(message + suffix).strip().lower()

        if not answer:
            return default
        if answer in ("y", "yes", "是", "好", "确认"):
            return True
        if answer in ("n", "no", "否", "不", "取消"):
            return False

        print("请输入 Y 或 N。")


def build_world_context(client, sampling_resolution):
    """从当前 CARLA world 重新获取所有地图相关对象。"""
    world = client.get_world()
    world_map = world.get_map()
    spectator = world.get_spectator()
    current_map_name = map_short_name(world_map)

    print(f"\n正在为地图 {current_map_name} 构建 GlobalRoutePlanner ...")
    planner = GlobalRoutePlanner(world_map, sampling_resolution)
    print("GlobalRoutePlanner 已就绪。")

    return world, world_map, spectator, current_map_name, planner


def switch_world_map(client, map_identifier, sampling_resolution):
    """
    加载地图并完整刷新 world / map / spectator / planner。
    load_world() 会创建新的 CARLA world，因此旧引用不能继续使用。
    """
    target_short_name = map_name_from_value(map_identifier)

    print(f"\n正在加载地图：{target_short_name}")
    world = client.load_world(map_identifier)
    world_map = world.get_map()
    spectator = world.get_spectator()
    current_map_name = map_short_name(world_map)

    if current_map_name != target_short_name:
        print(
            "提示：CARLA 返回的地图短名称为 "
            f"'{current_map_name}'，将以实际加载结果为准。"
        )

    print(f"地图已加载：{current_map_name}")
    print("正在重建 GlobalRoutePlanner ...")
    planner = GlobalRoutePlanner(world_map, sampling_resolution)
    print("GlobalRoutePlanner 已就绪。")

    return world, world_map, spectator, current_map_name, planner


def choose_existing_route():
    """列出 configs/routes 下已有路线并返回所选路径；回车取消。"""
    routes_dir = CONFIG_PATH.parent / "routes"
    route_paths = sorted(
        routes_dir.glob("*.yaml"),
        key=lambda p: p.stem.lower(),
    )

    if not route_paths:
        print("\nconfigs/routes 中没有找到任何 .yaml 路线文件。")
        return None

    print("\n================ 已有路线 ================")
    for index, path in enumerate(route_paths, start=1):
        print(f"[{index:02d}] {path.stem}")
    print("==========================================")

    while True:
        raw = input(
            "\n请输入路线序号或名称（直接回车取消）："
        ).strip()

        if not raw:
            return None

        if raw.isdigit():
            index = int(raw)
            if 1 <= index <= len(route_paths):
                return route_paths[index - 1]
            print("序号超出范围，请重新输入。")
            continue

        try:
            wanted = normalize_route_name(raw)
        except ValueError as exc:
            print(exc)
            continue

        candidate = route_path_for_name(wanted)
        if candidate.exists():
            return candidate

        print("没有找到该路线，请重新输入。")


def print_editor_help(
    route_name,
    route_path,
    current_map_name,
    sampling_resolution,
    altitude_m,
    lookahead_m,
    anchors,
    planned_path,
):
    print("\n============================================")
    print("UAV 道路路线编辑器")
    print("============================================")
    print("路线名称:", route_name)
    print("路线文件:")
    print(route_path)
    print("当前 CARLA 地图:", current_map_name)
    print("规划采样间距:", f"{sampling_resolution:.3f} m")
    print("UAV 高度:", f"road_z + {altitude_m:.3f} m")
    print("朝向前视距离:", f"{lookahead_m:.3f} m")
    print()
    print("在 CARLA 窗口移动 Spectator 到目标道路附近，然后切回终端按键：")
    print()
    print("A     将 Spectator 投影到最近 Driving lane，并添加 Anchor")
    print("U     撤销最后一个 Anchor，并重新规划")
    print("P     打印当前 Anchor / planned_path 信息")
    print("M     更换 CARLA 地图")
    print("N     新建路线（同名保存时直接覆盖）")
    print("R     加载已有路线")
    print("S     保存 anchors + 完整 planned_path")
    print("C     清空当前工作路线")
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


########################## 程序入口：连接 CARLA、规划并固化路线 ################################


def main():
    parser = argparse.ArgumentParser(
        description=(
            "编辑当前 CARLA UAV 路线；也可以传入一个名称来新建 "
            "configs/routes/<name>.yaml。"
        )
    )
    parser.add_argument(
        "route_name",
        nargs="?",
        help=(
            "新路线名称，例如 route_02。省略时编辑 UAVdataset.yaml 当前选择的路线。"
            "同名文件在保存时直接覆盖，不额外确认。"
        ),
    )
    args = parser.parse_args()

    config = load_yaml(CONFIG_PATH)

    editing_existing_route = False

    if args.route_name is None:
        route_name = route_name_from_config(config)
        route_path = route_path_for_name(route_name)

        if route_path.exists():
            route_config = load_yaml(route_path)
            route = validate_route_schema(route_config, route_path)

            if route["name"] != route_name:
                raise ValueError(
                    "路线文件名和 route.name 不一致："
                    f"{route_path.name} vs {route['name']}"
                )

            editing_existing_route = True
        else:
            route_config = make_empty_route_config(route_name)
            route = route_config["route"]
    else:
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
        raise ValueError("uav.altitude_above_road_m 必须 > 0")
    if sampling_resolution <= 0:
        raise ValueError("route_planner.sampling_resolution_m 必须 > 0")
    if lookahead_m <= 0:
        raise ValueError("uav.heading_lookahead_m 必须 > 0")

    anchors = copy.deepcopy(route.get("anchors") or [])
    planned_path = copy.deepcopy(route.get("planned_path") or [])

    client = carla.Client("localhost", 2000)
    client.set_timeout(20.0)

    world, world_map, spectator, current_map_name, planner = (
        build_world_context(client, sampling_resolution)
    )

    saved_map_name = route.get("map")

    # 已有路线属于另一张地图时，优先询问是否自动切到路线地图。
    if (
        editing_existing_route
        and saved_map_name
        and str(saved_map_name) != current_map_name
    ):
        print(
            "\n当前路线保存在地图 "
            f"'{saved_map_name}'，但 CARLA 当前地图是 '{current_map_name}'。"
        )

        map_identifier = find_map_identifier(client, saved_map_name)

        if map_identifier is None:
            print(
                "CARLA 当前安装的地图列表中找不到该路线所需地图。"
                "为避免误用其他地图坐标，编辑器将清空内存中的工作路线；"
                "磁盘上的 YAML 不会被修改。"
            )
            anchors = []
            planned_path = []
        elif prompt_yes_no(
            f"是否自动切换到路线地图 '{saved_map_name}'？",
            default=True,
        ):
            (
                world,
                world_map,
                spectator,
                current_map_name,
                planner,
            ) = switch_world_map(
                client,
                map_identifier,
                sampling_resolution,
            )
        else:
            print(
                "未切换地图。为避免把旧地图坐标误当成当前地图坐标，"
                "内存中的 anchors/planned_path 已清空；磁盘 YAML 保持不变。"
            )
            anchors = []
            planned_path = []

    # 新建/空路线启动时提供一次地图选择。
    if not editing_existing_route or not anchors:
        print(
            "\n新建或空路线可以现在选择地图；"
            "直接回车则继续使用当前 CARLA 地图。"
        )
        selected = choose_map(
            client,
            current_map_name,
            allow_cancel=False,
        )

        if selected is not None:
            selected_short, selected_identifier = selected

            if selected_short != current_map_name:
                (
                    world,
                    world_map,
                    spectator,
                    current_map_name,
                    planner,
                ) = switch_world_map(
                    client,
                    selected_identifier,
                    sampling_resolution,
                )

    dirty = False

    print_editor_help(
        route_name,
        route_path,
        current_map_name,
        sampling_resolution,
        altitude_m,
        lookahead_m,
        anchors,
        planned_path,
    )

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

        ########################## 添加 Anchor ################################

        if key == "a":
            spectator_location = spectator.get_transform().location

            waypoint = world_map.get_waypoint(
                spectator_location,
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )

            if waypoint is None:
                print("\nSpectator 附近没有找到 Driving waypoint。")
                continue

            new_anchor = waypoint_to_anchor(waypoint)
            candidate_anchors = anchors + [new_anchor]

            try:
                candidate_path = plan_full_route(
                    planner,
                    candidate_anchors,
                )
                validate_planned_path_geometry(
                    candidate_path,
                    sampling_resolution,
                )
            except RuntimeError as exc:
                print(f"\n规划失败：{exc}")
                print("本次 Anchor 未添加。")
                continue

            anchors = candidate_anchors
            planned_path = candidate_path
            dirty = True

            print("\n已添加：")
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
                    f"已重新规划：{len(planned_path)} 个点，"
                    f"XY 长度 {polyline_xy_length(planned_path):.3f} m"
                )

        ########################## 撤销 ################################

        elif key == "u":
            if not anchors:
                print("\n当前没有可撤销的 Anchor。")
                continue

            removed = anchors[-1]
            candidate_anchors = anchors[:-1]

            try:
                candidate_path = plan_full_route(
                    planner,
                    candidate_anchors,
                )
                validate_planned_path_geometry(
                    candidate_path,
                    sampling_resolution,
                )
            except RuntimeError as exc:
                print(f"\n撤销后重新规划失败：{exc}")
                print("Anchor 未删除。")
                continue

            anchors = candidate_anchors
            planned_path = candidate_path
            dirty = True

            print("\n已撤销：")
            print(removed)
            print(
                f"剩余 anchors={len(anchors)}, "
                f"planned_path_points={len(planned_path)}"
            )

        ########################## 打印 ################################

        elif key == "p":
            print_route(
                route_name,
                current_map_name,
                anchors,
                planned_path,
                altitude_m,
                sampling_resolution,
                lookahead_m,
            )

        ########################## 更换地图 ################################

        elif key == "m":
            selected = choose_map(
                client,
                current_map_name,
                allow_cancel=True,
            )

            if selected is None:
                print("\n已取消更换地图。")
                continue

            selected_short, selected_identifier = selected

            if selected_short == current_map_name:
                print("\n选择的就是当前地图，不需要重新加载。")
                continue

            if anchors or planned_path:
                if not prompt_yes_no(
                    f"当前工作路线有 {len(anchors)} 个 Anchor。"
                    "切换地图会清空内存中的工作路线，磁盘 YAML 不会改变。"
                    "是否继续？",
                    default=False,
                ):
                    print("已取消更换地图。")
                    continue

            try:
                (
                    world,
                    world_map,
                    spectator,
                    current_map_name,
                    planner,
                ) = switch_world_map(
                    client,
                    selected_identifier,
                    sampling_resolution,
                )
            except Exception as exc:
                print(f"\n地图加载失败：{exc}")
                continue

            anchors = []
            planned_path = []
            dirty = True
            last_draw_time = 0.0

            print(
                f"\n已切换到 {current_map_name}。"
                "工作路线已清空；原路线 YAML 尚未被修改。"
            )

        ########################## 新建路线 ################################

        elif key == "n":
            if dirty and (anchors or planned_path):
                if not prompt_yes_no(
                    "当前路线有尚未保存的修改。新建路线会丢弃这些内存修改，"
                    "是否继续？",
                    default=False,
                ):
                    print("已取消新建路线。")
                    continue

            raw_name = input(
                "\n请输入新路线名称（直接回车取消）："
            ).strip()

            if not raw_name:
                print("已取消新建路线。")
                continue

            try:
                new_route_name = normalize_route_name(raw_name)
            except ValueError as exc:
                print(exc)
                continue

            route_name = new_route_name
            route_path = route_path_for_name(route_name)
            route_config = make_empty_route_config(route_name)
            route = route_config["route"]
            anchors = []
            planned_path = []
            dirty = False

            if route_path.exists():
                print(
                    f"\n注意：{route_path.name} 已存在。"
                    "后续按 S/Q 保存时会按你的习惯直接覆盖同名文件。"
                )

            print(
                f"\n已新建空路线：{route_name}\n"
                f"当前地图：{current_map_name}\n"
                "如需换地图，请按 M。"
            )

        ########################## 加载已有路线 ################################

        elif key == "r":
            if dirty and (anchors or planned_path):
                if not prompt_yes_no(
                    "当前路线有尚未保存的修改。加载其他路线会丢弃这些内存修改，"
                    "是否继续？",
                    default=False,
                ):
                    print("已取消加载路线。")
                    continue

            selected_route_path = choose_existing_route()

            if selected_route_path is None:
                print("已取消加载路线。")
                continue

            try:
                selected_route_config = load_yaml(selected_route_path)
                selected_route = validate_route_schema(
                    selected_route_config,
                    selected_route_path,
                )
                selected_route_name = normalize_route_name(
                    selected_route["name"]
                )

                if (
                    selected_route_path.stem
                    != selected_route_name
                ):
                    raise ValueError(
                        "路线文件名和 route.name 不一致："
                        f"{selected_route_path.name} vs "
                        f"{selected_route_name}"
                    )
            except Exception as exc:
                print(f"\n路线读取失败：{exc}")
                continue

            selected_map_name = selected_route.get("map")

            if (
                selected_map_name
                and selected_map_name != current_map_name
            ):
                map_identifier = find_map_identifier(
                    client,
                    selected_map_name,
                )

                if map_identifier is None:
                    print(
                        f"\n路线需要地图 '{selected_map_name}'，"
                        "但 CARLA 当前可用地图中没有找到它。"
                    )
                    continue

                if not prompt_yes_no(
                    f"该路线属于地图 '{selected_map_name}'。"
                    "是否自动切换到该地图并加载路线？",
                    default=True,
                ):
                    print("已取消加载路线。")
                    continue

                try:
                    (
                        new_world,
                        new_world_map,
                        new_spectator,
                        new_current_map_name,
                        new_planner,
                    ) = switch_world_map(
                        client,
                        map_identifier,
                        sampling_resolution,
                    )
                except Exception as exc:
                    print(f"\n路线地图加载失败：{exc}")
                    continue

                world = new_world
                world_map = new_world_map
                spectator = new_spectator
                current_map_name = new_current_map_name
                planner = new_planner

            candidate_anchors = copy.deepcopy(
                selected_route.get("anchors") or []
            )
            candidate_path = copy.deepcopy(
                selected_route.get("planned_path") or []
            )

            try:
                validate_planned_path_geometry(
                    candidate_path,
                    sampling_resolution,
                )
            except RuntimeError as exc:
                print(f"\n路线几何检查失败：{exc}")
                continue

            route_name = selected_route_name
            route_path = selected_route_path.resolve()
            route_config = selected_route_config
            route = selected_route
            anchors = candidate_anchors
            planned_path = candidate_path
            dirty = False
            last_draw_time = 0.0

            print(
                f"\n路线已加载：{route_name}\n"
                f"地图：{current_map_name}\n"
                f"Anchor 数量：{len(anchors)}\n"
                f"planned_path 点数：{len(planned_path)}"
            )

        ########################## 保存 ################################

        elif key == "s":
            if save_current_route(
                route_path,
                config,
                route_name,
                anchors,
                planned_path,
                current_map_name,
                sampling_resolution,
            ):
                dirty = False
                route["map"] = current_map_name
                route["anchors"] = copy.deepcopy(anchors)
                route["planned_path"] = copy.deepcopy(planned_path)

        ########################## 清空 ################################

        elif key == "c":
            if not anchors and not planned_path:
                print("\n当前工作路线已经是空的。")
                continue

            anchors.clear()
            planned_path.clear()
            dirty = True
            print("\n当前工作路线已清空；磁盘 YAML 尚未修改。")

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
                dirty = False
                running = False

        ########################## 不保存退出 ################################

        elif ord(key) == 27:
            if dirty and (anchors or planned_path):
                if not prompt_yes_no(
                    "当前有尚未保存的修改。确定不保存并退出吗？",
                    default=False,
                ):
                    print("继续编辑。")
                    continue

            print("\n不保存退出。")
            running = False

    print("\n路线编辑器已关闭。")


if __name__ == "__main__":
    main()
