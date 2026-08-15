import carla

import copy
import time
import msvcrt

from pathlib import Path
from ruamel.yaml import YAML


# ============================================================
# PATH
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "UAVdataset.yaml"
)


# ============================================================
# YAML
# ============================================================

yaml = YAML()
yaml.preserve_quotes = True
yaml.indent(
    mapping=2,
    sequence=4,
    offset=2
)


def load_yaml(path):

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        return yaml.load(f)


def save_yaml(path, data):

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:

        yaml.dump(
            data,
            f
        )


# ============================================================
# ROUTE DRAWING
# ============================================================

def make_location(point):

    return carla.Location(
        x=float(point["x"]),
        y=float(point["y"]),
        z=float(point["z"])
    )


def draw_route(
    world,
    waypoints,
    life_time=1.0
):

    if not waypoints:
        return

    debug = world.debug

    point_color = carla.Color(
        0,
        255,
        0
    )

    line_color = carla.Color(
        0,
        180,
        255
    )

    text_color = carla.Color(
        255,
        255,
        255
    )

    arrow_color = carla.Color(
        255,
        180,
        0
    )

    # ========================================================
    # POINTS
    # ========================================================

    for i, point in enumerate(
        waypoints
    ):

        location = make_location(
            point
        )

        # waypoint 球
        debug.draw_point(
            location,
            size=0.25,
            color=point_color,
            life_time=life_time
        )

        # WP 编号
        text_location = carla.Location(
            x=location.x,
            y=location.y,
            z=location.z + 0.8
        )

        debug.draw_string(
            text_location,
            f"WP{i:02d}",
            draw_shadow=True,
            color=text_color,
            life_time=life_time
        )

    # ========================================================
    # ROUTE LINES
    # ========================================================

    if len(waypoints) < 2:
        return

    for i in range(
        len(waypoints) - 1
    ):

        start = make_location(
            waypoints[i]
        )

        end = make_location(
            waypoints[i + 1]
        )

        # 航线
        debug.draw_line(
            start,
            end,
            thickness=0.08,
            color=line_color,
            life_time=life_time
        )

        # 飞行方向箭头
        debug.draw_arrow(
            start,
            end,
            thickness=0.08,
            arrow_size=0.25,
            color=arrow_color,
            life_time=life_time
        )


# ============================================================
# PRINT ROUTE
# ============================================================

def print_route(route):

    waypoints = route[
        "waypoints"
    ]

    print(
        "\n============================================"
    )

    print(
        "Route:",
        route.get(
            "name",
            "unnamed"
        )
    )

    print(
        "Waypoints:",
        len(waypoints)
    )

    print(
        "Speed:",
        route.get(
            "speed_mps",
            0
        ),
        "m/s"
    )

    print(
        "============================================"
    )

    for i, p in enumerate(
        waypoints
    ):

        print(
            f"WP{i:02d}: "
            f"x={float(p['x']):8.2f}  "
            f"y={float(p['y']):8.2f}  "
            f"z={float(p['z']):6.2f}"
        )

    print()


# ============================================================
# MAIN
# ============================================================

def main():

    # ========================================================
    # MAIN CONFIG
    # ========================================================

    config = load_yaml(
        CONFIG_PATH
    )

    route_file = (
        config["uav"]["route_file"]
    )

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

    route = route_config[
        "route"
    ]

    if route.get(
        "waypoints"
    ) is None:

        route["waypoints"] = []

    # 工作副本
    waypoints = copy.deepcopy(
        route["waypoints"]
    )

    # ========================================================
    # CARLA
    # ========================================================

    client = carla.Client(
        "localhost",
        2000
    )

    client.set_timeout(
        10.0
    )

    world = client.get_world()

    spectator = (
        world.get_spectator()
    )

    print(
        "\n============================================"
    )

    print(
        "UAV ROUTE EDITOR"
    )

    print(
        "============================================"
    )

    print(
        "Route file:"
    )

    print(
        route_path
    )

    print(
        "\n在 CARLA 窗口里移动 Spectator。"
    )

    print(
        "然后切回这个终端按下面的按键："
    )

    print()

    print(
        "A     添加当前位置为航点"
    )

    print(
        "U     撤销最后一个航点"
    )

    print(
        "P     打印当前全部航点"
    )

    print(
        "S     保存路线"
    )

    print(
        "C     清空所有航点"
    )

    print(
        "Q     保存并退出"
    )

    print(
        "ESC   不保存退出"
    )

    print(
        "\n当前航点数量:",
        len(waypoints)
    )

    print(
        "\n等待操作..."
    )

    # ========================================================
    # EDIT LOOP
    #
    # 每隔一小段时间重新绘制一次路线。
    #
    # life_time 很短，所以删除 waypoint 后，
    # 旧的显示很快就会自动消失。
    # ========================================================

    last_draw_time = 0.0

    running = True

    while running:

        current_time = (
            time.time()
        )

        # ----------------------------------------------------
        # 持续显示路线
        # ----------------------------------------------------

        if (
            current_time
            - last_draw_time
            > 0.35
        ):

            draw_route(
                world,
                waypoints,
                life_time=0.8
            )

            last_draw_time = (
                current_time
            )

        # ----------------------------------------------------
        # Keyboard
        # ----------------------------------------------------

        if not msvcrt.kbhit():

            time.sleep(
                0.02
            )

            continue

        key = msvcrt.getwch()

        key = key.lower()

        # ====================================================
        # ADD
        # ====================================================

        if key == "a":

            tf = (
                spectator
                .get_transform()
            )

            point = {

                "x": round(
                    tf.location.x,
                    3
                ),

                "y": round(
                    tf.location.y,
                    3
                ),

                "z": round(
                    tf.location.z,
                    3
                )
            }

            waypoints.append(
                point
            )

            print(
                "\nAdded:"
            )

            print(
                f"WP{len(waypoints)-1:02d}  "
                f"x={point['x']:.3f}  "
                f"y={point['y']:.3f}  "
                f"z={point['z']:.3f}"
            )

        # ====================================================
        # UNDO
        # ====================================================

        elif key == "u":

            if waypoints:

                removed = (
                    waypoints.pop()
                )

                print(
                    "\nRemoved:"
                )

                print(
                    removed
                )

            else:

                print(
                    "\nNo waypoint to remove."
                )

        # ====================================================
        # PRINT
        # ====================================================

        elif key == "p":

            temp_route = dict(
                route
            )

            temp_route[
                "waypoints"
            ] = waypoints

            print_route(
                temp_route
            )

        # ====================================================
        # SAVE
        # ====================================================

        elif key == "s":

            route[
                "waypoints"
            ] = copy.deepcopy(
                waypoints
            )

            save_yaml(
                route_path,
                route_config
            )

            print(
                "\nRoute saved:"
            )

            print(
                route_path
            )

        # ====================================================
        # CLEAR
        # ====================================================

        elif key == "c":

            waypoints.clear()

            print(
                "\nAll waypoints cleared."
            )

        # ====================================================
        # SAVE + QUIT
        # ====================================================

        elif key == "q":

            route[
                "waypoints"
            ] = copy.deepcopy(
                waypoints
            )

            save_yaml(
                route_path,
                route_config
            )

            print(
                "\nRoute saved."
            )

            running = False

        # ====================================================
        # ESC: QUIT WITHOUT SAVE
        # ====================================================

        elif ord(key) == 27:

            print(
                "\nExit without saving."
            )

            running = False

    print(
        "\nRoute editor closed."
    )


if __name__ == "__main__":
    main()
