import carla
from pathlib import Path
from ruamel.yaml import YAML


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "UAVdataset.yaml"


def main():

    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)

    world = client.get_world()

    spectator = world.get_spectator()

    transform = spectator.get_transform()

    print("\nCurrent spectator:")
    print(transform)

    yaml = YAML()
    yaml.preserve_quotes = True

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.load(f)

    pose = config["uav"]["initial_pose"]

    pose["x"] = round(transform.location.x, 4)
    pose["y"] = round(transform.location.y, 4)
    pose["z"] = round(transform.location.z, 4)

    # UAV 本身保持水平。
    #
    # Spectator 的 pitch 是“你眼睛朝哪看”，
    # 并不是无人机机体应该倾斜多少。
    #
    # 所以只保存 spectator 的 yaw 作为 UAV 朝向。

    pose["pitch"] = 0.0
    pose["yaw"] = round(transform.rotation.yaw, 4)
    pose["roll"] = 0.0

    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.dump(config, f)

    print("\nSaved UAV position to:")
    print(CONFIG_PATH)

    print("\nUAV pose:")
    print(
        f"x={pose['x']:.2f}, "
        f"y={pose['y']:.2f}, "
        f"z={pose['z']:.2f}, "
        f"yaw={pose['yaw']:.2f}"
    )


if __name__ == "__main__":
    main()
