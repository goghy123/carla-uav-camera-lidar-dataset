from pathlib import Path

from ruamel.yaml import YAML
from recording_options import parse_args, run_directory, WEATHERS


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "configs"
CONFIG_PATH = CONFIG_DIR / "UAVdataset.yaml"
ROUTES_DIR = CONFIG_DIR / "routes"


def load_yaml(path):
    yaml = YAML(typ="safe")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.load(f)


def discover_routes(routes_dir=ROUTES_DIR, config_dir=CONFIG_DIR):
    entries = []
    seen_names = set()

    for route_path in sorted(routes_dir.glob("*.yaml")):
        data = load_yaml(route_path)
        route = data.get("route") if isinstance(data, dict) else None
        route_name = route.get("name") if isinstance(route, dict) else None

        if not isinstance(route_name, str) or not route_name.strip():
            raise ValueError(f"Route file has no valid route.name: {route_path}")
        if route_name in seen_names:
            raise ValueError(f"Duplicate route.name: {route_name}")

        seen_names.add(route_name)
        entries.append(
            {
                "name": route_name,
                "route_file": route_path.relative_to(config_dir).as_posix(),
            }
        )

    if not entries:
        raise FileNotFoundError(f"No route YAML files found in: {routes_dir}")

    return entries


def nonempty_scene_dirs(output_root, entries):
    return [
        output_root / entry["name"]
        for entry in entries
        if (output_root / entry["name"]).is_dir()
        and any((output_root / entry["name"]).iterdir())
    ]


def main(argv=None):
    options = parse_args(argv, batch=True)
    config = load_yaml(CONFIG_PATH)
    entries = discover_routes()
    if options.route_file:
        path = (CONFIG_DIR / options.route_file).resolve()
        route = load_yaml(path)["route"]
        entries = [{"name": route["name"], "route_file": str(path)}]
    output_root = (PROJECT_ROOT / str(options.output_root or config["output"]["root"])).resolve()
    weathers = WEATHERS if options.weather == "all" else (options.weather,)
    jobs, targets = [], set()
    for entry in entries:
        route = load_yaml(CONFIG_DIR / entry["route_file"])["route"]
        for weather in weathers:
            target = run_directory(output_root, route["map"], weather, options.run_id)
            if target in targets:
                raise ValueError(f"Multiple routes target {target}; record route versions under different run IDs")
            targets.add(target)
            jobs.append((entry, weather, target))
    conflicts = [path for path in targets if path.exists() and (not path.is_dir() or any(path.iterdir()))]

    if conflicts:
        raise FileExistsError(
            "One-click recording requires every target scene directory to be empty.\n"
            + "\n".join(str(path) for path in conflicts)
        )

    print("\nOne-click UAV dataset recording")
    print("Output root:", output_root)
    print("Runs:", len(jobs))
    for index, entry in enumerate(entries, start=1):
        print(f"  [{index:02d}] {entry['name']}")

    from collect_uavdataset import main as collect_route

    for index, (entry, weather, target) in enumerate(jobs, start=1):
        print("\n" + "#" * 72)
        print(f"RUN {index}/{len(jobs)}: {entry['name']} / {weather}")
        print("#" * 72)
        options.weather = weather
        collect_route(route_file_override=entry["route_file"], options=options)

    print("\nAll recording plans saved:" if options.plan_only else "\nAll routes recorded successfully:")
    print(output_root)


if __name__ == "__main__":
    main()
