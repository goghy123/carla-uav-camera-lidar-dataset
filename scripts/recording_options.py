"""Shared collection CLI; importing --help does not require CARLA."""
import argparse
import re
from pathlib import Path


WEATHERS = ("clear", "overcast", "night", "dawn", "rain")


def parse_args(argv=None, batch=False):
    parser = argparse.ArgumentParser(description="Record synchronized UAV dataset v3")
    parser.add_argument("--output-root", help="Absolute path or path relative to project root")
    parser.add_argument("--run-id", default="run_0001", help="Route version, e.g. run_0002")
    parser.add_argument("--weather", choices=(*WEATHERS, "all") if batch else WEATHERS,
                        default="all" if batch else None)
    parser.add_argument("--route-file", help="Route YAML, absolute or relative to configs/")
    parser.add_argument("--num-frames", type=int, help="Positive trial length, or -1 for full route")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--disturbance", dest="disturbance", action="store_true")
    group.add_argument("--no-disturbance", dest="disturbance", action="store_false")
    parser.set_defaults(disturbance=None)
    parser.add_argument("--plan-only", action="store_true", help="Save plans without connecting to CARLA")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"run_[0-9]{4,}", args.run_id):
        parser.error("--run-id must be run_ followed by at least four digits")
    if args.num_frames is not None and args.num_frames != -1 and args.num_frames <= 0:
        parser.error("--num-frames must be -1 or positive")
    return args


def run_directory(output_root, map_name, weather, run_id):
    if Path(map_name).name != map_name or map_name in ("", ".", ".."):
        raise ValueError("route.map must be a safe directory name")
    return Path(output_root) / (map_name + "_" + weather) / run_id
