"""
Self-contained CARLA global route planner used by route_editor.py.

This module is adapted from CARLA 0.9.16's GlobalRoutePlanner design.
It intentionally avoids CARLA's external ``agents`` package and ``networkx``
so that a working ``import carla`` is sufficient.

CARLA GlobalRoutePlanner source:
PythonAPI/carla/agents/navigation/global_route_planner.py
Copyright (c) 2018-2020 CVC. MIT licensed.
"""

import heapq
import math
from collections import defaultdict
from enum import IntEnum

import carla


class RoadOption(IntEnum):
    VOID = -1
    LEFT = 1
    RIGHT = 2
    STRAIGHT = 3
    LANEFOLLOW = 4
    CHANGELANELEFT = 5
    CHANGELANERIGHT = 6


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _norm(v):
    return math.sqrt(_dot(v, v))


def _cross_z(a, b):
    return a[0] * b[1] - a[1] * b[0]


def _distance3(a, b):
    return math.sqrt(
        (a[0] - b[0]) ** 2
        + (a[1] - b[1]) ** 2
        + (a[2] - b[2]) ** 2
    )


class GlobalRoutePlanner:
    """CARLA-compatible high-level route planner.

    Public API:
        planner = GlobalRoutePlanner(carla_map, sampling_resolution)
        route = planner.trace_route(origin, destination)

    ``route`` is a list of ``(carla.Waypoint, RoadOption)`` tuples.
    """

    def __init__(self, wmap, sampling_resolution):
        sampling_resolution = float(sampling_resolution)
        if sampling_resolution <= 0.0:
            raise ValueError("sampling_resolution must be > 0")

        self._sampling_resolution = sampling_resolution
        self._wmap = wmap
        self._topology = []

        # Lightweight directed graph replacing networkx.DiGraph.
        self._nodes = {}
        self._edges = {}
        self._successors = defaultdict(set)
        self._id_map = {}
        self._road_id_to_edge = {}

        self._intersection_end_node = -1
        self._previous_decision = RoadOption.VOID

        self._build_topology()
        self._build_graph()
        self._find_loose_ends()
        self._lane_change_link()

    def trace_route(self, origin, destination):
        route_trace = []
        route = self._path_search(origin, destination)
        current_waypoint = self._wmap.get_waypoint(origin)
        destination_waypoint = self._wmap.get_waypoint(destination)

        if current_waypoint is None or destination_waypoint is None:
            raise ValueError(
                "Origin or destination cannot be projected to a CARLA waypoint"
            )

        for i in range(len(route) - 1):
            road_option = self._turn_decision(i, route)
            edge = self._edge(route[i], route[i + 1])

            if edge["type"] not in (
                RoadOption.LANEFOLLOW,
                RoadOption.VOID,
            ):
                route_trace.append((current_waypoint, road_option))
                exit_wp = edge["exit_waypoint"]
                n1, n2 = self._road_id_to_edge[
                    exit_wp.road_id
                ][exit_wp.section_id][exit_wp.lane_id]
                next_edge = self._edge(n1, n2)
                if next_edge["path"]:
                    closest_index = self._find_closest_in_list(
                        current_waypoint,
                        next_edge["path"],
                    )
                    closest_index = min(
                        len(next_edge["path"]) - 1,
                        closest_index + 5,
                    )
                    current_waypoint = next_edge["path"][closest_index]
                else:
                    current_waypoint = next_edge["exit_waypoint"]
                route_trace.append((current_waypoint, road_option))
            else:
                path = (
                    [edge["entry_waypoint"]]
                    + edge["path"]
                    + [edge["exit_waypoint"]]
                )
                closest_index = self._find_closest_in_list(
                    current_waypoint,
                    path,
                )

                for waypoint in path[closest_index:]:
                    current_waypoint = waypoint
                    route_trace.append(
                        (current_waypoint, road_option)
                    )

                    if (
                        len(route) - i <= 2
                        and waypoint.transform.location.distance(
                            destination
                        )
                        < 2 * self._sampling_resolution
                    ):
                        break

                    if (
                        len(route) - i <= 2
                        and current_waypoint.road_id
                        == destination_waypoint.road_id
                        and current_waypoint.section_id
                        == destination_waypoint.section_id
                        and current_waypoint.lane_id
                        == destination_waypoint.lane_id
                    ):
                        destination_index = (
                            self._find_closest_in_list(
                                destination_waypoint,
                                path,
                            )
                        )
                        if closest_index > destination_index:
                            break

        return route_trace

    def _add_node(self, node_id, vertex):
        self._nodes[node_id] = {"vertex": vertex}

    def _add_edge(self, n1, n2, **attributes):
        self._edges[(n1, n2)] = attributes
        self._successors[n1].add(n2)
        self._successors.setdefault(n2, set())

    def _edge(self, n1, n2):
        return self._edges[(n1, n2)]

    def _build_topology(self):
        self._topology = []

        for wp1, wp2 in self._wmap.get_topology():
            l1 = wp1.transform.location
            l2 = wp2.transform.location

            # Same node quantization idea as CARLA 0.9.16.
            entry_xyz = (
                round(l1.x),
                round(l1.y),
                round(l1.z),
            )
            exit_xyz = (
                round(l2.x),
                round(l2.y),
                round(l2.z),
            )

            segment = {
                "entry": wp1,
                "exit": wp2,
                "entryxyz": entry_xyz,
                "exitxyz": exit_xyz,
                "path": [],
            }

            endloc = wp2.transform.location
            if (
                wp1.transform.location.distance(endloc)
                > self._sampling_resolution
            ):
                next_wps = wp1.next(
                    self._sampling_resolution
                )
                if not next_wps:
                    continue

                w = next_wps[0]
                while (
                    w.transform.location.distance(endloc)
                    > self._sampling_resolution
                ):
                    segment["path"].append(w)
                    next_wps = w.next(
                        self._sampling_resolution
                    )
                    if not next_wps:
                        break
                    w = next_wps[0]
            else:
                next_wps = wp1.next(
                    self._sampling_resolution
                )
                if not next_wps:
                    continue
                segment["path"].append(next_wps[0])

            self._topology.append(segment)

    def _build_graph(self):
        for segment in self._topology:
            entry_xyz = segment["entryxyz"]
            exit_xyz = segment["exitxyz"]
            path = segment["path"]
            entry_wp = segment["entry"]
            exit_wp = segment["exit"]
            intersection = entry_wp.is_junction

            road_id = entry_wp.road_id
            section_id = entry_wp.section_id
            lane_id = entry_wp.lane_id

            for vertex in (entry_xyz, exit_xyz):
                if vertex not in self._id_map:
                    new_id = len(self._id_map)
                    self._id_map[vertex] = new_id
                    self._add_node(new_id, vertex)

            n1 = self._id_map[entry_xyz]
            n2 = self._id_map[exit_xyz]

            self._road_id_to_edge.setdefault(
                road_id,
                {},
            )
            self._road_id_to_edge[
                road_id
            ].setdefault(section_id, {})
            self._road_id_to_edge[
                road_id
            ][section_id][lane_id] = (n1, n2)

            entry_vec = (
                entry_wp.transform.rotation
                .get_forward_vector()
            )
            exit_vec = (
                exit_wp.transform.rotation
                .get_forward_vector()
            )
            net_vec = (
                (
                    exit_wp.transform.location
                    - entry_wp.transform.location
                )
                .make_unit_vector()
            )

            self._add_edge(
                n1,
                n2,
                length=len(path) + 1,
                path=path,
                entry_waypoint=entry_wp,
                exit_waypoint=exit_wp,
                entry_vector=(
                    entry_vec.x,
                    entry_vec.y,
                    entry_vec.z,
                ),
                exit_vector=(
                    exit_vec.x,
                    exit_vec.y,
                    exit_vec.z,
                ),
                net_vector=(
                    net_vec.x,
                    net_vec.y,
                    net_vec.z,
                ),
                intersection=intersection,
                type=RoadOption.LANEFOLLOW,
            )

    def _find_loose_ends(self):
        count_loose_ends = 0
        hop_resolution = self._sampling_resolution

        for segment in self._topology:
            end_wp = segment["exit"]
            exit_xyz = segment["exitxyz"]
            road_id = end_wp.road_id
            section_id = end_wp.section_id
            lane_id = end_wp.lane_id

            known = (
                road_id in self._road_id_to_edge
                and section_id
                in self._road_id_to_edge[road_id]
                and lane_id
                in self._road_id_to_edge[
                    road_id
                ][section_id]
            )
            if known:
                continue

            count_loose_ends += 1
            self._road_id_to_edge.setdefault(
                road_id,
                {},
            )
            self._road_id_to_edge[
                road_id
            ].setdefault(section_id, {})

            n1 = self._id_map[exit_xyz]
            n2 = -count_loose_ends
            self._road_id_to_edge[
                road_id
            ][section_id][lane_id] = (n1, n2)

            next_wp = end_wp.next(hop_resolution)
            path = []

            while (
                next_wp
                and next_wp[0].road_id == road_id
                and next_wp[0].section_id
                == section_id
                and next_wp[0].lane_id == lane_id
            ):
                path.append(next_wp[0])
                next_wp = next_wp[0].next(
                    hop_resolution
                )

            if path:
                last = path[-1].transform.location
                n2_xyz = (
                    last.x,
                    last.y,
                    last.z,
                )
                self._add_node(n2, n2_xyz)
                self._add_edge(
                    n1,
                    n2,
                    length=len(path) + 1,
                    path=path,
                    entry_waypoint=end_wp,
                    exit_waypoint=path[-1],
                    entry_vector=None,
                    exit_vector=None,
                    net_vector=None,
                    intersection=end_wp.is_junction,
                    type=RoadOption.LANEFOLLOW,
                )

    def _lane_change_link(self):
        for segment in self._topology:
            left_found = False
            right_found = False

            for waypoint in segment["path"]:
                if segment["entry"].is_junction:
                    continue

                if (
                    waypoint.right_lane_marking
                    and (
                        waypoint.right_lane_marking.lane_change
                        & carla.LaneChange.Right
                    )
                    and not right_found
                ):
                    next_waypoint = (
                        waypoint.get_right_lane()
                    )
                    if (
                        next_waypoint is not None
                        and next_waypoint.lane_type
                        == carla.LaneType.Driving
                        and waypoint.road_id
                        == next_waypoint.road_id
                    ):
                        next_segment = self._localize(
                            next_waypoint
                            .transform.location
                        )
                        if next_segment is not None:
                            self._add_edge(
                                self._id_map[
                                    segment[
                                        "entryxyz"
                                    ]
                                ],
                                next_segment[0],
                                entry_waypoint=waypoint,
                                exit_waypoint=next_waypoint,
                                intersection=False,
                                exit_vector=None,
                                path=[],
                                length=0,
                                type=(
                                    RoadOption
                                    .CHANGELANERIGHT
                                ),
                                change_waypoint=(
                                    next_waypoint
                                ),
                            )
                            right_found = True

                if (
                    waypoint.left_lane_marking
                    and (
                        waypoint.left_lane_marking.lane_change
                        & carla.LaneChange.Left
                    )
                    and not left_found
                ):
                    next_waypoint = (
                        waypoint.get_left_lane()
                    )
                    if (
                        next_waypoint is not None
                        and next_waypoint.lane_type
                        == carla.LaneType.Driving
                        and waypoint.road_id
                        == next_waypoint.road_id
                    ):
                        next_segment = self._localize(
                            next_waypoint
                            .transform.location
                        )
                        if next_segment is not None:
                            self._add_edge(
                                self._id_map[
                                    segment[
                                        "entryxyz"
                                    ]
                                ],
                                next_segment[0],
                                entry_waypoint=waypoint,
                                exit_waypoint=next_waypoint,
                                intersection=False,
                                exit_vector=None,
                                path=[],
                                length=0,
                                type=(
                                    RoadOption
                                    .CHANGELANELEFT
                                ),
                                change_waypoint=(
                                    next_waypoint
                                ),
                            )
                            left_found = True

                if left_found and right_found:
                    break

    def _localize(self, location):
        waypoint = self._wmap.get_waypoint(location)
        if waypoint is None:
            return None

        try:
            return self._road_id_to_edge[
                waypoint.road_id
            ][waypoint.section_id][waypoint.lane_id]
        except KeyError:
            return None

    def _distance_heuristic(self, n1, n2):
        return _distance3(
            self._nodes[n1]["vertex"],
            self._nodes[n2]["vertex"],
        )

    def _astar_path(self, source, target):
        if (
            source not in self._nodes
            or target not in self._nodes
        ):
            raise ValueError(
                "Route endpoints are not present "
                "in the CARLA road graph"
            )

        if source == target:
            return [source]

        queue = []
        order = 0
        g_score = {source: 0.0}
        came_from = {}

        heapq.heappush(
            queue,
            (
                self._distance_heuristic(
                    source,
                    target,
                ),
                order,
                0.0,
                source,
            ),
        )

        while queue:
            _, _, popped_g, current = (
                heapq.heappop(queue)
            )

            if (
                popped_g
                > g_score.get(
                    current,
                    float("inf"),
                )
                + 1e-12
            ):
                continue

            if current == target:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path

            base_cost = g_score[current]

            for neighbor in self._successors.get(
                current,
                (),
            ):
                edge = self._edge(
                    current,
                    neighbor,
                )
                tentative = (
                    base_cost
                    + float(edge["length"])
                )

                if (
                    tentative
                    >= g_score.get(
                        neighbor,
                        float("inf"),
                    )
                ):
                    continue

                came_from[neighbor] = current
                g_score[neighbor] = tentative
                order += 1

                f_score = (
                    tentative
                    + self._distance_heuristic(
                        neighbor,
                        target,
                    )
                )
                heapq.heappush(
                    queue,
                    (
                        f_score,
                        order,
                        tentative,
                        neighbor,
                    ),
                )

        raise ValueError(
            "No drivable route exists "
            "between the selected anchors"
        )

    def _path_search(self, origin, destination):
        start = self._localize(origin)
        end = self._localize(destination)

        if start is None or end is None:
            raise ValueError(
                "Cannot localize one of the anchors "
                "in the CARLA road graph"
            )

        route = self._astar_path(
            start[0],
            end[0],
        )
        route.append(end[1])
        return route

    def _successive_last_intersection_edge(
        self,
        index,
        route,
    ):
        last_intersection_edge = None
        last_node = None

        for node1, node2 in (
            (route[i], route[i + 1])
            for i in range(
                index,
                len(route) - 1,
            )
        ):
            candidate_edge = self._edge(
                node1,
                node2,
            )

            if node1 == route[index]:
                last_intersection_edge = (
                    candidate_edge
                )

            if (
                candidate_edge["type"]
                == RoadOption.LANEFOLLOW
                and candidate_edge[
                    "intersection"
                ]
            ):
                last_intersection_edge = (
                    candidate_edge
                )
                last_node = node2
            else:
                break

        return (
            last_node,
            last_intersection_edge,
        )

    def _turn_decision(
        self,
        index,
        route,
        threshold=math.radians(35.0),
    ):
        previous_node = route[index - 1]
        current_node = route[index]
        next_node = route[index + 1]
        next_edge = self._edge(
            current_node,
            next_node,
        )
        decision = None

        if index > 0:
            if (
                self._previous_decision
                != RoadOption.VOID
                and isinstance(
                    self._intersection_end_node,
                    int,
                )
                and self._intersection_end_node > 0
                and self._intersection_end_node
                != previous_node
                and next_edge["type"]
                == RoadOption.LANEFOLLOW
                and next_edge["intersection"]
            ):
                decision = self._previous_decision
            else:
                self._intersection_end_node = -1
                current_edge = self._edge(
                    previous_node,
                    current_node,
                )

                calculate_turn = (
                    current_edge["type"]
                    == RoadOption.LANEFOLLOW
                    and not current_edge[
                        "intersection"
                    ]
                    and next_edge["type"]
                    == RoadOption.LANEFOLLOW
                    and next_edge[
                        "intersection"
                    ]
                )

                if calculate_turn:
                    (
                        last_node,
                        tail_edge,
                    ) = (
                        self
                        ._successive_last_intersection_edge(
                            index,
                            route,
                        )
                    )
                    self._intersection_end_node = (
                        last_node
                        if last_node is not None
                        else -1
                    )

                    if tail_edge is not None:
                        next_edge = tail_edge

                    current_vec = current_edge[
                        "exit_vector"
                    ]
                    next_vec = next_edge[
                        "exit_vector"
                    ]

                    if (
                        current_vec is None
                        or next_vec is None
                    ):
                        return next_edge["type"]

                    cross_list = []
                    for neighbor in (
                        self._successors.get(
                            current_node,
                            (),
                        )
                    ):
                        select_edge = self._edge(
                            current_node,
                            neighbor,
                        )
                        if (
                            select_edge["type"]
                            == RoadOption.LANEFOLLOW
                            and neighbor
                            != route[index + 1]
                        ):
                            side_vec = select_edge[
                                "net_vector"
                            ]
                            if side_vec is not None:
                                cross_list.append(
                                    _cross_z(
                                        current_vec,
                                        side_vec,
                                    )
                                )

                    next_cross = _cross_z(
                        current_vec,
                        next_vec,
                    )
                    denominator = (
                        _norm(current_vec)
                        * _norm(next_vec)
                    )

                    if denominator <= 1e-12:
                        deviation = 0.0
                    else:
                        cosine = max(
                            -1.0,
                            min(
                                1.0,
                                _dot(
                                    current_vec,
                                    next_vec,
                                )
                                / denominator,
                            ),
                        )
                        deviation = math.acos(
                            cosine
                        )

                    if not cross_list:
                        cross_list.append(0.0)

                    if deviation < threshold:
                        decision = (
                            RoadOption.STRAIGHT
                        )
                    elif (
                        next_cross
                        < min(cross_list)
                    ):
                        decision = RoadOption.LEFT
                    elif (
                        next_cross
                        > max(cross_list)
                    ):
                        decision = RoadOption.RIGHT
                    elif next_cross < 0:
                        decision = RoadOption.LEFT
                    elif next_cross > 0:
                        decision = RoadOption.RIGHT
                    else:
                        decision = (
                            RoadOption.STRAIGHT
                        )
                else:
                    decision = next_edge["type"]
        else:
            decision = next_edge["type"]

        self._previous_decision = decision
        return decision

    @staticmethod
    def _find_closest_in_list(
        current_waypoint,
        waypoint_list,
    ):
        min_distance = float("inf")
        closest_index = -1

        for i, waypoint in enumerate(
            waypoint_list
        ):
            distance = (
                waypoint.transform.location.distance(
                    current_waypoint
                    .transform.location
                )
            )
            if distance < min_distance:
                min_distance = distance
                closest_index = i

        return closest_index
