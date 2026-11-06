import math


def motion_intervals_at_frame(frame_index, move_frames, hover_frames):
    cycle_frames = move_frames + hover_frames
    cycles, remainder = divmod(int(frame_index), cycle_frames)
    return cycles * move_frames + min(remainder, move_frames)


def is_hovering_at_frame(frame_index, move_frames, hover_frames):
    if hover_frames <= 0:
        return False
    return int(frame_index) % (move_frames + hover_frames) >= move_frames


def required_frames(total_length_m, distance_per_frame_m, move_frames, hover_frames):
    movement_intervals = math.ceil(
        float(total_length_m) / float(distance_per_frame_m)
    )
    hover_events = max(0, movement_intervals - 1) // move_frames
    return movement_intervals + hover_events * hover_frames + 1


def altitude_above_road_at_distance(
    distance_m,
    speed_mps,
    minimum_m,
    maximum_m,
    cycle_motion_seconds,
):
    midpoint = (float(minimum_m) + float(maximum_m)) / 2.0
    amplitude = (float(maximum_m) - float(minimum_m)) / 2.0
    motion_time = float(distance_m) / float(speed_mps)
    phase = 2.0 * math.pi * motion_time / float(cycle_motion_seconds)
    return midpoint + amplitude * math.cos(phase)
