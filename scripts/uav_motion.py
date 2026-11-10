import math
import random


def build_motion_plan(length_m, fps, speed, hover, num_frames=-1):
    """Integrate one deterministic cruise/brake/hover/recover timeline."""
    low, high = float(speed["low_mps"]), float(speed["high_mps"])
    stage_s = float(speed["stage_seconds"])
    every, hold = float(hover["after_motion_seconds"]), float(hover["duration_seconds"])
    brake, recover = float(hover["brake_seconds"]), float(hover["recover_seconds"])
    values = (length_m, fps, low, high, stage_s, every, brake, recover)
    if not all(math.isfinite(v) and v > 0 for v in values) or high < low or not math.isfinite(hold) or hold < 0:
        raise ValueError("Invalid motion profile")
    if num_frames != -1 and num_frames < 1:
        raise ValueError("num_frames must be -1 or positive")
    if not all(math.isclose(v * fps, round(v * fps)) for v in (stage_s, every, hold, brake, recover)):
        raise ValueError("Motion phase durations must be integer multiples of 1/fps")
    dt = 1.0 / fps
    distance = moving = cruise = since_hover = elapsed = 0.0
    state, hover_index, resume_speed = "cruise", 0, low
    rows = []
    while True:
        phase = cruise % (4 * stage_s)
        stage = min(3, int(phase / stage_s))
        ratio = (phase - stage * stage_s) / stage_s
        smooth = (1 - math.cos(math.pi * ratio)) / 2
        target = (low, low + (high-low)*smooth, high, high - (high-low)*smooth)[stage]
        if state == "cruise":
            velocity = target
        elif state == "brake":
            velocity = resume_speed * (1 + math.cos(math.pi * elapsed / brake)) / 2
        elif state == "recover":
            velocity = resume_speed * (1 - math.cos(math.pi * elapsed / recover)) / 2
        else:
            velocity = 0.0
        finished = distance >= length_m - 1e-8
        if finished:
            velocity = 0.0
        rows.append(dict(frame_index=len(rows), dataset_time_s=len(rows)/fps,
                         route_distance_m=distance, motion_time_s=moving,
                         cruise_time_s=cruise, speed_mps=velocity,
                         motion_state="finished" if finished else state,
                         speed_stage=("low", "accelerate", "high", "decelerate")[stage],
                         is_hovering=state == "hover" and not finished,
                         hover_index=hover_index if state == "hover" and not finished else None))
        if (num_frames == -1 and finished) or (num_frames > 0 and len(rows) >= num_frames):
            break
        if not finished:
            step = min(velocity * dt, length_m - distance)
            distance += step
            if velocity > 1e-8:
                moving += step / velocity
            if state == "cruise":
                cruise += dt
                since_hover += dt
                if hold > 0 and since_hover >= every - 1e-8:
                    state, elapsed, resume_speed = "brake", 0.0, target
                    hover_index += 1
            else:
                elapsed += dt
                duration = {"brake": brake, "hover": hold, "recover": recover}[state]
                if elapsed >= duration - 1e-8:
                    state = {"brake": "hover", "hover": "recover", "recover": "cruise"}[state]
                    elapsed = 0.0
                    if state == "cruise":
                        since_hover = 0.0
    for i, row in enumerate(rows):
        row["acceleration_mps2"] = (row["speed_mps"] - rows[i-1]["speed_mps"]) * fps if i else 0.0
    return rows


def build_attitude_schedule(num_frames, fps, config):
    rng = random.Random(int(config["seed"]))
    probability = float(config["probability_per_second"])
    duration_min, duration_max = map(float, config["duration_seconds"])
    gap = float(config["min_gap_seconds"])
    bounds = [float(config[key]) for key in ("roll_deg", "pitch_deg", "yaw_deg")]
    if not (math.isfinite(fps) and fps > 0 and 0 <= probability <= 1 and
            0 < duration_min <= duration_max and gap >= 0 and
            all(math.isfinite(v) and v >= 0 for v in bounds + [duration_min, duration_max, gap])):
        raise ValueError("Invalid disturbance profile")
    rows = [dict(frame_index=i, roll_deg=0.0, pitch_deg=0.0, yaw_deg=0.0,
                 event_index=None) for i in range(num_frames)]
    events, tick = [], 0
    if config["enabled"]:
        while round(tick * fps) < num_frames:
            if rng.random() >= probability:
                tick += 1
                continue
            start = round(tick * fps)
            count = max(2, round(rng.uniform(duration_min, duration_max) * fps))
            if start + count >= num_frames:
                break  # Do not leave an unfinished disturbance at the route endpoint.
            amplitudes = [rng.uniform(-v, v) for v in bounds]
            cycles = [rng.uniform(0.5, 1.5) for _ in bounds]
            phases = [rng.uniform(-math.pi, math.pi) for _ in bounds]
            event = dict(index=len(events), start_frame=start, end_frame=start+count,
                         amplitudes_deg=amplitudes, cycles=cycles, phases_rad=phases)
            events.append(event)
            for j in range(count+1):
                u = j / count
                envelope = math.sin(math.pi*u)**2 if 0 < j < count else 0.0
                for axis, amplitude, cycle, phase in zip(("roll_deg", "pitch_deg", "yaw_deg"), amplitudes, cycles, phases):
                    rows[start+j][axis] = amplitude * envelope * math.sin(2*math.pi*cycle*u + phase)
                rows[start+j]["event_index"] = event["index"]
            tick = math.ceil((start+count)/fps + gap)
    return dict(enabled=bool(config["enabled"]), seed=int(config["seed"]),
                config=dict(config), events=events, frames=rows)


def scene_index(motion, fps, seconds=20.0):
    from collections import Counter
    size = round(seconds * fps)
    if size < 1 or not math.isclose(size, seconds * fps):
        raise ValueError("scene duration must contain an integer number of frames")
    scenes = []
    for start in range(0, len(motion), size):
        rows = motion[start:start+size]
        scenes.append(dict(scene_id="scene_%04d" % len(scenes), start_frame=start,
                           end_frame_exclusive=start+len(rows), num_frames=len(rows),
                           start_time_s=start/fps, duration_s=len(rows)/fps,
                           partial=len(rows) != size,
                           motion_state_counts=dict(Counter(r["motion_state"] for r in rows)),
                           speed_stage_counts=dict(Counter(r["speed_stage"] for r in rows))))
    return dict(format_version=1, fps=fps, scene_seconds=seconds, scenes=scenes)


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
