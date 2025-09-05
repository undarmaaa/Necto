import os
import json
import argparse
from datetime import datetime
import numpy as np

# Sharded log utilities
from log_loader import (
    iter_shard_paths,
    shard_output_path_for_date,
    shard_date_from_path,
)

GOAL_Y = 5100  # Approximate y position for goal line
GROUND_Z = 18  # Ball is on ground if z <= this
CAR_MAX_SPEED = 2300  # Unreal units per second


def distance(a, b):
    # Euclidean distance between two points
    return ((a["x"] - b["x"]) ** 2 + (a["y"] - b["y"]) ** 2 + (a["z"] - b["z"]) ** 2) ** 0.5


def detect_goal(window, clearing_team, blue_score, orange_score):
    """
    Returns (score_diff, who_scored)
    who_scored: 1 if clearing team, -1 if opponent, 0 if no goal
    """
    # Sort window by time to ensure correct order
    window_sorted = sorted(window, key=lambda s: s.get("time", 0))
    last_blue = blue_score
    last_orange = orange_score
    for state in window_sorted:
        if "blue_score" in state and "orange_score" in state:
            blue = state["blue_score"]
            orange = state["orange_score"]
        else:
            blue = last_blue
            orange = last_orange
        if blue > last_blue:
            return (blue - last_blue, 1 if clearing_team == 0 else -1)
        if orange > last_orange:
            return (orange - last_orange, 1 if clearing_team == 1 else -1)
        last_blue, last_orange = blue, orange
    # Fallback: check ball y position in last frame only
    last_state = window_sorted[-1]
    y = last_state["ball"]["y"]
    if y > GOAL_Y:
        return (0, 1 if clearing_team == 1 else -1)
    elif y < -GOAL_Y:
        return (0, 1 if clearing_team == 0 else -1)
    return (0, 0)


def predict_landing_point(window, clear_time=None):
    """
    Return the ball position at the first frame after the clear event.
    If not found, use the last state in window.
    """
    if clear_time is not None:
        for state in window:
            if state["time"] > clear_time:
                return state["ball"]
    # Fallback: use last state
    return window[-1]["ball"]


def time_to_reach(car, target):
    """
    Estimate time to reach the target for a car.
    Considers straight-line distance, boost, and facing direction.
    - If boost < 20, time is increased by 20%.
    - If car is facing away from the ball (>90 deg), time is doubled.
    """
    d = distance(car, target)
    time = d / CAR_MAX_SPEED

    # Penalize low boost
    if car.get("boost", 100) < 20:
        time *= 1.2

    # Penalize cars not facing the ball
    # Compute angle between car's facing direction and vector to ball
    dx = target["x"] - car["x"]
    dy = target["y"] - car["y"]
    car_yaw = car.get("rot_yaw", 0)
    # Car's forward vector
    fx = np.cos(car_yaw)
    fy = np.sin(car_yaw)
    # Normalize vectors
    norm = np.hypot(dx, dy)
    if norm > 1e-3:
        dx /= norm
        dy /= norm
        dot = fx * dx + fy * dy
        angle = np.arccos(np.clip(dot, -1, 1)) * 180 / np.pi
        if angle > 90:
            time *= 2.0
    return time


def extract_clear_frame(window, clear_time):
    """
    Find the frame in window closest to clear_time.
    Fallbacks to first frame if window is empty or clear_time is None.
    """
    if not window:
        return None
    if clear_time is None:
        return window[0]
    # Choose frame with min |time - clear_time|
    idx = int(np.argmin([abs(s.get("time", 0) - clear_time) for s in window]))
    return window[idx]


def compute_features(window, clear):
    """
    Compute feature vector at clear time:
      - ball_x, ball_y, ball_z
      - vel_x, vel_y, vel_z
      - ally_boost_total, enemy_boost_total
    """
    clear_time = clear.get("time", None)
    frame = extract_clear_frame(window, clear_time)
    if frame is None:
        return None

    ball = frame["ball"]
    # Velocity may not exist in some windows; fallback to zeros
    vx = ball.get("vel_x", 0.0)
    vy = ball.get("vel_y", 0.0)
    vz = ball.get("vel_z", 0.0)

    clearing_team = clear.get("clearing_team", 0)
    ally_boost_total = 0.0
    enemy_boost_total = 0.0
    for car in frame.get("cars", []):
        if car.get("team", -1) == clearing_team:
            ally_boost_total += float(car.get("boost", 0.0))
        else:
            enemy_boost_total += float(car.get("boost", 0.0))

    return {
        "ball_x": float(ball["x"]),
        "ball_y": float(ball["y"]),
        "ball_z": float(ball["z"]),
        "vel_x": float(vx),
        "vel_y": float(vy),
        "vel_z": float(vz),
        "ally_boost_total": float(ally_boost_total),
        "enemy_boost_total": float(enemy_boost_total),
    }


def label_file(input_path, output_path, max_lines=None, verbose=False):
    """
    Read clears from input_path (raw JSONL), label them, attach features,
    and write to output_path (labeled JSONL).
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    count = 0
    with open(input_path, "r", encoding="utf-8") as infile, open(output_path, "w", encoding="utf-8") as outfile:
        for i, line in enumerate(infile):
            if max_lines is not None and i >= max_lines:
                break
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except Exception as e:
                if verbose:
                    print(f"Skipping line {i+1}: {e}")
                continue

            clear = data.get("clear_event")
            window = data.get("game_window")
            if not window or not clear:
                continue

            last_state = window[-1]
            ball = last_state["ball"]
            cars = last_state["cars"]

            # Get initial scores
            blue_score = clear.get("blue_score", 0)
            orange_score = clear.get("orange_score", 0)

            # Check for goals in the window
            score_diff, who_scored = detect_goal(window, clear["clearing_team"], blue_score, orange_score)

            # Predict ball landing point (immediately after clear event)
            landing = predict_landing_point(window, clear.get("time", None))

            # For each car, estimate time to reach landing point
            reach_times = []
            for car in cars:
                t = time_to_reach(car, landing)
                reach_times.append((t, car["team"], car["index"]))

            # Use last_touch info to determine who actually reached the ball first after the clear event
            clear_time = clear.get("time", None)
            first_touch_after_clear = None
            if clear_time is not None:
                prev_touch = None
                for state in window:
                    if "last_touch" in state and state["last_touch"]:
                        touch = state["last_touch"]
                        if prev_touch is not None and (
                            touch["player_index"] != prev_touch["player_index"]
                            or touch["time_seconds"] != prev_touch["time_seconds"]
                        ):
                            # Ball was touched by a new player
                            if state["time"] >= clear_time:
                                first_touch_after_clear = touch
                                break
                        prev_touch = touch

            # Find min time for each team (for fallback)
            min_ally = min((t for t, team, idx in reach_times if team == clear["clearing_team"]), default=float("inf"))
            min_enemy = min((t for t, team, idx in reach_times if team != clear["clearing_team"]), default=float("inf"))

            # Scoring logic
            score = 0.0
            label = ""
            # Penalty/bonus for goals
            if who_scored == -1:
                score -= 1.0  # Opponent scored
                label = "goal_against"
            elif who_scored == 1:
                score += 1.0  # Our team scored
                label = "goal_for"
            else:
                # Use actual touch info if available
                if first_touch_after_clear is not None:
                    if first_touch_after_clear["team"] != clear["clearing_team"]:
                        score -= 0.5
                        label = "trapped"
                    else:
                        score += 0.5
                        label = "good_pass"
                else:
                    # Fallback: Ball landing point logic
                    if min_enemy < min_ally:
                        score -= 0.5
                        label = "trapped"
                    elif min_ally < min_enemy:
                        score += 0.5
                        label = "good_pass"
                    else:
                        score += 0.0
                        label = "trap_prevented"

            # Attach classification artifacts
            data["label"] = label
            data["score"] = score
            data["landing_point"] = landing

            # Feature vector at clear time for ML
            feats = compute_features(window, clear)
            if feats is not None:
                data["features"] = feats

            outfile.write(json.dumps(data) + "\n")
            count += 1

    print(f"Labeled {count} clears -> {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Label Rocket League clears. Supports date-sharded raw logs (logs/nexto_clears_YYYY-MM-DD.jsonl)."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--date", type=str, help='Single date to process, format "YYYY-MM-DD" (UTC).')
    group.add_argument("--range", nargs=2, metavar=("START_DATE", "END_DATE"),
                       help='Date range inclusive, format "YYYY-MM-DD YYYY-MM-DD" (UTC).')
    parser.add_argument("--max-lines", type=int, default=None, help="If set, process at most N lines per file.")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging for skipped lines, etc.")
    parser.add_argument(
        "--legacy",
        action="store_true",
        help="Process legacy file nexto_clears.jsonl and write nexto_clears_labeled.jsonl (backward-compatible).",
    )
    args = parser.parse_args()

    # Legacy mode for backward compatibility
    if args.legacy or (not args.date and not args.range):
        # Legacy input/output co-located with this script
        base_dir = os.path.dirname(__file__)
        input_path = os.path.join(base_dir, "nexto_clears.jsonl")
        output_path = os.path.join(base_dir, "nexto_clears_labeled.jsonl")
        if not os.path.isfile(input_path):
            # If there is a latest shard, prefer that automatically
            if not args.legacy:
                # Try latest sharded file if available
                processed_any = False
                if args.date:
                    start, end = args.date, args.date
                elif args.range:
                    start, end = args.range[0], args.range[1]
                else:
                    start = end = None
                paths = list(iter_shard_paths(start, end, labeled=False))
                for raw_path in paths:
                    date_str = shard_date_from_path(raw_path, labeled=False) or "unknown"
                    out_path = shard_output_path_for_date(date_str, labeled=True)
                    label_file(raw_path, out_path, max_lines=args.max_lines, verbose=args.verbose)
                    processed_any = True
                if processed_any:
                    return
            print(f"Legacy input not found: {input_path}")
            return
        label_file(input_path, output_path, max_lines=args.max_lines, verbose=args.verbose)
        return

    # Date-sharded mode
    if args.date:
        start = end = args.date
    else:
        start, end = args.range[0], args.range[1]

    processed = 0
    for raw_path in iter_shard_paths(start, end, labeled=False):
        date_str = shard_date_from_path(raw_path, labeled=False) or "unknown"
        out_path = shard_output_path_for_date(date_str, labeled=True)
        label_file(raw_path, out_path, max_lines=args.max_lines, verbose=args.verbose)
        processed += 1

    if processed == 0:
        print("No matching shards found in logs/. Ensure logging is generating nexto_clears_YYYY-MM-DD.jsonl")


if __name__ == "__main__":
    main()
