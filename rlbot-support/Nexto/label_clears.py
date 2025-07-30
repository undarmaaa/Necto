import json
import numpy as np

input_path = "nexto_clears.jsonl"
output_path = "nexto_clears_labeled.jsonl"
max_lines = 100

GOAL_Y = 5100  # Approximate y position for goal line
GROUND_Z = 18  # Ball is on ground if z <= this
CAR_MAX_SPEED = 2300  # Unreal units per second

def distance(a, b):
    # Euclidean distance between two points
    return ((a["x"]-b["x"])**2 + (a["y"]-b["y"])**2 + (a["z"]-b["z"])**2) ** 0.5

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

with open(input_path, "r", encoding="utf-8") as infile, open(output_path, "w", encoding="utf-8") as outfile:
    count = 0
    for i, line in enumerate(infile):
        if i >= max_lines:
            break
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except Exception as e:
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
        print(f"DEBUG: clear_time={clear.get('time', None)}, clearing_team={clear['clearing_team']}, blue_score={blue_score}, orange_score={orange_score}, score_diff={score_diff}, who_scored={who_scored}")

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
                        touch["player_index"] != prev_touch["player_index"] or
                        touch["time_seconds"] != prev_touch["time_seconds"]
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

        # Situation at clear time: measure how "bad" it is (e.g., ball in our half, enemies closer to ball)
        initial_state = window[0]
        ball_y = initial_state["ball"]["y"]
        our_side = (clear["clearing_team"] == 0 and ball_y < 0) or (clear["clearing_team"] == 1 and ball_y > 0)
        if our_side:
            score -= 0.5  # Ball is in our half at clear time

        # If enemies are closer to ball at clear time, penalize more
        cars0 = initial_state["cars"]
        min_ally0 = min((distance(car, initial_state["ball"]) for car in cars0 if car["team"] == clear["clearing_team"]), default=float("inf"))
        min_enemy0 = min((distance(car, initial_state["ball"]) for car in cars0 if car["team"] != clear["clearing_team"]), default=float("inf"))
        if min_enemy0 < min_ally0:
            score -= 0.5

        # Add results to output
        data["label"] = label
        data["score"] = score
        data["landing_point"] = landing
        data["min_ally_time"] = min_ally
        data["min_enemy_time"] = min_enemy
        outfile.write(json.dumps(data) + "\n")
        count += 1

print(f"Labeled {count} clears in {output_path}")
