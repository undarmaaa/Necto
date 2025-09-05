#!/usr/bin/env python3
"""
Convert rattletrap JSON replays into raw "clear events" JSONL compatible with rlbot-support/Nexto/label_clears.py.

Input: One or more JSON files produced by rattletrap:
  rattletrap.exe --input match.replay --output match.json

Output: Sharded JSONL at rlbot-support/Nexto/logs/nexto_clears_YYYY-MM-DD.jsonl
  Each line:
    {
      "clear_event": { "time": float_seconds, "clearing_team": 0|1, "blue_score": int, "orange_score": int },
      "game_window": [
        { "time": float, "ball": { "x","y","z","vel_x","vel_y","vel_z" }, "cars": [] },
        ...
      ]
    }

Notes:
- This implementation focuses on ball tracking via "TAGame.RBActor_TA:ReplicatedRBState".
- Car details/last_touch are omitted for robustness; label_clears.py can still operate with empty cars lists.
- Timebase is derived from header.properties.RecordFPS (default 30 if missing).
- A "clear" is detected when ball.y crosses midfield (sign change).
- Minimal velocity threshold is approximated using discrete differences; tweak thresholds via CLI flags.

Usage examples:
  # Single file
  python tools/rattletrap_to_clears.py --json replays/3v3_2020/sample.json

  # Directory (recursively finds *.json)
  python tools/rattletrap_to_clears.py --json-dir replays/3v3_2020

  # Adjust windows/thresholds
  python tools/rattletrap_to_clears.py --json-dir replays/3v3_2020 --pre 1.0 --post 3.0 --min-dy 50
"""

import argparse
import json
import math
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Iterable

# Defaults aligned with earlier converter
DEF_PRE_WINDOW = 1.0   # seconds before the clear event to include
DEF_POST_WINDOW = 3.0  # seconds after the clear event to include
DEF_MIN_DY = 50.0      # minimal absolute delta Y (uu) at crossing between frames to accept as "clear"
DEF_FPS = 30.0         # fallback FPS if header missing


def read_json(path: str) -> Optional[dict]:
    """
    Load rattletrap JSON with a tolerant fallback:
      - Some files may contain trailing commas (non-strict JSON). We strip trailing commas
        before '}' or ']' and try to parse again.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        try:
            return json.loads(text)
        except Exception:
            # Lenient cleanup: remove trailing commas before } or ]
            import re
            cleaned = re.sub(r",\s*([}\]])", r"\1", text)
            # Also collapse any accidental ',]' or ',}' patterns
            cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)
            return json.loads(cleaned)
    except Exception:
        return None


def ensure_logs_dir(base_out_dir: str) -> str:
    os.makedirs(base_out_dir, exist_ok=True)
    return base_out_dir


def default_shard_path(base_out_dir: str) -> str:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    fname = f"nexto_clears_{today}.jsonl"
    return os.path.join(base_out_dir, fname)


def header_properties_map(j: dict) -> Dict[str, Any]:
    """
    Build a simple dict from header.properties.elements which is an array of [name, {value: ...}] pairs.
    """
    out: Dict[str, Any] = {}
    try:
        elems = j["content"]["header"]["body"]["properties"]["elements"]
        for pair in elems:
            if not isinstance(pair, list) or len(pair) != 2:
                continue
            k = pair[0]
            v = pair[1]
            if isinstance(v, dict) and "value" in v:
                out[k] = v["value"]
    except Exception:
        pass
    return out


def extract_fps(j: dict) -> float:
    props = header_properties_map(j)
    try:
        v = props.get("RecordFPS", {}).get("float", None)
        if isinstance(v, (int, float)) and v > 0:
            return float(v)
    except Exception:
        pass
    return DEF_FPS


def extract_initial_scores(j: dict) -> Tuple[int, int]:
    props = header_properties_map(j)
    blue = 0
    orange = 0
    try:
        b = props.get("Team0Score", {}).get("int", None)
        o = props.get("Team1Score", {}).get("int", None)
        if isinstance(b, int):
            blue = b
        if isinstance(o, int):
            orange = o
    except Exception:
        pass
    return blue, orange


def iter_json_files(json_file: Optional[str], json_dir: Optional[str]) -> Iterable[str]:
    if json_file:
        yield json_file
        return
    if json_dir:
        for root, _, files in os.walk(json_dir):
            for f in files:
                if f.lower().endswith(".json"):
                    yield os.path.join(root, f)


def find_ball_actor_id_from_frame(frame: dict) -> Optional[int]:
    """
    Scan a frame's replications for a spawned actor of class Ball_TA and return its actor_id.value.
    """
    reps = frame.get("replications", [])
    for r in reps:
        try:
            val = r["value"]
            if "spawned" in val:
                sp = val["spawned"]
                class_name = sp.get("class_name", "")
                if isinstance(class_name, str) and "Ball_TA" in class_name:
                    actor_id = r["actor_id"]["value"]
                    if isinstance(actor_id, int):
                        return actor_id
        except Exception:
            continue
    return None


def extract_rbstate_location(update_entry: dict) -> Optional[Tuple[float, float, float]]:
    """
    Given an 'updated' replication entry, extract location (x,y,z) if present.
    """
    try:
        name = update_entry.get("name", "")
        if name != "TAGame.RBActor_TA:ReplicatedRBState":
            return None
        rb = update_entry["value"]["rigid_body_state"]
        loc = rb["location"]
        x = float(loc["x"])
        y = float(loc["y"])
        z = float(loc["z"])
        return (x, y, z)
    except Exception:
        return None


def extract_ball_position_from_frame(frame: dict, ball_actor_id: Optional[int]) -> Optional[Tuple[float, float, float]]:
    """
    If ball_actor_id is known, scan replications for that actor and extract updated location.
    Also fallback to initialization location if no update seen yet.
    """
    reps = frame.get("replications", [])
    for r in reps:
        try:
            aid = r["actor_id"]["value"]
        except Exception:
            continue

        # If ball just spawned in this frame with init location
        try:
            val = r["value"]
            if "spawned" in val:
                sp = val["spawned"]
                class_name = sp.get("class_name", "")
                if isinstance(class_name, str) and "Ball_TA" in class_name:
                    init = sp.get("initialization", {})
                    loc = init.get("location", None)
                    if loc and all(k in loc for k in ("x", "y", "z")):
                        return (float(loc["x"]), float(loc["y"]), float(loc["z"]))
        except Exception:
            pass

        # Extract from updated if actor matches known ball id
        if ball_actor_id is not None and aid == ball_actor_id:
            try:
                val = r["value"]
                if "updated" in val:
                    for upd in val["updated"]:
                        xyz = extract_rbstate_location(upd)
                        if xyz is not None:
                            return xyz
            except Exception:
                continue

    return None


def build_ball_timeseries(j: dict, fps: float) -> Tuple[List[float], List[float], List[float], List[float]]:
    """
    Produce sequences t[], x[], y[], z[] for the ball across frames.
    Strategy:
      - Track ball_actor_id from spawn events; update when new spawn encountered (goal resets).
      - For each frame, carry forward last known position if no update.
      - Time advances by 1/fps per frame index.
    """
    frames = j.get("content", {}).get("body", {}).get("frames", [])
    t: List[float] = []
    X: List[float] = []
    Y: List[float] = []
    Z: List[float] = []

    ball_actor_id: Optional[int] = None
    last_xyz: Optional[Tuple[float, float, float]] = None

    for i, fr in enumerate(frames):
        # If a new ball spawn appears, update ball_actor_id immediately
        new_ball_id = find_ball_actor_id_from_frame(fr)
        if new_ball_id is not None:
            ball_actor_id = new_ball_id

        xyz = extract_ball_position_from_frame(fr, ball_actor_id)
        if xyz is None:
            # No new update; carry forward last known
            xyz = last_xyz

        # Advance time index regardless
        t_val = i / max(fps, 1e-6)
        t.append(t_val)

        if xyz is None:
            # Unknown yet; use zeros, will be smoothed out by deltas (no clear detection until known)
            X.append(0.0)
            Y.append(0.0)
            Z.append(0.0)
        else:
            X.append(float(xyz[0]))
            Y.append(float(xyz[1]))
            Z.append(float(xyz[2]))
            last_xyz = xyz

    return t, X, Y, Z


def _quat_to_yaw_z(quat: dict) -> float:
    """
    Convert quaternion dict {w,x,y,z} to yaw around Z in radians.
    """
    try:
        w = float(quat.get("w", 1.0))
        x = float(quat.get("x", 0.0))
        y = float(quat.get("y", 0.0))
        z = float(quat.get("z", 0.0))
        # Yaw around Z (intrinsic), common formula:
        # yaw = atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
        import math as _m
        return float(_m.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))
    except Exception:
        return 0.0


def _extract_rbstate_xyz_yaw_from_update(update_entry: dict) -> Optional[Tuple[float, float, float, float]]:
    """
    From an 'updated' replication entry, extract (x,y,z,yaw) for RBActor state if present.
    """
    try:
        if update_entry.get("name", "") != "TAGame.RBActor_TA:ReplicatedRBState":
            return None
        rb = update_entry["value"]["rigid_body_state"]
        loc = rb["location"]
        x = float(loc["x"])
        y = float(loc["y"])
        z = float(loc["z"])
        yaw = 0.0
        rot = rb.get("rotation", {})
        quat = rot.get("quaternion", None)
        if isinstance(quat, dict):
            yaw = _quat_to_yaw_z(quat)
        return (x, y, z, yaw)
    except Exception:
        return None


def _scan_cars_in_frame(frame: dict, known_car_ids: set) -> List[int]:
    """
    Return list of actor_id (ints) that are Car_TA in this frame's 'spawned' replications.
    """
    ids: List[int] = []
    for r in frame.get("replications", []):
        try:
            val = r["value"]
            if "spawned" in val:
                sp = val["spawned"]
                class_name = sp.get("class_name", "")
                if isinstance(class_name, str) and "Car_TA" in class_name:
                    aid = r["actor_id"]["value"]
                    if isinstance(aid, int):
                        known_car_ids.add(aid)
                        ids.append(aid)
        except Exception:
            continue
    return ids


def build_cars_timeseries(j: dict, fps: float, heuristic_frames: int = 30) -> Tuple[List[List[dict]], Dict[int, int]]:
    """
    Build per-frame car snapshots and assign teams heuristically:
      - Identify Car_TA actors by 'spawned' entries.
      - For each frame, extract (x,y,z,yaw) for known car actors from RBState updates; carry forward last known.
      - Assign team per actor by mean y over the first 'heuristic_frames' frames: y<0 -> team 0, else team 1.
      - Build per-frame cars list with dicts: {team,index,x,y,z,rot_yaw,boost}
        (index is stable per team based on actor id ordering; boost is set to 0.0 placeholder).

    Returns:
      cars_per_frame: list indexed by frame, each a list of car dicts
      team_map: actor_id -> team (0 or 1)
    """
    frames = j.get("content", {}).get("body", {}).get("frames", [])
    # Discover car actor ids
    car_ids: set = set()
    for fr in frames:
        _scan_cars_in_frame(fr, car_ids)

    # Track last known state per car actor
    last_state: Dict[int, Tuple[float, float, float, float]] = {}
    # Collect y history for heuristic team assignment
    y_hist: Dict[int, List[float]] = {aid: [] for aid in car_ids}

    cars_per_frame: List[List[dict]] = []

    for i, fr in enumerate(frames):
        # Extract updates per known car
        states_this_frame: Dict[int, Tuple[float, float, float, float]] = {}

        # If new cars spawned this frame, ensure they are in sets
        _scan_cars_in_frame(fr, car_ids)

        for r in fr.get("replications", []):
            try:
                aid = r["actor_id"]["value"]
            except Exception:
                continue
            if aid not in car_ids:
                continue
            try:
                val = r["value"]
                if "updated" in val:
                    for upd in val["updated"]:
                        xyz_yaw = _extract_rbstate_xyz_yaw_from_update(upd)
                        if xyz_yaw is not None:
                            states_this_frame[aid] = xyz_yaw
            except Exception:
                continue

        # Carry forward last known
        for aid in list(car_ids):
            state = states_this_frame.get(aid, last_state.get(aid))
            if state is not None:
                last_state[aid] = state

        # Build cars list for this frame
        cars_list: List[dict] = []
        for aid, state in last_state.items():
            x, y, z, yaw = state
            cars_list.append({
                "_actor_id": int(aid),  # internal field for building team/index, will be dropped later
                "x": float(x),
                "y": float(y),
                "z": float(z),
                "rot_yaw": float(yaw)
            })
            # Collect y hist for heuristic (only first N frames)
            if i < heuristic_frames:
                y_hist.setdefault(aid, []).append(float(y))

        cars_per_frame.append(cars_list)

    # Heuristic team assignment
    team_map: Dict[int, int] = {}
    for aid in car_ids:
        ys = y_hist.get(aid, [])
        mean_y = sum(ys) / len(ys) if ys else 0.0
        team_map[aid] = 0 if mean_y < 0.0 else 1

    # Build per-team index mapping (stable order by actor id)
    by_team: Dict[int, List[int]] = {0: [], 1: []}
    for aid in sorted(car_ids):
        by_team[team_map[aid]].append(aid)
    index_map: Dict[int, int] = {}
    for team in (0, 1):
        for idx, aid in enumerate(by_team[team]):
            index_map[aid] = idx

    # Inject team/index and drop internal id
    for cars_list in cars_per_frame:
        for car in cars_list:
            aid = car.pop("_actor_id", None)
            team = team_map.get(aid, -1)
            car["team"] = int(team)
            car["index"] = int(index_map.get(aid, 0))

    return cars_per_frame, team_map


def detect_clears_from_y(t: List[float], y: List[float], min_dy: float) -> List[int]:
    """
    Return frame indices where y crosses midfield (sign change) and abs(delta y) >= min_dy.
    """
    idxs: List[int] = []
    for i in range(1, len(y)):
        y0 = y[i - 1]
        y1 = y[i]
        if y0 == 0.0 or y1 == 0.0:
            # Avoid ambiguous boundary; require strict sign change
            continue
        if (y0 < 0.0 and y1 > 0.0) or (y0 > 0.0 and y1 < 0.0):
            if abs(y1 - y0) >= float(min_dy):
                idxs.append(i)
    return idxs


def build_window_records(i_evt: int, t: List[float], X: List[float], Y: List[float], Z: List[float],
                         fps: float, pre_s: float, post_s: float,
                         cars_per_frame: Optional[List[List[dict]]] = None) -> List[dict]:
    """
    Extract frames within [t_evt - pre_s, t_evt + post_s], and compute simple finite-difference velocities.
    """
    if i_evt < 0 or i_evt >= len(t):
        return []

    t_evt = t[i_evt]
    t0 = t_evt - pre_s
    t1 = t_evt + post_s

    # Map time bounds to index bounds
    pre_n = max(0, int(math.floor(t0 * fps)))
    post_n = min(len(t) - 1, int(math.ceil(t1 * fps)))

    frames: List[dict] = []
    for j in range(pre_n, post_n + 1):
        dt = (t[j] - t[j - 1]) if j > 0 else (1.0 / fps)
        dx = (X[j] - X[j - 1]) if j > 0 else 0.0
        dy = (Y[j] - Y[j - 1]) if j > 0 else 0.0
        dz = (Z[j] - Z[j - 1]) if j > 0 else 0.0
        if dt <= 0:
            vx = vy = vz = 0.0
        else:
            vx = dx / dt
            vy = dy / dt
            vz = dz / dt

        cars = []
        if cars_per_frame is not None and 0 <= j < len(cars_per_frame):
            # Shallow copy to avoid accidental mutation downstream
            cars = [{k: v for k, v in car.items()} for car in cars_per_frame[j]]

        frames.append({
            "time": float(t[j]),
            "ball": {
                "x": float(X[j]),
                "y": float(Y[j]),
                "z": float(Z[j]),
                "vel_x": float(vx),
                "vel_y": float(vy),
                "vel_z": float(vz),
            },
            "cars": cars
        })

    return frames


def infer_clearing_team(y_before: float) -> int:
    """
    If y_before < 0 => blue team (0), else orange (1).
    """
    return 0 if y_before < 0.0 else 1


def process_json_to_clears(j: dict, pre: float, post: float, min_dy: float, verbose: bool) -> List[dict]:
    fps = extract_fps(j)
    blue_score, orange_score = extract_initial_scores(j)

    t, X, Y, Z = build_ball_timeseries(j, fps)
    # Build car snapshots and heuristic teams across all frames
    cars_per_frame, _team_map = build_cars_timeseries(j, fps=fps)
    if verbose:
        print(f"[info] frames: {len(t)}, fps: {fps}, cars_tracked: {len(_team_map)}")

    clear_idxs = detect_clears_from_y(t, Y, min_dy=min_dy)
    if verbose:
        print(f"[info] detected crossings: {len(clear_idxs)}")

    records: List[dict] = []
    for idx in clear_idxs:
        # Determine clearing team from the side before crossing
        prev_idx = max(0, idx - 1)
        clearing_team = infer_clearing_team(Y[prev_idx])

        window_frames = build_window_records(idx, t, X, Y, Z, fps=fps, pre_s=pre, post_s=post, cars_per_frame=cars_per_frame)
        if not window_frames:
            continue

        rec = {
            "clear_event": {
                "time": float(t[idx]),
                "clearing_team": int(clearing_team),
                "blue_score": int(blue_score),
                "orange_score": int(orange_score),
            },
            "game_window": window_frames
        }
        records.append(rec)

    return records


def main() -> int:
    ap = argparse.ArgumentParser(description="Convert rattletrap JSON to raw clear events JSONL for Nexto labeler.")
    ap.add_argument("--json", type=str, default=None, help="Path to a single rattletrap JSON file.")
    ap.add_argument("--json-dir", type=str, default=None, help="Directory containing rattletrap JSON files (recursively).")
    ap.add_argument("--out-dir", type=str, default="rlbot-support/Nexto/logs", help="Output logs directory.")
    ap.add_argument("--pre", type=float, default=DEF_PRE_WINDOW, help="Seconds before crossing to include.")
    ap.add_argument("--post", type=float, default=DEF_POST_WINDOW, help="Seconds after crossing to include.")
    ap.add_argument("--min-dy", type=float, default=DEF_MIN_DY, help="Minimum |delta Y| between consecutive frames at crossing, in unreal units.")
    ap.add_argument("--verbose", action="store_true", help="Verbose logging.")
    args = ap.parse_args()

    if not args.json and not args.json_dir:
        print("Error: provide --json or --json-dir")
        return 2

    ensure_logs_dir(args.out_dir)
    shard_path = default_shard_path(args.out_dir)

    total_inputs = 0
    total_events = 0
    with open(shard_path, "a", encoding="utf-8") as out_fh:
        for path in iter_json_files(args.json, args.json_dir):
            total_inputs += 1
            j = read_json(path)
            if j is None:
                if args.verbose:
                    print(f"[skip] {os.path.basename(path)}: failed to read/parse JSON.")
                continue

            try:
                recs = process_json_to_clears(j, pre=args.pre, post=args.post, min_dy=args.min_dy, verbose=args.verbose)
            except Exception as e:
                if args.verbose:
                    print(f"[skip] {os.path.basename(path)}: processing error: {e}")
                continue

            for r in recs:
                out_fh.write(json.dumps(r) + "\n")
            total_events += len(recs)
            if args.verbose:
                print(f"[done] {os.path.basename(path)} -> {len(recs)} events")

    print(f"Processed {total_inputs} inputs, wrote {total_events} clear events -> {shard_path}")
    print("Next step: label the clears with:")
    print("  python rlbot-support/Nexto/label_clears.py --date " + datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
