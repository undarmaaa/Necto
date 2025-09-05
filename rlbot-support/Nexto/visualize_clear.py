import json
import random
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import os
import argparse
from datetime import datetime
import re
from log_loader import shard_output_path_for_date

def sanitize_label(label):
    # Remove non-alphanumeric characters for safe filenames
    return re.sub(r'[^a-zA-Z0-9_-]', '_', label)

def load_random_clear(jsonl_path):
    with open(jsonl_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    if not lines:
        raise ValueError("No clears found in the file.")
    line = random.choice(lines)
    return json.loads(line)

def plot_clear(clear_data, save_gif=False, gif_path="clear_animation.gif"):
    game_window = clear_data["game_window"]
    label = clear_data.get("label", "Unknown")
    clear_event = clear_data.get("clear_event", {})
    landing_point = clear_data.get("landing_point", None)

    # Field dimensions (Rocket League: 8192 x 10240, but we'll auto-scale)
    fig, ax = plt.subplots(figsize=(8, 6))
    # ax.set_title(f"Clear Visualization - Label: {label}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    # Get all positions for scaling
    all_x = []
    all_y = []
    for frame in game_window:
        all_x.append(frame["ball"]["x"])
        all_y.append(frame["ball"]["y"])
        for car in frame["cars"]:
            all_x.append(car["x"])
            all_y.append(car["y"])
    if landing_point:
        all_x.append(landing_point["x"])
        all_y.append(landing_point["y"])

    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    pad = 500
    ax.set_xlim(min_x - pad, max_x + pad)
    ax.set_ylim(min_y - pad, max_y + pad)

    # Prepare plot elements
    ball_dot, = ax.plot([], [], 'o', color='red', markersize=12, label='Ball')
    car_dots = []
    car_texts = []
    max_cars = max((len(frame["cars"]) for frame in game_window), default=0)
    max_cars = max(1, min(max_cars, 12))
    for i in range(max_cars):  # Dynamically allocate markers for cars (cap at 12)
        dot, = ax.plot([], [], 'o', color='blue', markersize=8)
        car_dots.append(dot)
        text = ax.text(0, 0, "", fontsize=8, color='black')
        car_texts.append(text)
    # Info container outside play zone (above plot)
    label_text = ax.text(0.01, 1.02, "", transform=ax.transAxes, fontsize=8,
                        verticalalignment='bottom', ha='left', bbox=dict(boxstyle="round", fc="w"))
    clearer_text = ax.text(0.8, 1.02, "", transform=ax.transAxes, fontsize=10,
                        verticalalignment='bottom', ha='left')

    # Draw scoreboard (will be updated per frame)
    scoreboard_text = ax.text(
        -0.08, -0.1,
        "",  # Will be set in update()
        transform=ax.transAxes,
        fontsize=10, ha='left', va='bottom', color='red', bbox=dict(boxstyle="round", fc="w", alpha=0.7)
    )

    # Draw goal lines and labels
    field_xmin, field_xmax = ax.get_xlim()
    # Blue goal at y ≈ -5120, Orange goal at y ≈ +5120
    blue_goal_y = -5120
    orange_goal_y = 5120
    ax.axhline(blue_goal_y, color='blue', linestyle='--', linewidth=2, alpha=0.5)
    ax.axhline(orange_goal_y, color='orange', linestyle='--', linewidth=2, alpha=0.5)
    ax.text((field_xmin + field_xmax) / 2, blue_goal_y - 200, "Blue Goal", color='blue',
            fontsize=10, ha='center', va='top', fontweight='bold', bbox=dict(boxstyle="round", fc="w", alpha=0.7))
    ax.text((field_xmin + field_xmax) / 2, orange_goal_y + 200, "Orange Goal", color='orange',
            fontsize=10, ha='center', va='bottom', fontweight='bold', bbox=dict(boxstyle="round", fc="w", alpha=0.7))

    # Draw landing point marker in play zone if available
    landing_text = None
    if landing_point:
        ax.plot(landing_point["x"], landing_point["y"], 'x', color='red', markersize=10, label='Landing Point')
    ax.legend()

    def init():
        ball_dot.set_data([], [])
        for dot in car_dots:
            dot.set_data([], [])
        for text in car_texts:
            text.set_text("")
        label_text.set_text("")
        elements = [ball_dot] + car_dots + car_texts + [label_text, clearer_text, scoreboard_text]
        if landing_text:
            elements.append(landing_text)
        return elements

    # Find the frame index closest to the clear event time
    clear_time = clear_event.get("time", None)
    clear_idx = None
    if clear_time is not None:
        times = [abs(frame["time"] - clear_time) for frame in game_window]
        clear_idx = times.index(min(times))

    pause_frames = 20  # Number of extra frames to pause at clear event

    def update(anim_idx):
        # Map anim_idx to the correct frame (pause at clear event)
        if clear_idx is not None and anim_idx > clear_idx:
            if anim_idx <= clear_idx + pause_frames:
                frame_idx = clear_idx
            else:
                frame_idx = anim_idx - pause_frames
        else:
            frame_idx = anim_idx
        if frame_idx >= len(game_window):
            frame_idx = len(game_window) - 1
        frame = game_window[frame_idx]
        # Ball
        ball_dot.set_data(frame["ball"]["x"], frame["ball"]["y"])
        # Cars
        used = min(len(frame["cars"]), len(car_dots))
        for i in range(used):
            car = frame["cars"][i]
            color = "blue" if car.get("team", 0) == 0 else "orange"
            car_dots[i].set_data(car.get("x", 0.0), car.get("y", 0.0))
            car_dots[i].set_color(color)
            car_texts[i].set_position((car.get("x", 0.0), car.get("y", 0.0)))
            display_name = car.get("name") or f"Car {car.get('team', 0)}-{car.get('index', i)}"
            car_texts[i].set_text(display_name)
        # Hide unused car dots/texts
        for j in range(used, len(car_dots)):
            car_dots[j].set_data([], [])
            car_texts[j].set_text("")
        # Label and info (with colored clearer line)
        clearer_name = clear_event.get('clearing_player_name', 'N/A')
        clearer_team = None
        for car in frame["cars"]:
            if car.get("name") == clearer_name:
                clearer_team = car.get("team")
                break
        if clearer_team == 0:
            clearer_color = "blue"
        elif clearer_team == 1:
            clearer_color = "orange"
        else:
            clearer_color = "black"
        # Show label, time, and score (time is back in main label area)
        clear_time_safe = clear_event.get("time", frame.get("time", 0.0))
        label_text.set_text(
            f"Label: {label}\n"
            f"Time: {frame.get('time', 0.0):.2f}\n"
            f"Clear time: {clear_time_safe:.2f}\n"
            f"Score: {clear_data.get('score', 'N/A')}"
        )
        clearer_text.set_text(f"Clearer: {clearer_name}")
        clearer_text.set_color(clearer_color)
        # Update scoreboard per frame
        blue_score = frame.get("blue_score", clear_event.get("blue_score", 0))
        orange_score = frame.get("orange_score", clear_event.get("orange_score", 0))
        scoreboard_text.set_text(f"Blue: {blue_score} | Orange: {orange_score}")
        # Highlight the clearer at the clear event frame
        if clear_idx is not None and frame_idx == clear_idx:
            # Find the clearer in this frame
            clearer_name = clear_event.get("clearing_player_name", None)
            for i, car in enumerate(frame["cars"]):
                if car.get("name") == clearer_name:
                    # Draw a circle around the clearer
                    circle = plt.Circle((car["x"], car["y"]), 200, color='red', fill=False, linewidth=3, zorder=10)
                    # Remove previous circle if any
                    [p.remove() for p in ax.patches]
                    ax.add_patch(circle)
                    break
        else:
            # Remove highlight circle if not at clear event
            [p.remove() for p in ax.patches]
        return [ball_dot] + car_dots + car_texts + [label_text, scoreboard_text, clearer_text]

    total_frames = len(game_window)
    if clear_idx is not None:
        total_frames += pause_frames

    ani = animation.FuncAnimation(
        fig, update, frames=total_frames, init_func=init,
        blit=True, interval=100, repeat=False
    )

    if save_gif:
        ani.save(gif_path, writer='pillow')
        print(f"Animation saved to {gif_path}")
    else:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize a labeled Rocket League clear and optionally save as GIF.")
    parser.add_argument("--gif", action="store_true", help="Save the animation as a GIF instead of displaying it.")
    parser.add_argument("--output", type=str, default=None, help="Output GIF filename (overrides folder/timestamp logic if set)")
    parser.add_argument("--folder", type=str, default="clear_gifs", help="Output folder for GIFs (default: clear_gifs)")
    parser.add_argument("--max-clears", type=int, default=None, help="If set, create GIFs for the first N clears in the file (or first N matching labels if --labels is used)")
    parser.add_argument("--start-index", type=int, default=None, help="Start index (1-based, inclusive) of clears to visualize in batch mode (applies after label filtering)")
    parser.add_argument("--end-index", type=int, default=None, help="End index (1-based, inclusive) of clears to visualize in batch mode (applies after label filtering)")
    parser.add_argument("--date", type=str, default=None, help='Select labeled shard by date "YYYY-MM-DD" (UTC) from logs/')
    parser.add_argument("--labels", nargs="+", type=str, default=None, help="Filter clears by label(s), e.g., --labels good_pass trapped")
    parser.add_argument("--count", type=int, default=None, help="Randomly sample this many clears from the filtered set")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility of sampling")
    args = parser.parse_args()

    # Path to labeled clears file
    base_dir = os.path.dirname(__file__)
    if args.date:
        jsonl_path = shard_output_path_for_date(args.date, labeled=True)
    else:
        jsonl_path = os.path.join(base_dir, "nexto_clears_labeled.jsonl")
    if not os.path.isfile(jsonl_path):
        print(f"File not found: {jsonl_path}")
        exit(1)

    # Optional reproducibility
    if args.seed is not None:
        random.seed(args.seed)

    # Load and optionally filter lines by label(s)
    with open(jsonl_path, "r", encoding="utf-8") as f:
        all_lines = f.readlines()

    lines = all_lines
    if args.labels:
        wanted = {s.lower() for s in args.labels}
        filtered = []
        for ln in all_lines:
            try:
                obj = json.loads(ln)
            except Exception:
                continue
            if str(obj.get("label", "")).lower() in wanted:
                filtered.append(ln)
        lines = filtered

    if not lines:
        if args.labels:
            print(f"No matching clears found for labels: {', '.join(args.labels)} in {jsonl_path}")
        else:
            print(f"No clears found in {jsonl_path}")
        exit(1)

    # Selection logic
    selected_lines = None
    offset = 0

    # Random sampling by count from filtered set
    if args.count is not None:
        k = max(0, min(args.count, len(lines)))
        if k == 0:
            print("Requested count is 0; nothing to visualize.")
            exit(0)
        selected_lines = random.sample(lines, k)

    # Range or first-N selection (applies after filtering); only triggers original batch mode when --gif was used previously,
    # but we allow it regardless now.
    elif (args.start_index is not None and args.end_index is not None):
        start = max(args.start_index - 1, 0)
        end = min(args.end_index, len(lines))
        if end <= start:
            print(f"Invalid range after filtering: start={args.start_index}, end={args.end_index}, total={len(lines)}")
            exit(1)
        selected_lines = lines[start:end]
        offset = start

    elif args.max_clears is not None:
        selected_lines = lines[:args.max_clears]
        offset = 0

    # Default: single random clear
    else:
        selected_lines = [random.choice(lines)]
        offset = 0

    # Output/visualization
    if args.gif:
        # Single-selection with explicit output path
        if len(selected_lines) == 1 and args.output:
            try:
                clear_data = json.loads(selected_lines[0])
            except Exception as e:
                print(f"Skipping due to JSON error: {e}")
                exit(1)
            label = clear_data.get('label', 'Unknown')
            print(f"Saving GIF for label: {label} -> {args.output}")
            # Ensure parent directory exists
            out_dir = os.path.dirname(args.output)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            plot_clear(clear_data, save_gif=True, gif_path=args.output)
        else:
            out_folder = os.path.join(os.path.dirname(__file__), args.folder)
            os.makedirs(out_folder, exist_ok=True)

            for i, line in enumerate(selected_lines):
                try:
                    clear_data = json.loads(line)
                except Exception as e:
                    print(f"Skipping clear {offset + i + 1}: {e}")
                    continue
                label = clear_data.get('label', 'Unknown')
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                safe_label = sanitize_label(label)
                gif_path = os.path.join(out_folder, f"{safe_label}_{timestamp}_{offset + i + 1}.gif")
                print(f"Saving GIF for clear {offset + i + 1} with label: {label} -> {gif_path}")
                plot_clear(clear_data, save_gif=True, gif_path=gif_path)

            if len(selected_lines) > 1:
                print(f"Saved {len(selected_lines)} GIF(s) to {out_folder}")
    else:
        # Display one-by-one
        for i, line in enumerate(selected_lines):
            try:
                clear_data = json.loads(line)
            except Exception as e:
                print(f"Skipping visualization {offset + i + 1}: {e}")
                continue
            label = clear_data.get('label', 'Unknown')
            print(f"Visualizing clear {offset + i + 1} with label: {label}")
            plot_clear(clear_data, save_gif=False)
