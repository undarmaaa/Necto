# Nexto Clear Visualizer — Filtering and Random Sampling

This extends `visualize_clear.py` with options to pick which types of clears to visualize (e.g., only `good_pass`) and how many to show (randomly sampled), with reproducibility.

Available labels (from labeling logic):
- good_pass
- trapped
- goal_for
- goal_against

## Dependencies

The visualizer requires:
- matplotlib (for plotting/animation)
- pillow (for GIF saving)

Install:
- Option A: Per-project (recommended)
  - Add to/ensure these lines exist in `rlbot-support/Nexto/requirements.txt`:
    - `matplotlib`
    - `pillow`
  - Then run: `python -m pip install -r rlbot-support/Nexto/requirements.txt`
- Option B: Ad-hoc
  - `python -m pip install matplotlib pillow`

## Input File

By default, the script reads `rlbot-support/Nexto/nexto_clears_labeled.jsonl` (legacy path).
You can also point it to date-sharded logs labeled output via `--date YYYY-MM-DD`.

If you do not have the labeled file yet, create it:
- Legacy single file:
  - `python rlbot-support/Nexto/label_clears.py --legacy`
  - Produces: `rlbot-support/Nexto/nexto_clears_labeled.jsonl`
- Date-sharded:
  - Label a specific date: `python rlbot-support/Nexto/label_clears.py --date 2025-08-27`
  - The visualizer will auto-resolve the labeled shard with `--date` as well.

## New CLI Options (visualize_clear.py)

- `--labels [LABEL ...]`
  - Filter clears by label(s). Case-insensitive.
  - Example: `--labels good_pass trapped`

- `--count N`
  - Randomly sample N clears from the filtered set.
  - If omitted, defaults to a single random clear.

- `--seed S`
  - Random seed for reproducible sampling used with `--count`.

- `--date YYYY-MM-DD`
  - Use the labeled shard for the given date (UTC), resolved via `log_loader.shard_output_path_for_date`.

- Existing options (now applied after label filtering):
  - `--max-clears N` — take first N of the filtered set.
  - `--start-index I --end-index J` — take a 1-based inclusive slice of the filtered set.

- Output options:
  - `--gif` — save output(s) as GIF(s) instead of displaying.
  - `--folder NAME` — output directory for GIFs (default: `clear_gifs`).
  - `--output PATH` — explicit single GIF filename (only used when exactly one clear is selected).
  - Without `--gif`, selections are displayed one-by-one in a window.

## Examples

- Display a single random good_pass:
  ```
  python rlbot-support/Nexto/visualize_clear.py --labels good_pass
  ```

- Save 10 random good_pass clears as GIFs (reproducible), into `clear_gifs/`:
  ```
  python rlbot-support/Nexto/visualize_clear.py ^
    --labels good_pass ^
    --count 10 ^
    --gif ^
    --folder clear_gifs ^
    --seed 1337
  ```

- Save 5 random clears across multiple labels:
  ```
  python rlbot-support/Nexto/visualize_clear.py --labels good_pass trapped --count 5 --gif
  ```

- Use labeled shard for a given date (UTC):
  ```
  python rlbot-support/Nexto/visualize_clear.py --date 2025-08-27 --labels goal_for --count 3 --gif
  ```

- Take a range after filtering (first 25 trapped clears):
  ```
  python rlbot-support/Nexto/visualize_clear.py --labels trapped --start-index 1 --end-index 25 --gif
  ```

- Save exactly one selected clear to a specific file:
  ```
  python rlbot-support/Nexto/visualize_clear.py --labels good_pass --count 1 --gif --output rlbot-support/Nexto/clear_gifs/good_pass_one.gif
  ```

## Notes

- If no clears match the specified labels, the script will exit with a helpful message.
- When multiple selections are displayed (no `--gif`), the script will open a window for each selection sequentially.
- When `--gif` is used without `--output`, files are named: `<label>_<timestamp>_<index>.gif` in the output folder.
- The animation pauses briefly at the clear moment to highlight the clearer and show scoreboard context.
