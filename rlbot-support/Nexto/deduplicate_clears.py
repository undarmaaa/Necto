import os
import json
import argparse
from typing import Set, Tuple

# Sharded log utilities
from log_loader import (
    iter_shard_paths,
    shard_date_from_path,
    deduped_output_path_for_date,
)

def clear_key(record) -> Tuple[int, str, int]:
    """
    Deduplicate by (clearing_team, clearing_player_name, clearing_second)
    """
    clear_event = record.get("clear_event", {})
    team = clear_event.get("clearing_team", None)
    player = clear_event.get("clearing_player_name", None)
    time_sec = int(clear_event.get("time", 0))
    return (team, player, time_sec)

def deduplicate_stream(infile_path: str, outfile_path: str, verbose: bool = False) -> int:
    """
    Read JSONL from infile_path, write unique lines to outfile_path based on clear_key.
    Returns number of unique records written.
    """
    os.makedirs(os.path.dirname(outfile_path), exist_ok=True)
    seen: Set[Tuple[int, str, int]] = set()
    written = 0
    with open(infile_path, "r", encoding="utf-8") as infile, open(outfile_path, "w", encoding="utf-8") as outfile:
        for i, line in enumerate(infile, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except Exception as e:
                if verbose:
                    print(f"[{os.path.basename(infile_path)}] Skipping invalid line {i}: {e}")
                continue
            key = clear_key(record)
            if key not in seen:
                seen.add(key)
                outfile.write(json.dumps(record) + "\n")
                written += 1
    return written

def deduplicate_file(input_path: str, output_path: str, verbose: bool = False):
    count = deduplicate_stream(input_path, output_path, verbose=verbose)
    print(f"Deduplicated {count} clears -> {output_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Deduplicate clears. Supports date-sharded raw logs (logs/nexto_clears_YYYY-MM-DD.jsonl)."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--date", type=str, help='Single date to process, format "YYYY-MM-DD" (UTC).')
    group.add_argument("--range", nargs=2, metavar=("START_DATE", "END_DATE"),
                       help='Date range inclusive, format "YYYY-MM-DD YYYY-MM-DD" (UTC).')
    parser.add_argument("--verbose", action="store_true", help="Verbose logging for skipped lines, etc.")
    parser.add_argument(
        "--legacy",
        action="store_true",
        help="Process legacy file nexto_clears.jsonl and write nexto_clears_deduped.jsonl (backward-compatible).",
    )
    args = parser.parse_args()

    base_dir = os.path.dirname(__file__)

    # Legacy mode or default fallback
    if args.legacy or (not args.date and not args.range):
        legacy_in = os.path.join(base_dir, "nexto_clears.jsonl")
        legacy_out = os.path.join(base_dir, "nexto_clears_deduped.jsonl")
        if os.path.isfile(legacy_in):
            deduplicate_file(legacy_in, legacy_out, verbose=args.verbose)
            return
        # If no legacy file, try latest shard(s) if CLI didn't explicitly set --legacy
        if not args.legacy:
            processed_any = False
            for raw_path in iter_shard_paths(None, None, labeled=False):
                date_str = shard_date_from_path(raw_path, labeled=False) or "unknown"
                out_path = deduped_output_path_for_date(date_str)
                deduplicate_file(raw_path, out_path, verbose=args.verbose)
                processed_any = True
            if processed_any:
                return
        print(f"Legacy input not found: {legacy_in}")
        return

    # Date-sharded mode
    if args.date:
        start = end = args.date
    else:
        start, end = args.range[0], args.range[1]

    processed = 0
    for raw_path in iter_shard_paths(start, end, labeled=False):
        date_str = shard_date_from_path(raw_path, labeled=False) or "unknown"
        out_path = deduped_output_path_for_date(date_str)
        deduplicate_file(raw_path, out_path, verbose=args.verbose)
        processed += 1

    if processed == 0:
        print("No matching shards found in logs/. Ensure logging is generating nexto_clears_YYYY-MM-DD.jsonl")

if __name__ == "__main__":
    main()
