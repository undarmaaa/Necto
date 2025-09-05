#!/usr/bin/env python3
"""
Batch convert Rocket League .replay files to JSON using rattletrap.

Requirements:
- rattletrap.exe installed (Windows) or rattletrap binary available on PATH.
  Example download (Windows, PowerShell):
    curl -L https://github.com/tfausak/rattletrap/releases/download/12.2.0/rattletrap-12.2.0-windows-x86_64.exe -o tools/bin/rattletrap.exe

Usage examples:
  # Convert a specific folder recursively, writing JSON next to each replay:
  python tools/batch_rattletrap.py --rattletrap "tools/bin/rattletrap.exe" --replay-dir "replays/3v3_2020"

  # Convert, but place all JSON into a separate mirror directory:
  python tools/batch_rattletrap.py --rattletrap "tools/bin/rattletrap.exe" --replay-dir "replays/3v3_2020" --json-out-dir "replays/3v3_2020_json"

  # Dry run to see what would be converted:
  python tools/batch_rattletrap.py --rattletrap "tools/bin/rattletrap.exe" --replay-dir "replays/3v3_2020" --dry-run
"""

import argparse
import os
import subprocess
import sys
from typing import List, Tuple


def find_replays(root: str) -> List[str]:
    out: List[str] = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith(".replay"):
                out.append(os.path.join(dirpath, fn))
    out.sort()
    return out


def out_json_path(replay_path: str, replay_root: str, json_root: str) -> str:
    """
    Map a replay file under replay_root to a JSON path under json_root, preserving relative structure.
    If json_root is None, place JSON next to the replay (replace extension).
    """
    if not json_root:
        base, _ = os.path.splitext(replay_path)
        return base + ".json"
    rel = os.path.relpath(replay_path, replay_root)
    rel_no_ext, _ = os.path.splitext(rel)
    return os.path.join(json_root, rel_no_ext + ".json")


def ensure_parent(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def run_rattletrap(rattletrap: str, replay: str, out_json: str, dry_run: bool = False) -> Tuple[bool, str]:
    ensure_parent(out_json)
    if dry_run:
        return True, f"[dry-run] {replay} -> {out_json}"
    try:
        # rattletrap usage: --input <replay> --output <json>
        cmd = [rattletrap, "--input", replay, "--output", out_json]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            return False, f"[err] {os.path.basename(replay)}: rattletrap failed ({result.returncode}): {result.stderr.strip() or result.stdout.strip()}"
        return True, f"[ok] {replay} -> {out_json}"
    except FileNotFoundError:
        return False, f"[err] rattletrap not found at: {rattletrap}"
    except Exception as e:
        return False, f"[err] {os.path.basename(replay)}: {e}"


def main() -> int:
    ap = argparse.ArgumentParser(description="Batch convert .replay to JSON via rattletrap.")
    ap.add_argument("--rattletrap", required=True, help="Path to rattletrap executable (e.g., tools/bin/rattletrap.exe)")
    ap.add_argument("--replay-dir", required=True, help="Directory containing .replay files (recursively searched).")
    ap.add_argument("--json-out-dir", default=None, help="Directory to place JSON outputs (mirror tree). Default: next to each .replay")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing JSON files (default: skip).")
    ap.add_argument("--dry-run", action="store_true", help="List actions without converting.")
    args = ap.parse_args()

    replays = find_replays(args.replay_dir)
    if not replays:
        print(f"No .replay files found under {args.replay_dir}")
        return 1

    total = len(replays)
    done = 0
    skipped = 0
    failed = 0

    print(f"Found {total} .replay files under {args.replay_dir}")
    for idx, rp in enumerate(replays, start=1):
        outp = out_json_path(rp, args.replay_dir, args.json_out_dir)
        if os.path.exists(outp) and not args.overwrite:
            print(f"[skip] exists: {outp}")
            skipped += 1
            continue
        ok, msg = run_rattletrap(args.rattletrap, rp, outp, dry_run=args.dry_run)
        print(msg)
        if ok:
            done += 1
        else:
            failed += 1

    print(f"Done. converted={done}, skipped={skipped}, failed={failed}, total={total}")
    if failed > 0:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
