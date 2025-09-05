#!/usr/bin/env python3
"""
Ballchasing.com replay downloader (focused on 3v3)
- Downloads .replay files at scale without needing Rocket League or Windows.
- Auth with BALLCHASING_API_KEY env var (create an API key in your ballchasing.com profile).

Quick start:
  1) Create an API key: https://ballchasing.com/profile
  2) Set env var (Windows cmd):
       setx BALLCHASING_API_KEY your_api_key_here
     or (PowerShell):
       [Environment]::SetEnvironmentVariable("BALLCHASING_API_KEY","your_api_key_here","User")
     or (Linux/macOS):
       export BALLCHASING_API_KEY=your_api_key_here
  3) Run (download latest 3v3 ranked standard replays since 2025-01-01):
       python tools/ballchasing_fetch.py --out-dir replays/3v3 --team-size 3 --playlist ranked-standard --min-date 2025-01-01 --count 500

Notes on complexity and file formats:
- Replay format: Rocket League .replay (binary) is different from your current JSONL logs.
- You do NOT parse .replay yourself; use a library like:
    * carball (Python) to parse into protobuf / pandas for physics/events
    * rattletrap (Haskell) to produce raw JSON, then postprocess in Python
- Work estimate:
    * Fetcher (this script): ~15 minutes to configure/use.
    * Basic parsing to your existing JSONL schema (clear_event + game_window): 1–2 days of mapping with carball.
    * After parsing, you can reuse rlbot-support/Nexto/label_clears.py unchanged to label clears.

API docs: https://ballchasing.com/doc/api

Examples:
  - Latest casual 3v3 across all ranks, 100 replays:
      python tools/ballchasing_fetch.py --out-dir replays/casual3s --playlist unranked --team-size 3 --count 100
  - Ranked Standard Champ2+ in a date range:
      python tools/ballchasing_fetch.py --out-dir replays/ranked3s --playlist ranked-standard --team-size 3 --min-rank c2 --min-date 2025-01-01 --max-date 2025-08-01 --count 2000
  - Dry run to preview:
      python tools/ballchasing_fetch.py --out-dir replays/preview --dry-run --count 50

Tip: Ballchasing supports many filters (rank, season, playlist, uploader, pro, map, etc.). This script exposes common ones and lets you pass raw query parameters as JSON if needed (--extra-params).

"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, Optional, Tuple
from urllib.parse import urljoin

import requests


API_BASE = "https://ballchasing.com/api/"
LIST_ENDPOINT = urljoin(API_BASE, "replays")
DETAIL_ENDPOINT = urljoin(API_BASE, "replays/{id}")
FILE_ENDPOINT = urljoin(API_BASE, "replays/{id}/file")


def get_session(api_key: str, timeout: int = 30) -> requests.Session:
    s = requests.Session()
    # Ballchasing expects Authorization header to be the raw API key (no "Bearer" prefix").
    s.headers.update({"Authorization": f"{api_key}", "Accept": "application/json", "User-Agent": "NectoFetcher/1.0"})
    s.timeout = timeout
    return s


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download Rocket League replays from ballchasing.com")
    p.add_argument("--out-dir", required=True, help="Directory to save .replay files and metadata.json")
    p.add_argument("--count", type=int, default=500, help="Max number of replays to download (default 500)")
    p.add_argument("--team-size", type=int, choices=[1, 2, 3], default=3, help="Team size filter (default 3)")
    p.add_argument(
        "--playlist",
        type=str,
        default="ranked-standard",
        help="Playlist filter (e.g., ranked-standard, unranked, private, hoops, rumble, etc.)",
    )
    p.add_argument("--min-date", type=str, default=None, help='Earliest match date (UTC), format YYYY-MM-DD')
    p.add_argument("--max-date", type=str, default=None, help='Latest match date (UTC), format YYYY-MM-DD')
    p.add_argument("--min-rank", type=str, default=None, help="Minimum rank (e.g., g1,g2,g3,p1,p2,p3,d1,d2,d3,c1,c2,c3, gc1,ssl)")
    p.add_argument("--max-rank", type=str, default=None, help="Maximum rank")
    p.add_argument("--uploader", type=str, default=None, help="Filter by uploader player id (from ballchasing profile)")
    p.add_argument(
        "--sort-by",
        type=str,
        default="created",
        choices=["created", "replay-date"],
        help="Sort criterion (default created). Allowed by API: created, replay-date",
    )
    p.add_argument("--sort-dir", type=str, default="desc", choices=["asc", "desc"], help="Sort direction (default desc)")
    p.add_argument("--sleep", type=float, default=0.5, help="Sleep seconds between downloads to respect rate limits")
    p.add_argument("--dry-run", action="store_true", help="List what would be downloaded, but do not fetch files")
    p.add_argument(
        "--extra-params",
        type=str,
        default=None,
        help='Raw JSON of extra query params to include, e.g. \'{"season": "15"}\'',
    )
    p.add_argument("--api-key", type=str, default=None, help="Override BALLCHASING_API_KEY environment variable")
    p.add_argument("--replay-after", type=str, default=None, help="Replay date lower bound (UTC), format YYYY-MM-DD")
    p.add_argument("--replay-before", type=str, default=None, help="Replay date upper bound (UTC), format YYYY-MM-DD")
    # Preferred API filters for replay date range per ballchasing docs:
    p.add_argument("--replay-date-after", type=str, default=None, help="Replay date lower bound (UTC), format YYYY-MM-DD")
    p.add_argument("--replay-date-before", type=str, default=None, help="Replay date upper bound (UTC), format YYYY-MM-DD")
    return p.parse_args()


def build_query(args: argparse.Namespace, page_count: int = 200, after: Optional[str] = None) -> Dict[str, str]:
    # See ballchasing API docs for all filters
    q: Dict[str, str] = {
        "count": str(page_count),  # max allowed per page is usually 200
        "sort-by": args.sort_by,
        "sort-dir": args.sort_dir,
        "team-size": str(args.team_size),
        "playlist": args.playlist,
    }
    if args.min_date:
        q["min-date"] = args.min_date
    if args.max_date:
        q["max-date"] = args.max_date
    # Legacy/custom keys if provided
    if getattr(args, "replay_after", None):
        q["replay-after"] = args.replay_after
    if getattr(args, "replay_before", None):
        q["replay-before"] = args.replay_before
    # Official replay-date range filters
    if getattr(args, "replay_date_after", None):
        q["replay-date-after"] = args.replay_date_after
    if getattr(args, "replay_date_before", None):
        q["replay-date-before"] = args.replay_date_before
    if args.min_rank:
        q["min-rank"] = args.min_rank
    if args.max_rank:
        q["max-rank"] = args.max_rank
    if args.uploader:
        q["uploader"] = args.uploader
    if after:
        q["after"] = after  # pagination cursor returned by API

    if args.extra_params:
        try:
            extra = json.loads(args.extra_params)
            for k, v in extra.items():
                q[str(k)] = str(v)
        except Exception as e:
            print(f"Warning: failed to parse --extra-params JSON: {e}", file=sys.stderr)
    return q


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(path: str, data: dict) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def sanitize_filename(s: str) -> str:
    return "".join(c for c in s if c.isalnum() or c in ("-", "_", ".", " ")).rstrip()


def infer_filename(meta: dict) -> str:
    # Optional pretty filename: {replay-date}_{blue}-{orange}_{id}.replay
    rid = meta.get("id", "unknown")
    date_str = meta.get("created", meta.get("upload_date", "")) or meta.get("date", "")
    # Try to use match date if available
    dt_part = ""
    for key in ("date", "replay_date", "created", "upload_date"):
        v = meta.get(key)
        if v:
            # API returns ISO8601 e.g. "2025-08-14T12:34:56Z"
            try:
                if v.endswith("Z"):
                    dt = datetime.strptime(v, "%Y-%m-%dT%H:%M:%SZ")
                else:
                    # fallback: allow fractional seconds/timezone naive
                    dt = datetime.fromisoformat(v.replace("Z", ""))
                dt_part = dt.strftime("%Y%m%d-%H%M%S")
                break
            except Exception:
                continue
    blue = meta.get("blue", {}).get("name") or "blue"
    orange = meta.get("orange", {}).get("name") or "orange"
    blue = sanitize_filename(blue)
    orange = sanitize_filename(orange)
    if dt_part:
        return f"{dt_part}_{blue}-vs-{orange}_{rid}.replay"
    return f"{rid}.replay"


def list_replays(session: requests.Session, args: argparse.Namespace, after: Optional[str]) -> Tuple[list, Optional[str]]:
    params = build_query(args, page_count=200, after=after)
    r = session.get(LIST_ENDPOINT, params=params)
    if r.status_code == 401:
        raise RuntimeError(f"List replays failed: 401 Unauthorized. Check BALLCHASING_API_KEY (no 'Bearer' prefix) or pass --api-key. Response: {r.text}")
    if r.status_code != 200:
        raise RuntimeError(f"List replays failed: {r.status_code} {r.text}")
    data = r.json()
    items = data.get("list", []) or data.get("items", []) or []
    next_cursor = data.get("next", None) or data.get("cursor", None)
    return items, next_cursor


def fetch_metadata(session: requests.Session, replay_id: str) -> dict:
    url = DETAIL_ENDPOINT.format(id=replay_id)
    r = session.get(url)
    if r.status_code != 200:
        raise RuntimeError(f"Get replay metadata failed: {r.status_code} {r.text}")
    return r.json()


def download_replay_file(session: requests.Session, replay_id: str, out_path: str) -> None:
    url = FILE_ENDPOINT.format(id=replay_id)
    with session.get(url, stream=True) as r:
        if r.status_code != 200:
            raise RuntimeError(f"Download failed: {r.status_code} {r.text}")
        tmp = f"{out_path}.part"
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)
        os.replace(tmp, out_path)


def main() -> int:
    args = parse_args()
    api_key = args.api_key or os.getenv("BALLCHASING_API_KEY")
    if not api_key:
        print("Error: BALLCHASING_API_KEY not set and no --api-key provided.", file=sys.stderr)
        return 2

    ensure_dir(args.out_dir)
    meta_dir = os.path.join(args.out_dir, "_meta")
    ensure_dir(meta_dir)

    session = get_session(api_key)
    total = 0
    after: Optional[str] = None

    print(f"Query: team-size={args.team_size}, playlist={args.playlist}, min-date={args.min_date}, max-date={args.max_date}, min-rank={args.min_rank}, max-rank={args.max_rank}")
    if args.extra_params:
        print(f"Extra params: {args.extra_params}")

    while total < args.count:
        try:
            items, after = list_replays(session, args, after)
        except Exception as e:
            print(f"List error: {e}", file=sys.stderr)
            break
        if not items:
            print("No more items from API.")
            break

        for it in items:
            if total >= args.count:
                break
            replay_id = it.get("id")
            if not replay_id:
                continue

            # fetch detailed metadata (has team names, dates, etc.)
            try:
                meta = fetch_metadata(session, replay_id)
            except Exception as e:
                print(f"[{replay_id}] metadata error: {e}", file=sys.stderr)
                continue

            filename = infer_filename(meta)
            out_path = os.path.join(args.out_dir, filename)
            meta_path = os.path.join(meta_dir, f"{replay_id}.json")

            if os.path.exists(out_path):
                print(f"[skip] exists: {out_path}")
                # Still ensure meta is saved/updated
                try:
                    save_json(meta_path, meta)
                except Exception as e:
                    print(f"[{replay_id}] failed to save metadata: {e}", file=sys.stderr)
                continue

            print(f"[{total+1}/{args.count}] {replay_id} -> {out_path}")
            if args.dry_run:
                # Save metadata even in dry-run so you can inspect filters/results
                try:
                    save_json(meta_path, meta)
                except Exception as e:
                    print(f"[{replay_id}] failed to save metadata: {e}", file=sys.stderr)
                total += 1
                continue

            try:
                download_replay_file(session, replay_id, out_path)
                save_json(meta_path, meta)
                total += 1
                time.sleep(max(0.0, args.sleep))
            except Exception as e:
                print(f"[{replay_id}] download error: {e}", file=sys.stderr)
                # Backoff slightly on errors to be nice to the API
                time.sleep(1.5)

        if after is None:
            # No further pages
            break

    print(f"Done. Downloaded {total} replays into {args.out_dir}")
    print("Next steps:")
    print("  - Parse .replay to frames with carball:")
    print("      pip install carball")
    print("      # pseudo:")
    print("      # from carball.analysis.analysis_manager import AnalysisManager")
    print("      # proto, df = AnalysisManager().create_analysis(path).get_protobuf_data(), AnalysisManager().df")
    print("  - Map frames to your JSONL schema (clear_event + game_window), then run:")
    print("      python rlbot-support/Nexto/label_clears.py --legacy  # or --date/--range with sharded output")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
