import json
import hashlib
import os

def round_floats(obj, ndigits=3):
    if isinstance(obj, float):
        return round(obj, ndigits)
    elif isinstance(obj, dict):
        return {k: round_floats(v, ndigits) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [round_floats(x, ndigits) for x in obj]
    else:
        return obj

def hash_clear(record):
    # Hash the clear_event and first/last frame of game_window for uniqueness
    m = hashlib.sha256()
    m.update(json.dumps(record["clear_event"], sort_keys=True).encode("utf-8"))
    window = record.get("game_window", [])
    if window:
        first = round_floats(window[0])
        last = round_floats(window[-1])
        m.update(json.dumps(first, sort_keys=True).encode("utf-8"))
        m.update(json.dumps(last, sort_keys=True).encode("utf-8"))
    return m.hexdigest()

def deduplicate_jsonl(input_path, output_path):
    seen = set()
    with open(input_path, "r", encoding="utf-8") as infile, \
         open(output_path, "w", encoding="utf-8") as outfile:
        for line in infile:
            try:
                record = json.loads(line)
                h = hash_clear(record)
                if h not in seen:
                    seen.add(h)
                    outfile.write(json.dumps(record) + "\n")
            except Exception as e:
                print(f"Skipping invalid line: {e}")

if __name__ == "__main__":
    input_path = os.path.join(os.path.dirname(__file__), "nexto_clears.jsonl")
    output_path = os.path.join(os.path.dirname(__file__), "nexto_clears_deduped.jsonl")
    deduplicate_jsonl(input_path, output_path)
    print(f"Deduplicated file written to {output_path}")
