# qmod/decode_chart.py
import argparse, base64, json
from pathlib import Path
from typing import Any, Optional

def _find_base64(obj: Any) -> Optional[str]:
    """
    Try common layouts:
      1) ChartDTO-like: {"format":"png_base64","data":"..."}
      2) Composite payload: {"chart":{"format":"png_base64","data":"..."}}
      3) Orchestrator: {"visual":{"chart":{...}}} or {"visual":{...}}
      4) Generic: scan shallow dicts/lists for a dict with keys {"format","data"}
    """
    if isinstance(obj, dict):
        # 1) direct ChartDTO
        if "data" in obj and isinstance(obj.get("format"), str):
            return obj["data"]
        # 2) composite payload
        if "chart" in obj and isinstance(obj["chart"], dict) and "data" in obj["chart"]:
            return obj["chart"]["data"]
        # 3) orchestrator-style
        for key in ("visual", "chart", "composite", "rsi", "macd"):
            if key in obj:
                b64 = _find_base64(obj[key])
                if b64:
                    return b64
        # 4) shallow scan
        for v in obj.values():
            b64 = _find_base64(v)
            if b64:
                return b64
    elif isinstance(obj, list):
        for item in obj:
            b64 = _find_base64(item)
            if b64:
                return b64
    return None

def main():
    p = argparse.ArgumentParser(description="Decode base64 chart from JSON to PNG")
    # accept both styles: flags or positional
    p.add_argument("--in", dest="infile", nargs="?", help="Input JSON path")
    p.add_argument("--out", dest="outfile", nargs="?", help="Output PNG path")
    p.add_argument("pos_in", nargs="?", help="(positional) input JSON path")
    p.add_argument("pos_out", nargs="?", help="(positional) output PNG path")
    args = p.parse_args()

    infile = args.infile or args.pos_in
    outfile = args.outfile or args.pos_out
    if not infile or not outfile:
        raise SystemExit("Usage: python -m qmod.decode_chart --in INPUT.json --out OUTPUT.png  (or positional)")

    in_path = Path(infile)
    obj = json.loads(in_path.read_text(encoding="utf-8-sig"))

    b64 = _find_base64(obj)
    if not b64:
        raise ValueError(f"No base64 image data found in {infile}")

    Path(outfile).write_bytes(base64.b64decode(b64))
    print(f"Saved chart to {outfile}")

if __name__ == "__main__":
    main()
