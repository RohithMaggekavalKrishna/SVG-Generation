"""
preprocess_v2.py — Advanced preprocessing for train.csv.

Improvements over v1:
- Normalizes ALL SVGs to width="256" height="256" while PRESERVING the natural
  viewBox. This makes the training signal consistent: the model always sees
  width=256 height=256 and learns to match the coordinate space to the viewBox.
- Strips px/pt units from width/height attributes before normalizing.
- Groups coordinate-space stats so the model sees a clear distribution.
- Removes duplicate SVGs (by svg hash, not just prompt).
- Stricter length filter: removes SVGs under 200 chars (too trivial to learn from).

Usage:
    python preprocess_v2.py --input data/train.csv --output data/train_v2.csv
"""

import argparse
import re
import xml.etree.ElementTree as ET
from collections import defaultdict

import pandas as pd

# ── Competition constraints ────────────────────────────────────────────────────
ALLOWED_TAGS = {
    "svg", "g", "path", "rect", "circle", "ellipse",
    "line", "polyline", "polygon", "defs", "use",
    "symbol", "clippath", "mask", "lineargradient",
    "radialgradient", "stop", "text", "tspan", "title",
    "desc", "style", "pattern", "marker", "filter",
}

DISALLOWED_TAGS = {
    "script", "animate", "animatetransform", "animatemotion",
    "animatecolor", "set", "foreignobject",
}

MAX_SVG_LEN = 16000
MAX_PATHS   = 256
MIN_SVG_LEN = 200  # skip trivially short SVGs

SVG_REGEX = re.compile(r"<svg[\s\S]*?</svg>", re.IGNORECASE)
WH_UNIT   = re.compile(r"([\d.]+)(px|pt|em|rem|%)?")


def strip_units(val: str) -> float:
    """Extract numeric part from '200px', '256pt', '256', etc."""
    m = WH_UNIT.match(val.strip())
    return float(m.group(1)) if m else None


def normalize_svg(svg: str) -> str:
    """
    Set width="256" height="256" on the <svg> root.
    Preserve the existing viewBox (this is critical — the model learns to
    generate coordinates in whatever space the training data uses, and
    the renderer scales them to fill the 256x256 canvas via the viewBox).
    If no viewBox exists, infer it from original width/height.
    """
    # Extract original width/height to infer viewBox if needed
    orig_w = orig_h = None
    w_match = re.search(r'\bwidth=["\']([^"\']+)["\']', svg, re.IGNORECASE)
    h_match = re.search(r'\bheight=["\']([^"\']+)["\']', svg, re.IGNORECASE)
    if w_match:
        orig_w = strip_units(w_match.group(1))
    if h_match:
        orig_h = strip_units(h_match.group(1))

    # Replace width/height with 256
    svg = re.sub(r'\bwidth=["\'][^"\']*["\']', 'width="256"', svg, count=1, flags=re.IGNORECASE)
    svg = re.sub(r'\bheight=["\'][^"\']*["\']', 'height="256"', svg, count=1, flags=re.IGNORECASE)
    if 'width=' not in svg:
        svg = svg.replace("<svg", '<svg width="256"', 1)
    if 'height=' not in svg:
        svg = svg.replace("<svg", '<svg height="256"', 1)

    # Ensure xmlns
    if 'xmlns=' not in svg:
        svg = svg.replace("<svg", '<svg xmlns="http://www.w3.org/2000/svg"', 1)

    # Add viewBox if missing — infer from original dimensions
    has_vb = bool(re.search(r'\bviewBox=', svg, re.IGNORECASE))
    if not has_vb:
        if orig_w and orig_h:
            vb = f'viewBox="0 0 {orig_w:.4g} {orig_h:.4g}"'
        else:
            vb = 'viewBox="0 0 256 256"'
        svg = svg.replace("<svg", f"<svg {vb}", 1)

    return svg


def validate_svg(svg: str) -> tuple[bool, str]:
    if not svg or not isinstance(svg, str):
        return False, "empty"
    svg = svg.strip()
    if not svg.lower().startswith("<svg"):
        m = SVG_REGEX.search(svg)
        if m:
            svg = m.group(0).strip()
        else:
            return False, "no_svg_tag"
    if len(svg) < MIN_SVG_LEN:
        return False, "too_short"
    if len(svg) > MAX_SVG_LEN:
        return False, "too_long"
    try:
        root = ET.fromstring(svg)
    except ET.ParseError as e:
        return False, f"parse_error"
    if not root.tag.lower().endswith("svg"):
        return False, "root_not_svg"
    path_count = 0
    for elem in root.iter():
        local = (elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag).lower()
        if local in DISALLOWED_TAGS:
            return False, f"disallowed_tag:{local}"
        if local not in ALLOWED_TAGS:
            return False, f"unknown_tag:{local}"
        for attr in elem.attrib:
            if attr.lower().startswith("on"):
                return False, f"event_handler:{attr}"
        for attr_name in ("href", "{http://www.w3.org/1999/xlink}href"):
            val = elem.attrib.get(attr_name, "")
            if val.startswith("http") or val.startswith("//"):
                return False, "external_ref"
        if local == "path":
            path_count += 1
    if path_count > MAX_PATHS:
        return False, f"too_many_paths:{path_count}"
    return True, "ok"


def clean_and_normalize(svg: str) -> str:
    svg = svg.strip()
    if not svg.lower().startswith("<svg"):
        m = SVG_REGEX.search(svg)
        if m:
            svg = m.group(0).strip()
    return normalize_svg(svg)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input",  default="data/train.csv")
    p.add_argument("--output", default="data/train_v2.csv")
    return p.parse_args()


def main():
    args = parse_args()

    print(f"Loading {args.input} ...")
    df = pd.read_csv(args.input)
    print(f"Loaded {len(df):,} rows. Columns: {list(df.columns)}")

    prompt_col = next((c for c in df.columns if c.lower() in ("prompt", "description", "text", "caption")), None)
    svg_col    = next((c for c in df.columns if c.lower() in ("svg", "svg_code", "output", "completion")), None)
    if not prompt_col or not svg_col:
        raise ValueError(f"Cannot detect columns from: {list(df.columns)}")

    df = df[[prompt_col, svg_col]].rename(columns={prompt_col: "prompt", svg_col: "svg"})
    initial = len(df)

    df = df.dropna(subset=["prompt", "svg"])
    df["prompt"] = df["prompt"].astype(str).str.strip()
    df["svg"]    = df["svg"].astype(str).str.strip()
    df = df[df["prompt"].str.len() > 0]

    # Normalize SVGs
    print("Normalizing SVGs (width=256, height=256, preserve viewBox) ...")
    df["svg"] = df["svg"].apply(clean_and_normalize)

    # Validate
    print("Validating SVGs ...")
    results      = df["svg"].apply(validate_svg)
    df["_valid"]  = results.apply(lambda x: x[0])
    df["_reason"] = results.apply(lambda x: x[1])

    reasons = defaultdict(int)
    for r in df[~df["_valid"]]["_reason"]:
        reasons[r.split(":")[0]] += 1

    valid_df = df[df["_valid"]][["prompt", "svg"]].reset_index(drop=True)

    # Dedup by both prompt and svg hash
    before_dedup = len(valid_df)
    valid_df = valid_df.drop_duplicates(subset=["prompt"]).reset_index(drop=True)
    valid_df = valid_df.drop_duplicates(subset=["svg"]).reset_index(drop=True)

    # ── ViewBox distribution report ────────────────────────────────────────────
    vb_pattern = re.compile(r'viewBox=["\'][\d.\s-]+\s+([\d.]+)\s+([\d.]+)["\']', re.IGNORECASE)
    vb_sizes = defaultdict(int)
    for svg in valid_df["svg"]:
        m = vb_pattern.search(svg)
        if m:
            w, h = round(float(m.group(1))), round(float(m.group(2)))
            vb_sizes[f"{w}x{h}"] += 1

    print("\n" + "="*55)
    print("PREPROCESSING V2 REPORT")
    print("="*55)
    print(f"  Initial rows         : {initial:>8,}")
    print(f"  After validation     : {len(df[df['_valid']]):>8,}")
    print(f"  After dedup          : {len(valid_df):>8,}")
    print(f"\n  Rejection reasons:")
    for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
        print(f"    {reason:<30} {count:>6,}")
    print(f"\n  ViewBox coordinate spaces (top 10):")
    for k, v in sorted(vb_sizes.items(), key=lambda x: -x[1])[:10]:
        pct = 100 * v / len(valid_df)
        print(f"    {k:<12} {v:>6,}  ({pct:.1f}%)")
    svg_lens = valid_df["svg"].str.len()
    print(f"\n  SVG length: min={svg_lens.min():.0f}  median={svg_lens.median():.0f}"
          f"  p95={svg_lens.quantile(0.95):.0f}  max={svg_lens.max():.0f}")
    print("="*55)

    valid_df.to_csv(args.output, index=False)
    print(f"\nSaved {len(valid_df):,} clean rows → {args.output}")


if __name__ == "__main__":
    main()
