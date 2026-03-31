"""
preprocess.py — One-time preprocessing for train.csv.

Validates SVGs against competition rules, normalizes attributes,
filters bad rows, and saves data/train_clean.csv.

Usage:
    python preprocess.py --input data/train.csv --output data/train_clean.csv
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

SVG_REGEX = re.compile(r"<svg[\s\S]*?</svg>", re.IGNORECASE)


# ── Validation & cleaning ──────────────────────────────────────────────────────

def fix_svg(svg: str) -> str:
    """Add required attributes if missing."""
    if 'xmlns=' not in svg:
        svg = svg.replace("<svg", '<svg xmlns="http://www.w3.org/2000/svg"', 1)
    if 'width=' not in svg:
        svg = svg.replace("<svg", '<svg width="256"', 1)
    if 'height=' not in svg:
        svg = svg.replace("<svg", '<svg height="256"', 1)
    if 'viewBox=' not in svg:
        svg = svg.replace("<svg", '<svg viewBox="0 0 256 256"', 1)
    return svg


def validate_svg(svg: str) -> tuple[bool, str]:
    """Returns (is_valid, reason_if_invalid)."""
    if not svg or not isinstance(svg, str):
        return False, "empty"

    svg = svg.strip()

    if not svg.lower().startswith("<svg"):
        # Try to extract SVG from surrounding text
        m = SVG_REGEX.search(svg)
        if m:
            svg = m.group(0).strip()
        else:
            return False, "no_svg_tag"

    if len(svg) > MAX_SVG_LEN:
        return False, "too_long"

    try:
        root = ET.fromstring(svg)
    except ET.ParseError as e:
        return False, f"parse_error:{e}"

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


def clean_svg(svg: str) -> str:
    """Extract and fix SVG string."""
    svg = svg.strip()
    if not svg.lower().startswith("<svg"):
        m = SVG_REGEX.search(svg)
        if m:
            svg = m.group(0).strip()
    return fix_svg(svg)


# ── Main ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input",  default="data/train.csv")
    p.add_argument("--output", default="data/train_clean.csv")
    return p.parse_args()


def main():
    args = parse_args()

    print(f"Loading {args.input} ...")
    df = pd.read_csv(args.input)
    print(f"Loaded {len(df):,} rows. Columns: {list(df.columns)}")

    # Detect columns
    prompt_col = next((c for c in df.columns if c.lower() in ("prompt", "description", "text", "caption")), None)
    svg_col    = next((c for c in df.columns if c.lower() in ("svg", "svg_code", "output", "completion")), None)

    if not prompt_col or not svg_col:
        raise ValueError(f"Cannot detect prompt/svg columns from: {list(df.columns)}")

    print(f"Prompt column: '{prompt_col}' | SVG column: '{svg_col}'")

    df = df[[prompt_col, svg_col]].rename(columns={prompt_col: "prompt", svg_col: "svg"})
    initial = len(df)

    # Drop nulls
    df = df.dropna(subset=["prompt", "svg"])
    df["prompt"] = df["prompt"].astype(str).str.strip()
    df["svg"]    = df["svg"].astype(str).str.strip()
    after_nulls  = len(df)

    # Drop empty prompts
    df = df[df["prompt"].str.len() > 0]
    after_empty  = len(df)

    # Clean SVG (fix attributes, extract from surrounding text)
    print("Cleaning SVG strings ...")
    df["svg"] = df["svg"].apply(clean_svg)

    # Validate
    print("Validating SVGs ...")
    results  = df["svg"].apply(validate_svg)
    df["_valid"]  = results.apply(lambda x: x[0])
    df["_reason"] = results.apply(lambda x: x[1])

    # Stats on rejection reasons
    reasons = defaultdict(int)
    for r in df[~df["_valid"]]["_reason"]:
        key = r.split(":")[0]  # group by category
        reasons[key] += 1

    valid_df = df[df["_valid"]][["prompt", "svg"]].reset_index(drop=True)

    # Drop duplicates
    before_dedup = len(valid_df)
    valid_df = valid_df.drop_duplicates(subset=["prompt"]).reset_index(drop=True)

    # ── Report ─────────────────────────────────────────────────────────────────
    print("\n" + "="*50)
    print("PREPROCESSING REPORT")
    print("="*50)
    print(f"  Initial rows         : {initial:>8,}")
    print(f"  After null drop      : {after_nulls:>8,}  (-{initial - after_nulls:,})")
    print(f"  After empty prompt   : {after_empty:>8,}  (-{after_nulls - after_empty:,})")
    print(f"  After SVG validation : {len(df[df['_valid']]):>8,}  (-{after_empty - len(df[df['_valid']]):,})")
    print(f"  After dedup          : {len(valid_df):>8,}  (-{before_dedup - len(valid_df):,})")
    print(f"\n  Rejection reasons:")
    for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
        print(f"    {reason:<30} {count:>6,}")

    # Token length stats (rough: chars / 3.5)
    svg_lens = valid_df["svg"].str.len()
    print(f"\n  SVG length stats (chars):")
    print(f"    min={svg_lens.min():.0f}  median={svg_lens.median():.0f}  "
          f"p95={svg_lens.quantile(0.95):.0f}  max={svg_lens.max():.0f}")

    prompt_lens = valid_df["prompt"].str.len()
    print(f"  Prompt length stats (chars):")
    print(f"    min={prompt_lens.min():.0f}  median={prompt_lens.median():.0f}  "
          f"p95={prompt_lens.quantile(0.95):.0f}  max={prompt_lens.max():.0f}")
    print("="*50)

    # Save
    valid_df.to_csv(args.output, index=False)
    print(f"\nSaved {len(valid_df):,} clean rows → {args.output}")


if __name__ == "__main__":
    main()
