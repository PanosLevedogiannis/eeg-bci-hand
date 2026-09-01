"""
=============================================================================
  Appendix figures — downsample per-subject plots for the thesis PDF
=============================================================================
  The analysis writes 300-dpi PNGs (2-4 MB each). Fourteen subjects of
  ERD + time-frequency plots would be ~90 MB, which no Overleaf project
  wants to carry. A thesis page is ~16 cm wide, so anything past roughly
  1800 px adds file size without adding visible detail.

  Copies erd_* and tf_* into the thesis figures/ folder, downsampled and
  flattened onto white (RGBA PNGs can render with black boxes in some
  PDF viewers).

  Usage:
    python prepare_appendix_figures.py --dest /path/to/thesis/figures
=============================================================================
"""

import argparse
import glob
import os

from PIL import Image

HERE     = os.path.dirname(os.path.abspath(__file__))
FIG_DIR  = os.path.join(HERE, "eeg_data", "figures")
DEST_DIR = "/Users/panoslevedogiannis/Downloads/thesis_v7/figures"
MAX_W    = 1800


def convert(src, dest_dir, max_w=MAX_W):
    """
    Line plots (ERD) stay PNG — flat colour areas compress well losslessly
    and sharp text stays crisp. Spectrograms (TF) are continuous-tone, where
    PNG is the wrong format entirely; JPEG cuts them to a fraction with no
    visible difference at this scale.
    """
    im = Image.open(src)

    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        flat = Image.new("RGB", im.size, "white")
        flat.paste(im, mask=im.split()[-1])
        im = flat
    else:
        im = im.convert("RGB")

    if im.width > max_w:
        h = round(im.height * max_w / im.width)
        im = im.resize((max_w, h), Image.LANCZOS)

    base = os.path.basename(src)
    if base.startswith("tf_"):
        out = os.path.join(dest_dir, os.path.splitext(base)[0] + ".jpg")
        im.save(out, "JPEG", quality=85, optimize=True, progressive=True)
    else:
        out = os.path.join(dest_dir, base)
        im.save(out, "PNG", optimize=True)

    return out, os.path.getsize(src), os.path.getsize(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", default=DEST_DIR)
    parser.add_argument("--max-width", type=int, default=MAX_W)
    args = parser.parse_args()

    os.makedirs(args.dest, exist_ok=True)

    srcs = sorted(glob.glob(os.path.join(FIG_DIR, "erd_S*.png"))) + \
           sorted(glob.glob(os.path.join(FIG_DIR, "tf_S*.png")))
    srcs = [s for s in srcs if "comparison" not in os.path.basename(s)]

    before = after = 0
    for s in srcs:
        _, b, a = convert(s, args.dest, args.max_width)
        before += b
        after  += a
        print(f"  {os.path.basename(s):20s} {b/1e6:6.2f} MB -> {a/1e6:5.2f} MB")

    print(f"\n  {len(srcs)} figures   {before/1e6:.1f} MB -> {after/1e6:.1f} MB "
          f"({after/before:.0%} of original)")
    print(f"  written to {args.dest}")


if __name__ == "__main__":
    main()
