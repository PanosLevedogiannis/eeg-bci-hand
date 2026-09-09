"""Turn a phone or screen recording of the demo into the clip the site embeds.

    python3 site/prepare_video.py ~/Movies/demo.mov
    python3 site/prepare_video.py demo.mov --start 4 --duration 40
    python3 site/prepare_video.py demo.mov --preset Preset960x540   # smaller

Writes site/assets/hand_demo.mp4 and a poster frame next to it. Rebuild the
page afterwards and the clip is in it:

    python3 site/build_site.py                        # embedded as a data URI
    python3 site/build_site.py --pages --with-trials  # copied to docs/media/

Only tools that ship with macOS are used: avconvert for the transcode (H.264
in an MPEG-4 container, which every browser plays) and qlmanage for the poster.
ffmpeg is not needed and is not installed on this machine.

The self-contained build carries the clip inside the HTML, so keep it short.
Around 30 to 60 seconds is enough to show a few trials, and under 9 MB leaves
room under the 16 MB an Artifact page is allowed. --duration trims it.
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "assets")

# H.264 + AAC in an MPEG-4 container. The HEVC presets make smaller files but
# Chrome and Firefox will not reliably play them, so they are not offered.
PRESETS = ["Preset1280x720", "Preset960x540", "Preset1920x1080", "Preset640x480"]


def run(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        sys.exit("failed: %s\n%s%s" % (" ".join(cmd), p.stdout, p.stderr))
    return p.stdout


def poster_from(video, out_jpg, width):
    """Pull a still out of the clip with Quick Look, then convert it to JPEG."""
    tmp = os.path.join(os.path.dirname(os.path.abspath(out_jpg)), "_poster_tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    try:
        run(["qlmanage", "-t", "-s", str(width), "-o", tmp, video])
        made = glob.glob(os.path.join(tmp, "*.png"))
        if not made:
            return False
        run(["sips", "-s", "format", "jpeg", "-s", "formatOptions", "72",
             made[0], "--out", out_jpg])
        return True
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", help="the recording, any format QuickTime opens")
    ap.add_argument("--preset", default=PRESETS[0], choices=PRESETS,
                    help="output size, default %(default)s")
    ap.add_argument("--start", type=float, default=None, help="skip this many seconds")
    ap.add_argument("--duration", type=float, default=None, help="keep this many seconds")
    ap.add_argument("--out", default=os.path.join(ASSETS, "hand_demo.mp4"))
    ap.add_argument("--no-poster", action="store_true",
                    help="keep the existing poster frame, or go without one")
    args = ap.parse_args()

    if not os.path.exists(args.source):
        sys.exit("no such file: " + args.source)
    os.makedirs(ASSETS, exist_ok=True)

    # avconvert picks the container from the extension, and .m4v is MPEG-4
    tmp_out = os.path.splitext(args.out)[0] + ".m4v"
    cmd = ["avconvert", "--source", args.source, "--output", tmp_out,
           "--preset", args.preset, "--replace", "--multiPass"]
    if args.start is not None:
        cmd += ["--start", str(args.start)]
    if args.duration is not None:
        cmd += ["--duration", str(args.duration)]
    print("transcoding with", args.preset, "...")
    run(cmd)
    os.replace(tmp_out, args.out)          # same container, the name says mp4

    mb = os.path.getsize(args.out) / 1e6
    print("wrote %s, %.1f MB" % (args.out, mb))
    if mb > 9:
        print("  the self-contained build wants under 9 MB: trim it with --duration,\n"
              "  or drop to --preset Preset960x540")

    poster = os.path.splitext(args.out)[0] + ".jpg"
    if not args.no_poster:
        if poster_from(args.out, poster, 1280):
            print("wrote %s, %.0f KB" % (poster, os.path.getsize(poster) / 1e3))
        else:
            print("no poster frame came out of qlmanage; the page will show the "
                  "first video frame instead")

    print("\nnow rebuild:\n  python3 site/build_site.py"
          "\n  python3 site/build_site.py --pages --with-trials")


if __name__ == "__main__":
    main()
