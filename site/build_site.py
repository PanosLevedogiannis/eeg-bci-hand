"""Assemble the presentation site.

Two builds from one template:

  python3 site/build_site.py                 -> site/thesis-site.html
      Self-contained. Per-trial participant EEG embedded. This is the file
      published as the private Artifact and shared with the committee.

  python3 site/build_site.py --pages         -> docs/index.html
      For GitHub Pages. The page itself carries no waveforms; it asks for
      demo_data.json at runtime and degrades to the 3D hand alone when that
      file is absent.

  python3 site/build_site.py --pages --with-trials
      Also writes docs/demo_data.json, the released EEG subset. What it
      contains and why it is releasable: site/DATA_STATEMENT.md

The demo clip of the physical hand (assets/hand_demo.mp4 + hand_demo.jpg, made
by prepare_video.py) is embedded as a data URI in the self-contained build and
copied to docs/media/ for Pages. It is optional: without it the panel says so
and the live camera tab still works.

English text comes from i18n/en.json, keyed by the normalised Greek. Keys that
are absent fall back to Greek, and the build reports how many were found.

Aggregate figures (the ERD grand averages, the 14-subject results) are in
every build — they are summary statistics, not recordings.
"""
import argparse, base64, json, mimetypes, os, shutil

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
A = os.path.join(HERE, "assets")

ap = argparse.ArgumentParser()
ap.add_argument("--pages", action="store_true", help="build docs/index.html for GitHub Pages")
ap.add_argument("--with-trials", action="store_true",
                help="also write docs/demo_data.json (publishes participant waveforms)")
ap.add_argument("--video", default=os.path.join(A, "hand_demo.mp4"),
                help="demo clip of the physical hand (default: assets/hand_demo.mp4)")
ap.add_argument("--no-video", action="store_true", help="build without the clip")
args = ap.parse_args()

VIDEO = None if args.no_video else (args.video if os.path.exists(args.video) else None)
POSTER = None
if VIDEO:
    cand = os.path.splitext(VIDEO)[0] + ".jpg"
    POSTER = cand if os.path.exists(cand) else None

demo = json.load(open(os.path.join(A, "demo_data.json"), encoding="utf-8"))
trials = {k: v.pop("trials") for k, v in demo.items()}      # demo is now metadata + ERD only

html = open(os.path.join(HERE, "site.tpl.html"), encoding="utf-8").read()
html = html.replace("/*__DEMO_META__*/{}", json.dumps(demo, ensure_ascii=False, separators=(",", ":")))
html = html.replace("/*__RESULTS__*/[]", open(os.path.join(A, "results.json"), encoding="utf-8").read())

def b64(name):
    with open(os.path.join(A, name), "rb") as f:
        return "data:image/jpeg;base64," + base64.b64encode(f.read()).decode()


def data_uri(path):
    mime = mimetypes.guess_type(path)[0] or "application/octet-stream"
    with open(path, "rb") as f:
        return "data:" + mime + ";base64," + base64.b64encode(f.read()).decode()


# ---- English text -------------------------------------------------------
i18n_path = os.path.join(HERE, "i18n", "en.json")
if os.path.exists(i18n_path):
    i18n = json.load(open(i18n_path, encoding="utf-8"))
    html = html.replace("/*__I18N__*/null",
                        json.dumps(i18n, ensure_ascii=False, separators=(",", ":")))
    print("english: %d passages, %d runtime strings"
          % (len(i18n.get("nodes", {})), len(i18n.get("js", {}))))
else:
    print("note: no", i18n_path, "— the page will be Greek only")

# ---- three.js ------------------------------------------------------------
# Inlined so the page also draws the hand with no network: the defence room
# cannot be counted on. Delete assets/three.min.js to go back to the CDN tag.
#   curl -o site/assets/three.min.js \
#        https://cdnjs.cloudflare.com/ajax/libs/three.js/0.160.1/three.min.js
CDN_TAG = ('<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/'
           '0.160.1/three.min.js"></script>')
three = os.path.join(A, "three.min.js")
if os.path.exists(three):
    assert html.count(CDN_TAG) == 1, "the three.js tag in the template moved"
    lib = open(three, encoding="utf-8").read()
    assert "</script" not in lib, "the library would close its own script tag"
    html = html.replace(CDN_TAG, "<script>\n" + lib + "\n</script>")
    print("three.js: inlined, %.0f KB" % (os.path.getsize(three) / 1e3))
else:
    print("note: no", three, "— the page will load three.js from the CDN and\n"
          "      will not draw the hand without a network connection")

for token, name in (("__IMG_COMPLETE__", "complete.jpg"),
                    ("__IMG_TENDONS__",  "tendons.jpg"),
                    ("__IMG_SERVOS__",   "servos.jpg")):
    html = html.replace(token, b64(name))

# ---- the demo clip of the physical hand ---------------------------------
if VIDEO and args.pages:
    docs_media = os.path.join(ROOT, "docs", "media")
    os.makedirs(docs_media, exist_ok=True)
    shutil.copy2(VIDEO, os.path.join(docs_media, "hand_demo.mp4"))
    html = html.replace("__VIDEO_SRC__", "media/hand_demo.mp4")
    if POSTER:
        shutil.copy2(POSTER, os.path.join(docs_media, "hand_demo.jpg"))
        html = html.replace("__VIDEO_POSTER__", "media/hand_demo.jpg")
    print("video: copied to docs/media/, %.1f MB" % (os.path.getsize(VIDEO) / 1e6))
elif VIDEO:
    mb = os.path.getsize(VIDEO) / 1e6
    html = html.replace("__VIDEO_SRC__", data_uri(VIDEO))
    if POSTER:
        html = html.replace("__VIDEO_POSTER__", data_uri(POSTER))
    print("video: embedded, %.1f MB source -> %.1f MB of base64" % (mb, mb * 4 / 3))
    if mb > 9:
        print("  WARNING: an Artifact page must stay under 16 MB in total. Shorten the clip\n"
              "           or lower the quality with prepare_video.py --preset Preset960x540.")
else:
    print("note: no demo clip — the panel will say so, the camera tab still works")

# an absent clip or poster leaves the token empty, which the page reads as absent
html = html.replace("__VIDEO_SRC__", "").replace("__VIDEO_POSTER__", "")

if args.pages:
    html = html.replace("/*__DEMO_TRIALS__*/null", "null")
    page = ('<!doctype html>\n<html lang="el">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
            '<meta name="description" content="Διπλωματική εργασία: έλεγχος του ρομποτικού χεριού InMoov i2 '
            'μέσω διεπαφής εγκεφάλου-υπολογιστή βασισμένης σε φαντασία κίνησης.">\n'
            + html + "\n</head>\n<body>\n</body>\n</html>\n")
    docs = os.path.join(ROOT, "docs")
    os.makedirs(docs, exist_ok=True)
    out = os.path.join(docs, "index.html")
    open(out, "w", encoding="utf-8").write(page)
    open(os.path.join(docs, ".nojekyll"), "w").close()
    data_path = os.path.join(docs, "demo_data.json")
    if args.with_trials:
        json.dump(trials, open(data_path, "w", encoding="utf-8"), separators=(",", ":"))
        print("wrote", data_path, "— PARTICIPANT WAVEFORMS, read site/DATA_STATEMENT.md")
    elif os.path.exists(data_path):
        print("note:", data_path, "already exists and will be served")
else:
    html = html.replace("/*__DEMO_TRIALS__*/null",
                        json.dumps(trials, ensure_ascii=False, separators=(",", ":")))
    out = os.path.join(HERE, "thesis-site.html")
    open(out, "w", encoding="utf-8").write(html)

print("wrote", out, round(os.path.getsize(out) / 1e6, 2), "MB")
