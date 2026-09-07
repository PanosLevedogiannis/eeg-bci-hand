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

Aggregate figures (the ERD grand averages, the 14-subject results) are in
every build — they are summary statistics, not recordings.
"""
import argparse, base64, json, os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
A = os.path.join(HERE, "assets")

# Visitor counting for the public build. Put your GoatCounter code here — the
# word you pick when you sign up at https://www.goatcounter.com — and rebuild.
# Leave it empty and no analytics script is added at all.
# Dashboard: https://<code>.goatcounter.com
ANALYTICS_CODE = os.environ.get("GOATCOUNTER_CODE", "")

ap = argparse.ArgumentParser()
ap.add_argument("--pages", action="store_true", help="build docs/index.html for GitHub Pages")
ap.add_argument("--with-trials", action="store_true",
                help="also write docs/demo_data.json (publishes participant waveforms)")
args = ap.parse_args()

demo = json.load(open(os.path.join(A, "demo_data.json"), encoding="utf-8"))
trials = {k: v.pop("trials") for k, v in demo.items()}      # demo is now metadata + ERD only

html = open(os.path.join(HERE, "site.tpl.html"), encoding="utf-8").read()
html = html.replace("/*__DEMO_META__*/{}", json.dumps(demo, ensure_ascii=False, separators=(",", ":")))
html = html.replace("/*__RESULTS__*/[]", open(os.path.join(A, "results.json"), encoding="utf-8").read())

def b64(name):
    with open(os.path.join(A, name), "rb") as f:
        return "data:image/jpeg;base64," + base64.b64encode(f.read()).decode()

for token, name in (("__IMG_COMPLETE__", "complete.jpg"),
                    ("__IMG_TENDONS__",  "tendons.jpg"),
                    ("__IMG_SERVOS__",   "servos.jpg")):
    html = html.replace(token, b64(name))

if args.pages:
    html = html.replace("/*__DEMO_TRIALS__*/null", "null")
    if ANALYTICS_CODE:
        beacon = ('<script data-goatcounter="https://%s.goatcounter.com/count" '
                  'async src="//gc.zgo.at/count.js"></script>\n' % ANALYTICS_CODE)
        html = html.replace("<!--__ANALYTICS_NOTE__-->",
                            "Μέτρηση επισκέψεων με GoatCounter, χωρίς cookies και χωρίς "
                            "αποθήκευση διευθύνσεων IP.")
        print("analytics: counting to https://%s.goatcounter.com" % ANALYTICS_CODE)
    else:
        beacon = ""
        html = html.replace("<!--__ANALYTICS_NOTE__-->", "")
        print("analytics: none — set GOATCOUNTER_CODE to count visitors "
              "(see site/README.md)")
    page = ('<!doctype html>\n<html lang="el">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
            '<meta name="description" content="Διπλωματική εργασία: έλεγχος του ρομποτικού χεριού InMoov i2 '
            'μέσω διεπαφής εγκεφάλου-υπολογιστή βασισμένης σε φαντασία κίνησης.">\n'
            + beacon + html + "\n</head>\n<body>\n</body>\n</html>\n")
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
    html = html.replace("<!--__ANALYTICS_NOTE__-->", "")
    html = html.replace("/*__DEMO_TRIALS__*/null",
                        json.dumps(trials, ensure_ascii=False, separators=(",", ":")))
    out = os.path.join(HERE, "thesis-site.html")
    open(out, "w", encoding="utf-8").write(html)

print("wrote", out, round(os.path.getsize(out) / 1e6, 2), "MB")
